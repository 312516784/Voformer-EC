import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import pairwise_distances
import math
import copy
import os
import gc
import time

# Define the Config class, including all configuration parameters
class Config:
    # Data Directory
    data_dir = '.'
    data_path = 'combined_precipitation_temperature_data.npy'
    
    # Voformer configuration Voformer-Extreme Clustering
    train_voformer = False # Whether to train the Voformer model
    load_voformer = True # Whether to load an existing Voformer model
    voformer_model_path = 'best_voformer_ec_ablation.pth' # Voformer model save path
    
    # Clustering configuration
    perform_clustering = True # Whether to perform clustering
    clustering_results_path = 'clustering_results_ablation.npy' # Clustering result save path
    extracted_features_path = 'extracted_features_ablation.npy' # Extracted feature save path
    
    #  Visualization
    visualize_clusters = True       # Whether to visualize the clustering results
    visualization_output_dir = 'visualizations'  # Visualization output directory
    
    # Voformer parameters
    input_dim = 2
    d_model = 256
    n_heads = 8
    num_layers = 6
    d_ff = 1024
    dropout = 0.2
    batch_size_voformer = 64
    num_epochs = 70
    learning_rate = 1e-5
    
    neighborhood_radius = 5 # Clustering neighborhood radius
    ex_neighborhood_radius = 0.2
    DC_reference_distance = 0.04
    Noise_filtering_threshold = 0.05
    patience = 10

    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = Config()

# Extreme Clustering
def extreme_clustering(features, neighborhood_radius=0.2):
    # Handle both PyTorch tensors and NumPy arrays
    if isinstance(features, np.ndarray):
        data = features
    else:  # Assume PyTorch tensor
        data = features.detach().cpu().numpy()
    
    number = data.shape[0]
    dim = data.shape[1]

    # Calculate the distance matrix
    dist1 = pdist(data, metric='euclidean')
    dist = squareform(dist1)

    # Sorting distance and index
    sorted_indices = np.argsort(dist, axis=1)
    sorted_distances = np.sort(dist, axis=1)

    # Calculate distance
    position = int(round(number * configs.DC_reference_distance)) - 1
    sda = np.sort(dist, axis=0)
    dc = sda[position % number, position // number]

    # Calculate density
    density = np.zeros(number)
    for i in range(number - 1):
        for j in range(i + 1, number):
            tmp = np.exp(-((dist[i, j] / dc) ** 2))
            density[i] += tmp
            density[j] += tmp

    # Finding extreme points
    extreme_points = []
    state = np.zeros(number)
    for i in range(number):
        if state[i] == 0:
            j = 1
            while j < number and density[i] >= density[sorted_indices[i, j]] and sorted_distances[i, j] < neighborhood_radius:
                if density[i] == density[sorted_indices[i, j]]:
                    state[sorted_indices[i, j]] = 1
                j += 1
            if j < number and sorted_distances[i, j] >= neighborhood_radius:
                extreme_points.append(i)

    # Allocation Category
    clustering_result = np.zeros(number, dtype=int) - 1
    for idx, point in enumerate(extreme_points):
        clustering_result[point] = idx + 1
        j = 1
        while j < number and sorted_distances[point, j] < neighborhood_radius:
            if density[point] == density[sorted_indices[point, j]]:
                clustering_result[sorted_indices[point, j]] = idx + 1
            j += 1

    # Allocate the remaining points
    for i in range(number):
        if clustering_result[i] == -1:
            queue = [i]
            while True:
                current_point = queue[-1]
                j = 0
                while j < number and density[current_point] >= density[sorted_indices[current_point, j]]:
                    j += 1
                if j >= number:
                    break  # 防止j超出范围
                if clustering_result[sorted_indices[current_point, j]] == -1:
                    queue.append(sorted_indices[current_point, j])
                else:
                    break
                if len(queue) >= number:
                    break
            if j < number:
                label = clustering_result[sorted_indices[current_point, j]]
                for point in queue:
                    clustering_result[point] = label

    # Remove noise points
    unique_labels, counts = np.unique(clustering_result, return_counts=True)
    mean_count = np.mean(counts[unique_labels != -1])
    noise_labels = unique_labels[counts < mean_count * configs.Noise_filtering_threshold]
    for label in noise_labels:
        clustering_result[clustering_result == label] = -1

    # Renumber
    unique_labels = np.unique(clustering_result)
    label_map = {label: idx for idx, label in enumerate(unique_labels) if label != -1}
    for old_label, new_label in label_map.items():
        clustering_result[clustering_result == old_label] = new_label

    return clustering_result

def compute_clustering_loss(features, alpha=1.0, beta=0.1):
    """
    结合特征分布和聚类结构的损失
    """
    batch_size, feature_dim = features.shape
    
    # 1. 特征紧凑性损失
    feature_mean = features.mean(dim=0, keepdim=True)
    compactness_loss = F.mse_loss(features, feature_mean.expand_as(features))
    
    # 2. 特征多样性损失（避免特征坍塌）
    features_norm = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features_norm, features_norm.T)
    diversity_loss = similarity_matrix.mean()
    
    # 3. 聚类结构损失
    pairwise_distances = torch.cdist(features, features)
    # 鼓励形成明显的聚类结构
    structure_loss = -torch.var(pairwise_distances)
    
    total_loss = alpha * compactness_loss + beta * diversity_loss + 0.01 * structure_loss
    return total_loss

# Voformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Volatilite(nn.Module):
    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        deviation = (mean_x - x) ** 2
        mean_deviation = torch.mean(deviation, dim=1, keepdim=True)  # Shape: (batch_size, 1, ...)
        volatility = torch.sqrt(mean_deviation)  # Shape: (batch_size, 1, ...)
        scaled_x = x * volatility
        return scaled_x

class ProbSparseAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        # Calculate scores (QK^T / sqrt(d_k)) and apply top-k sparsity
        scale = self.scale or 1.0 / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        # Top-k selection (sparse attention focus)
        U_part = L_K
        top_k = max(1, int(U_part / 4))  # Use 25% sparsity
        idx = torch.topk(scores, top_k, dim=-1)[1]  # Get top-k indices
        mask_topk = torch.zeros_like(scores).scatter_(-1, idx, 1.0).bool()
        sparse_scores = scores.masked_fill(~mask_topk, -float('inf'))

        # Apply softmax, dropout, and compute attention outputs
        attn = self.dropout(torch.softmax(sparse_scores, dim=-1))
        outputs = torch.matmul(attn, values)

        return outputs

class InformerAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InformerAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / (self.d_k ** 0.5)

        self.qkv_projection = nn.Linear(d_model, d_model * 3)  # Query, key, value
        self.out_projection = nn.Linear(d_model, d_model)

        self.attention = ProbSparseAttention(scale=self.scale, attention_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape  # Batch size, Sequence length, Feature dimension (d_model)

        # Compute Q, K, V
        qkv = self.qkv_projection(x).view(B, L, 3, self.n_heads, self.d_k)
        queries, keys, values = qkv.unbind(dim=2)  # Split Q, K, V (B, L, H, Dk)

        # Reshape for multi-head attention
        queries = queries.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)
        keys = keys.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)
        values = values.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)

        # Apply ProbSparseAttention
        outputs = self.attention(queries, keys, values, mask)

        # Reshape back to original dimensions
        outputs = outputs.view(B, self.n_heads, L, self.d_k).permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(B, L, -1)  # (B, L, d_model)

        # Final projections with dropout and residual connection
        outputs = self.dropout(self.out_projection(outputs))
        outputs = self.norm(outputs + x)  # Residual connection ensures same shape (B, L, d_model)

        return outputs

class VoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(VoformerEncoderLayer, self).__init__()
        self.self_attn = InformerAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Volatilite(),  # Volatilite Activation Function
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, mask=src_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + src2
        return src

class VoformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(VoformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)
        return self.norm(src)

class Voformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, d_ff, dropout=0.1):
        super(Voformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = VoformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = VoformerEncoder(encoder_layer, num_layers)
        self.feature_projection = nn.Linear(configs.d_model, configs.d_model)
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        return x

# Voformer Training
# Training Voformer
def train_voformer_ec(configs):
    # Loading data
    data = np.load(configs.data_path)

    # Data preprocessing
    num_samples, seq_length, num_features = data.shape
    data_reshaped = data.reshape(num_samples, seq_length * num_features)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    # Reshape data_scaled back to (2415, 276, 2)
    data_scaled_3d = data_scaled.reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled_3d, dtype=torch.float32) # (2415, 276, 2)

    # Creating a dataset and data loader
    dataset = TensorDataset(X)
    # Divide the training set, validation set and test set (for example, 70% training, 15% validation, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_voformer, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size_voformer, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size_voformer, shuffle=False)

    # Initialize the model
    model = Voformer(configs.input_dim, configs.d_model, configs.n_heads, configs.num_layers, configs.d_ff, configs.dropout)
    model = model.to(configs.device)
    
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()
                           
    for epoch in range(configs.num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            batch_X = batch[0].to(configs.device)

            optimizer.zero_grad()
            features = model(batch_X)
            
            loss = compute_clustering_loss(features.mean(dim=1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)

        # Verification
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_X = batch[0].to(configs.device)
                features = model(batch_X)
                
                loss = compute_clustering_loss(features.mean(dim=1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{configs.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), configs.voformer_model_path)
            print("保存当前最佳Voformer模型")
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print("早停触发，停止训练...")
                break

    print("Voformer model training completed and saved")
    return model

# Voformer-EC
# Loading Voformer
def load_voformer_model(configs):
    model = Voformer(configs.input_dim, configs.d_model, configs.n_heads, configs.num_layers, configs.d_ff, configs.dropout)
    model.load_state_dict(torch.load(configs.voformer_model_path, map_location=configs.device))
    model = model.to(configs.device)
    model.eval()
    print("Voformer loaded")
    return model

# Extract features by Voformer
def extract_features_with_voformer_ec(data, model, configs):
    """使用训练好的Voformer-EC模型提取特征并进行聚类"""
    model.eval()
    features_list = []
    
    # 数据预处理
    num_samples, seq_length, num_features = data.shape
    data_reshaped = data.reshape(num_samples, seq_length * num_features)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    data_scaled_3d = data_scaled.reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled_3d, dtype=torch.float32)
    
    # 创建数据加载器
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size_voformer, shuffle=False)
    
    # 提取特征
    with torch.no_grad():
        for batch in dataloader:
            batch_X = batch[0].to(configs.device)
            batch_features = model(batch_X)
            features_list.append(batch_features.cpu().numpy())
    
    # 合并所有特征
    features = np.concatenate(features_list, axis=0)
    
    features_2d = features.mean(axis=1)
    print(f"特征形状转换: {features.shape} -> {features_2d.shape}")
    # 执行极值聚类
    print("开始执行Extreme Clustering...")
    clustering = extreme_clustering(features_2d, neighborhood_radius=configs.ex_neighborhood_radius)
    
    return features_2d, clustering

def within_cluster_ss(X, labels, centroids):
    """计算簇内平方和（惯性）"""
    inertia = 0
    for i in range(len(centroids)):
        cluster_mask = labels == i
        if np.sum(cluster_mask) > 0:  # 确保簇不为空
            inertia += np.sum((X[cluster_mask] - centroids[i]) ** 2)
    return inertia

def mean_distance_to_nearest_cluster_member(X, labels):
    """计算到最近簇成员的平均距离"""
    unique_labels = np.unique(labels[labels != -1])  # 排除噪声点
    mean_distances = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            mean_distances.append(np.mean(min_distances))
    
    return np.mean(mean_distances) if mean_distances else 0

def dunn_index(X, labels, centroids):
    """计算邓恩指数"""
    unique_labels = np.unique(labels[labels != -1])  # 排除噪声点
    num_clusters = len(unique_labels)
    
    if num_clusters < 2:
        return 0
    
    inter_cluster_dists = pdist(centroids, 'euclidean')
    inter_cluster_dists = squareform(inter_cluster_dists)
    np.fill_diagonal(inter_cluster_dists, np.inf)
    
    min_inter_cluster_dist = np.min(inter_cluster_dists)
    
    max_intra_cluster_dist = 0
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        
        if len(cluster_points) <= 1:
            continue
        
        intra_cluster_dists = pdist(cluster_points, 'euclidean')
        max_dist = np.max(intra_cluster_dists) if len(intra_cluster_dists) > 0 else 0
        
        if max_dist > max_intra_cluster_dist:
            max_intra_cluster_dist = max_dist
    
    if max_intra_cluster_dist == 0:
        return float('inf')
    
    dunn = min_inter_cluster_dist / max_intra_cluster_dist
    return dunn

def calculate_centroids(X, labels):
    """计算每个簇的中心点"""
    unique_labels = np.unique(labels[labels != -1])  # 排除噪声点
    centroids = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    return np.array(centroids)

def evaluate_clustering(features, labels):
    """评估聚类结果"""
    # 过滤噪声点
    valid_indices = labels != -1
    if np.sum(valid_indices) < 2:
        print("警告：有效聚类点数量不足，无法计算评估指标")
        return {}
    
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]
    
    # 检查聚类数量
    n_clusters = len(np.unique(valid_labels))
    if n_clusters < 2:
        print("警告：聚类数量不足，无法计算评估指标")
        return {}
    
    try:
        # 计算质心
        centroids = calculate_centroids(features, labels)
        
        # 计算传统聚类评估指标
        silhouette = silhouette_score(valid_features, valid_labels)
        davies_bouldin = davies_bouldin_score(valid_features, valid_labels)
        calinski_harabasz = calinski_harabasz_score(valid_features, valid_labels)
        
        # 计算新增的三个指标
        inertia = within_cluster_ss(features, labels, centroids)
        mean_nearest_distance = mean_distance_to_nearest_cluster_member(features, labels)
        dunn = dunn_index(features, labels, centroids)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'within_cluster_ss': inertia,
            'mean_nearest_distance': mean_nearest_distance,
            'dunn_index': dunn,
            'n_clusters': n_clusters,
            'n_noise_points': np.sum(labels == -1),
            'n_valid_points': np.sum(valid_indices)
        }
        
        print(f"聚类评估结果:")
        print(f"  轮廓系数 (Silhouette Score): {silhouette:.4f}")
        print(f"  Davies-Bouldin指数: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz指数: {calinski_harabasz:.4f}")
        print(f"  簇内平方和 (Within-cluster SS): {inertia:.4f}")
        print(f"  最近簇成员平均距离: {mean_nearest_distance:.4f}")
        print(f"  邓恩指数 (Dunn Index): {dunn:.4f}")
        print(f"  聚类数量: {n_clusters}")
        print(f"  噪声点数量: {np.sum(labels == -1)}")
        print(f"  有效点数量: {np.sum(valid_indices)}")
        
        return metrics
    except Exception as e:
        print(f"计算聚类评估指标时出错: {e}")
        return {}

# Voformer-EC visualization
def visualize_clustering_results(features, labels, configs, save_path=None):
    """可视化聚类结果"""
    # 确保输出目录存在
    if save_path is None:
        os.makedirs(configs.visualization_output_dir, exist_ok=True)
        save_path = os.path.join(configs.visualization_output_dir, 'clustering_results.png')
    
    # 使用PCA进行降维以便可视化
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        print(f"PCA降维完成，解释方差比: {pca.explained_variance_ratio_}")
    else:
        features_2d = features
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 获取唯一标签
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 绘制每个聚类
    for i, label in enumerate(unique_labels):
        if label == -1:
            # 噪声点用黑色表示
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.6, label=f'Noise')
        else:
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {label}')
    
    plt.title('Voformer-EC Result', fontsize=16)
    plt.xlabel('Feature 1' if features.shape[1] > 2 else 'First principal component', fontsize=12)
    plt.ylabel('Feature 2' if features.shape[1] > 2 else 'Second principal component', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"聚类可视化结果已保存到: {save_path}")

def plot_training_history(train_losses, val_losses, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.title('Voformer-EC Training Track', fontsize=16)
    plt.xlabel('Train count', fontsize=12)
    plt.ylabel('Loss value', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def perform_clustering_with_voformer_ec(model, data, configs):
    """执行完整的聚类流程"""
    print("开始提取特征并执行聚类...")
    
    # 提取特征和执行聚类
    features, clustering_results = extract_features_with_voformer_ec(data, model, configs)
    
    # 保存结果
    np.save(configs.extracted_features_path, features)
    np.save(configs.clustering_results_path, clustering_results)
    print(f"特征已保存到: {configs.extracted_features_path}")
    print(f"聚类结果已保存到: {configs.clustering_results_path}")
    
    # 评估聚类结果
    metrics = evaluate_clustering(features, clustering_results)
    
    # 可视化结果
    if configs.visualize_clusters:
        visualize_clustering_results(features, clustering_results, configs)
    
    return clustering_results, metrics

def save_results_summary(configs, metrics, clustering_results):
    """保存结果摘要"""
    summary = {
        'model_config': {
            'input_dim': configs.input_dim,
            'd_model': configs.d_model,
            'n_heads': configs.n_heads,
            'num_layers': configs.num_layers,
            'neighborhood_radius': configs.ex_neighborhood_radius,
            'DC_reference_distance': configs.DC_reference_distance,
            'Noise_filtering_threshold': configs.Noise_filtering_threshold
        },
        'clustering_metrics': metrics,
        'clustering_summary': {
            'total_points': len(clustering_results),
            'unique_clusters': len(np.unique(clustering_results[clustering_results != -1])),
            'noise_points': np.sum(clustering_results == -1)
        }
    }

def main():
    """主函数"""
    configs = Config()
    
    print("=== Voformer-EC 纯聚类模型 ===")
    print(f"设备: {configs.device}")
    print(f"数据路径: {configs.data_path}")
    
    # 加载数据
    if not os.path.exists(configs.data_path):
        print(f"错误: 数据文件 {configs.data_path} 不存在!")
        return
    
    print("加载数据...")
    data = np.load(configs.data_path)
    print(f"数据形状: {data.shape}")
    
    # 训练或加载Voformer-EC模型
    if configs.train_voformer:
        print("开始训练Voformer-EC模型...")
        voformer_model = train_voformer_ec(configs)
    else:
        if os.path.exists(configs.voformer_model_path):
            print("加载预训练的Voformer-EC模型...")
            voformer_model = load_voformer_model(configs)
        else:
            print(f"模型文件 {configs.voformer_model_path} 不存在，开始训练...")
            voformer_model = train_voformer_ec(configs)
    
    # 执行聚类
    if configs.perform_clustering:
        print("\n=== 开始聚类分析 ===")
        clustering_results, metrics = perform_clustering_with_voformer_ec(voformer_model, data, configs)
        
        # 保存结果摘要
        save_results_summary(configs, metrics, clustering_results)
        
        print("\n=== 聚类分析完成 ===")
        print("所有结果文件已保存完成!")
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("程序执行完成!")

if __name__ == "__main__":
    main()
