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
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import math
import copy
import os
import gc
import time
import random

# Define the Config class, including all configuration parameters
class Config:
    # Data Directory
    data_dir = '.'
    data_path = 'combined_precipitation_temperature_data.npy'
    
    # Voformer configuration Voformer-Extreme Clustering
    train_voformer = True # Whether to train the Voformer model
    load_voformer = False # Whether to load an existing Voformer model
    voformer_model_path = 'best_voformer_ec_updated_1.pth' # Voformer model save path
    
    # Clustering configuration
    perform_clustering = True # Whether to perform clustering
    clustering_results_path = 'clustering_results_updated_1.npy' # Clustering result save path
    extracted_features_path = 'extracted_features_updated_1.npy' # Extracted feature save path
    
    #  Visualization
    visualize_clusters = True       # Whether to visualize the clustering results
    visualization_output_dir = 'visualizations'  # Visualization output directory
    
    # Voformer parameters
    n_clusters = 11
    input_dim = 2
    d_model = 256
    n_heads = 4
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    proj_dim = 128
    
    batch_size_voformer = 256
    num_epochs = 200
    learning_rate = 1e-4
    warmup_epochs = 20

    # Loss Weights
    gamma = 1.0         # Clustering KL weight
    lambda_cl = 1.0     # Contrastive Loss weight
    
    neighborhood_radius = 5 # Clustering neighborhood radius
    ex_neighborhood_radius = 0.2
    DC_reference_distance = 0.04
    Noise_filtering_threshold = 0.001
    
    patience = 10

    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = Config()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"The random seed has been fixed as: {seed}")
seed_everything(2026)

## Extreme Clustering
def extreme_clustering(features, percentile=2.0):
    dists = pdist(features, metric='euclidean')
    
    # Dynamically calculate dc (Cut-off distance)
    # Instead of using a fixed 0.2, it takes the top 2% quantiles of the distance distribution.
    # This ensures that the density definition is always reasonable regardless of the feature value.
    dc = np.percentile(dists, percentile)
    
    dists_matrix = squareform(dists)
    num_samples = features.shape[0]
    
    # 1. Calculate the local density rho
    # Using the Gaussian Kernel will result in a smoother and more robust texture than the original Step Kernel.
    rho = np.zeros(num_samples)
    for i in range(num_samples):
        # Gaussian kernel: exp(-(d/dc)^2)
        rho[i] = np.sum(np.exp(-(dists_matrix[i] / dc) ** 2))
        
    # 2. Calculate delta (the nearest distance to higher density points).
    delta = np.zeros(num_samples)
    max_dist = np.max(dists_matrix)
    
    # Sort by density in descending order
    rho_order = np.argsort(-rho)
    
    for i in range(num_samples):
        idx = rho_order[i]
        if i == 0:
            delta[idx] = max_dist
        else:
            # Among people with a higher density than it, the one closest to it
            higher_density_indices = rho_order[:i]
            delta[idx] = np.min(dists_matrix[idx, higher_density_indices])
            
    # 3. Determine cluster centers
    # gamma_score = rho * delta
    gamma_score = rho * delta
    
    # Automatically find centers using the gap, or directly return `gamma_score` for external filtering
    # For simplicity, we'll use the logic of `initialize_voformer_centers` to truncate the data.
    # We only need to return 'pseudo-labels' sorted by gamma for the initialization function to reference.
    # Here we still return preliminary labels.
    # Simple strategy: take the K points with the largest gamma values ​​as centers (K can be defined externally; we'll tentatively use a loose value here).
    # But actually, `initialize_voformer_centers` only uses it to determine centers, so we don't need to fix it here.
    # To maintain interface compatibility, return labels assigned based on nearest neighbors.
    # These labels are just to provide the `initialize` function with clues about "which are the centers".
    # In fact, the rewritten `initialize` function is more intelligent.
    # Let's return a compatible array of labels.
    # We take the 15 points with the largest gamma values ​​as candidate centers.
    n_candidates = 15 
    centers_idx = np.argsort(-gamma_score)[:n_candidates]
    
    labels = np.full(num_samples, -1, dtype=int)
    for i, center_id in enumerate(centers_idx):
        labels[center_id] = i # Mark center
        
    # Assign the remaining points to the nearest center (simple Voronoi partition, used only for initialization).
    for i in range(num_samples):
        if labels[i] == -1:
            dists_to_centers = dists_matrix[i, centers_idx]
            labels[i] = np.argmin(dists_to_centers)
            
    return labels

## Voformer
# 1. Activation & Encoding
class NNVAF(nn.Module):
    def __init__(self, d_hidden):
        super(NNVAF, self).__init__()
        self.volatility_gate = nn.Sequential(
            nn.Linear(1, d_hidden // 2),
            nn.Tanh(), 
            nn.Linear(d_hidden // 2, d_hidden),
            nn.Sigmoid()
        )
        self.relu = nn.GELU() 

    def forward(self, x):
        volatility = torch.std(x, dim=-1, keepdim=True) 
        gate = self.volatility_gate(volatility)
        return self.relu(x * gate)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 2. Extraction Modules
class ConvEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.layer = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = F.softmax(self.layer(x), dim=1)
        return torch.sum(x * weights, dim=1)

class PhysicsInjector(nn.Module):
    def __init__(self, input_dim, d_model):
        super(PhysicsInjector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        mean_stat = x.mean(dim=1)       
        std_stat = x.std(dim=1)         
        max_stat = x.max(dim=1).values  
        stats = torch.cat([mean_stat, std_stat, max_stat], dim=1)
        return self.proj(stats)

class ExtremeClusteringLayer(nn.Module):
    def __init__(self, n_clusters, n_features, alpha=1.0):
        super(ExtremeClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, n_features))
        nn.init.xavier_normal_(self.cluster_centers)

    def init_centers(self, centers):
        self.cluster_centers.data = centers.to(self.cluster_centers.device)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

# 3.Voformer-EC Pro
class Voformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, d_ff, dropout=0.1, n_clusters=11):
        super(Voformer, self).__init__()
        
        self.physics_injector = PhysicsInjector(input_dim, d_model)
        self.embedding = ConvEmbedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = []
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
            encoder_layers.append(layer)
            encoder_layers.append(NNVAF(d_model))
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.pooling = AttentionPooling(d_model)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )
        
        self.clustering_layer = ExtremeClusteringLayer(n_clusters, d_model)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            NNVAF(d_model * 2), 
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            NNVAF(d_model) 
        )
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        seq_len = x.size(1)
        phys_vec = self.physics_injector(x) 
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        enc_out = self.encoder(x_emb)
        deep_vec = self.pooling(enc_out)
        
        # Concat Fusion
        combined = torch.cat([deep_vec, phys_vec], dim=1)
        latent_vector = self.fusion_layer(combined)
        
        # Proj & Cluster
        proj = self.proj_head(latent_vector)
        q = self.clustering_layer(latent_vector)
        
        latent_expanded = latent_vector.unsqueeze(1).repeat(1, seq_len, 1)
        dec_out = self.decoder_fc(latent_expanded)
        recon = self.output_projection(dec_out)
        
        return recon, q, latent_vector, proj

    def extract_features(self, x):
        phys_vec = self.physics_injector(x)
        x_emb = self.embedding(x)
        x_emb = self.pos_encoder(x_emb)
        enc_out = self.encoder(x_emb)
        deep_vec = self.pooling(enc_out)
        combined = torch.cat([deep_vec, phys_vec], dim=1)
        return self.fusion_layer(combined)

def initialize_voformer_centers(model, data_loader, device):
    print("Initializing cluster centers using Density Peaks (Extreme Clustering)...")
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[0].to(device)
            features = model.extract_features(batch_x)
            all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    
    labels = extreme_clustering(all_features) 
    
    n_clusters = model.clustering_layer.cluster_centers.shape[0]
    d_model = model.clustering_layer.cluster_centers.shape[1]
    initial_centers = torch.zeros(n_clusters, d_model)
    
    valid_labels = labels[labels != -1]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    unique_labels = unique_labels[sorted_indices]
    
    print(f"Found {len(unique_labels)} clusters from Extreme Clustering logic.")
    
    for i in range(n_clusters):
        if i < len(unique_labels):
            lbl = unique_labels[i]
            cluster_points = all_features[labels == lbl]
            center = torch.tensor(np.mean(cluster_points, axis=0))
            initial_centers[i] = center
        else:
            nn.init.xavier_normal_(initial_centers[i].unsqueeze(0))
            
    model.clustering_layer.init_centers(initial_centers)
    print("Initialization Done.")

## Voformer Training
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# 2. Data augmentation functions (for SimCLR)
def augment_timeseries(x):
    B, L, F_dim = x.shape
    device = x.device
    # Random scaling (preserves the waveform but changes the amplitude to simulate rainfall variations).
    scales = torch.rand(B, 1, 1).to(device) * 0.2 + 0.9
    # Random noise
    noise = torch.randn_like(x) * 0.02
    return x * scales + noise

# 3. SimCLR Loss
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    return loss / 2

def train_voformer_ec(configs):
    data = np.load(configs.data_path)
    num_samples, seq_length, num_features = data.shape
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, num_features)).reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled, dtype=torch.float32)

    dataset = TensorDataset(X)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_voformer, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size_voformer, shuffle=False)

    # --- B. Initial ---
    model = Voformer(
        configs.input_dim, configs.d_model, configs.n_heads, 
        configs.num_layers, configs.d_ff, configs.dropout,
        n_clusters=configs.n_clusters
    ).to(configs.device)
    
    # --- Phase 1: Pretraining (Reconstruction) ---
    print("\n=== Phase 1: Pretraining (Reconstruction) ===")
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    
    for epoch in range(configs.warmup_epochs): 
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch_X = batch[0].to(configs.device)
            optimizer.zero_grad()
            recon, _, _, _ = model(batch_X)
            loss = F.mse_loss(recon, batch_X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Pretrain Epoch {epoch+1}/{configs.warmup_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    # --- Phase 2: Init Centers ---
    print("\n=== Phase 2: Initializing Cluster Centers ===")
    initialize_voformer_centers(model, train_loader, configs.device)

    # --- Phase 3: Joint Training (Rec + KL + SimCLR) ---
    print("\n=== Phase 3: Joint Training (Rec + Clustering + Contrastive) ===")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    lambda_rec = 1.0
    lambda_kl = configs.gamma      
    lambda_cl = configs.lambda_cl
    
    for epoch in range(configs.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch_X = batch[0].to(configs.device)
            optimizer.zero_grad()
            
            # 1. Raw data forward
            recon, q, _, _ = model(batch_X)
            
            # 2. Contrastive learning forward (augmented view)
            view1 = augment_timeseries(batch_X)
            view2 = augment_timeseries(batch_X)
            _, _, _, proj1 = model(view1) # 只取 proj
            _, _, _, proj2 = model(view2)
            
            # 3. Loss
            loss_rec = F.mse_loss(recon, batch_X)
            
            # The p-distribution needed to calculate the KL divergence
            p = target_distribution(q).detach()
            loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
            
            loss_cl = nt_xent_loss(proj1, proj2)
            
            loss = lambda_rec * loss_rec + lambda_kl * loss_kl + lambda_cl * loss_cl
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_X = batch[0].to(configs.device)
                recon, q, _, _ = model(batch_X) 
                loss_rec = F.mse_loss(recon, batch_X)
                
                p = target_distribution(q).detach()
                loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
                
                val_loss += (loss_rec + lambda_kl * loss_kl).item()
        
        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}, Train: {avg_train:.4f}, Val: {avg_val:.4f}')

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), configs.voformer_model_path)
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print("Early stop...")
                break

    print("Voformer model training completed and saved")
    return model

## Voformer-EC
# Loading Voformer
def load_voformer_model(configs):
    model = Voformer(
        configs.input_dim, 
        configs.d_model, 
        configs.n_heads, 
        configs.num_layers, 
        configs.d_ff, 
        configs.dropout,
        n_clusters=configs.n_clusters
    )
    model.load_state_dict(torch.load(configs.voformer_model_path, map_location=configs.device))
    model = model.to(configs.device)
    model.eval()
    print("Voformer loaded")
    return model

# Extract features by Voformer
def extract_features_with_voformer_ec(data, model, configs):
    model.eval()
    features_list = []
    
    num_samples, seq_length, num_features = data.shape
    data_reshaped = data.reshape(num_samples, seq_length * num_features)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    data_scaled_3d = data_scaled.reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled_3d, dtype=torch.float32)
    
    # Create a data loader
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=configs.batch_size_voformer, shuffle=False)
    
    # Feature extraction
    with torch.no_grad():
        for batch in dataloader:
            batch_X = batch[0].to(configs.device)
            batch_features = model(batch_X)
            features_list.append(batch_features.cpu().numpy())
    
    # Merge features
    features = np.concatenate(features_list, axis=0)
    
    features_2d = features.mean(axis=1)
    print(f"Feature shape conversion: {features.shape} -> {features_2d.shape}")
    # Perform extreme value clustering
    print("Extreme Clustering...")
    clustering = extreme_clustering(features_2d, neighborhood_radius=configs.ex_neighborhood_radius)
    
    return features_2d, clustering

# Evaluation Metrics (Latent Space Version)
def map_clusters_to_proxy_classes(X_original, pred_labels, n_clusters=11):
    cluster_means = []
    mapping = {}
    for i in range(n_clusters):
        indices = np.where(pred_labels == i)[0]
        if len(indices) == 0:
            cluster_means.append(-999)
            continue
        cluster_data = X_original[indices] 
        mean_val = np.mean(cluster_data) 
        cluster_means.append(mean_val)
    return mapping

def evaluate_voformer_metrics(model, data_loader, true_labels, device):
    model.eval()
    
    all_preds = []
    all_latents = []  # Save Latent Space
    
    with torch.no_grad():
        for batch in data_loader:
            batch_X = batch[0].to(device)

            _, q, latent, _ = model(batch_X)
            
            preds = torch.argmax(q, dim=1).cpu().numpy()
            all_preds.append(preds)
            
            # collect latent vector
            all_latents.append(latent.cpu().numpy())
            
    # stitching together Latent Space data
    X_latent_space = np.concatenate(all_latents, axis=0)
    y_pred_fine = np.concatenate(all_preds, axis=0)
    y_true = np.array(true_labels)

    print(f"Evaluation feature space: Latent Space (Shape: {X_latent_space.shape})")
    print(f"Number of categories: {len(np.unique(y_pred_fine))} (Expect {model.clustering_layer.n_clusters})")
    print("Internal Metrics (based Latent Space)...")
    
    ss = silhouette_score(X_latent_space, y_pred_fine)
    ch = calinski_harabasz_score(X_latent_space, y_pred_fine)
    db = davies_bouldin_score(X_latent_space, y_pred_fine)
    
    print("Mapping is being performed to match the Precip Proxy baseline....")
    y_pred_mapped = np.zeros_like(y_pred_fine)
    unique_pred_labels = np.unique(y_pred_fine)
    for label in unique_pred_labels:
        indices = np.where(y_pred_fine == label)[0]
        if len(indices) == 0: continue
        true_labels_in_cluster = y_true[indices]
        counts = np.bincount(true_labels_in_cluster)
        dominant_label = np.argmax(counts)
        y_pred_mapped[indices] = dominant_label
        
    print("Mapping complete. Calculating External Metrics (vs Precip Proxy)...")
    nmi = normalized_mutual_info_score(y_true, y_pred_mapped)
    ari = adjusted_rand_score(y_true, y_pred_mapped)
    
    return {
        "Silhouette Score": ss,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db,
        "NMI": nmi,
        "ARI": ari
    }

def generate_proxy_labels(data):
    print("Precip Proxy Labels (benchmark label)...")
    precip_mean = data[:, :, 0].mean(axis=1)
    q33 = np.percentile(precip_mean, 33.3)
    q66 = np.percentile(precip_mean, 66.6)
    N = data.shape[0]
    proxy_labels = np.zeros(N, dtype=int)
    proxy_labels[precip_mean <= q33] = 0
    proxy_labels[(precip_mean > q33) & (precip_mean <= q66)] = 1
    proxy_labels[precip_mean > q66] = 2
    print(f"label distribution: Class 0={np.sum(proxy_labels==0)}, Class 1={np.sum(proxy_labels==1)}, Class 2={np.sum(proxy_labels==2)}")
    return proxy_labels

def visualize_clustering_results(features, labels, configs, save_path=None):
    if save_path is None:
        os.makedirs(configs.visualization_output_dir, exist_ok=True)
        save_path = os.path.join(configs.visualization_output_dir, 'clustering_results_latent.png')
    
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        print(f"PCA dimensionality reduction complete, explaining variance ratio.: {pca.explained_variance_ratio_}")
    else:
        features_2d = features
    
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c='black', marker='x', s=50, alpha=0.6, label='Noise')
        else:
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[i%len(colors)]], s=50, alpha=0.7, label=f'Cluster {label}')
    
    plt.title('Voformer-EC Latent Space Visualization', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Save the visualization to: {save_path}")

def plot_training_history(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.title('Voformer-EC Training Track', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def perform_clustering_with_voformer_ec(model, data, configs):
    print("Start extracting features and performing clustering...")
    
    # Feature extraction and clustering
    features, clustering_results = extract_features_with_voformer_ec(data, model, configs)
    
    # Save
    np.save(configs.extracted_features_path, features)
    np.save(configs.clustering_results_path, clustering_results)
    print(f"Features have been saved to: {configs.extracted_features_path}")
    print(f"Clustering results have been saved to: {configs.clustering_results_path}")
    
    # Evaluate clustering results
    metrics = evaluate_clustering(features, clustering_results)
    
    # Visualization results
    if configs.visualize_clusters:
        visualize_clustering_results(features, clustering_results, configs)
    
    return clustering_results, metrics

def save_results_summary(configs, metrics, clustering_results):
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
    configs = Config()
    
    print("=== Voformer-EC (Aligned Evaluation) ===")
    print(f"Equipment: {configs.device}")
    
    # 1. Loading data
    if not os.path.exists(configs.data_path):
        print(f"Error: Data file {configs.data_path} not exist!")
        return
    
    print("Loading data...")
    data = np.load(configs.data_path) # Shape
    print(f"Data Shape: {data.shape}")
    
    # 2. Ground Truth (Precip Proxy)
    y_true = generate_proxy_labels(data)

    # 3. Data preprocessing
    from sklearn.preprocessing import StandardScaler
    N, T, F = data.shape
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, F)).reshape(N, T, F)
    X_tensor = torch.tensor(data_scaled, dtype=torch.float32)

    # 4. Training or loading a model
    if configs.train_voformer:
        print("\nStart training the Voformer-EC...")
        voformer_model = train_voformer_ec(configs)
    else:
        if os.path.exists(configs.voformer_model_path):
            print("\nLoading pre-trained Voformer-EC...")
            voformer_model = load_voformer_model(configs)
        else:
            print(f"\nModel pth {configs.voformer_model_path} not exist, force start training...")
            voformer_model = train_voformer_ec(configs)
    
    # 5. Implement a new evaluation process
    if configs.perform_clustering:
        print("\n=== Start cluster evaluation (Aligned with iTransformer) ===")
        perform_clustering_with_voformer_ec
        eval_dataset = TensorDataset(X_tensor)
        eval_loader = DataLoader(eval_dataset, batch_size=configs.batch_size_voformer, shuffle=False)
        
        metrics = evaluate_voformer_metrics(
            model=voformer_model, 
            data_loader=eval_loader, 
            true_labels=y_true, 
            device=configs.device
        )

        print("\n" + "="*60)
        print("Voformer-EC Final Results (Original Space vs Precip Proxy)")
        print("="*60)
        print(f"Silhouette Score (SS):  {metrics['Silhouette Score']:.4f} (Higher is better)")
        print(f"Calinski-Harabasz (CH): {metrics['Calinski-Harabasz']:.4f} (Higher is better)")
        print(f"Davies-Bouldin (DB):    {metrics['Davies-Bouldin']:.4f} (Lower is better)")
        print("-" * 60)
        print(f"NMI (vs Precip Proxy):  {metrics['NMI']:.4f} (Higher is better)")
        print(f"ARI (vs Precip Proxy):  {metrics['ARI']:.4f} (Higher is better)")
        print("="*60)
        
        with open('voformer_final_metrics.txt', 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        print("\nThe results have been saved to voformer_final_metrics.txt")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Program execution complete!")

if __name__ == "__main__":
    main()
