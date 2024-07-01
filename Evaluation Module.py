# Evaluation Module

# Load data
data = embeddings_2
# Within-cluster SS (inertia)
def within_cluster_ss(X, labels, centroids):
    inertia = 0
    for i in range(len(centroids)):
        inertia += np.sum((X[labels == i] - centroids[i]) ** 2)
    return inertia


def mean_distance_to_nearest_cluster_member(X, labels):
    unique_labels = np.unique(labels)
    mean_distances = []
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            mean_distances.append(np.mean(min_distances))
    
    return np.mean(mean_distances)

def dunn_index(X, labels, centroids):
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    inter_cluster_dists = pdist(centroids, 'euclidean')
    inter_cluster_dists = squareform(inter_cluster_dists)
    np.fill_diagonal(inter_cluster_dists, np.inf)

    min_inter_cluster_dist = np.min(inter_cluster_dists)

    max_intra_cluster_dist = 0
    for i in range(num_clusters):
        cluster_points = X[labels == i]
        
        if len(cluster_points) == 0:  # Skip empty clusters
            continue
        
        intra_cluster_dists = pdist(cluster_points, 'euclidean')
        
        if len(intra_cluster_dists) == 0:  # If there's only one point in the cluster, max distance should be 0
            max_dist = 0
        else:
            max_dist = np.max(intra_cluster_dists)
        
        if max_dist > max_intra_cluster_dist:
            max_intra_cluster_dist = max_dist

    dunn = min_inter_cluster_dist / max_intra_cluster_dist
    return dunn

# Calculate centroids
def get_centroids(data, labels):
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), data.shape[1]))
    
    for i, label in enumerate(unique_labels):
        cluster_points = data[labels == label]
        centroids[i] = np.mean(cluster_points, axis=0)
    
    return centroids

centroids = get_centroids(data, cluster_result)

# Calculate evaluation indices
inertia = within_cluster_ss(data, cluster_result, centroids)
silhouette = silhouette_score(data, cluster_result)
davies_bouldin = davies_bouldin_score(data, cluster_result)
calinski_harabasz = calinski_harabasz_score(data, cluster_result)
dunn = dunn_index(data, cluster_result, centroids)
mean_distance = mean_distance_to_nearest_cluster_member(data, cluster_result)

# Print evaluation indices
print(f'Silhouette Score: {silhouette}')
print(f'Calinski-Harabasz Index: {calinski_harabasz}')
print(f'Davies-Bouldin Index: {davies_bouldin}')
print(f'Within-cluster SS (Inertia): {inertia}')
print(f'Dunn Index: {dunn}')
print(f'Mean Distance to Nearest Cluster Member: {mean_distance}')

max_k = 50
silhouette_scores = []
calinski_harabasz_scores = []
davies_bouldin_scores = []
within_cluster_sses = []
mean_distance_scores = []

def calculate_scores(k):
    cluster_result = Extreme_Clustreing(data, neighbuorhood_radius=k*0.1)
    unique_labels = np.unique(cluster_result)
    centroids = [np.mean(data[cluster_result == label], axis=0) for label in unique_labels]

    silhouette = silhouette_score(data, cluster_result)
    calinski_harabasz = calinski_harabasz_score(data, cluster_result)
    davies_bouldin = davies_bouldin_score(data, cluster_result)
    within_cluster_ss_value = within_cluster_ss(data, cluster_result, centroids)
    mean_distance = mean_distance_to_nearest_cluster_member(data, cluster_result)

    return silhouette, calinski_harabasz, davies_bouldin, within_cluster_ss_value, mean_distance

# Setting CPU cores number
results = Parallel(n_jobs=-1)(delayed(calculate_scores)(k) for k in range(0, max_k + 1))

for silhouette, calinski_harabasz, davies_bouldin, within_cluster_ss_value, mean_distance in results:
    silhouette_scores.append(silhouette)
    calinski_harabasz_scores.append(calinski_harabasz)
    davies_bouldin_scores.append(davies_bouldin)
    within_cluster_sses.append(within_cluster_ss_value)
    mean_distance_scores.append(mean_distance)

# Find the indices of maximum Silhouette Score and Calinski-Harabasz Score, and minimum Davies-Bouldin Score
max_silhouette_index = np.argmax(silhouette_scores)
max_calinski_harabasz_index = np.argmax(calinski_harabasz_scores)
max_within_cluster_ss = np.argmax(within_cluster_sses)
min_davies_bouldin_index = np.argmin(davies_bouldin_scores)
min_mean_distance_index = np.argmin(mean_distance_scores)

# Print the i values
print("i value for maximum Silhouette Score:", max_silhouette_index + 1)
print("i value for maximum Calinski-Harabasz Score:", max_calinski_harabasz_index + 1)
print("i value for maximum Within-Cluster SS (Inertia):", max_within_cluster_ss + 1)
print("i value for minimum Davies-Bouldin Score:", min_davies_bouldin_index + 1)
print("i value for minimum Mean Distance to Nearest Cluster Member:", min_mean_distance_index + 1)

# Plot the performance metrics
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].plot([i * 0.1 for i in range(0, max_k + 1)], silhouette_scores)
axs[0, 0].set_title('Silhouette Score')
axs[0, 0].set_xlabel('Neighbuorhood Radius Multiplier')
axs[0, 0].set_ylabel('Score')

axs[0, 1].plot([i * 0.1 for i in range(0, max_k + 1)], calinski_harabasz_scores)
axs[0, 1].set_title('Calinski-Harabasz Score')
axs[0, 1].set_xlabel('Neighbuorhood Radius Multiplier')
axs[0, 1].set_ylabel('Score')

axs[1, 0].plot([i * 0.1 for i in range(0, max_k + 1)], davies_bouldin_scores)
axs[1, 0].set_title('Davies-Bouldin Score')
axs[1, 0].set_xlabel('Neighbuorhood Radius Multiplier')
axs[1, 0].set_ylabel('Score')

axs[1, 1].plot([i * 0.1 for i in range(0, max_k + 1)], within_cluster_sses)
axs[1, 1].set_title('Within-Cluster SS (Inertia)')
axs[1, 1].set_xlabel('Neighbuorhood Radius Multiplier')
axs[1, 1].set_ylabel('Score')

plt.tight_layout()
plt.show()

# The scores of the four indicators when the Davies-Bouldin Score is the smallest
min_davies_bouldin_index = np.argmin(davies_bouldin_scores)
silhouette_score_at_min_db = silhouette_scores[min_davies_bouldin_index]
calinski_harabasz_at_min_db = calinski_harabasz_scores[min_davies_bouldin_index]
within_cluster_ss_at_min_db = within_cluster_sses[min_davies_bouldin_index]
min_mean_distance_index = np.argmin(mean_distance_scores)

# Print the i values
print("Silhouette Score when the Davies-Bouldin Score is the smallest:", silhouette_score_at_min_db)
print("Calinski-Harabasz Score when the Davies-Bouldin Score is the smallest:", calinski_harabasz_at_min_db)
print("Within-Cluster SS (Inertia) when the Davies-Bouldin Score is the smallest:", within_cluster_ss_at_min_db)
print("Minimum Davies-Bouldin Score:", davies_bouldin_scores[min_davies_bouldin_index])
print("Minimum Mean Distance to Nearest Cluster Member:", mean_distance_scores[min_mean_distance_index])
