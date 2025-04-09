import numpy as np
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
                             homogeneity_score, completeness_score, v_measure_score, confusion_matrix,
                             silhouette_score, calinski_harabasz_score, davies_bouldin_score)
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering


def compute_purity(true_labels, pred_labels):
    contingency_matrix = confusion_matrix(true_labels, pred_labels)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity


def compute_aligned_accuracy(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # Maximize matches
    aligned_accuracy = cm[row_ind, col_ind].sum() / np.sum(cm)
    return aligned_accuracy


def compute_clustering_metrics(true_labels, pred_labels, embeddings=None):
    """
    Compute various clustering evaluation metrics.

    Parameters:
    - true_labels: Ground truth labels (1D array of shape [N])
    - pred_labels: Predicted cluster labels (1D array of shape [N])
    - embeddings: Optional data points (2D array of shape [N, d]), e.g., KPCA embeddings

    Returns:
    - Dictionary containing all computed metrics
    """
    metrics = {}

    # Metrics based on labels only
    metrics['Adjusted Rand Index (ARI)'] = adjusted_rand_score(true_labels, pred_labels)
    metrics['Normalized Mutual Information (NMI)'] = normalized_mutual_info_score(true_labels, pred_labels)
    metrics['Fowlkes-Mallows Index (FMI)'] = fowlkes_mallows_score(true_labels, pred_labels)
    metrics['Homogeneity'] = homogeneity_score(true_labels, pred_labels)
    metrics['Completeness'] = completeness_score(true_labels, pred_labels)
    metrics['V-measure'] = v_measure_score(true_labels, pred_labels)
    metrics['Purity'] = compute_purity(true_labels, pred_labels)
    metrics['Aligned Accuracy'] = compute_aligned_accuracy(true_labels, pred_labels)

    # Metrics requiring data points (embeddings)
    if embeddings is not None:
        metrics['Silhouette Score'] = silhouette_score(embeddings, pred_labels, metric='euclidean')
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(embeddings, pred_labels)
        metrics['Davies-Bouldin Index (DBI)'] = davies_bouldin_score(embeddings, pred_labels)

    return metrics



if __name__ == "__main__":
    # Example data (replace with your actual data)
    N = 100  # Number of samples
    n_clusters = 2  # Number of clusters
    K = np.random.rand(N, N)  # Dummy kernel matrix (symmetric, positive semi-definite)
    K = (K + K.T) / 2  # Ensure symmetry
    true_labels = np.random.randint(0, 2, N)  # Dummy true labels (0 or 1)
    embeddings = np.random.rand(N, 3)  # Dummy KPCA embeddings (e.g., 3D)

    # Perform spectral clustering
    model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize',
                               random_state=42)
    pred_labels = model.fit_predict(K)

    # Compute all metrics
    metrics = compute_clustering_metrics(true_labels, pred_labels, embeddings)

    # Display results neatly
    print("Clustering Evaluation Metrics:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")