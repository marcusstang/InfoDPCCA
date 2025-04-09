import os
import logging
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from scipy.io import loadmat
from itertools import combinations
from tck.TCK import TCK
from sklearn.decomposition import KernelPCA
import numpy as np
import plotly.express as px
import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils import *


def standardize_time_series(data):
    """
    Standardizes each time series in the dataset to have zero mean and unit variance
    along the time axis for each dimension.
    """
    # Compute mean and standard deviation along the time axis (axis=1)
    mean = np.mean(data, axis=1, keepdims=True)  # Shape: (94, 1, 116)
    std = np.std(data, axis=1, keepdims=True)  # Shape: (94, 1, 116)

    # Avoid division by zero by replacing zero std with 1
    std[std == 0] = 1.0

    # Standardize data
    standardized_data = (data - mean) / std
    return standardized_data


class EnumeratedAALDataset(Dataset):
    def __init__(self, data, labels, single_batch=False):
        """
        Args:
            data (np.ndarray): Time series data of shape (N, T, D).
            labels (np.ndarray): Labels corresponding to the data.
            single_batch (bool): If True, all samples will be returned as one batch.
        """
        self.data = data
        self.labels = labels
        self.single_batch = single_batch

        # Get indices for each label
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 3)[0]

        # Generate all unique pairs of indices within each label group
        self.pairs = []
        self.pairs += [(i, j, 1) for i, j in combinations(pos_indices, 2)]  # Positive class pairs
        self.pairs += [(i, j, -1) for i, j in combinations(neg_indices, 2)]  # Negative class pairs
        # print(len(pos_indices), len(neg_indices), len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get the indexed pair
        i, j, label = self.pairs[idx]

        # Retrieve corresponding samples
        x1 = self.data[i]
        x2 = self.data[j]

        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


def custom_collate(batch):
    """Collate function to return all data in a single batch if enabled."""
    x1_list, x2_list, y_list = zip(*batch)

    # Stack all elements together into a single tensor
    x1_tensor = torch.stack(x1_list)
    x2_tensor = torch.stack(x2_list)
    y_tensor = torch.stack(y_list)

    return x1_tensor, x2_tensor, y_tensor


def save_model(model, optimizer, epoch, checkpoint_dir="checkpoints"):
    # Paths for saving the model and optimizer states
    model_name = f"AAL_D2_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    # Save optimizer state
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)
    logging.info("Done saving model and optimizer checkpoints to disk.")


def load_model(model, optimizer, checkpoint_dir="checkpoints", epoch=None):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"AAL_D2")]
    if not model_files:
        logging.info("No checkpoints found. Starting from scratch.")
        return 0

    # Sort checkpoints by epoch number (assuming filenames follow the correct format)
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    if epoch is None:
        # Load the latest checkpoint if no epoch is specified
        model_file = model_files[-1]
        epoch = int(model_file.split('_')[-1].split('.')[0])
    else:
        # Find the checkpoint for the specified epoch
        model_file = f"AAL_D2_{epoch}.pth"
        if model_file not in model_files:
            logging.error(
                f"Checkpoint for epoch {epoch} not found. Available epochs: {[int(f.split('_')[-1].split('.')[0]) for f in model_files]}")
            return 0

    model_path = os.path.join(checkpoint_dir, model_file)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_file}")

    # Load model state
    logging.info(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    # Load optimizer state
    logging.info(f"Loading optimizer states from {optimizer_path}...")
    optimizer.load(optimizer_path)
    logging.info("Done loading model and optimizer states.")
    return epoch


def main():
    # load AAL data
    AAL_path = './datasets/AAL116.mat'
    AAL = loadmat(AAL_path)['timecourse']
    # load ECEO labels
    AAL_label_path = './datasets/AAL116_label.mat'
    AAL_label = loadmat(AAL_label_path)['AAL116_label']

    processed_AAL = []
    for sample in AAL:
        if isinstance(sample, np.ndarray) and len(sample) == 1:
            processed_AAL.append(sample[0])  # Extract the inner array
        else:
            processed_AAL.append(sample)

    # Stack into a single NumPy array if possible
    AAL = np.array(processed_AAL)
    AAL = AAL.transpose(2, 0, 1)
    AAL_label = AAL_label.squeeze()

    # remove the samples with label 2
    mask = AAL_label != 2
    AAL = AAL[mask]
    AAL_label = AAL_label[mask]
    AAL = standardize_time_series(AAL)

    # print(AAL.shape)
    # print(AAL_label.shape)

    # ---------------------------- Create Dataset ----------------------------
    torch.manual_seed(42)

    dataset = EnumeratedAALDataset(AAL, AAL_label)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ---------------------------- PCA ----------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    single_batch_mode = True  # Set to False for normal batching
    dataset = EnumeratedAALDataset(AAL, AAL_label, single_batch=single_batch_mode)
    batch_size = len(dataset) if single_batch_mode else 8  # Use full dataset if single_batch=True
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    for x1, x2, y in test_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        break

    #######

    x = torch.cat((x1, x2), dim=2)  # shape [batch, T, 2*D]
    recon_z0 = torch.empty(x.shape[0], x.shape[1], 5, device=x.device)

    for t in range(x.shape[1]):
        x_t = x[:, t, :]  # shape [batch, 2*D]
        x_t_centered = x_t - x_t.mean(dim=0)
        U, S, V = torch.pca_lowrank(x_t_centered, q=5)
        projected_t = x_t_centered @ V  # shape [batch, 5]
        recon_z0[:, t, :] = projected_t

    recon_z0 = recon_z0.cpu().numpy()
    y = y.cpu().detach().numpy()

    new_K = False
    if new_K:
        tck = TCK(G=20, C=10)
        K = tck.fit(recon_z0).predict(mode='tr-tr')
        print(f"K shape: {K.shape}")
        np.save("K_pca_AAL.npy", K)
    else:
        K = np.load("K_pca_AAL.npy")

    idx_sorted = np.argsort(y)
    K_sorted = K[:, idx_sorted][idx_sorted, :]
    fig = plt.figure(figsize=(6, 6))
    h = plt.imshow(K_sorted)
    plt.title("PCA: TCK matrix (sorted)")
    plt.colorbar(h)
    class_num, _ = np.histogram(y, bins=len(np.unique(y)))
    pos = np.cumsum(class_num)
    plt.xticks(pos, np.unique(y))
    plt.yticks(pos, np.unique(y))
    plt.xlabel("MTS class")
    plt.ylabel("MTS class")
    plt.show()

    kpca = KernelPCA(n_components=3, kernel='precomputed')
    embeddings_pca = kpca.fit_transform(K)

    # Convert embeddings and labels to a DataFrame for Plotly
    df = pd.DataFrame({
        'PC1': embeddings_pca[:, 0],
        'PC2': embeddings_pca[:, 1],
        'PC3': embeddings_pca[:, 2],
        'Label': y
    })

    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Label',
                        color_continuous_scale='Viridis', size_max=10)

    fig.update_layout(title='3D Kernel PCA Embeddings')

    # Save the interactive plot as an HTML file
    fig.write_html('scatter_3d_pca.html')

    # To display in Jupyter or directly in the notebook
    # fig.show()
    # plt.show()

    # -------------------------- Compute clustering metrics --------------------------
    n_clusters = len(np.unique(y))
    true_labels = y  # Dummy true labels (0 or 1)
    embeddings = embeddings_pca  # Dummy KPCA embeddings (e.g., 3D)

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


if __name__ == '__main__':
    main()