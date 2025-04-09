import os
import logging
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from pyro.infer import (
    SVI,
    Trace_ELBO,
)
from pyro.optim import (
    Adam,
    ClippedAdam,
)
import numpy as np
from tck.TCK import TCK
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
from sklearn.metrics import v_measure_score
from sklearn.decomposition import KernelPCA
import plotly.express as px
import plotly.io as pio
import pandas as pd
from sklearn.model_selection import train_test_split
from InfoDPCCA_GRU import *

def standardize_time_series(data):
    """
    Standardizes each time series in the dataset to have zero mean and unit variance
    along the time axis for each dimension.
    """
    # Compute mean and standard deviation along the time axis (axis=1)
    mean = np.mean(data, axis=1, keepdims=True)  # Shape: (94, 1, 116)
    std = np.std(data, axis=1, keepdims=True)    # Shape: (94, 1, 116)

    # Avoid division by zero by replacing zero std with 1
    std[std == 0] = 1.0

    # Standardize data
    standardized_data = (data - mean) / std
    return standardized_data

def process_and_concatenate(hi_path, lo_path):
    data_hi = pd.read_csv(hi_path)  # Shape (2880, 19)
    data_lo = pd.read_csv(lo_path)  # Shape (2880, 18)

    subjects_hi = sorted(data_hi['Label'].unique())
    subjects_lo = sorted(data_lo['Label'].unique())

    # Process hi data
    filtered_hi = []
    for subject in subjects_hi:
        subject_data = data_hi[data_hi['Label'] == subject]
        filtered_hi.append(subject_data.iloc[:59])  # Take first 59 windows
    data_hi_processed = pd.concat(filtered_hi)  # Shape (2832, 19)

    # Process lo data
    filtered_lo = []
    for subject in subjects_lo:
        subject_data = data_lo[data_lo['Label'] == subject]
        filtered_lo.append(subject_data.iloc[:59])  # Take first 59 windows
    data_lo_processed = pd.concat(filtered_lo)  # Shape (2832, 18)

    # Extract features
    features_hi = data_hi_processed.iloc[:, 1:-1].values  # Shape (2832, 17)
    features_lo = data_lo_processed.iloc[:, :-1].values  # Shape (2832, 17)

    # Reshape
    tensor_hi = features_hi.reshape(48, 59, 17)
    tensor_lo = features_lo.reshape(48, 59, 17)

    # Concatenate
    tensor_concatenated = np.concatenate((tensor_hi, tensor_lo), axis=0)
    return tensor_concatenated


class STEW_DATA(Dataset):
    def __init__(self, data):
        """
        Args:
            data (np.ndarray): Time series data of shape (N, T, D).
            labels (np.ndarray): Labels corresponding to the data.
        """
        self.data = data

        label_1 = [0, 15, 16, 19, 21, 25, 29, 33, 35, 36, 38, 43, 44, 46]
        label_0 = [1, 2, 3, 6, 10, 18, 22, 24, 26, 32, 34, 37, 39]
        #label_1 = [15, 16, 33]
        #label_0 = [24, 26, 34]
        #label_1 = [16, 25, 44]
        #label_0 = [28, 40, 17, 5]

        self.pairs = []
        self.pairs += [(i, i+48, 1) for i in label_1]
        self.pairs += [(i, i+48, 0) for i in label_0]


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


def save_model(model, optimizer, epoch, step=1, checkpoint_dir="checkpoints_ECEO_uai"):
    # Paths for saving the model and optimizer states
    model_name = f"STEW_M{step}_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    # Save optimizer state
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)
    logging.info("Done saving model and optimizer checkpoints to disk.")


def load_model(model, optimizer, step=1, checkpoint_dir="checkpoints_ECEO_uai", epoch=None):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"STEW_M{step}")]
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
        model_file = f"STEW_M{step}_{epoch}.pth"
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
    checkpoint_dir = "STEW_checkpoints"

    # Example usage
    hi_path = './datasets/extractedFeatures_hi.csv'
    lo_path = './datasets/extractedFeatures_lo.csv'
    result = process_and_concatenate(hi_path, lo_path)
    #result = standardize_time_series(result)
    #print(f"Resulting tensor shape: {result.shape}")  # Should output: (96, 59, 17)

    """
    plt.figure(figsize=(10, 6))
    plt.plot(result[50,:,10:14], label='z0', linestyle='--', marker='x')
    plt.title('InfoDPCCA: Shared Latent State')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    torch.manual_seed(42)

    train_dataset = STEW_DATA(result)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # train loader in training

    """
    # Iterate through the DataLoader
    for x1, x2, y in train_loader:
        print("x1 shape:", x1.shape)  # (batch_size, 235, 116)
        print("x2 shape:", x2.shape)  # (batch_size, 235, 116)
        print("y:", y)  # Labels (1 or -1)
        break  # Display one batch
    """

    # ---------------------------- Training Step 1 ----------------------------
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M1 = InfoDPCCA(
        x_dim=17,  # x dimensions
        y_dim=17,  # y dimensions
        z0_dim=3,  # z dimensions
        emission_x_dim=50,
        emission_y_dim=50,
        rnn_x_dim=100,  # RNN hidden dimensions
        rnn_y_dim=100,
        num_layers=1,  # RNN layers
        rnn_dropout_rate=0.1,
        beta=.1,
        alpha=1,
    ).to(device)

    # setup optimizer
    adam_params = {
        "lr": 0.0003,  # 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 10.0,  # 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam = ClippedAdam(adam_params)
    svi = SVI(M1.model, M1.guide, adam, Trace_ELBO())

    # Load checkpoint if it exists
    start_epoch = load_model(M1, adam, step=1, checkpoint_dir=checkpoint_dir)

    # resume training
    NUM_EPOCHS = 500
    print_every = 1
    save_every = 10

    if start_epoch < NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            total_epoch_loss_train = train(svi, train_loader, true_latent=True)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M1, adam, epoch, step=1, checkpoint_dir=checkpoint_dir)

    # ---------------------------- Training Step 2 ----------------------------
    M2 = InfoDPCCA_2(
        x_dim=17,  # x dimensions
        y_dim=17,  # y dimensions
        z0_dim=3,
        z1_dim=5,
        z2_dim=5,
        emission_x_dim=50,
        emission_y_dim=50,
        rnn_x_dim=100,  # RNN hidden dimensions
        rnn_y_dim=100,
        num_layers=1,  # RNN layers
        rnn_dropout_rate=0.1,
        rnn_x=M1.rnn_x,
        rnn_y=M1.rnn_y,
        h_x_0=M1.h_x_0,
        h_y_0=M1.h_y_0,
        combiner_12_0=M1.combiner_12_0,
        res_con=True,
        old_emitter_x=M1.emitter_x,
        old_emitter_y=M1.emitter_y,
        rnn4vi=False,
        rnn_vi_dim=200,
    ).to(device)

    adam2 = ClippedAdam(adam_params)
    svi2 = SVI(M2.model, M2.guide, adam2, Trace_ELBO())

    # Load checkpoint if it exists
    start_epoch = load_model(M2, adam2, step=2, checkpoint_dir=checkpoint_dir)

    NUM_EPOCHS = -1500
    print_every = 5
    save_every = 10

    # training loop
    if start_epoch < NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            total_epoch_loss_train = train(svi2, train_loader, true_latent=True)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M2, adam2, epoch, step=2, checkpoint_dir=checkpoint_dir)

    # recon_fig(M2, train_loader, 5)
    # print(recon_loss(M2, train_loader, True))
    # recon_fig_z0(M1, train_loader, 2, 5)
    # recon_fig_z0(M1, train_loader, 0, 7)

    # ------------------------------- Use Step I Model ----------------------------
    for x1, x2, y in train_loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        break

    recon_z0 = M1(x1, x2)
    recon_z0 = recon_z0.detach().numpy()
    y = y.cpu().detach().numpy()

    new_K = True
    if new_K:
        tck = TCK(G=20, C=10)
        K = tck.fit(recon_z0).predict(mode='tr-tr')
        print(f"K shape: {K.shape}")
        np.save("K_stew.npy", K)
    else:
        K = np.load("K_stew.npy")

    idx_sorted = np.argsort(y)
    K_sorted = K[:, idx_sorted][idx_sorted, :]
    fig = plt.figure(figsize=(6, 6))
    h = plt.imshow(K_sorted)
    plt.title("InfoDPCCA Step I: TCK matrix (sorted)")
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
    fig.write_html('scatter_3d_stew.html')

    fig.show()
    plt.show()



if __name__ == "__main__":
    main()