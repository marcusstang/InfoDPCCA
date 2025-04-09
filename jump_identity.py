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
from scipy.io import loadmat
from itertools import combinations
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
from InfoDPCCA import *


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

class ECEO_identity_jump(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.ndarray): Time series data of shape (N, T, D).
            labels (np.ndarray): Labels corresponding to the data.
        """
        self.data = data
        self.labels = labels

        self.pairs = []
        self.pairs += [(i, i+23, 1) for i in range(23)]


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
    model_name = f"ECEO_M{step}_{epoch}.pth"
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
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"ECEO_M{step}")]
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
        model_file = f"ECEO_M{step}_{epoch}.pth"
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
    checkpoint_dir = "IJ_checkpoints"
    # load ECEO data
    ECEO_path = './datasets/ECEO.mat'
    ECEO = loadmat(ECEO_path)['BOLD']
    # load ECEO labels
    ECEO_label_path = './datasets/ECEO_label.mat'
    ECEO_label = loadmat(ECEO_label_path)['label']

    processed_ECEO = []
    for sample in ECEO:
        if isinstance(sample, np.ndarray) and len(sample) == 1:
            processed_ECEO.append(sample[0])  # Extract the inner array
        else:
            processed_ECEO.append(sample)

    # Stack into a single NumPy array if possible
    ECEO = np.array(processed_ECEO)
    ECEO_label = ECEO_label.squeeze()

    #ECEO = standardize_time_series(ECEO)
    #print(ECEO.shape)
    #print(ECEO_label.shape)

    selected = np.ones(94)
    selected[0] = 0
    selected[47] = 0
    selected = selected.astype(bool)

    # Apply the selection
    ECEO = ECEO[selected]
    ECEO_label = ECEO_label[selected]

    even_slices = ECEO[::2]  # 0, 2, 4, ..., 90  (shape: 46, T, D)
    odd_slices = ECEO[1::2]  # 1, 3, 5, ..., 91  (shape: 46, T, D)
    ECEO = np.concatenate((even_slices, odd_slices), axis=1)

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # ---------------------------- Create Dataset ----------------------------
    torch.manual_seed(42)

    train_dataset = ECEO_identity_jump(ECEO, ECEO_label)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # train loader in training


    # ---------------------------- Training Step 1 ----------------------------
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M1 = InfoDPCCA(
        x_dim=116,  # x dimensions
        y_dim=116,  # y dimensions
        z0_dim=5,  # z dimensions
        emission_x_dim=300,
        emission_y_dim=300,
        rnn_x_dim=500,  # RNN hidden dimensions
        rnn_y_dim=500,
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
    NUM_EPOCHS = -500
    print_every = 1
    save_every = 10

    if start_epoch < NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            total_epoch_loss_train = train(svi, train_loader, true_latent=True)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M1, adam, epoch, step=1, checkpoint_dir=checkpoint_dir)

    # ---------------------------- Training Step 2 ----------------------------
    M2 = InfoDPCCA_2(
        x_dim=116,  # x dimensions
        y_dim=116,  # y dimensions
        z0_dim=5,
        z1_dim=20,
        z2_dim=20,
        emission_x_dim=300,
        emission_y_dim=300,
        rnn_x_dim=500,  # RNN hidden dimensions
        rnn_y_dim=500,
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
        rnn4vi=True,
        rnn_vi_dim=600,
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
        for epoch in range(start_epoch, NUM_EPOCHS+1):
            total_epoch_loss_train = train(svi2, train_loader, true_latent=True)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M2, adam2, epoch, step=2, checkpoint_dir=checkpoint_dir)

    #recon_fig(M2, train_loader, 5)
    #print(recon_loss(M2, train_loader, True))
    #recon_fig_z0(M1, train_loader, 2, 5)
    #recon_fig_z0(M1, train_loader, 0, 7)



if __name__ == '__main__':
    main()