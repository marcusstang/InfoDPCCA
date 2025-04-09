import os
import logging
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import (
    SVI,
    Trace_ELBO,
)
from pyro.optim import (
    Adam,
    ClippedAdam,
)
from InfoDPCCA_GRU import *

def load_data(data_path, sequence_length, test_size):
    df = pd.read_csv(data_path)

    def process_data(df, start_col, end_col):
        # Extract data and convert to numpy array
        data = df.iloc[:, start_col:end_col].astype(float).values

        return data

    # Extract raw data for normalization
    x1_raw = process_data(df, 1, 11)  # Columns 1-11
    x2_raw = process_data(df, 11, 21)  # Columns 11-21


    # Number of sequences in each dataset
    test_start_idx = x1_raw.shape[0] - sequence_length - test_size + 1

    # Split the data into training and testing sets
    train_x1_raw, test_x1_raw = x1_raw[:test_start_idx], x1_raw[test_start_idx:]
    train_x2_raw, test_x2_raw = x2_raw[:test_start_idx], x2_raw[test_start_idx:]


    # Calculate global standard deviation for the entire training set
    global_std = np.concatenate([train_x1_raw, train_x2_raw], axis=0).std(axis=0)

    # Normalize and create sequences for training data
    def create_sequences(data, global_std, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            # Extract each sequence
            sequence = data[i:i + sequence_length]

            # Normalize each segmented sequence using global standard deviation of the training set
            mean = sequence.mean(axis=0)
            normalized_sequence = (sequence - mean) / global_std
            #normalized_sequence = sequence

            sequences.append(normalized_sequence)

        return torch.tensor(np.array(sequences), dtype=torch.float32)

    # Create sequences for training and testing sets
    train_x1 = create_sequences(train_x1_raw, global_std, sequence_length)
    train_x2 = create_sequences(train_x2_raw, global_std, sequence_length)

    test_x1 = create_sequences(test_x1_raw, global_std, sequence_length)
    test_x2 = create_sequences(test_x2_raw, global_std, sequence_length)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(train_x1, train_x2)
    test_dataset = TensorDataset(test_x1, test_x2)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Check the shapes of the datasets
    print(f"Train dataset size: {len(train_dataset)} sequences")
    print(f"Test dataset size: {len(test_dataset)} sequences")

    return train_loader, test_loader


def save_model(model, optimizer, epoch, step=1, checkpoint_dir="checkpoints"):
    # Paths for saving the model and optimizer states
    model_name = f"fin_M{step}_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)
    # Save optimizer state
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)
    logging.info("Done saving model and optimizer checkpoints to disk.")


def load_model(model, optimizer, step=1, checkpoint_dir="checkpoints", epoch=None):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"fin_M{step}")]
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
        model_file = f"fin_M{step}_{epoch}.pth"
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


def get_last_saved_epoch(elbo_file_path):
    """Retrieve the last saved epoch from the ELBO pickle file. If the file does not exist, create an empty one."""
    try:
        with open(elbo_file_path, "rb") as f:
            elbo_data = pickle.load(f)
            if not elbo_data:
                return -1  # No epochs saved yet
            # Get the last epoch number
            last_epoch = elbo_data[-1][0]  # Assuming elbo_data is a list of (epoch, elbo) tuples
            return last_epoch
    except (FileNotFoundError, EOFError):
        # File does not exist or is empty, create an empty file
        with open(elbo_file_path, "wb") as f:
            pickle.dump([], f)  # Initialize with an empty list
        return -1  # No epochs saved yet

def train_fin(svi, train_loader, num_sectors=2):
    batch_size = train_loader.batch_size
    N_mini_batches = len(train_loader)
    epoch_nll = 0.0
    for which_mini_batch, mini_batch_list in enumerate(train_loader):
        mini_batch_list = mini_batch_list[:num_sectors]
        epoch_nll += svi.step(mini_batch_list[0], mini_batch_list[1])

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_nll / normalizer_train
    return total_epoch_loss_train

def main():
    data_path = "datasets/stock_prices.csv"
    checkpoint_dir = "NASDAQ_checkpoints"
    sequence_length = 30  # T
    test_size = 20  # Use the last 20 sequences for testing
    train_loader, test_loader = load_data(data_path, sequence_length, test_size)

    mode = "infodpcca"

    # ---------------------------- Training Step 1 ----------------------------
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M1 = InfoDPCCA(
        x_dim=10,  # x dimensions
        y_dim=10,  # y dimensions
        z0_dim=1,  # z dimensions
        emission_x_dim=20,
        emission_y_dim=20,
        rnn_x_dim=30,  # RNN hidden dimensions
        rnn_y_dim=30,
        num_layers=1,  # RNN layers
        rnn_dropout_rate=0.1,
        beta=.00001,
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
    NUM_EPOCHS = -30
    print_every = 1
    save_every = 5

    if start_epoch < NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            total_epoch_loss_train = train_fin(svi, train_loader, num_sectors=2)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M1, adam, epoch, step=1)

    # ---------------------------- Training Step II ----------------------------
    M2 = InfoDPCCA_2(
        x_dim=10,  # x dimensions
        y_dim=10,  # y dimensions
        z0_dim=1,
        z1_dim=2,
        z2_dim=2,
        emission_x_dim=20,
        emission_y_dim=20,
        rnn_x_dim=30,  # RNN hidden dimensions
        rnn_y_dim=30,
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
        rnn_vi_dim=60,
    ).to(device)

    adam_params2 = {
        "lr": 0.0003,  # 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 5.0,  # 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam2 = ClippedAdam(adam_params2)
    svi2 = SVI(M2.model, M2.guide, adam2, Trace_ELBO())

    # Load checkpoint if it exists
    start_epoch = load_model(M2, adam2, step=2, checkpoint_dir=checkpoint_dir)

    NUM_EPOCHS = -1500
    print_every = 1
    save_every = 20

    # training loop
    if start_epoch < NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            total_epoch_loss_train = train_fin(svi2, train_loader, num_sectors=2)

            # Print training loss every 10 epochs
            if epoch % print_every == 0:
                print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # Save the model every 10 epochs
            if epoch > 0 and epoch % save_every == 0:
                save_model(M2, adam2, epoch, step=2)

    # Evaluation
    # 1. RMSE
    print(recon_loss(M2, test_loader, true_latent=False, fin=True))
    # 2. Visualization
    recon_fig_fin(M1, test_loader, 0, 12)
    recon_fig(M2, test_loader, 5, 12, True)

    # recon_fig_fin(M1, test_loader, 0, 5)




if __name__ == '__main__':
    main()

