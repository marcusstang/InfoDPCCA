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
from D2PCCA import *

def load_data(data_path, sequence_length, test_size):
    df = pd.read_csv(data_path)

    def process_data(df, start_col, end_col):
        # Extract data and convert to numpy array
        data = df.iloc[:, start_col:end_col].astype(float).values

        return data

    # Extract raw data for normalization
    x1_raw = process_data(df, 1, 11)  # Columns 1-11
    x2_raw = process_data(df, 11, 21)  # Columns 11-21
    x3_raw = process_data(df, 21, 31)  # Columns 21-31
    x4_raw = process_data(df, 31, 41)  # Columns 31-41
    x5_raw = process_data(df, 41, 51)  # Columns 41-51

    # Number of sequences in each dataset
    test_start_idx = x1_raw.shape[0] - sequence_length - test_size + 1

    # Split the data into training and testing sets
    train_x1_raw, test_x1_raw = x1_raw[:test_start_idx], x1_raw[test_start_idx:]
    train_x2_raw, test_x2_raw = x2_raw[:test_start_idx], x2_raw[test_start_idx:]
    train_x3_raw, test_x3_raw = x3_raw[:test_start_idx], x3_raw[test_start_idx:]
    train_x4_raw, test_x4_raw = x4_raw[:test_start_idx], x4_raw[test_start_idx:]
    train_x5_raw, test_x5_raw = x5_raw[:test_start_idx], x5_raw[test_start_idx:]

    # Calculate global standard deviation for the entire training set
    global_std = np.concatenate([train_x1_raw, train_x2_raw, train_x3_raw, train_x4_raw, train_x5_raw], axis=0).std(axis=0)

    # Normalize and create sequences for training data
    def create_sequences(data, global_std, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            # Extract each sequence
            sequence = data[i:i + sequence_length]

            # Normalize each segmented sequence using global standard deviation of the training set
            mean = sequence.mean(axis=0)
            normalized_sequence = (sequence - mean) / global_std

            sequences.append(normalized_sequence)

        return torch.tensor(np.array(sequences), dtype=torch.float32)

    # Create sequences for training and testing sets
    train_x1 = create_sequences(train_x1_raw, global_std, sequence_length)
    train_x2 = create_sequences(train_x2_raw, global_std, sequence_length)
    train_x3 = create_sequences(train_x3_raw, global_std, sequence_length)
    train_x4 = create_sequences(train_x4_raw, global_std, sequence_length)
    train_x5 = create_sequences(train_x5_raw, global_std, sequence_length)

    test_x1 = create_sequences(test_x1_raw, global_std, sequence_length)
    test_x2 = create_sequences(test_x2_raw, global_std, sequence_length)
    test_x3 = create_sequences(test_x3_raw, global_std, sequence_length)
    test_x4 = create_sequences(test_x4_raw, global_std, sequence_length)
    test_x5 = create_sequences(test_x5_raw, global_std, sequence_length)

    # Create TensorDataset for training and testing
    train_dataset = TensorDataset(train_x1, train_x2, train_x3, train_x4, train_x5)
    test_dataset = TensorDataset(test_x1, test_x2, test_x3, test_x4, test_x5)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Check the shapes of the datasets
    print(f"Train dataset size: {len(train_dataset)} sequences")
    print(f"Test dataset size: {len(test_dataset)} sequences")

    return train_loader, test_loader


def save_checkpoint(model, optimizer, epoch, mode, checkpoint_dir="checkpoints"):
    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Paths for saving the model and optimizer states
    model_name = f"{mode}_epoch_{epoch}.pth"
    model_path = os.path.join(checkpoint_dir, model_name)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{model_name}")

    # Save model state
    logging.info(f"Saving model to {model_path}...")
    torch.save(model.state_dict(), model_path)

    # Save optimizer state using Pyro's save method
    logging.info(f"Saving optimizer states to {optimizer_path}...")
    optimizer.save(optimizer_path)

    logging.info("Done saving model and optimizer checkpoints to disk.")


def load_checkpoint(model, optimizer, mode, checkpoint_dir="checkpoints"):
    # Find the latest checkpoint for the dataset
    model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"{mode}")]
    if not model_files:
        logging.info("No checkpoints found. Starting from scratch.")
        return 0

    # Sort checkpoints by epoch number (assuming filenames follow the correct format)
    model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    latest_model_file = model_files[-1]
    epoch = int(latest_model_file.split('_')[-1].split('.')[0])

    model_path = os.path.join(checkpoint_dir, latest_model_file)
    optimizer_path = os.path.join(checkpoint_dir, f"optimizer_{latest_model_file}")

    # Load model state
    logging.info(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path))

    # Load optimizer state using Pyro's load method
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

def train(svi, train_loader, num_sectors=2):
    batch_size = train_loader.batch_size
    N_mini_batches = len(train_loader)
    epoch_nll = 0.0
    for which_mini_batch, mini_batch_list in enumerate(train_loader):
        mini_batch_list = mini_batch_list[:num_sectors]
        epoch_nll += svi.step(mini_batch_list)

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

    mode = "dpcca"

    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cca = D2PCCA(
        x_dims=[10, 10],  # Dimensions for x1 and x2
        emission_dims=[20, 20],  # Hidden dimensions in emission networks
        z_dims=[1, 2, 2],  # Latent state dimensions: shared z, zx, zy
        transition_dims=[5, 10, 10],  # Hidden dimensions in transition networks
        rnn_dim=50,  # RNN hidden dimension
        rnn_layers=1,  # Number of RNN layers
        rnn_dropout_rate=0.1,  # RNN dropout rate
    ).to(device)

    # setup optimizer
    adam_params = {
        "lr": 0.0003,
        "betas": (0.96, 0.999),
        "clip_norm": 10.0,
        "lrd": 0.99996,
        "weight_decay": 2.0,
    }

    adam = ClippedAdam(adam_params)
    svi = SVI(cca.model, cca.guide, adam, Trace_ELBO())

    # Load checkpoint if it exists
    start_epoch = load_checkpoint(cca, adam, mode, checkpoint_dir)


    NUM_EPOCHS = -300  # 1500


    # training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        total_epoch_loss_train = train(svi, train_loader, num_sectors=2)
        elbo = -total_epoch_loss_train / sequence_length

        print("[epoch %03d]  average training loss: %.4f" % (epoch, -elbo))

        def eval_test_elbo():
            cca.rnn.eval()
            for mini_batch_list in test_loader:
                pass
            test_elbo = -svi.evaluate_loss(mini_batch_list) / (test_size * sequence_length)
            cca.rnn.train()
            return test_elbo

        #elbo2 = eval_test_elbo()
        #print("[epoch %03d]  average test loss: %.4f" % (epoch, -elbo2))

        # Save the model every 10 epochs
        if epoch > 0 and epoch % 10 == 0:
            save_checkpoint(cca, adam, epoch, mode)

    # Evaluation
    # 1. RMSE
    print(recon_loss_D2PCCA(cca, test_loader, true_latent=False, fin=True))
    # 2. Visualization
    recon_fig_D2PCCA(cca, test_loader, 5, 12, True)





if __name__ == '__main__':
    main()

