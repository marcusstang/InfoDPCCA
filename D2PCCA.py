import torch
import torch.nn.functional as F
import torch.nn as nn

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import matplotlib.pyplot as plt

# p(x_t | zs_t, zu_t)
class Emitter(nn.Module):
    def __init__(self, x_dim, zs_dim, zu_dim, emission_dim):
        # x_dim:        dimensions of the output
        # zs_dim:       dimensions of the shared latent state
        # zu_dim:       dimensions of the unique latent state
        # emission_dim: dimensions of the hidden layer of the emission network
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(zs_dim + zu_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, x_dim)
        self.relu = nn.ReLU()

    def forward(self, zs_t, zc_t):
        z_combined = torch.cat((zs_t, zc_t), dim=-1)
        h1 = self.relu(self.lin_z_to_hidden(z_combined))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        x_loc = self.lin_hidden_to_loc(h2)
        x_scale = torch.exp(self.lin_hidden_to_scale(h2))
        return x_loc, x_scale

# p(z_t | z_{t-1})
# The latent state z_t is tructured as (zs_t, z1_t, z2_t, ..., zd_t), where zs_t
# is the latent state shared by each observed variable, and zi_t is the latent
# state unique to the i-th observed variable, for i = 1,...,d.
class GatedTransition(nn.Module):
    def __init__(self, z_dims, transition_dims):
        # z_dims:           a variable-length tuple containing the dimensions of
        #                   zs_t, z1_t, z2_t, ..., and zd_t, respectively.
        # transition_dims:  dimensions of the hidden layers of the transition
        #                   network. It may take a tuple of the same shape as the
        #                   z_dims, or a single integer if the hidden dimensions
        #                   are of the same size.
        super().__init__()
        self.d = len(z_dims)
        self.z_dims = z_dims

        if isinstance(transition_dims, int):
            transition_dims = [transition_dims] * self.d
        else:
            assert len(transition_dims) == self.d

        self.shared_transition = self._build_transition_layers(z_dims[0], transition_dims[0])

        # Create transition layers for each observed variable's latent state
        self.individual_transitions = nn.ModuleList([
            self._build_transition_layers(z_dims[i], transition_dims[i]) for i in range(1, self.d)
        ])

    def forward(self, z_t):
        # Assert that the input dimension matches the expected sum of z_dims
        assert z_t.shape[-1] == sum(self.z_dims), (
            f"Expected z_t to have last dimension {sum(self.z_dims)}, "
            f"but got {z_t.shape[-1]} instead."
        )

        # Extract the shared latent state (zs_t)
        zs_t = z_t[:, :self.z_dims[0]]
        shared_loc, shared_scale = self._compute_transition(zs_t, self.shared_transition)

        # For each individual latent state (z1_t, z2_t, ..., zd_t)
        individual_locs = []
        individual_scales = []
        start_idx = self.z_dims[0]  # Start index after the shared latent state
        for i in range(1, self.d):
            zi_t = z_t[:, start_idx:start_idx + self.z_dims[i]]
            zi_loc, zi_scale = self._compute_transition(zi_t, self.individual_transitions[i - 1])
            individual_locs.append(zi_loc)
            individual_scales.append(zi_scale)
            start_idx += self.z_dims[i]  # Update the start index for the next individual latent state

        # Concatenate the locations and scales for zs_t and each zi_t
        cca_loc = torch.cat([shared_loc] + individual_locs, dim=-1)
        cca_scale = torch.cat([shared_scale] + individual_scales, dim=-1)

        return cca_loc, cca_scale


    def _build_transition_layers(self, z_dim, transition_dim):
        layers = nn.ModuleDict({
            'lin_gate_z_to_hidden': nn.Linear(z_dim, transition_dim),
            'lin_gate_hidden_to_z': nn.Linear(transition_dim, z_dim),
            'lin_proposed_mean_z_to_hidden': nn.Linear(z_dim, transition_dim),
            'lin_proposed_mean_hidden_to_z': nn.Linear(transition_dim, z_dim),
            'lin_sig': nn.Linear(z_dim, z_dim),
            'lin_z_to_loc': nn.Linear(z_dim, z_dim),
        })

        # Initialize `lin_z_to_loc` to be an identity mapping initially
        nn.init.eye_(layers['lin_z_to_loc'].weight)
        nn.init.zeros_(layers['lin_z_to_loc'].bias)

        return layers

    def _compute_transition(self, z_t, layers):
        # g_t: gating units
        _gate = F.relu(layers['lin_gate_z_to_hidden'](z_t))
        gate = torch.sigmoid(layers['lin_gate_hidden_to_z'](_gate))

        # h_t: proposed mean
        _proposed_mean = F.relu(layers['lin_proposed_mean_z_to_hidden'](z_t))
        proposed_mean = layers['lin_proposed_mean_hidden_to_z'](_proposed_mean)

        # loc, scale
        z_loc = (1 - gate) * layers['lin_z_to_loc'](z_t) + gate * proposed_mean
        z_scale = F.softplus(layers['lin_sig'](F.relu(proposed_mean))) + 1e-6

        return z_loc, z_scale


# q(z_t|z_{t-1}, h_t^r)
class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        #loc = self.lin_hidden_to_loc(h_combined)
        loc = self.tanh(self.lin_hidden_to_loc(h_combined))
        # scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        scale = self.softplus(self.lin_hidden_to_scale(h_combined)) + 1e-6
        return loc, scale


class D2PCCA(nn.Module):
    def __init__(self, x_dims, emission_dims, z_dims, transition_dims, rnn_dim,
                 rnn_layers=1, rnn_dropout_rate=0.0):
        # x_dims and emission_dims are defined similarly to z_dims and transition_dims
        # rnn_dim: hidden size of RNN used in the inference process.
        # rnn_layers: number of layers in the RNN.
        # rnn_dropout_rate: dropout rate used in the RNN.
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")

        # Validate input dimensions
        assert len(x_dims) == len(emission_dims), "x_dims and emission_dims must have the same length."
        assert len(z_dims) == len(transition_dims), "z_dims and transition_dims must have the same length."

        self.d = len(z_dims)
        self.x_dims = x_dims
        self.z_dims = z_dims
        self.z_shared_dim = z_dims[0]  # The first dimension is for the shared latent state (z)
        self.z_individual_dims = z_dims[1:]  # Remaining dimensions are for individual latent states (zx, zy, etc.)

        # Initialize emitters for each output variable
        self.emitters = nn.ModuleList([
            Emitter(x_dims[i], z_dims[0], z_dims[i + 1], emission_dims[i]) for i in range(self.d - 1)
        ])

        # Initialize the transition model
        self.trans = GatedTransition(z_dims, transition_dims)

        # Initialize the combiner for inference
        self.combiner = Combiner(sum(z_dims), rnn_dim)

        # RNN configuration
        self.rnn = nn.RNN( #nn.GRU(#!!!!!!!!!!!!!
            input_size=sum(x_dims),
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=rnn_layers,
            dropout=rnn_dropout_rate if rnn_layers > 1 else 0.0,
        )

        # p(z_0) initialization
        self.z_0 = nn.Parameter(torch.zeros(sum(z_dims)))
        # q(z_0)
        self.z_q_0 = nn.Parameter(torch.zeros(sum(z_dims)))
        # Initial hidden state of the RNN
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

    # p(z_t, x_t|z_{t-1}, x_{t-1})
    def model(self, mini_batch_list, annealing_factor=1.0):
        pyro.module("D2PCCA", self)
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size

        # Ensure all mini-batches have the same shape
        for mini_batch in mini_batch_list:
            assert mini_batch.size(1) == T_max and mini_batch.size(0) == batch_size, \
                "All mini-batches must have the same sequence length and batch size."

        # p(z_0)
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))  # Replicate z_0 for the batch

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # p(z_t | z_{t-1})
                z_all_loc, z_all_scale = self.trans(z_prev)
                z_all_t = pyro.sample(f"z_{t}", dist.Normal(z_all_loc, z_all_scale).to_event(1))

                # Extract individual latent states (z, zx, zy, etc.)
                z_t = z_all_t[:, :self.z_shared_dim]
                individual_latents = []
                start_idx = self.z_shared_dim
                for i in range(self.d - 1):
                    individual_latents.append(z_all_t[:, start_idx:start_idx + self.z_individual_dims[i]])
                    start_idx += self.z_individual_dims[i]

                # Compute emissions for each output variable
                for i, emitter in enumerate(self.emitters):
                    x_loc, x_scale = emitter(z_t, individual_latents[i])
                    pyro.sample(f"obs_x{i + 1}_{t}", dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch_list[i][:, t - 1, :])

                # Update time step
                z_prev = z_all_t


    def guide(self, mini_batch_list, annealing_factor=1.0):
        pyro.module("D2PCCA", self)
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]
        # Use the first mini-batch to determine sequence length and batch size
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size
        # Combine all mini-batches along the feature dimension for RNN input
        mini_batch_combined = torch.cat(mini_batch_list, dim=-1)
        # Expand h_0 to fit batch size
        h_0_contig = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        # Reverse the combined mini-batch
        mini_batch_reversed = torch.flip(mini_batch_combined, dims=[1])

        # Pass through the RNN
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                # ST-R: q(z_t | z_{t-1}, x_{t:T}, y_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                z_dist = dist.Normal(z_loc, z_scale).to_event(1)
                z_t = pyro.sample(f"z_{t}", z_dist)
                # Update time step
                z_prev = z_t

    def forward(self, mini_batch_list):
        # Ensure mini_batch_list is on the correct device
        mini_batch_list = [mini_batch.to(self.device) for mini_batch in mini_batch_list]

        # Use the first mini-batch to determine sequence length and batch size
        T_max = mini_batch_list[0].size(1)  # Sequence length (T)
        batch_size = mini_batch_list[0].size(0)  # Batch size
        # Combine all mini-batches along the feature dimension for RNN input
        mini_batch_combined = torch.cat(mini_batch_list, dim=-1)
        # Expand h_0 to fit batch size
        h_0_contig = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        # Reverse the combined mini-batch
        mini_batch_reversed = torch.flip(mini_batch_combined, dims=[1])

        # Pass through the RNN
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        rnn_output = torch.flip(rnn_output, dims=[1])
        # q(z_0)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        reconstructions = []
        scales = []
        # tensor containing extracted z
        recon_z_loc = torch.zeros(batch_size, T_max, sum(self.z_dims))
        recon_z_scale = torch.zeros(batch_size, T_max, sum(self.z_dims))

        # Generate latent states using the guide
        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(1, T_max + 1)):
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
                z_all_t = z_loc

                recon_z_loc[:, t - 1, :] = z_loc  # the first spots of recon_z_loc and recon_z_scale are empty.
                recon_z_scale[:, t - 1, :] = z_scale

                # Extract individual latent states (z, zx, zy, etc.)
                z_t = z_all_t[:, :self.z_shared_dim]
                individual_latents = []
                start_idx = self.z_shared_dim
                for i in range(self.d - 1):
                    individual_latents.append(z_all_t[:, start_idx:start_idx + self.z_individual_dims[i]])
                    start_idx += self.z_individual_dims[i]

                # Compute emissions for each output variable
                for i, emitter in enumerate(self.emitters):
                    x_loc, x_scale = emitter(z_t, individual_latents[i])
                    reconstructions.append(x_loc)
                    scales.append(x_scale)

                # Update time step
                z_prev = z_all_t

        #print(x_loc.shape)

        # Group reconstructions by each observation x1, x2, ..., x5
        grouped_reconstructions = [
            torch.stack([reconstructions[t] for t in range(i, len(reconstructions), self.d - 1)], dim=1)
            for i in range(self.d - 1)]
        grouped_scales = [
            torch.stack([scales[t] for t in range(i, len(scales), self.d - 1)], dim=1)
            for i in range(self.d - 1)
        ]

        return grouped_reconstructions, grouped_scales, recon_z_loc, recon_z_scale

def train_D2PCCA(svi, train_loader, num_sectors=2, true_latent=False):
    epoch_loss = 0.

    if true_latent:
        for x, y, _ in train_loader:  # x is mini-batch
            epoch_loss += svi.step([x, y])
    else:
        for x, y in train_loader:  # x is mini-batch
            epoch_loss += svi.step([x, y])

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def recon_loss_D2PCCA(model, test_loader, true_latent=False, fin=False):
    model.eval()
    total_squared_error = 0.0
    total_elements = 0

    if not fin:
        if true_latent:
            for mini_batch_x, mini_batch_y, _ in test_loader:
                mini_batch_x = mini_batch_x.to(model.device)
                mini_batch_y = mini_batch_y.to(model.device)
                # Get the model's reconstructions
                recon_groups, _, _, _ = model([mini_batch_x, mini_batch_y])
                recon_x_loc = recon_groups[0]
                recon_y_loc = recon_groups[1]
                # Concatenate the reconstructed sequences
                recon = torch.cat([recon_x_loc, recon_y_loc], dim=2).to(model.device)
                recon = recon[:, 1:, :]
                # Concatenate the original and target sequences along the feature dimension
                original = torch.cat([mini_batch_x, mini_batch_y], dim=2).to(model.device)
                original = original[:, 1:, :]  # Remove the first time step
                # Compute the squared differences
                squared_diff = (original - recon) ** 2
                total_squared_error += torch.sum(squared_diff).item()
                total_elements += squared_diff.numel()

        else:
            for mini_batch_x, mini_batch_y in test_loader:
                mini_batch_x = mini_batch_x.to(model.device)
                mini_batch_y = mini_batch_y.to(model.device)
                # Get the model's reconstructions
                recon_groups, _, _, _ = model([mini_batch_x, mini_batch_y])
                recon_x_loc = recon_groups[0]
                recon_y_loc = recon_groups[1]
                # Concatenate the reconstructed sequences
                recon = torch.cat([recon_x_loc, recon_y_loc], dim=2).to(model.device)
                recon = recon[:, 1:, :]
                # Concatenate the original and target sequences along the feature dimension
                original = torch.cat([mini_batch_x, mini_batch_y], dim=2).to(model.device)
                original = original[:, 1:, :]  # Remove the first time step
                # Compute the squared differences
                squared_diff = (original - recon) ** 2
                total_squared_error += torch.sum(squared_diff).item()
                total_elements += squared_diff.numel()
    else:
        for which_mini_batch, mini_batch_list in enumerate(test_loader):
            mini_batch_x = mini_batch_list[0].to(model.device)
            mini_batch_y = mini_batch_list[1].to(model.device)
            # Get the model's reconstructions
            recon_groups, _, _, _ = model([mini_batch_x, mini_batch_y])
            recon_x_loc = recon_groups[0]
            recon_y_loc = recon_groups[1]
            # Concatenate the reconstructed sequences
            recon = torch.cat([recon_x_loc, recon_y_loc], dim=2).to(model.device)
            recon = recon[:, 1:, :]
            # Concatenate the original and target sequences along the feature dimension
            original = torch.cat([mini_batch_x, mini_batch_y], dim=2).to(model.device)
            original = original[:, 1:, :]  # Remove the first time step
            # Compute the squared differences
            squared_diff = (original - recon) ** 2
            total_squared_error += torch.sum(squared_diff).item()
            total_elements += squared_diff.numel()

    # Compute RMSE
    mse = total_squared_error / total_elements
    rmse = mse ** 0.5

    return rmse


def elbo_loss_D2PCCA(svi, model, test_loader, true_latent=False):
    model.eval()
    total_elbo = 0.0
    multiplier = 0

    if true_latent:
        for mini_batch_x, mini_batch_y, _ in test_loader:
            mini_batch_x = mini_batch_x.to(model.device)
            mini_batch_y = mini_batch_y.to(model.device)
            total_elbo += svi.evaluate_loss([mini_batch_x, mini_batch_y])
            seq_length = mini_batch_x.size(1)
            batch_size = mini_batch_x.size(0)
            multiplier += seq_length * batch_size
    else:
        for mini_batch_x, mini_batch_y in test_loader:
            mini_batch_x = mini_batch_x.to(model.device)
            mini_batch_y = mini_batch_y.to(model.device)
            total_elbo += svi.evaluate_loss([mini_batch_x, mini_batch_y])
            seq_length = mini_batch_x.size(1)
            batch_size = mini_batch_x.size(0)
            multiplier += seq_length * batch_size

    model.train()
    return -total_elbo / multiplier


def _pearson_corr_1d(x, y):
    """
    Compute the 'global' Pearson correlation between two 1D tensors x and y.
    (i.e., correlation with global means over all samples in x and y).
    """
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered**2).sum()) * torch.sqrt((y_centered**2).sum())
    if denominator == 0.0:
        # In degenerate cases, define correlation = 0
        return torch.tensor(0.0, device=x.device)
    return numerator / denominator


def smgc_metric(Z, Zhat):
    """
    SMGC = (1/D) * sum_{j=1..D} [ max_{k in [1..D]} |rho_{j,k}| ]
    where rho_{j,k} is the 'global' correlation between dimension j of Z and
    dimension k of Zhat, across the entire [B*T] samples in this batch.

    Args:
        Z: shape [B, T, D]
        Zhat: shape [B, T, D]
    Returns:
        Scalar (float) with SMGC metric
    """
    B, T, D = Z.shape

    # Flatten [B,T] -> [B*T]
    Z_flat   = Z.reshape(B*T, D)
    Zhat_flat= Zhat.reshape(B*T, D)

    # For each j in [0..D-1], compute correlation with each k
    # and take the max of absolute correlations
    max_corrs_per_j = []

    for j in range(D):
        # The j-th dimension in the *true* data
        z_j = Z_flat[:, j]

        corrs_j = []
        for k in range(D):
            zhat_k = Zhat_flat[:, k]
            # Global correlation
            r_jk = _pearson_corr_1d(z_j, zhat_k)
            corrs_j.append(torch.abs(r_jk))

        # max_{k} |r_{j,k}|
        max_corr_j = torch.stack(corrs_j).max()
        max_corrs_per_j.append(max_corr_j)

    smgc_val = torch.stack(max_corrs_per_j).mean()
    return smgc_val.item()  # return as float


def smlc_metric(Z, Zhat):
    """
    SMLC = (1/D) * sum_{j=1..D} [ max_{k in [1..D]} |(1/B)*sum_n corr_jk^n| ]

    But carefully: we define:
      tilde(rho_{j,k}) = (1/B) sum_{n=1..B} [ correlation( Z[n,:,j], Zhat[n,:,k] ) ]

    Then we do:
      SMLC = (1/D)* sum_j [ max_k |tilde(rho_{j,k})| ].

    Args:
        Z: shape [B, T, D]
        Zhat: shape [B, T, D]
    Returns:
        Scalar (float) with SMLC metric
    """
    B, T, D = Z.shape

    # We'll build up an average correlation matrix: [D, D].
    # For each j,k, we average correlation across n in [0..B-1].
    corr_matrix = torch.zeros(D, D, device=Z.device)

    for j in range(D):
        for k in range(D):
            # Collect correlation for each sequence n
            # then average
            corrs_for_n = []
            for n in range(B):
                z_j_n     = Z[n, :, j]      # shape [T]
                zhat_k_n  = Zhat[n, :, k]   # shape [T]
                # local correlation for this sequence:
                r_jk_n = _pearson_corr_1d(z_j_n, zhat_k_n)
                corrs_for_n.append(r_jk_n)

            # \tilde{rho}_{j,k} = average of r_jk^n across n
            corrs_for_n = torch.stack(corrs_for_n)
            avg_corr_jk = corrs_for_n.mean()

            corr_matrix[j, k] = avg_corr_jk

    # Now compute sum_{j=1..D} [ max_{k in [1..D]} |corr_matrix[j,k]| ], then / D
    max_corrs_per_j = []
    for j in range(D):
        row_j = corr_matrix[j, :]           # all k for fixed j
        max_corr_j = torch.max(torch.abs(row_j))
        max_corrs_per_j.append(max_corr_j)

    smlc_val = torch.stack(max_corrs_per_j).mean()
    return smlc_val.item()


def smc_loss_D2PCCA(model, test_loader, device='cpu'):
    """
    Computes average SMGC and SMLC over all batches in test_loader.
    If your batches are of roughly the same size, you can simple-average.
    Otherwise, you might do a weighted average by batch size.
    """
    model.eval()

    total_smgc = 0.0
    total_smlc = 0.0
    num_batches = 0

    with torch.no_grad():
        for mini_batch_x, mini_batch_y, mini_batch_z in test_loader:
            # Move data to device
            mini_batch_x = mini_batch_x.to(device)
            mini_batch_y = mini_batch_y.to(device)
            mini_batch_z = mini_batch_z.to(device)

            # Forward pass -> get reconstructions
            # Suppose your model returns recon_z in shape [B, T, D]
            _, _, recon_z, _ = model([mini_batch_x, mini_batch_y])

            # If necessary, slice out the relevant parts:
            recon_z0 = recon_z[:, :-1, :model.z_shared_dim]
            true_z0 = mini_batch_z[:, :-1, :model.z_shared_dim]
            # For demonstration, let's assume you want the entire recon_z vs mini_batch_z:
            Z = true_z0
            Z_hat = recon_z0

            # Compute SMGC and SMLC for this batch
            batch_smgc = smgc_metric(Z, Z_hat)
            batch_smlc = smlc_metric(Z, Z_hat)

            total_smgc += batch_smgc
            total_smlc += batch_smlc
            num_batches += 1

    avg_smgc = total_smgc / num_batches
    avg_smlc = total_smlc / num_batches
    return avg_smgc, avg_smlc

def recon_fig_D2PCCA(model, test_loader, dim_idx, data_idx=0, fin=False):
    # by default, we select first test sample of x_1

    model.eval()

    if fin:
        for which_mini_batch, mini_batch_list in enumerate(test_loader):
            mini_batch_x = mini_batch_list[0].to(model.device)
            mini_batch_y = mini_batch_list[1].to(model.device)
            break
    else:
        for mini_batch_x, mini_batch_y, mini_batch_label in test_loader:
            mini_batch_x = mini_batch_x.to(model.device)
            mini_batch_y = mini_batch_y.to(model.device)
            mini_batch_label = mini_batch_label.to(model.device)
            break

    # Get the model's reconstructions
    recon_group_loc, recon_group_scale, recon_z_loc, recon_z_scale = model([mini_batch_x[data_idx:data_idx+1, :, :], mini_batch_y[data_idx:data_idx+1, :, :]])

    x1 = mini_batch_x[data_idx, :, dim_idx].cpu().detach().numpy()
    x1_recon_loc = recon_group_loc[0][0, :, dim_idx].cpu().detach().numpy()
    x1_recon_scale = recon_group_scale[0][0, :, dim_idx].cpu().detach().numpy()

    # Calculate 95% CI for the reconstructed sequence
    lower_bound = x1_recon_loc - 2 * x1_recon_scale
    upper_bound = x1_recon_loc + 2 * x1_recon_scale

    # Plot the original and reconstructed sequences for the first feature of x1
    plt.figure(figsize=(10, 6))
    plt.plot(x1, label='Original x1', linestyle='-', marker='o')
    plt.plot(x1_recon_loc, label='Reconstructed x1', linestyle='--', marker='x')
    plt.fill_between(range(len(x1_recon_loc)), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% CI')
    #print(mini_batch_label[data_idx])
    #plt.plot(recon_z_loc[0, :, dim_idx].detach().numpy(), label='Original x1[0,:,0]', linestyle='-', marker='o')

    plt.title('DPCCA: Original vs Reconstructed Sequence for x1')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    squared_diff = (x1 - x1_recon_loc) ** 2
    #print(x1[-2], x1_recon_loc[-2], (x1[-2]- x1_recon_loc[-2]))
    #return squared_diff
    return sum(squared_diff)
