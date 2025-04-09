import torch
import torch.nn.functional as F
import torch.nn as nn

import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#------------------------------------------------ NN Modules ------------------------------------------------#

# p(x_t | z_t) or p(y_t | z_t)
class Emitter(nn.Module):
    def __init__(self, output_dim, z_dim, emission_dim):
        super().__init__()
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t):
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        x_loc = self.lin_hidden_to_loc(h2)
        x_scale = torch.exp(self.lin_hidden_to_scale(h2))
        return x_loc, x_scale

# q(z_t| h_t^r)
class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, h_rnn):
        h_combined = h_rnn
        loc = self.lin_hidden_to_loc(h_combined)
        #loc = self.tanh(self.lin_hidden_to_loc(h_combined))
        # scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        scale = self.softplus(self.lin_hidden_to_scale(h_combined)) #+ 1e-6 # small value for stability
        return loc, scale

# p(x | z0, z1/z2)
class Emitter_Res_Con(nn.Module):
    def __init__(self, output_dim, z0_dim, z1_dim, emission_dim, emitter1):
        super().__init__()

        # Freeze the parameters of emitter1
        self.emitter1 = emitter1
        for param in self.emitter1.parameters():
            param.requires_grad = False  # Freeze emitter1

        # Gating units for each dimension
        self.lin_gate_z_to_hidden = nn.Linear(z0_dim + z1_dim, emission_dim)
        self.lin_gate_hidden_to_z = nn.Linear(emission_dim, output_dim)  # Gate for each dimension

        # Emitter2
        self.lin_z_to_hidden = nn.Linear(z0_dim + z1_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, output_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.z0_dim = z0_dim

    def forward(self, z_t):
        # z_t is assumed to be of shape [batch_size, z0_dim + z1_dim]
        z0 = z_t[:, :self.z0_dim]  # First part for z0

        # Compute the mean using emitter1 (frozen parameters)
        loc1, _ = self.emitter1(z0)

        # Gate calculation (vector gate)
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t))  # [batch_size, emission_dim]
        gate = self.sigmoid(self.lin_gate_hidden_to_z(_gate))  # [batch_size, output_dim]

        # Emitter2 computations
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        loc2 = self.lin_hidden_to_loc(h2)

        # Apply the vector gate to the mean (element-wise)
        x_loc = (1 - gate) * loc1 + gate * loc2  # Weighted combination of loc1 and loc2

        # Variance (scale) does not change and is solely based on Emitter2
        x_scale = torch.exp(self.lin_hidden_to_scale(h2))  # log-variance

        return x_loc, x_scale

#------------------------------------------------ Step 1 ------------------------------------------------#

class InfoDPCCA(nn.Module):
    def __init__(
        self,
        x_dim=100,                # x dimensions
        y_dim=100,                # y dimensions
        z0_dim=100,               # z dimensions
        emission_x_dim=10,        # hidden dimensions of emitter for x
        emission_y_dim=10,        # hidden dimensions of emitter for y
        rnn_x_dim=600,            # hidden dimensions of RNN for x
        rnn_y_dim=600,            # hidden dimensions of RNN for y
        num_layers=1,             # RNN layers
        rnn_dropout_rate=0.0,     # RNN dropout rate
        beta=.01,                 # beta
        alpha=.01,                # alpha
    ):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")

        self.emitter_x = Emitter(x_dim, z0_dim, emission_x_dim)
        self.emitter_y = Emitter(y_dim, z0_dim, emission_y_dim)
        self.combiner_1_0 = Combiner(z0_dim, rnn_x_dim)
        self.combiner_2_0 = Combiner(z0_dim, rnn_y_dim)
        self.combiner_12_0 = Combiner(z0_dim, rnn_x_dim + rnn_y_dim)
        rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
        self.rnn_x = nn.GRU(#nn.RNN(#
            input_size=x_dim,
            hidden_size=rnn_x_dim,
            #nonlinearity="relu",#
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )
        self.rnn_y = nn.GRU(#nn.RNN(#
            input_size=y_dim,
            hidden_size=rnn_y_dim,
            #nonlinearity="relu",#
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_dropout_rate,
        )
        self.h_x_0 = nn.Parameter(torch.zeros(1, 1, rnn_x_dim))
        self.h_y_0 = nn.Parameter(torch.zeros(1, 1, rnn_y_dim))
        self.z0_dim = z0_dim
        self.emission_x_dim = emission_x_dim
        self.emission_y_dim = emission_y_dim
        self.beta = beta
        self.alpha = alpha
        #self.x_dim = x_dim
        #self.y_dim = y_dim

    def model(self, mini_batch_x, mini_batch_y):
        pyro.module("infodpcca", self)
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size

        h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
        rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
        h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
        rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)): # !!!!!!!
                with pyro.poutine.scale(scale=self.alpha):
                    prior_mean = torch.zeros(batch_size, self.z0_dim, device=self.device)
                    prior_scale = torch.ones(batch_size, self.z0_dim, device=self.device)
                    z_t = pyro.sample("z0_%d" % t, dist.Normal(prior_mean, prior_scale).to_event(1))

                z_10_loc, z_10_scale = self.combiner_1_0(rnn_x_output[:, t - 2, :])
                z_10_dist = dist.Normal(z_10_loc, z_10_scale).to_event(1)
                z_20_loc, z_20_scale = self.combiner_2_0(rnn_y_output[:, t - 2, :])
                z_20_dist = dist.Normal(z_20_loc, z_20_scale).to_event(1)

                with pyro.poutine.scale(scale=self.beta):
                    pyro.factor("z_10_%d" % t, z_10_dist.log_prob(z_t).sum(-1))
                    pyro.factor("z_20_%d" % t, z_20_dist.log_prob(z_t).sum(-1))

                x_loc, x_scale = self.emitter_x(z_t)
                x_t = pyro.sample("obs_x_%d" % t, dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch_x[:, t - 1, :])
                y_loc, y_scale = self.emitter_y(z_t)
                y_t = pyro.sample("obs_y_%d" % t, dist.Normal(y_loc, y_scale).to_event(1), obs=mini_batch_y[:, t - 1, :])

    def guide(self, mini_batch_x, mini_batch_y):
        pyro.module("infodpcca", self)
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size

        h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
        rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
        h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
        rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)
        rnn_output = torch.cat([rnn_x_output, rnn_y_output], dim=2)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                # ST-R: q(z_t | z_{t-1}, x_{t:T}, y_{t:T})
                z_loc, z_scale = self.combiner_12_0(rnn_output[:, t - 2, :])
                z_dist = dist.Normal(z_loc, z_scale)
                with pyro.poutine.scale(scale=self.alpha + 2*self.beta):
                    z_t = pyro.sample("z0_%d" % t, z_dist.to_event(1))

    def forward(self, mini_batch_x, mini_batch_y):
        # this return shared latent states
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        T_max = mini_batch_x.size(1)  # T
        batch_size = mini_batch_x.size(0)  # batch size

        h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
        rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
        h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
        rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)
        rnn_output = torch.cat([rnn_x_output, rnn_y_output], dim=2)

        recon_z_loc = torch.zeros(batch_size, T_max-1, self.z0_dim)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                # ST-R: q(z_t | z_{t-1}, x_{t:T}, y_{t:T})
                z_loc, _ = self.combiner_12_0(rnn_output[:, t - 2, :])
                recon_z_loc[:, t - 2, :] = z_loc

        return recon_z_loc


#------------------------------------------------ Step 2 ------------------------------------------------#
class InfoDPCCA_2(nn.Module):
    def __init__(
        self,
        x_dim=100,                # x dimensions
        y_dim=100,                # y dimensions
        z0_dim=100,               # z0 dimensions
        z1_dim=100,               # z1 dimensions
        z2_dim=100,               # z2 dimensions
        emission_x_dim=10,        # hidden dimensions of emitter for x
        emission_y_dim=10,        # hidden dimensions of emitter for y
        rnn_x_dim=600,            # hidden dimensions of RNN for x
        rnn_y_dim=600,            # hidden dimensions of RNN for y
        num_layers=1,             # RNN layers
        rnn_dropout_rate=0.0,  # RNN dropout rate
        rnn_x = None,             # RNN on x used in step I
        rnn_y = None,             # RNN on y used in step I
        h_x_0 = None,             # initial hidden state of RNN on x used in step I
        h_y_0 = None,             # initial hidden state of RNN on y used in step I
        combiner_12_0 = None,     # q_0^{12}(z_t^0|x_{1:t}^{1:2}) used in step I
        res_con = False,          # use of residual connection
        old_emitter_x = None,     # emitter for x used in step I
        old_emitter_y = None,     # emitter for y used in step I
        rnn4vi = True,            # use of a new RNN for VI
        rnn_vi_dim=600,           # hidden dimensions of RNN used for VI
    ):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.cuda()
        else:
            self.device = torch.device("cpu")

        # Use of residual connections
        self.res_con = res_con
        if res_con:
            self.emitter_x = Emitter_Res_Con(x_dim, z0_dim, z1_dim, emission_x_dim, old_emitter_x)
            self.emitter_y = Emitter_Res_Con(y_dim, z0_dim, z2_dim, emission_y_dim, old_emitter_y)
        else:
            self.emitter_x = Emitter(x_dim, z0_dim + z1_dim, emission_x_dim)
            self.emitter_y = Emitter(y_dim, z0_dim + z2_dim, emission_y_dim)

        # p_1^1(z^1|x^1)
        self.combiner_1_1 = Combiner(z1_dim, rnn_x_dim)
        # p_2^2(z^2|x^2)
        self.combiner_2_2 = Combiner(z2_dim, rnn_y_dim)

        # register attributes
        self.z0_dim = z0_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.emission_x_dim = emission_x_dim
        self.emission_y_dim = emission_y_dim
        self.res_con = res_con
        self.rnn4vi = rnn4vi
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Inherit trained modules from step I.
        self.combiner_12_0 = combiner_12_0
        self.rnn_x = rnn_x
        self.rnn_y = rnn_y
        self.h_x_0 = h_x_0
        self.h_y_0 = h_y_0

        # Freeze their parameters.
        for param in self.combiner_12_0.parameters():
            param.requires_grad = False
        for param in self.rnn_x.parameters():
            param.requires_grad = False
        for param in self.rnn_y.parameters():
            param.requires_grad = False
        self.h_x_0.requires_grad = False
        self.h_y_0.requires_grad = False

        # use of RNN backbone in VI
        if rnn4vi:
            rnn_dropout_rate = 0.0 if num_layers == 1 else rnn_dropout_rate
            self.rnn_vi = nn.GRU(#nn.RNN(#
                input_size=x_dim+y_dim,
                hidden_size=rnn_vi_dim,
                #nonlinearity="relu",#
                batch_first=True,
                bidirectional=False,
                num_layers=num_layers,
                dropout=rnn_dropout_rate,
            )
            self.h_vi_0 = nn.Parameter(torch.zeros(1, 1, rnn_vi_dim))
            # q(z_t^{0:2}|x_{1:t+1}^{1:2})
            self.combiner_vi = Combiner(z0_dim + z1_dim + z2_dim, rnn_vi_dim)
        else:
            #self.combiner_vi = Combiner2inputs(z0_dim + z1_dim + z2_dim, x_dim + y_dim, rnn_vi_dim)
            self.combiner_vi = Combiner(z0_dim + z1_dim + z2_dim, rnn_x_dim + rnn_y_dim)



    def model(self, mini_batch_x, mini_batch_y):
        pyro.module("infodpcca2", self)
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size

        h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
        rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
        h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
        rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)
        rnn_output = torch.cat([rnn_x_output, rnn_y_output], dim=2)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)): # !!!!!!!
                # z
                z0_loc, z0_scale = self.combiner_12_0(rnn_output[:, t - 2, :])
                z1_loc, z1_scale = self.combiner_1_1(rnn_x_output[:, t - 2, :])
                z2_loc, z2_scale = self.combiner_2_2(rnn_y_output[:, t - 2, :])
                zt_loc = torch.cat([z0_loc, z1_loc, z2_loc], dim=1)
                zt_scale = torch.cat([z0_scale, z1_scale, z2_scale], dim=1)
                zt_dist = dist.Normal(zt_loc, zt_scale)
                zt = pyro.sample("z_%d" % t, zt_dist.to_event(1))

                z0 = zt[:, :self.z0_dim]
                z1 = zt[:, self.z0_dim:self.z0_dim + self.z1_dim]
                z2 = zt[:, self.z0_dim + self.z1_dim:]

                # x
                z01 = torch.cat([z0, z1], dim=1)
                x_loc, x_scale = self.emitter_x(z01)
                x_t = pyro.sample("obs_x_%d" % t, dist.Normal(x_loc, x_scale).to_event(1), obs=mini_batch_x[:, t - 1, :])

                # y
                z02 = torch.cat([z0, z2], dim=1)
                y_loc, y_scale = self.emitter_y(z02)
                y_t = pyro.sample("obs_y_%d" % t, dist.Normal(y_loc, y_scale).to_event(1), obs=mini_batch_y[:, t - 1, :])

    def guide(self, mini_batch_x, mini_batch_y):
        pyro.module("infodpcca2", self)
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        T_max = mini_batch_x.size(1) # T
        batch_size = mini_batch_x.size(0) # batch size

        if self.rnn4vi:
            h_vi_0_contig = self.h_vi_0.expand(1, batch_size, self.rnn_vi.hidden_size).contiguous()
            rnn_vi_output, _ = self.rnn_vi(torch.cat([mini_batch_x, mini_batch_y], dim=2), h_vi_0_contig)
        else:
            h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
            rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
            h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
            rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)
            rnn_vi_output = torch.cat([rnn_x_output, rnn_y_output], dim=2)

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                # q(z_t^{0:2} | h_{t+1})
                z_loc, z_scale = self.combiner_vi(rnn_vi_output[:, t - 1, :])
                z_dist = dist.Normal(z_loc, z_scale)
                zt = pyro.sample("z_%d" % t, z_dist.to_event(1))


    def forward(self, mini_batch_x, mini_batch_y):
        T_max = mini_batch_x.size(1) # T
        mini_batch_x = mini_batch_x.to(self.device)
        mini_batch_y = mini_batch_y.to(self.device)
        batch_size = mini_batch_x.size(0) # batch size

        if self.rnn4vi:
            h_vi_0_contig = self.h_vi_0.expand(1, batch_size, self.rnn_vi.hidden_size).contiguous()
            rnn_vi_output, _ = self.rnn_vi(torch.cat([mini_batch_x, mini_batch_y], dim=2), h_vi_0_contig)
        else:
            h_x_0_contig = self.h_x_0.expand(1, batch_size, self.rnn_x.hidden_size).contiguous()
            rnn_x_output, _ = self.rnn_x(mini_batch_x, h_x_0_contig)
            h_y_0_contig = self.h_y_0.expand(1, batch_size, self.rnn_y.hidden_size).contiguous()
            rnn_y_output, _ = self.rnn_y(mini_batch_y, h_y_0_contig)
            rnn_vi_output = torch.cat([rnn_x_output, rnn_y_output], dim=2)

        # tensor containing reconstructions of x and y
        recon_x_loc = torch.zeros(batch_size, T_max, self.x_dim)
        recon_x_scale = torch.zeros(batch_size, T_max, self.x_dim)
        recon_y_loc = torch.zeros(batch_size, T_max, self.y_dim)
        recon_y_scale = torch.zeros(batch_size, T_max, self.y_dim)

        # tensor containing extracted z
        recon_z_loc = torch.zeros(batch_size, T_max, self.z0_dim + self.z1_dim + self.z2_dim)
        recon_z_scale = torch.zeros(batch_size, T_max, self.z0_dim + self.z1_dim + self.z2_dim)

        # since the model starts at the second time stamp, we input the original x_1.
        recon_x_loc[:, 0, :] = mini_batch_x[:, 0, :]
        recon_y_loc[:, 0, :] = mini_batch_y[:, 0, :]

        with pyro.plate("z_minibatch", batch_size):
            for t in pyro.markov(range(2, T_max + 1)):
                # q(z_t^{0:2} | h_{t+1})
                z_loc, z_scale = self.combiner_vi(rnn_vi_output[:, t - 1, :])
                recon_z_loc[:, t - 1, :] = z_loc # the first spots of recon_z_loc and recon_z_scale are empty.
                recon_z_scale[:, t - 1, :] = z_scale
                #z_dist = dist.Normal(z_loc, z_scale)
                #zt = pyro.sample("z_%d" % t, z_dist.to_event(1))
                z0 = z_loc[:, :self.z0_dim]
                z1 = z_loc[:, self.z0_dim:self.z0_dim + self.z1_dim]
                z2 = z_loc[:, self.z0_dim + self.z1_dim:]

                # x
                z01 = torch.cat([z0, z1], dim=1)
                x_loc, x_scale = self.emitter_x(z01)
                recon_x_loc[:, t - 1, :] = x_loc
                recon_x_scale[:, t - 1, :] = x_scale

                # y
                z02 = torch.cat([z0, z2], dim=1)
                y_loc, y_scale = self.emitter_y(z02)
                recon_y_loc[:, t - 1, :] = y_loc
                recon_y_scale[:, t - 1, :] = y_scale

        return recon_x_loc, recon_y_loc, recon_z_loc, recon_x_scale, recon_y_scale, recon_z_scale


#------------------------------------------------ Utilities ------------------------------------------------#
# Training one epoch of the training set
def train(svi, train_loader, true_latent=False):
    epoch_loss = 0.

    if true_latent:
        for x, y, _ in train_loader: # x is mini-batch
            epoch_loss += svi.step(x,y)
    else:
        for x, y in train_loader: # x is mini-batch
            epoch_loss += svi.step(x,y)

    # return average epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def recon_loss(model, test_loader, true_latent=False, fin=False):
    model.eval()
    total_squared_error = 0.0
    total_elements = 0

    if not fin:
        if true_latent:
            for mini_batch_x, mini_batch_y, _ in test_loader:
                mini_batch_x = mini_batch_x.to(model.device)
                mini_batch_y = mini_batch_y.to(model.device)
                # Get the model's reconstructions
                recon_x_loc, recon_y_loc, _, _, _, _ = model(mini_batch_x, mini_batch_y)
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
                recon_x_loc, recon_y_loc, _, _, _, _ = model(mini_batch_x, mini_batch_y)
                recon = torch.cat([recon_x_loc, recon_y_loc], dim=2).to(model.device)
                recon = recon[:, 1:, :]
                original = torch.cat([mini_batch_x, mini_batch_y], dim=2).to(model.device)
                original = original[:, 1:, :]  # Remove the first time step
                squared_diff = (original - recon) ** 2
                total_squared_error += torch.sum(squared_diff).item()
                total_elements += squared_diff.numel()
    else:
        for which_mini_batch, mini_batch_list in enumerate(test_loader):
            mini_batch_x = mini_batch_list[0].to(model.device)
            mini_batch_y = mini_batch_list[1].to(model.device)
            recon_x_loc, recon_y_loc, _, _, _, _ = model(mini_batch_x, mini_batch_y)
            recon = torch.cat([recon_x_loc, recon_y_loc], dim=2).to(model.device)
            recon = recon[:, 1:, :]
            original = torch.cat([mini_batch_x, mini_batch_y], dim=2).to(model.device)
            original = original[:, 1:, :]  # Remove the first time step
            squared_diff = (original - recon) ** 2
            total_squared_error += torch.sum(squared_diff).item()
            total_elements += squared_diff.numel()

    # Compute RMSE
    mse = total_squared_error / total_elements
    rmse = mse ** 0.5

    return rmse


def elbo_loss(svi, model, test_loader, true_latent=False):
    model.eval()
    total_elbo = 0.0
    multiplier = 0

    if true_latent:
        for mini_batch_x, mini_batch_y, _ in test_loader:
            mini_batch_x = mini_batch_x.to(model.device)
            mini_batch_y = mini_batch_y.to(model.device)
            total_elbo += svi.evaluate_loss(mini_batch_x, mini_batch_y)
            seq_length = mini_batch_x.size(1)
            batch_size = mini_batch_x.size(0)
            multiplier += seq_length * batch_size
    else:
        for mini_batch_x, mini_batch_y in test_loader:
            mini_batch_x = mini_batch_x.to(model.device)
            mini_batch_y = mini_batch_y.to(model.device)
            total_elbo += svi.evaluate_loss(mini_batch_x, mini_batch_y)
            seq_length = mini_batch_x.size(1)
            batch_size = mini_batch_x.size(0)
            multiplier += seq_length * batch_size

    model.train()
    return -total_elbo / multiplier

# Metrics for correlations

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


def smc_loss(model, test_loader, device='cpu'):
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
            _, _, recon_z, _, _, _ = model(mini_batch_x, mini_batch_y)

            # If necessary, slice out the relevant parts:
            recon_z0 = recon_z[:, 1:, :model.z0_dim]
            true_z0 = mini_batch_z[:, :-1, :model.z0_dim]
            # For demonstration, let's assume you want the entire recon_z vs mini_batch_z:
            Z = true_z0
            Z_hat = recon_z0

            #print(Z.shape, Z_hat.shape)

            # Compute SMGC and SMLC for this batch
            batch_smgc = smgc_metric(Z, Z_hat)
            batch_smlc = smlc_metric(Z, Z_hat)

            total_smgc += batch_smgc
            total_smlc += batch_smlc
            num_batches += 1

    avg_smgc = total_smgc / num_batches
    avg_smlc = total_smlc / num_batches
    return avg_smgc, avg_smlc

#Visualizatoin of Reconstruction
def recon_fig(model, test_loader, dim_idx, data_idx=0, fin=False):
    # by default, we select first test sample of x_1
    model.eval()

    if fin == True:
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

    #print(mini_batch_label)

    # Get the model's reconstructions
    recon_x_loc, recon_y_loc, recon_z_loc, recon_x_scale, recon_y_scale, recon_z_scale = model(mini_batch_x[data_idx:data_idx+1, :, :], mini_batch_y[data_idx:data_idx+1, :, :])

    x1 = mini_batch_x[data_idx, :, dim_idx].cpu().detach().numpy()
    x1_recon_loc = recon_x_loc[0, :, dim_idx].cpu().detach().numpy()
    x1_recon_scale = recon_x_scale[0, :, dim_idx].cpu().detach().numpy()

    # Calculate 95% CI for the reconstructed sequence
    lower_bound = x1_recon_loc - 2 * x1_recon_scale
    upper_bound = x1_recon_loc + 2 * x1_recon_scale

    # Plot the original and reconstructed sequences for the first feature of x1
    plt.figure(figsize=(10, 6))
    plt.plot(x1, label='Original x1', linestyle='-', marker='o')
    plt.plot(x1_recon_loc, label='Reconstructed x1', linestyle='--', marker='x')
    plt.fill_between(range(len(x1_recon_loc)), lower_bound, upper_bound, color='gray', alpha=0.3,
                      label='95% CI')
    if fin:
        plt.axvline(x=17, color='red', linestyle='--')
    plt.title('InfoDPCCA: Original vs Reconstructed Sequence for x1')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    squared_diff = (x1 - x1_recon_loc) ** 2
    #print(x1[-2], x1_recon_loc[-2], (x1[-2]- x1_recon_loc[-2]))
    #return squared_diff
    return sum(squared_diff)


def recon_fig_fin(model, test_loader, dim_idx, data_idx=0):
    # by default, we select first test sample of x_1
    model.eval()


    for which_mini_batch, mini_batch_list in enumerate(test_loader):
        mini_batch_x = mini_batch_list[0].to(model.device)
        mini_batch_y = mini_batch_list[1].to(model.device)
        break


    #print(mini_batch_label)

    # Get the model's reconstructions
    recon_z_loc = model(mini_batch_x[data_idx:data_idx+1, :, :], mini_batch_y[data_idx:data_idx+1, :, :])
    z = recon_z_loc[0, :, dim_idx].cpu().detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(z, label='z0', linestyle='--', marker='x')
    plt.axvline(x=17, color='red', linestyle='--')
    plt.title('InfoDPCCA: Shared Latent State')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

