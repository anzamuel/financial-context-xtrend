# %% IMPORTS
#from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
#import datetime as dt
import torch
import random

#from empyrical import sharpe_ratio

#import os
#import json

from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm



# %% CONSTANTS
TESTING = False
# TRAIN_LAST_OUTPUT = 0  # more items in sequence is neccessary!
TRAIN_LAST_OUTPUT = 1
LATENT_PATH = False
VAL_DIVERSIFIED_SHARPE = True

# BATCH_SIZE = 64
BATCH_SIZE = 252
# NUM_CONTEXT = 10
NUM_CONTEXT = 10
ITERATIONS = 200
# MIN_SEQ_LEN = 5
# MIN_SEQ_LEN = 5

# REMOVE_FIRST_N_OUTPUTS = 3
REMOVE_FIRST_N_OUTPUTS = 0

N = 50

# LR = 0.001
LR = 0.01

# LEN_CONTEXT = 6
# BATCH_SIZE = 256

# EMBEDDING_DIM = 32
EMBEDDING_DIM = 64

ENCODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]
DECODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]

LATENT_DIM = 4
# LATENT_DIM = 4
USE_X_ATTENTION = True
USE_SELF_ATTENTION = True

VALID_START_YEAR = 2005  # valid 2015-208

TEST_START_YEAR = 2010  # valid 2015-208



# %% Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# %% READ CONTEXT DATA
# Removed if TESTING: ...
all_segments = pd.read_pickle("context_data.pkl")
all_segments['x'] = all_segments['x'].apply(lambda item: torch.from_numpy(item).unsqueeze(0))
all_segments['y'] = all_segments['y'].apply(lambda item: torch.from_numpy(item).unsqueeze(0))

# all_segments = all_segments[all_segments["seq_len"] >= MIN_SEQ_LEN].copy()
#print(all_segments['x'].iloc[0])



# %% ONLY CONSIDER CONTEXT BETWEEN VALID_START_YEAR, TEST_START_YEAR and with at least NUM_CONTEXT entries
all_segments.index = all_segments.groupby(["seq_len", "ticker"]).cumcount()
#print(all_segments.groupby(["seq_len", "ticker"]).cumcount())

# %%
all_segments["day"] = all_segments.groupby("date").ngroup()
#print(all_segments)

# %%
all_segments_day = all_segments.set_index("day")
all_segments_day_dict = {}
for s in all_segments_day.seq_len.unique():
    all_segments_day_dict[s] = all_segments_day[all_segments_day["seq_len"] == s]

# %%
targets = all_segments[all_segments.index > NUM_CONTEXT]
train_contexts = targets[targets.date.dt.year < VALID_START_YEAR]
valid_contexts = targets[
    (targets.date.dt.year >= VALID_START_YEAR)
    & (targets.date.dt.year < TEST_START_YEAR)
]



# %% FUNC PREPARE_BATCHES
def prepare_batches(targets, keep_partial_batches=False):
    # shuffled = train_contexts.sample(frac=1)
    shuffled = targets.sample(frac=1)
    shuffled["batch"] = shuffled.groupby("seq_len").cumcount() // BATCH_SIZE
    grouped = shuffled.groupby(["seq_len", "batch"])
    if not keep_partial_batches:
        shuffled = grouped.filter(lambda x: x["batch"].count() == BATCH_SIZE)
    shuffled.index = shuffled.groupby(["seq_len", "batch"]).ngroup()
    num_batches = shuffled.index.max()
    order = list(range(num_batches))
    random.shuffle(order)
    # print(order)
    # print(shuffled.loc[order])

    shuffled["contexts"] = shuffled.apply(
        lambda row: all_segments_day_dict[row.seq_len]
        # .loc[: (row.day - 1)]
        .loc[lambda df: df.index < row.day][["x", "y"]].sample(n=NUM_CONTEXT),
        axis=1,
    )

    shuffled["context_x"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["x"].tolist(), dim=3)
    )
    shuffled["context_y"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["y"].tolist(), dim=3)
    )

    # print(shuffled["context_x"].iloc[0].shape)
    batches = []
    for i in tqdm(order):
        batch = shuffled.loc[[i]]
        # print(batch)
        # print(torch.cat(batch["context_x"].tolist()).shape)
        batches.append(
            (
                batch["seq_len"].iloc[0],
                torch.cat(batch["context_x"].tolist()),
                torch.cat(batch["context_y"].tolist()),
                torch.cat(batch["x"].tolist()),
                torch.cat(batch["y"].tolist()),
                batch["date"].tolist(),
                batch["ticker"].tolist(),
            )
        )

    # print(shuffled["context_x"].iloc[0].shape)
    # print(batches[0])
    # print(list(map(lambda b: b[1].size()[0], batches)))
    # print(batches["x"].map(lambda x: len(x)))
    return batches

#batches = prepare_batches(train_contexts)

# %% Test batches dim
#batch = batches[1]
#print(batch[1].size(), batch[2].size())


# %% CLASS MLP
class MLP(nn.Module):
    '''
    Apply MLP to the final axis of a 3D tensor
    '''
    def __init__(self, input_size, output_size_list):
        '''
        Parameters:
        -input_size (int): number of dimensions for each point in each sequence.
        -output_size_list (list of ints): number of output dimensions for each layer.
        '''
        super().__init__()
        self.input_size = input_size  # e.g. 2
        self.output_size_list = output_size_list  # e.g. [128, 128, 128, 128]
        network_size_list = [input_size] + self.output_size_list  # e.g. [2, 128, 128, 128, 128]
        network_list = []

        # iteratively build neural network.
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i-1], network_size_list[i], bias=True))
            network_list.append(nn.ReLU())

        # Add final layer, create sequential container.
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        return self.mlp(x)



# %% CLASS AttnLinear
class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x



# %% CLASS Attention
class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type,
        n_heads=4,
        x_dim=1,
        rep="mlp",
        dropout=0,
        mlp_hidden_dim_list=[],
    ):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.mlp_k = MLP(x_dim, mlp_hidden_dim_list)
            self.mlp_q = MLP(x_dim, mlp_hidden_dim_list)

        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        elif attention_type == "ptmultihead":
            self._W = torch.nn.MultiheadAttention(
                hidden_dim, n_heads, bias=True, dropout=dropout
            )
            self._attention_func = self._pytorch_multihead_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.mlp_k(k)
            q = self.mlp_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep

    def _laplace_attention(self, k, v, q, scale=0.5):
        # TODO: re-implement in the future as this runs too slowly.

        b, n, _ = k.size()
        b, m, _ = q.size()

        weights = torch.empty(b, m, n)
        for h in range(b):
            print(h)
            for i in range(m):
                for j in range(n):
                    weights[h][i][j] = -1.0 * torch.sum((q[h][i] - k[h][j]), dim=-1).item()
                weights[h][i] = torch.softmax(weights[h][i], dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        o = self._W(q, k, v)[0]
        return o.permute(1, 0, 2)



# %% CLASS DeterministicLSTMEncoder
class DeterministicLSTMEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim=32,  # the dim of last axis of r..
        context_concat="stack",
        self_attention_type="dot",
        use_self_attn=False,
        use_x_attn=False,
        cross_attention_type="dot",
    ):
        super().__init__()
        self.concat = context_concat
        # stacking x and y and encoding
        if self.concat == "stack":
            self.input_dim = x_dim + y_dim

        # appending y to x and encoding
        elif self.concat == "append":
            self.input_dim = x_dim
        else:
            raise ValueError
        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.embedding_dim = embedding_dim
        self.use_self_attn = use_self_attn
        self.use_x_attn = use_x_attn

        self.num_rnn_layers = 1
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_key_enc = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)
        self.rnn_query_enc = nn.LSTM(x_dim, self.hidden_dim, self.num_rnn_layers)

        if embedding_dim != hidden_dim_list[-1]:
            print("Warning, Check the dim of latent z and the dim of mlp last layer!")

        if self.use_self_attn:
            self._self_attention = Attention(
                self.hidden_dim,
                attention_type=self_attention_type,
                rep="",  # can use mlp, have already encoded sequences using an LSTM so not needed
            )

        if self.use_x_attn:
            self._cross_attention = Attention(
                hidden_dim=self.hidden_dim,
                attention_type=cross_attention_type,
                x_dim=x_dim,
                rep="",  # can use mlp, have already encoded sequences using an LSTM so not needed
            )

    def forward(self, context_x, context_y, target_x=None):
        (
            batch_sz,
            context_len,
            y_dim,
            num_contexts,
        ) = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]

        if self.use_x_attn:
            # Encode the context_x and context_y as Values
            # Each time-series in the context is encoded such that a single
            # hidden state represents an encoding of the context
            # Handle different possible tensor shapes
            target_shape = target_x.size()
            _, target_len, _ = target_shape  # [batch_size, target_len, y_size]

            hidden_v = torch.zeros(
                batch_sz, context_len, num_contexts, self.hidden_dim
            ).to(device)
            for i in range(num_contexts):
                if self.concat == "stack":
                    # h0 = torch.randn(
                    #     self.num_rnn_layers, context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # hidden states
                    # c0 = torch.randn(
                    #     self.num_rnn_layers, context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                    )  # (b, seq_len, 2 * y_dim)
                elif self.concat == "append":
                    # h0 = torch.randn(
                    #     self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # hidden states
                    # c0 = torch.randn(
                    #     self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    # ).to(
                    #     device
                    # )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                    )  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(
                    # encoder_input, (h0, c0)
                    encoder_input
                )  # (b, context_seq_len, latent_dim)

                hidden_v[:, :, i, :] = hidden_r_i  # (b, hidden_dim)

            # added to do self attention for each timestep
            shape = hidden_v.shape
            hidden_v = hidden_v.reshape((shape[0] * shape[1], shape[2], shape[3]))

            # self attention over values
            if self.use_self_attn:
                hidden_v = self._self_attention(hidden_v, hidden_v, hidden_v)

            # KW maybe just keep hidden and cell as 0 for now...
            # # Encode the context_x as Keys
            # h0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(
            #     device
            # )
            # c0 = torch.randn(self.num_rnn_layers, context_len, self.hidden_dim).to(
            #     device
            # )

            hidden_k = torch.zeros(
                batch_sz, context_len, num_contexts, self.hidden_dim
            ).to(device)
            for i in range(num_contexts):
                # hidden_r_i, _ = self.rnn_key_enc(context_x[:, :, :, i], (h0, c0))
                hidden_k_i, _ = self.rnn_key_enc(context_x[:, :, :, i])
                hidden_k[:, :, i, :] = hidden_k_i

            hidden_k = hidden_k.reshape((shape[0] * shape[1], shape[2], shape[3]))

            # set for c0, h0 to 0 for now...
            # # Encoder the target_x as Query
            # h0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(
            #     device
            # )
            # c0 = torch.randn(self.num_rnn_layers, target_len, self.hidden_dim).to(
            #     device
            # )
            # hidden_r_i, _ = self.rnn_query_enc(
            #     target_x, (h0, c0)
            # )  # (b, num_contexts, hidden_dim)
            hidden_q, _ = self.rnn_query_enc(target_x)

            shape_q = hidden_q.shape
            hidden_q = hidden_q.reshape(shape_q[0] * shape_q[1], 1, shape_q[2])

            # hidden_k: (b, num_contexts, hidden_dim)
            # hidden_v: (b, num_contexts, hidden_dim)
            # hidden_q: (b, 1, hidden_dim)
            representation = self._cross_attention(
                hidden_k, hidden_v, hidden_q
            ).reshape(shape_q)

        else:
            # TODO only done X attention path so far
            representation = torch.zeros(batch_sz, self.hidden_dim).to(device)
            for i in range(num_contexts):
                if self.concat == "stack":
                    h0 = torch.randn(
                        self.num_rnn_layers, context_len, self.hidden_dim
                    ).to(
                        device
                    )  # hidden states
                    c0 = torch.randn(
                        self.num_rnn_layers, context_len, self.hidden_dim
                    ).to(
                        device
                    )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                    )  # (b, seq_len, 2 * y_dim)
                elif self.concat == "append":
                    h0 = torch.randn(
                        self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    ).to(
                        device
                    )  # hidden states
                    c0 = torch.randn(
                        self.num_rnn_layers, 2 * context_len, self.hidden_dim
                    ).to(
                        device
                    )  # cell states
                    encoder_input = torch.cat(
                        [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                    )  # (b, 2 * seq_len, y_dim)
                else:
                    raise ValueError

                hidden_r_i, _ = self.rnn(
                    encoder_input, (h0, c0)
                )  # (b, context_seq_len, latent_dim)

                representation += hidden_r_i[:, -1, :]  # (b, hidden_dim)
        return representation



# %% CLASS LatentLSTMEncoder
class LatentLSTMEncoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim,
        latent_dim,
        context_concat="stack",
        use_self_attn=False,
        self_attention_type="dot",
    ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.concat = context_concat
        if self.concat == "stack":
            self.input_dim = x_dim + y_dim
        # appending y to x and encoding
        elif self.concat == "append":
            self.input_dim = x_dim
        else:
            raise ValueError
        self.num_rnn_layers = 1
        self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, self.num_rnn_layers)
        self.mean_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.log_var_transform = nn.Linear(self.hidden_dim, self.latent_dim)
        self.use_self_attn = use_self_attn
        # self.self_attention = Attention(n_units, self_attention_type)

    def forward(self, context_x, context_y, state=None):

        (
            batch_sz,
            context_x_len,
            y_dim,
            num_contexts,
        ) = context_x.size()  # [batch_size, seq_len, y_size, num_contexts]
        (
            _,
            context_y_len,
            _,
            _,
        ) = context_y.size()  # [batch_size, seq_len, y_size, num_contexts]

        representation = torch.zeros(batch_sz, context_x_len, self.hidden_dim).to(
            device
        )
        for i in range(num_contexts):
            if self.concat == "stack":
                if state is None:
                    assert context_x_len == context_y_len
                    # h0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # hidden states
                    # c0 = torch.randn(self.num_rnn_layers, context_x_len, self.hidden_dim).to(device)  # cell states
                    # state = (h0, c0)
                encoder_input = torch.cat(
                    [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
                )  # (b, seq_len, 2 * y_dim)
            elif self.concat == "append":
                # if state is None:
                # h0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # hidden states
                # c0 = torch.randn(self.num_rnn_layers, context_x_len + context_y_len, self.hidden_dim).to(device)  # cell states
                # state = (h0, c0)
                encoder_input = torch.cat(
                    [context_x[:, :, :, i], context_y[:, :, :, i]], dim=1
                )  # (b, 2 * seq_len, y_dim)
            else:
                raise ValueError

            hidden_r_i, _ = self.rnn(encoder_input)  # (b, context_seq_len, latent_dim)

            if self.use_self_attn:
                pass

            # average
            # TODO can remove loop
            representation += hidden_r_i / num_contexts  # (b, hidden_dim)

        # r = self.self_attention(r, r, r)
        mu = self.mean_transform(representation)
        log_var = self.log_var_transform(representation)

        sigma = 0.1 + 0.9 * F.softplus(0.5 * log_var)

        dist = torch.distributions.Normal(mu, sigma)

        z = mu + sigma * torch.randn_like(mu)
        return z, dist, state



# %% CLASS DeterministicLSTMDecoder
class DeterministicLSTMDecoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim,  # the dim of last axis of x, r and z..
        x_attn_repr=False,
    ):
        """

        :params x_dim: int
        :params y_dim: int
        :params hidden_dim_list: list
        :params embedding_dim: int
        """
        super(DeterministicLSTMDecoder, self).__init__()

        self.hidden_dim_list = hidden_dim_list
        self.hidden_dim = hidden_dim_list[-1]
        self.input_dim = embedding_dim + x_dim
        self.x_attn_repr = x_attn_repr
        self.rnn = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, y_dim),
            nn.Tanh(),
        )

    def forward(
        self, r, target_x, target_y, z=None, testing=False, teacher_forcing=False
    ):
        # r:        (b, hidden_dim)
        # z:        (b, latent_dim)
        # target_x: (b, target_x_seq_len, x_dim)
        # target_y: (b, target_y_seq_len, x_dim)
        batch_size, len_target_x, _ = target_x.size()
        _, len_target_y, _ = target_y.size()

        # h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        # c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        # concatenate target_x and representation
        # r_tiled = torch.tile(
        #     r.unsqueeze(1), (1, len_target_x, 1)
        # )  # (b, target_len, hidden_dim)
        if z is not None:
            # z_tiled = torch.tile(
            #     z.unsqueeze(1), (1, len_target_x, 1)
            # )  # (b, target_len, hidden_dim)
            decoder_input = torch.cat(
                [z, r, target_x], dim=-1
            )  # (b, target_len, latent_dim + hidden_dim + x_dim)
        else:
            decoder_input = torch.cat(
                [r, target_x], dim=-1
            )  # (b, target_len, hidden_dim + x_dim)

        # encode the target sequence

        # TODO think we can do this without looping

        # h = torch.randn(batch_size, self.hidden_dim).to(device)  # hidden states
        # c = torch.randn(batch_size, self.hidden_dim).to(device)  # cell states

        h = torch.zeros((batch_size, self.hidden_dim)).to(device)  # hidden states
        c = torch.zeros((batch_size, self.hidden_dim)).to(device)  # cell states

        h_vector = torch.zeros(batch_size, len_target_x, self.hidden_dim).to(device)

        h_list = []
        for i in range(len_target_x):
            h, c = self.rnn(decoder_input[:, i, :], (h, c))  # (b, hidden_dim)
            h_list.append(h)

        h_vector = torch.stack(h_list).swapaxes(0, 1)

        # mu_sigma = self.out(h) # final prediction only (b, 2 * y_dim)

        # mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1) # mu/sigma (b, 1)

        # # # Bound the variance
        # sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # # Get the distribution
        # dist = torch.distributions.Normal(mu, sigma)

        positions = self.out(h_vector)  # final prediction only (b, 2 * y_dim)

        captured_positions = y_target * positions

        if testing:
            return captured_positions

        if TRAIN_LAST_OUTPUT:
            captured_positions = captured_positions[:, -TRAIN_LAST_OUTPUT:, :]

        if REMOVE_FIRST_N_OUTPUTS:
            captured_positions = captured_positions[:, REMOVE_FIRST_N_OUTPUTS:, :]


        # TODO replace with a tensor sqrt
        # sharpe = torch.mean(
        #     torch.mean(captured_positions, dim=1)
        #     / (torch.std(captured_positions, dim=1) + 1e-9)
        # ) * np.sqrt(252.0)
        sharpe = (
            torch.mean(captured_positions) / (torch.std(captured_positions) + 1e-9)
        ) * np.sqrt(252.0)
        return -sharpe



# %% CLASS ANP_RNN_Model
class ANP_RNN_Model(nn.Module):
    """
    (Attentive) Neural Process model
    https://github.com/VersElectronics/Neural-Processes/blob/master/neural_process_models/anp_rnn.py
    """

    def __init__(
        self,
        x_dim,
        y_dim,
        encoder_rnn_hidden_size_list,
        decoder_rnn_hidden_size_list,
        embedding_dim,
        latent_dim,
        context_concat="stack",
        use_self_attention=False,
        use_cross_attention=False,
        le_self_attention_type="dot",
        de_self_attention_type="dot",
        de_cross_attention_type="multihead",
        latent_path=False,
    ):
        """
        :params x_dim: int
        :params y_dim: int
        :params encoder_rnn_hidden_size_list: list
        :params decoder_rnn_hidden_size_list: list
        :params embedding_dim: int
        """
        super(ANP_RNN_Model, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.encoder_rnn_hidden_size_list = encoder_rnn_hidden_size_list
        self.decoder_rnn_hidden_size_list = decoder_rnn_hidden_size_list
        self.embedding_dim = embedding_dim
        self.num_rnn_layers = 1
        self.use_cross_attention = use_cross_attention
        self.latent_dim = latent_dim
        self.latent_path = latent_path

        self._deter_encoder = DeterministicLSTMEncoder(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            hidden_dim_list=self.encoder_rnn_hidden_size_list,
            embedding_dim=self.embedding_dim,  # the dim of last axis of r..
            context_concat=context_concat,
            use_x_attn=use_cross_attention,
            cross_attention_type=de_cross_attention_type,
            use_self_attn=use_self_attention,
            self_attention_type=de_self_attention_type,
        )

        self._decoder = DeterministicLSTMDecoder(
            x_dim=self.x_dim,
            y_dim=self.y_dim,
            hidden_dim_list=self.decoder_rnn_hidden_size_list,
            embedding_dim=self.embedding_dim + self.latent_dim
            if self.latent_path
            else self.embedding_dim,
            x_attn_repr=use_cross_attention,
        )

        if self.latent_path:
            self._lat_encoder = LatentLSTMEncoder(
                x_dim=self.x_dim,
                y_dim=self.y_dim,
                hidden_dim=self.encoder_rnn_hidden_size_list[-1],
                latent_dim=self.latent_dim,
                context_concat=context_concat,
            )

    def forward(self, context_x, context_y, target_x, target_y, testing=False):
        # If testing is True we unroll the LSTM and use the previous timestep predictions
        # as an input for a new prediction.

        r = self._deter_encoder(
            context_x, context_y, target_x if self.use_cross_attention else None
        )

        if self.latent_path:
            # TODO pass state from latent encoder context to latent encoding of the targets
            z, prior_dist, _ = self._lat_encoder(
                context_x, context_y
            )  # z (b, latent_dim)
            z_post, post_dist, _ = self._lat_encoder(
                target_x.unsqueeze(-1), target_y.unsqueeze(-1)
            )
        else:
            z = None
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        batch_size, len_target_y, _ = target_y.size()

        loss_or_pos_testing = self._decoder(r, target_x, target_y, z, testing=testing)

        return loss_or_pos_testing


# %% TRAINING LOOP
model = ANP_RNN_Model(
    x_dim=8,
    y_dim=1,
    encoder_rnn_hidden_size_list=ENCODER_OUTPUT_SIZES,
    decoder_rnn_hidden_size_list=DECODER_OUTPUT_SIZES,
    embedding_dim=EMBEDDING_DIM,
    latent_dim=LATENT_DIM,
    context_concat="stack",
    use_cross_attention=USE_X_ATTENTION,
    use_self_attention=USE_SELF_ATTENTION,
    de_cross_attention_type="dot",
    de_self_attention_type="dot",
    latent_path=LATENT_PATH,
)

model = model.to(device)

# Set up the optimizer and train step
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# probably need to separate the target data out earlier

best_valid_sharpe = 0
epoch_sharpes = []
for it in range(ITERATIONS):
    print(f"Iteration {it}")

    train_sharpes = []
    # train_mse = []

    print("Prepping")
    batches = prepare_batches(train_contexts)
    print(len(batches))
    print("Ready")
    pbar = tqdm(batches, desc=f"Iteration {it} | Train Sharpe: N/A", leave=True)
    for seq_len, x_context, y_context, x_target, y_target, _, _ in pbar:

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        train_loss = model.forward(
            x_context,
            y_context,
            x_target,
            y_target,
        )
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # train_mse.append(
        #     (torch.mean((pred_y[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
        # )
        curr_sharpe = (-train_loss).detach().item()
        train_sharpes.append(curr_sharpe)
        # Update progress bar with current stats
        avg_sharpe = np.mean(train_sharpes)
        pbar.set_description(f"Iteration {it} | Train Sharpe: {avg_sharpe:.4f} | Current: {curr_sharpe:.4f}")
        if TESTING:
            break

    # print(train_sharpes)
    # epoch_sharpes.append(np.mean(sharpes))
    print("Train Sharpe: ", np.mean(train_sharpes))

    valid_sharpes = []

    # train_mse = []

    all_returns = []
    targ = valid_contexts if not TESTING else train_contexts
    val_batches = prepare_batches(targ, keep_partial_batches=True)
    val_pbar = tqdm(val_batches, desc=f"Validation | Valid Sharpe: N/A", leave=True)
    for (
        seq_len,
        x_context,
        y_context,
        x_target,
        y_target,
        dates,
        tickers,
    ) in val_pbar:
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        valid_loss = model.forward(
            x_context, y_context, x_target, y_target, VAL_DIVERSIFIED_SHARPE
        )

        # valid_loss.backward()

        # valid_mse.append(
        #     (torch.mean((pred_y[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
        # )

        if VAL_DIVERSIFIED_SHARPE:
            captured_returns = valid_loss
            if captured_returns.shape[0] == BATCH_SIZE:
                valid_sharpes.append(
                    (
                        torch.mean(captured_returns)
                        / (torch.std(captured_returns) + 1e-9)
                        * np.sqrt(252.0)
                    )
                    .detach()
                    .item()
                )
            captured_returns = valid_loss[:, -1, 0].tolist()
            all_returns += zip(dates, tickers, captured_returns)

        else:
            curr_val_sharpe = (-valid_loss).detach().item()
            valid_sharpes.append(curr_val_sharpe)
            val_pbar.set_description(f"Validation | Valid Sharpe: {np.mean(valid_sharpes):.4f} | Current: {curr_val_sharpe:.4f}")

    # print(valid_sharpes)
    # epoch_sharpes.append(np.mean(sharpes))
    if VAL_DIVERSIFIED_SHARPE:
        diversified = (
            pd.DataFrame(all_returns, columns=["date", "ticker", "captured_returns"])
            .groupby("date")["captured_returns"]
            .sum()
        )
        val_sharpe = diversified.mean() * np.sqrt(252) / diversified.std()
        # print(valid_sharpes)
        print("Div Valid Sharpe: ", val_sharpe)
        print("Valid Single Sharpe: ", np.mean(valid_sharpes))
    else:
        print("Valid Sharpe: ", np.mean(valid_sharpes))

    # # TESTING
    # test_results = (
    #     test_data_prepped_all_segments[[]]
    #     .copy()
    #     .assign(captured_return=0.0)
    #     # .assign(mse=0.0)
    #     # .assign(pred=0.0)
    #     # .assign(std=0.0)
    #     .sort_index()
    # )

    # for key, test in test_data_prepped_all_segments.iterrows():
    #     seq_len, ticker, date = key
    #     context_points = train_data_prepped_all_segments.loc[(seq_len, ticker)].sample(
    #         n=NUM_CONTEXT
    #     )

    #     x_context = torch.cat(context_points["x"].tolist(), axis=3).to(device)
    #     y_context = torch.cat(context_points["y"].tolist(), axis=3).to(device)
    #     x_target = test["x"].to(device)
    #     y_target = test["y"].to(device)

    #     captured_pos = model.forward(
    #         x_context, y_context, x_target, y_target, testing=True
    #     )

    #     # test_results.loc[key, "mse"] = (
    #     #     (torch.mean((pred_y_test[:, :, :] - y_target[:, -1:, :]) ** 2)).detach().item()
    #     # )

    #     # test_results.loc[key, "pred"] = (
    #     #     torch.mean(pred_y_test[:, :, :])
    #     #     .detach()
    #     #     .item()
    #     # )

    #     # test_results.loc[key, "std"] = (
    #     #             torch.mean(sigma[:, :, :])
    #     #             .detach()
    #     #             .item()
    #     #         )

    #     test_results.loc[key, "captured_return"] = (
    #         captured_pos[-1, -1, -1].detach().item()
    #     )
    #     # break

    # test_results = test_results.reset_index()
    # valid_results = test_results[
    #     test_results["date"] < dt.datetime(TEST_START_YEAR, 1, 1)
    # ]
    # # test_results = test_results[
    # #     test_results["date"] >= dt.datetime(TEST_START_YEAR, 1, 1)
    # # ]

    # valid_results_port = valid_results.groupby("date")["captured_return"].sum() / N
    # # print(test_results)
    # valid_sharpe = (
    #     np.mean(valid_results_port) / np.std(valid_results_port) * np.sqrt(252)
    # )
    # print("Valid Sharpe Port: ", valid_sharpe)
    # print()

    # # test_results_port = test_results.groupby("date")["captured_return"].sum() / N
    # # test_sharpe = np.mean(test_results_port) / np.std(test_results_port) * np.sqrt(252)
    # # print("Test Sharpe Port: ", test_sharpe)

    # # if valid_sharpe >= best_valid_sharpe:
    # #     best_valid_sharpe = valid_sharpe

    # #     if not os.path.exists("results"):
    # #         os.mkdir("results")
    # #     valid_results.to_csv(os.path.join("results", RUN_NAME + "_valid.csv"))
    # #     test_results.to_csv(os.path.join("results", RUN_NAME + "_test.csv"))
    # #     with open(os.path.join("results", RUN_NAME + "_results.json"), "w", encoding="utf-8") as f:
    # #         # TODO other settings results
    # #         json.dump(
    # #             {
    # #                 "valid_sharpe": valid_sharpe,
    # #                 "test_sharpe": test_sharpe,
    # #                 "iteration": it,
    # #             },
    # #             f,
    # #             indent=4,
    #         )
