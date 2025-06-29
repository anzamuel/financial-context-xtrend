import torch
from torch import nn
from models.attention import Attention
from models.temporal_block import TemporalBlock


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim=32,  # the dim of last axis of r..
        self_attention_type="dot",
        cross_attention_type="dot",
        n_heads=4,
        static_dim=8,
        vsn_dim=32,
        ffn_dim=64,
        lstm_hidden_dim=32,
        dropout = 0.5,
    ):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.input_dim = x_dim + y_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_heads = n_heads
        self.hidden_dim = ffn_dim
        self.embedding_dim = embedding_dim


        self.temporal_block = TemporalBlock(
            input_dim=self.input_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
            dropout=dropout,
        )
        self.temporal_block_key = TemporalBlock(
            input_dim=x_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
            dropout=dropout,
        )

        self.temporal_block_query = TemporalBlock(
            input_dim=x_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
            dropout=dropout,
        )


        self._self_attention = Attention(
            self.hidden_dim,
            attention_type=self_attention_type,
            n_heads=self.n_heads,
        )

        self._cross_attention = Attention(
            hidden_dim=self.hidden_dim,
            attention_type=cross_attention_type,
            n_heads=n_heads,
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(self.hidden_dim)


    def forward(self, context_x, context_xi, target_x=None,
        embedding_context_s = None, embedding_target_s=None):
        batch_sz, context_len, _, num_contexts = context_x.shape

        # Vectorized context processing
        encoder_input = context_xi.permute(0, 3, 1, 2).reshape(batch_sz * num_contexts, context_len, -1)
        context_x_reshaped = context_x.permute(0, 3, 1, 2).reshape(batch_sz * num_contexts, context_len, -1)
        static_s_reshaped = embedding_context_s.reshape(batch_sz * num_contexts, -1)
        # Pass through temporal block and take the last LSTM cell output instead of mean
        v_i = self.temporal_block(encoder_input, static_s_reshaped)[:, -1, :]  # Take last time step
        Vt = v_i.view(batch_sz, num_contexts, -1)
        k_i = self.temporal_block_key(context_x_reshaped, static_s_reshaped)[:, -1, :]
        Kt = k_i.view(batch_sz, num_contexts, -1)

        Vt_prime = self._self_attention(Vt, Vt, Vt)
        Vt_prime_2 = self.ffn2(Vt_prime)

        qt = self.temporal_block_query(target_x, embedding_target_s)
        representation = self._cross_attention(Kt, Vt_prime_2, qt)
        output = self.ffn(representation)
        norm_output = self.layer_norm(output)
        return norm_output

if __name__ == "__main__":
    torch.manual_seed(0)

    # ----------- Configuration -----------
    batch_size = 4
    context_len = 20
    target_len = 10
    num_contexts = 5
    x_dim = 6
    y_dim = 1
    static_dim = 8

    # Encoder params
    vsn_dim = 64
    lstm_hidden_dim = 64
    ffn_dim = 64
    n_heads = 4

    # ----------- Create Dummy Data -----------
    # Simulated input sequences
    context_x = torch.randn(batch_size, context_len, x_dim, num_contexts)
    context_y = torch.randn(batch_size, context_len, y_dim, num_contexts)
    target_x = torch.randn(batch_size, target_len, x_dim)

    # Static embedding (e.g., asset metadata or ID embedding)
    static_s = torch.randn(batch_size, static_dim)

    # ----------- Initialize Encoder -----------
    encoder = Encoder(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dim_list=[64],  # Just placeholder, not used directly anymore
        embedding_dim=ffn_dim,
        self_attention_type='ptmultihead',
        cross_attention_type='ptmultihead',
        n_heads=n_heads,
        static_dim=static_dim,
        vsn_dim=vsn_dim,
        ffn_dim=ffn_dim,
        lstm_hidden_dim=lstm_hidden_dim
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    static_s = static_s.to(device)
    encoder = encoder.to(device)
    # ----------- Forward Pass -----------
    with torch.no_grad():
        output = encoder(context_x, context_y, target_x, static_s)

    # ----------- Output Check -----------
    print(f"Encoder output shape: {output.shape}")
