import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vsn import VSN

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, static_dim, vsn_hidden_dim, lstm_hidden_dim, ffn_hidden_dim):
        super().__init__()

        self.vsn = VSN(input_dim, vsn_hidden_dim, static_dim, use_static=True)

        self.lstm = nn.LSTM(
            input_size=vsn_hidden_dim,
            hidden_size=lstm_hidden_dim,
            batch_first=True
        )

        self.layer_norm_1 = nn.LayerNorm(lstm_hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(ffn_hidden_dim)

        self.ffn_2 = nn.Sequential(
            nn.Linear(lstm_hidden_dim, ffn_hidden_dim),
            nn.ELU()
        )
        self.static_proj = nn.Linear(static_dim, ffn_hidden_dim)
        self.final_proj = nn.Linear(ffn_hidden_dim, ffn_hidden_dim)

    def forward(self, x_seq, s, h_0=None, c_0=None):
        """
        x_seq: (batch, seq_len, input_dim)
        s: (batch, static_dim)
        h_0, c_0: optional LSTM states
        returns: (batch, seq_len, ffn_hidden_dim)
        """
        batch_size, seq_len, _ = x_seq.shape

        vsn_out = []
        for t in range(seq_len):
            vsn_t = self.vsn(x_seq[:, t], s)  # (batch, vsn_hidden_dim)
            vsn_out.append(vsn_t)
        x_prime = torch.stack(vsn_out, dim=1)  # (batch, seq_len, vsn_hidden_dim)

        lstm_out, _ = self.lstm(x_prime, (h_0, c_0) if h_0 is not None else None)  # (batch, seq_len, lstm_hidden_dim)

        residual_added = lstm_out + x_prime  # assuming same dims
        a_t = self.layer_norm_1(residual_added)  # (batch, seq_len, lstm_hidden_dim)

        static_proj = self.static_proj(s).unsqueeze(1)  # (batch, 1, ffn_hidden_dim)
        ffn_in = self.ffn_2(a_t) + static_proj  # broadcast (batch, seq_len, ffn_hidden_dim)
        ffn_out = self.final_proj(ffn_in)

        out = self.layer_norm_2(ffn_out + a_t)  # (batch, seq_len, ffn_hidden_dim)
        return out

if __name__ == "__main__":
    batch, seq_len = 16, 60
    input_dim = 10
    static_dim = 8
    vsn_dim = 64
    lstm_dim = 64
    ffn_dim = 64

    x = torch.randn(batch, seq_len, input_dim)
    s = torch.randn(batch, static_dim)

    block = TemporalBlock(input_dim, static_dim, vsn_dim, lstm_dim, ffn_dim)
    out = block(x, s)
    print(out.shape)  # (16, 60, 64)
