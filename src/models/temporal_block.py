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
            batch_first=True,
        )

        self.layer_norm_1 = nn.LayerNorm(lstm_hidden_dim)
        
        self.ffn_3 = nn.Sequential(
            nn.Linear(static_dim, lstm_hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        )
        
        self.ffn_4 = nn.Sequential(
            nn.Linear(static_dim, lstm_hidden_dim),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        )
        
        self.linear1 = nn.Linear(lstm_hidden_dim, ffn_hidden_dim)
        self.linear2 = nn.Linear(static_dim, ffn_hidden_dim)
        
        self.linear3 = nn.Sequential(
            nn.ELU(),
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim)
        )
        self.layer_norm_2 = nn.LayerNorm(ffn_hidden_dim)

    def forward(self, x_seq, s, h_0=None, c_0=None):
        """
        x_seq: (batch, seq_len, input_dim)
        s: (batch, static_dim)
        h_0, c_0: optional LSTM states
        returns: (batch, seq_len, ffn_hidden_dim)
        """
        batch_size, seq_len, _ = x_seq.shape
        x_flat = x_seq.reshape(-1, x_seq.shape[-1])  # (batch * seq_len, input_dim)
        s_expanded = s.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, s.shape[-1])  # (batch * seq_len, static_dim)
        vsn_out_flat = self.vsn(x_flat, s_expanded)  # (batch * seq_len, vsn_hidden_dim)
        x_prime = vsn_out_flat.view(batch_size, seq_len, -1)  # (batch, seq_len, vsn_hidden_dim)

        
        h_0, c_0 = (self.ffn_3(s), self.ffn_4(s))
        h_0 = h_0.unsqueeze(0)
        c_0 = c_0.unsqueeze(0)

        lstm_out, _ = self.lstm(x_prime, (h_0, c_0))  # (batch, seq_len, lstm_hidden_dim)

        residual_added = lstm_out + x_prime  # assuming same dims
        a_t = self.layer_norm_1(residual_added)  # (batch, seq_len, lstm_hidden_dim)
        
        static_expanded = s.unsqueeze(1).expand(-1, a_t.size(1), -1)
        ffn_out = self.linear3(self.linear1(a_t)+self.linear2(static_expanded))

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
