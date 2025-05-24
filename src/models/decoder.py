import torch
from torch import nn
import numpy as np
from models.vsn import VSN

class Decoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        encoder_hidden_dim,
        static_dim=8,
        ffn_dim=64,
        sharpe_dim=1,
        mle_dim=64,
        dropout=0.5,
    ):
        super().__init__()
        self.vsn = VSN(
            input_dim=x_dim,
            hidden_dim=y_dim,
            static_dim=static_dim,
            dropout = dropout,
        )
        self.ffn1 = nn.Sequential(
            nn.Linear(y_dim + encoder_hidden_dim, ffn_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, ffn_dim),
        )
        self.layer_norm_1 = nn.LayerNorm(ffn_dim)
        self.lstm = nn.LSTM(
            input_size=ffn_dim,
            hidden_size=ffn_dim,
            batch_first=True,
        )
        # FFN2 outputs mu and sigma directly
        self.layer_norm_2 = nn.LayerNorm(ffn_dim)
        
        self.linear1 = nn.Linear(ffn_dim, ffn_dim)
        self.linear2 = nn.Linear(static_dim, ffn_dim)
        
        self.linear3 = nn.Sequential(
            nn.ELU(),
            nn.Linear(ffn_dim, ffn_dim)
        )
        self.layer_norm_3 = nn.LayerNorm(ffn_dim)
        self.ffn3 = nn.Sequential(
            nn.Linear(ffn_dim, ffn_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, mle_dim * 2)
        )
        # Sharpe head takes [mu, sigma] as input
        self.sharpe_head = nn.Sequential(
            nn.Linear(mle_dim, mle_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mle_dim, sharpe_dim),
            nn.Tanh()
        )

    def forward(self, target_x, target_y, static_s, encoder_out, testing = False):
        """
        target_x: (batch, target_len, x_dim)
        target_y: (batch, target_len, y_dim)
        static_s: (batch, static_dim)
        """
        # VSN block
        batch_size, seq_len, _ = target_x.shape
        target_x_flat = target_x.reshape(-1, target_x.shape[-1])  # (batch * seq_len, x_dim)
        static_s_flat = static_s.unsqueeze(1).expand(-1, seq_len, -1).reshape(-1, static_s.shape[-1])  # (batch * seq_len, static_dim)
        vsn_out_flat = self.vsn(target_x_flat, static_s_flat)  # (batch * seq_len, y_dim)
        vsn_out = vsn_out_flat.view(batch_size, seq_len, -1)   # (batch, seq_len, y_dim)
        x_cat = torch.cat([vsn_out, encoder_out], dim=-1)
        # FFN
        x_prime = self.layer_norm_1(self.ffn1(x_cat)) 
        
        # LSTM
        
        lstm_out, _ = self.lstm(x_prime) 
        a_t = self.layer_norm_2(lstm_out + x_prime)
        static_expanded = static_s.unsqueeze(1).expand(-1, seq_len, -1)
        
        # FFN2 outputs mu and sigma
        last_fnn = self.linear3(self.linear1(a_t)+self.linear2(static_expanded))
        mle_out = self.ffn3(self.layer_norm_3(last_fnn+a_t))
        mu, logsigma = torch.chunk(mle_out, 2, dim=-1)  
        
        sigma = torch.exp(logsigma).clamp(min = -10, max = 10)
        # Reparameterization trick: sample z ~ N(mu, sigma^2)
        if testing:
            z = mu
        else:
            eps = torch.randn_like(mu)
            z = mu + sigma * eps  # (batch, target_len, mle_dim)
        positions = self.sharpe_head(z)  # (batch, target_len, mle_dim)
        
        captured_positions = target_y * positions
        
        if testing:
            return captured_positions
        
        captured_positions = captured_positions[:, -1:, :]
        
        sharpe = (
            torch.mean(captured_positions) / (torch.std(captured_positions) + 1e-9)
        ) * np.sqrt(252.0)
        return -sharpe, mu, logsigma