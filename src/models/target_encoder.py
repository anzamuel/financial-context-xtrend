import torch
import torch.nn as nn
from src.models.vsn import VariableSelectionNetwork
from src.models.lstm import LSTM

class TargetEncoder(nn.Module):
    def __init__(self, num_features: int, num_embeddings: int, embedding_dim: int, d_h: int, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.vsn = VariableSelectionNetwork(
            num_features=num_features,
            d_h=d_h,
            static_dim=embedding_dim,
            dropout=dropout
        )
        self.lstm = LSTM(
            embedding_dim=embedding_dim,
            input_dim=d_h,
            d_h=d_h,
            dropout=dropout
        )
        self.layer_norm_1 = nn.LayerNorm(d_h)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_h + embedding_dim, d_h),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_h, d_h)
        )
        self.layer_norm_2 = nn.LayerNorm(d_h)

    def forward(self, x_seq: torch.Tensor, s: torch.Tensor):
        """
        x_seq: (batch, seq_len, num_features)
        s: (batch,)
        returns encoding_seq: (batch, seq_len, d_h) – The target encoding sequence
        """
        batch_size, seq_len, _ = x_seq.shape
        ticker_embedding = self.embedding(s)

        x_flat = x_seq.view(batch_size * seq_len, -1)
        embedding_expanded = ticker_embedding.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
        x_t_prime_flat = self.vsn(x_flat, embedding_expanded)
        x_t_prime_seq = x_t_prime_flat.view(batch_size, seq_len, -1)

        h_t_seq, _ = self.lstm(x_t_prime_seq, ticker_embedding)

        a_t_seq = self.layer_norm_1(h_t_seq + x_t_prime_seq)

        ffn2_input = torch.cat([a_t_seq, ticker_embedding.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        ffn2_out = self.ffn2(ffn2_input)

        encoding_seq = self.layer_norm_2(ffn2_out + a_t_seq)

        return encoding_seq

if __name__ == "__main__":
    d_h = 64
    embedding_dim = 8
    num_embeddings = 50
    num_features = 8
    batch_size = 4
    seq_len = 126

    block = TargetEncoder(
        num_features=num_features,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        d_h=d_h
    )

    s = torch.randint(0, num_embeddings, (batch_size,))
    x_seq_fixed = torch.randn(batch_size, seq_len, num_features)
    encoding_seq_fixed = block(x_seq_fixed, s)
    print(encoding_seq_fixed.shape)
