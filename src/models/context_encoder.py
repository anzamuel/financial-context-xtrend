import torch
import torch.nn as nn
from src.models.vsn import VariableSelectionNetwork
from src.models.lstm import LSTM

class ContextEncoder(nn.Module):
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

    def forward(self, x_seq: torch.Tensor, s: torch.Tensor, seq_length: torch.Tensor = None):
        """
        x_seq (xi_seq for values): (batch, seq_len, num_features)
        s: (batch,)
        seq_length: (batch,) - Optional
        returns encoding: (batch, d_h) - The final context encoding
        """
        batch_size, seq_len, _ = x_seq.shape
        ticker_embedding = self.embedding(s)

        x_flat = x_seq.view(batch_size * seq_len, -1)
        embedding_expanded = ticker_embedding.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
        x_t_prime_flat = self.vsn(x_flat, embedding_expanded)
        x_t_prime_seq = x_t_prime_flat.view(batch_size, seq_len, -1)

        _, h_t = self.lstm(x_t_prime_seq, ticker_embedding, seq_length=seq_length)

        if seq_length is not None:
            last_indices = seq_length - 1
            x_t_prime = x_t_prime_seq[torch.arange(batch_size), last_indices]
        else:
            x_t_prime = x_t_prime_seq[:, -1, :]

        a_t = self.layer_norm_1(h_t + x_t_prime)

        ffn2_input = torch.cat([a_t, ticker_embedding], dim=-1)
        ffn2_out = self.ffn2(ffn2_input)

        encoding = self.layer_norm_2(ffn2_out + a_t)

        return encoding

if __name__ == "__main__":
    d_h = 64
    embedding_dim = 8
    num_embeddings = 50
    num_features = 8
    batch_size = 4
    seq_len = 126
    max_context_len = 21
    min_context_len = 5

    block = ContextEncoder(
        num_features=num_features,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        d_h=d_h
    )

    s = torch.randint(0, num_embeddings, (batch_size,))

    print("--- Testing with fixed-length sequences (e.g., targets) ---")
    x_seq_fixed = torch.randn(batch_size, seq_len, num_features)
    encoding_fixed = block(x_seq_fixed, s)
    print(encoding_fixed.shape)

    print("\n--- Testing with variable-length sequences (e.g., contexts) ---")
    x_seq_padded = torch.randn(batch_size, max_context_len, num_features)
    seq_length = torch.randint(min_context_len, max_context_len + 1, (batch_size,))
    print("Example seq_length:", seq_length[:4])
    encoding_padded = block(x_seq_padded, s, seq_length=seq_length)
    print(encoding_padded.shape)
