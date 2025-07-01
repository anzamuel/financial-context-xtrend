import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features: int, d_h: int, static_dim: int, dropout: float = 0.5):
        super().__init__()
        self.num_features = num_features
        self.d_h = d_h

        self.feature_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_h),
                nn.ELU(),
                nn.Dropout(p=dropout),
                nn.Linear(d_h, d_h),
            ) for _ in range(num_features)
        ])

        self.weight_generator = nn.Sequential(
            nn.Linear(num_features + static_dim, d_h),
            nn.ELU(),
            nn.Linear(d_h, num_features)
        )

    def forward(self, x_t: torch.Tensor, ticker_embedding: torch.Tensor) -> torch.Tensor:
        """
        x_t: (batch, num_features)
        ticker_embedding: (batch, static_dim)
        returns: (batch, d_h)
        """
        features = torch.stack(
            [self.feature_mlps[j](x_t[:, j:j+1]) for j in range(self.num_features)],
            dim=1
        )

        combined_input = torch.cat([x_t, ticker_embedding], dim=1)
        w_t = F.softmax(self.weight_generator(combined_input), dim=-1)

        w_t = w_t.unsqueeze(-1)
        x_t_prime = (features * w_t).sum(dim=1)

        return x_t_prime

if __name__=="__main__":
    num_features = 8
    batch_size = 4
    d_h = 64
    static_dim = 8

    x_t = torch.randn(batch_size, num_features)
    ticker_embedding = torch.randn(batch_size, static_dim)
    vsn = VariableSelectionNetwork(num_features, d_h, static_dim)
    x_t_prime = vsn(x_t, ticker_embedding)
    print(x_t_prime.shape)
