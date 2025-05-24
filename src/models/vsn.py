import torch
import torch.nn as nn
import torch.nn.functional as F

class VSN(nn.Module):
    def __init__(self, input_dim, hidden_dim, static_dim=None, use_static=True):
        """
        input_dim: number of input features |𝓧|
        hidden_dim: output embedding dimension per feature
        static_dim: dimension of optional static embedding (like asset ID embedding)
        use_static: if False, disables use of static information s
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_static = use_static

        self.feature_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ELU(),
                nn.Dropout(p=0.3),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(input_dim)
        ])

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(static_dim, hidden_dim)
        
        self.linear3 = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_t, s=None):
        """
        x_t: (batch, input_dim)
        s: (batch, static_dim) or None
        returns: (batch, hidden_dim)
        """
        batch_size = x_t.size(0)

        transformed = []
        for j in range(self.input_dim):
            x_j = x_t[:, j:j+1]  # (batch, 1)
            out_j = self.feature_mlps[j](x_j)  # (batch, hidden_dim)
            transformed.append(out_j)

        features = torch.stack(transformed, dim=1)  # (batch, input_dim, hidden_dim)

        ffn = self.linear3(self.linear1(x_t)+self.linear2(s))
        weights = F.softmax(ffn, dim=-1)  # (batch, input_dim)

        weights = weights.unsqueeze(-1)  # (batch, input_dim, 1)
        vsn_out = (features * weights).sum(dim=1)  # (batch, hidden_dim)

        return vsn_out

if __name__=="__main__":
    x_t = torch.randn(32, 10)       # 10 input features
    s = torch.randn(32, 8)          # optional static embedding
    vsn = VSN(input_dim=10, hidden_dim=64, static_dim=8)

    output = vsn(x_t, s)            
    print(output)