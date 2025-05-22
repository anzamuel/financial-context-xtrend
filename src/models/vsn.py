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
                nn.ELU()
            ) for _ in range(input_dim)
        ])

        self.attn_input_dim = input_dim + (static_dim if use_static else 0)
        self.attn_ffn = nn.Sequential(
            nn.Linear(self.attn_input_dim, input_dim),
            nn.ELU(),
            nn.Linear(input_dim, input_dim)
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

        if self.use_static and s is not None:
            combined = torch.cat([x_t, s], dim=-1)  # (batch, input_dim + static_dim)
        else:
            combined = x_t  # (batch, input_dim)

        logits = self.attn_ffn(combined)  # (batch, input_dim)
        weights = F.softmax(logits, dim=-1)  # (batch, input_dim)

        weights = weights.unsqueeze(-1)  # (batch, input_dim, 1)
        vsn_out = (features * weights).sum(dim=1)  # (batch, hidden_dim)

        return vsn_out

if __name__=="__main__":
    x_t = torch.randn(32, 10)       # 10 input features
    s = torch.randn(32, 8)          # optional static embedding
    vsn = VSN(input_dim=10, hidden_dim=64, static_dim=8)

    output = vsn(x_t, s)            
    print(output)