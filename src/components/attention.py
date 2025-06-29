import torch
from torch import nn

class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x



class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        attention_type = "ptmultihead",
        n_heads=4,
        dropout=0.0
    ):
        super().__init__()

        if attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_out = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self.n_heads = n_heads
            self._attention_func = self._multihead_attention
        elif attention_type == "ptmultihead":
            self._mha = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads = n_heads, dropout=dropout, batch_first=True
            )
            self._attention_func = self._pytorch_multihead_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.size(-1) ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, T_q, T_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v)  # (B, T_q, D)

    def _multihead_attention(self, k, v, q):
        heads = []
        for i in range(self.n_heads):
            k_i = self._W_k[i](k)
            v_i = self._W_v[i](v)
            q_i = self._W_q[i](q)
            head = self._dot_attention(k_i, v_i, q_i)
            heads.append(head)

        concat = torch.cat(heads, dim=-1)
        return self._W_out(concat)

    def _pytorch_multihead_attention(self, k, v, q):
        # Pytorch multiheaded attention takes inputs if diff order and permutation
        output, _ = self._mha(q,k,v)
        return output
