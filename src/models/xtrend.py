import torch
import torch.nn as nn
from src.models.context_encoder import ContextEncoder
from src.models.target_encoder import TargetEncoder
from src.models.target_decoder import TargetDecoder
from src.models.attention import Attention
from src.models.ptp import PTP
from src.models.sharpe import SharpeLoss

class XTrendModel(nn.Module):
    def __init__(self, num_features: int, num_embeddings: int, embedding_dim: int, d_h: int, n_heads: int = 4, dropout: float = 0.5, warmup_period_len: int = 63):
        super().__init__()
        self.num_features = num_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.d_h = d_h
        self.warmup_period_len = warmup_period_len

        self.key_encoder = ContextEncoder(
            num_features=num_features,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            d_h=d_h,
            dropout=dropout
        )
        self.value_encoder = ContextEncoder(
            num_features=num_features + 1, # For the additional value column in xi
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            d_h=d_h,
            dropout=dropout
        )
        self.query_encoder = TargetEncoder(
            num_features=num_features,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            d_h=d_h,
            dropout=dropout
        )
        self.cross_attention = Attention(
            hidden_dim=d_h,
            attention_type="ptmultihead",
            n_heads=n_heads,
            dropout=dropout
        )
        self.target_decoder = TargetDecoder(
            num_features=num_features,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            d_h=d_h,
            dropout=dropout
        )
        self.ptp = PTP(input_dim=d_h, d_h=d_h, dropout=dropout)

    def forward(self, context_x: torch.Tensor, context_xi: torch.Tensor, context_s: torch.Tensor, context_lens: torch.Tensor, target_x: torch.Tensor, target_s: torch.Tensor):
        """
        context_x: (batch, sample_size, seq_len, num_features)
        context_xi: (batch, sample_size, seq_len, num_features + 1)
        context_s: (batch, sample_size)
        context_lens: (batch, sample_size)
        target_x: (batch, target_len, num_features)
        target_s: (batch,)
        returns z_seq: (batch, target_len, 1)
        """
        batch_size, sample_size, _, _ = context_x.shape

        context_x_flat = context_x.view(batch_size * sample_size, context_x.size(2), context_x.size(3))
        context_xi_flat = context_xi.view(batch_size * sample_size, context_xi.size(2), context_xi.size(3))
        context_s_flat = context_s.view(batch_size * sample_size)
        context_lens_flat = context_lens.view(batch_size * sample_size)

        K = self.key_encoder(context_x_flat, context_s_flat, seq_length=context_lens_flat)
        V = self.value_encoder(context_xi_flat, context_s_flat, seq_length=context_lens_flat)

        K = K.view(batch_size, sample_size, -1)
        V = V.view(batch_size, sample_size, -1)

        q_seq = self.query_encoder(target_x, target_s)

        y_seq = self.cross_attention(K, V, q_seq)

        decoder_output_seq = self.target_decoder(target_x, target_s, y_seq)

        z_seq = self.ptp(decoder_output_seq)

        return z_seq

    def training_step(self, batch: dict, optimizer: torch.optim.Optimizer):
        self.train()
        z_seq = self.forward(
            context_x=batch["context_x"], context_xi=batch["context_xi"],
            context_s=batch["context_s"], context_lens=batch["context_lens"],
            target_x=batch["target_x"], target_s=batch["target_s"]
        )
        loss_fn = SharpeLoss(warmup_period= self.warmup_period_len)
        loss = loss_fn(z_seq, batch["target_y"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss

    def evaluate(self, batch: dict):
        self.eval()
        with torch.no_grad():
            z_seq = self.forward(
                context_x=batch["context_x"], context_xi=batch["context_xi"],
                context_s=batch["context_s"], context_lens=batch["context_lens"],
                target_x=batch["target_x"], target_s=batch["target_s"]
            )
            loss_fn = SharpeLoss(warmup_period= self.warmup_period_len)
            loss = loss_fn(z_seq, batch["target_y"])
        return loss, z_seq

if __name__ == "__main__":
    d_h = 64
    embedding_dim = 8
    num_embeddings = 50
    num_features = 8
    batch_size = 4
    target_len = 126
    context_sample_size = 20
    min_context_len = 5
    max_context_len = 21

    model = XTrendModel(
        num_features=num_features,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        d_h=d_h
    )

    context_x = torch.randn(batch_size, context_sample_size, max_context_len, num_features)
    context_xi = torch.randn(batch_size, context_sample_size, max_context_len, num_features + 1)
    context_s = torch.randint(0, num_embeddings, (batch_size, context_sample_size))
    context_lens = torch.randint(min_context_len, max_context_len + 1, (batch_size, context_sample_size))
    target_x = torch.randn(batch_size, target_len, num_features)
    target_s = torch.randint(0, num_embeddings, (batch_size,))

    z_seq = model(context_x, context_xi, context_s, context_lens, target_x, target_s)
    print("Final output shape (z_seq):", z_seq.shape)
