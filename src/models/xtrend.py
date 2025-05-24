import torch
from torch import nn
from models.encoder import Encoder
from models.decoder import Decoder

class XTrendModel(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        static_dim=8,
        encoder_hidden_dim=64,
        vsn_dim=32,
        ffn_dim=64,
        lstm_hidden_dim=64,
        n_heads=4,
        sharpe_dim=32,
        mle_dim=32,
        self_attention_type="ptmultihead",
        cross_attention_type="ptmultihead",
    ):
        super().__init__()
        self.encoder = Encoder(
            x_dim=x_dim,
            y_dim=y_dim,
            hidden_dim_list=[encoder_hidden_dim],
            embedding_dim=encoder_hidden_dim,
            self_attention_type=self_attention_type,
            cross_attention_type=cross_attention_type,
            n_heads=n_heads,
            static_dim=static_dim,
            vsn_dim=vsn_dim,
            ffn_dim=ffn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        )
        self.decoder = Decoder(
            x_dim=x_dim,
            y_dim=y_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            static_dim=static_dim,
            ffn_dim=ffn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            sharpe_dim=sharpe_dim,
            mle_dim=mle_dim,
        )

    def forward(self, context_x, context_y, target_x, target_y, static_s, testing=False):
        """
        context_x: (batch, context_len, x_dim, num_contexts)
        context_y: (batch, context_len, y_dim, num_contexts)
        target_x: (batch, target_len, x_dim)
        target_y: (batch, target_len, y_dim)
        static_s: (batch, static_dim)
        """
        # Encoder: get Kt, Vt, Vt_prime
        encoder_out = self.encoder(context_x, context_y, target_x, static_s)
        
        if testing:
            return self.decoder(target_x, target_y, static_s, encoder_out, testing=testing)
        # Decoder: get outputs
        sharpe_loss, mu, sigma = self.decoder(target_x, target_y, static_s, encoder_out)
        return sharpe_loss, mu, sigma

    def training_step(self, batch, optimizer, alpha=1.0):
        context_x, context_y, target_x, target_y, static_s = batch
        sharpe_loss, mu, sigma = self.forward(context_x, context_y, target_x, target_y, static_s)

        # MLE loss (negative log-likelihood)
        dist = torch.distributions.Normal(mu, sigma.clamp(min=1e-6))
        mle_loss = -dist.log_prob(target_y).mean()

        total_loss = sharpe_loss + alpha * mle_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), sharpe_loss.item(), mle_loss.item()

    def evaluate(self, batch, alpha=1.0):
        context_x, context_y, target_x, target_y, static_s = batch
        return self.forward(context_x, context_y, target_x, target_y, static_s, testing=True)
            