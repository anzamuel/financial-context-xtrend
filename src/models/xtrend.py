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
        vsn_dim=64,
        ffn_dim=64,
        lstm_hidden_dim=64,
        n_heads=4,
        sharpe_dim=64,
        mle_dim=64,
        self_attention_type="ptmultihead",
        cross_attention_type="ptmultihead",
        dropout=0.5,
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
            dropout=dropout
        )
        self.decoder = Decoder(
            x_dim=x_dim,
            y_dim=y_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            static_dim=static_dim,
            ffn_dim=ffn_dim,
            sharpe_dim=sharpe_dim,
            mle_dim=mle_dim,
            dropout=dropout
        )
        self.embedding = nn.Embedding(num_embeddings=50,embedding_dim=static_dim)

    def forward(self, target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list, testing=False):
        """
        context_x: (batch, context_len, x_dim, num_contexts)
        context_y: (batch, context_len, y_dim, num_contexts)
        target_x: (batch, target_len, x_dim)
        target_y: (batch, target_len, y_dim)
        static_s: (batch, static_dim)
        """
        # Encoder: get Kt, Vt, Vt_prime
        context_x_list = context_x_list.permute(0, 2, 3, 1)
        context_xi_list = context_xi_list.permute(0, 2, 3, 1)
        embedding_context_s = self.embedding(context_s_list)
        embedding_target_s = self.embedding(target_s)

        encoder_out = self.encoder(context_x_list, context_xi_list, target_x, embedding_context_s, embedding_target_s)
        if testing:
            return self.decoder(target_x, target_y, embedding_target_s, encoder_out, testing=testing)
        # Decoder: get outputs
        sharpe_loss, mu, logsigma = self.decoder(target_x, target_y, embedding_target_s, encoder_out)
        return sharpe_loss, mu, logsigma

    def training_step(self, batch, optimizer, alpha=1.0):
        self.train()
        target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list = batch
        sharpe_loss, mu, logsigma = self.forward(target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list)
        # MLE loss (negative log-likelihood)
        dist = torch.distributions.Normal(mu, torch.exp(logsigma.clamp(min=-10, max = 10)))
        mle_loss = -dist.log_prob(target_y.unsqueeze(-1)).mean()

        total_loss = sharpe_loss + alpha * mle_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return total_loss.item(), sharpe_loss.item(), mle_loss.item()

    def evaluate(self, batch):
        self.eval()
        with torch.no_grad():
            target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list = batch
            return self.forward(target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list, testing=True)
