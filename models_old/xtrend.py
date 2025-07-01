import torch
from torch import nn
from components.encoder import Encoder
from components.decoder import Decoder

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

    def forward(self, target_x, target_y, target_s, context_x, context_xi, context_s, context_lens, testing=False): # TODO: implement context_lens (!)
        """
        context_x: (batch, context_len, x_dim, num_contexts)
        target_x: (batch, target_len, x_dim)
        """
        # Encoder: get Kt, Vt, Vt_prime
        context_x = context_x.permute(0, 2, 3, 1)
        context_xi = context_xi.permute(0, 2, 3, 1)
        embedding_context_s = self.embedding(context_s)
        embedding_target_s = self.embedding(target_s)

        encoder_out = self.encoder(context_x, context_xi, target_x, embedding_context_s, embedding_target_s)
        if testing:
            return self.decoder(target_x, target_y, embedding_target_s, encoder_out, testing=testing)
        # Decoder: get outputs
        sharpe_loss, mu, logsigma = self.decoder(target_x, target_y, embedding_target_s, encoder_out)
        return sharpe_loss, mu, logsigma

    def training_step(self, batch, optimizer, alpha=1.0):
        self.train()
        sharpe_loss, mu, logsigma = self.forward(**batch)
        # MLE loss (negative log-likelihood)
        dist = torch.distributions.Normal(mu, torch.exp(logsigma.clamp(min=-10, max = 10)))
        mle_loss = -dist.log_prob(batch["target_y"].unsqueeze(-1)).mean()
        total_loss = sharpe_loss + alpha * mle_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        return sharpe_loss, mle_loss, total_loss

    def evaluate(self, batch):
        self.eval()
        with torch.no_grad():
            return self.forward(**batch, testing=True)
