import torch
from torch import nn
from models.attention import Attention
from models.temporal_block import TemporalBlock


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_dim_list,  # the dims of hidden starts of mlps
        embedding_dim=32,  # the dim of last axis of r..
        self_attention_type="dot",
        cross_attention_type="dot",
        n_heads=4,
        static_dim=8,
        vsn_dim=32,
        ffn_dim=64,
        lstm_hidden_dim=32,
    ):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.input_dim = x_dim + y_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_heads = n_heads
        self.hidden_dim = ffn_dim
        self.embedding_dim = embedding_dim
        
        
        self.temporal_block = TemporalBlock(
            input_dim=self.input_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
        )
        self.temporal_block_key = TemporalBlock(
            input_dim=x_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
        )
        
        self.temporal_block_query = TemporalBlock(
            input_dim=x_dim,
            static_dim=static_dim,
            vsn_hidden_dim=vsn_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            ffn_hidden_dim=ffn_dim,
        )

        if embedding_dim != hidden_dim_list[-1]:
            print("Warning, Check the dim of latent z and the dim of mlp last layer!")

        self._self_attention = Attention(
            self.hidden_dim,
            attention_type=self_attention_type,
            n_heads=self.n_heads,
        )

        self._cross_attention = Attention(
            hidden_dim=self.hidden_dim,
            attention_type=cross_attention_type,
            n_heads=n_heads,
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )


    def forward(self, context_x, context_y, target_x=None, static_s=None):
        batch_sz, context_len, _, num_contexts = context_x.shape
        _, target_len, _ = target_x.shape

        # 1. Compute Vt (values) and Kt (keys) for each context
        # We'll aggregate over context_len (sequence) for each context
        Vt = []
        Kt = []
        for i in range(num_contexts):
            # Value: concat x and y, then temporal block
            encoder_input = torch.cat(
                [context_x[:, :, :, i], context_y[:, :, :, i]], dim=-1
            )  # (batch, context_len, x_dim + y_dim)
            v_i = self.temporal_block(encoder_input, static_s)  # (batch, context_len, hidden_dim)
            # Aggregate over context_len (e.g., mean)
            v_i = v_i.mean(dim=1)  # (batch, hidden_dim)
            Vt.append(v_i.unsqueeze(1))  # (batch, 1, hidden_dim)

            # Key: just x, then temporal block
            k_i = self.temporal_block_key(context_x[:, :, :, i], static_s)  # (batch, context_len, hidden_dim)
            k_i = k_i.mean(dim=1)  # (batch, hidden_dim)
            Kt.append(k_i.unsqueeze(1))  # (batch, 1, hidden_dim)

        # Stack over contexts
        Vt = torch.cat(Vt, dim=1)  # (batch, num_contexts, hidden_dim)
        Kt = torch.cat(Kt, dim=1)  # (batch, num_contexts, hidden_dim)

        # 2. Self-attention over Vt (Eq. 17)
        Vt_prime = self._self_attention(Vt, Vt, Vt)  # (batch, num_contexts, hidden_dim)
        
        Vt_prime_2 = self.ffn2(Vt_prime)

        # 3. Compute queries qt for each target
        qt = self.temporal_block_query(target_x, static_s)  # (batch, target_len, hidden_dim)

        # 4. Cross-attention: for each target, attend over Kt (keys) and Vt' (values)
        # Output: (batch, target_len, hidden_dim)
        representation = self._cross_attention(Kt, Vt_prime_2, qt)
        output = self.ffn(representation)
        return output  # (batch, target_len, hidden_dim)
    

if __name__ == "__main__":
    torch.manual_seed(0)

    # ----------- Configuration -----------
    batch_size = 4
    context_len = 20
    target_len = 10
    num_contexts = 5
    x_dim = 6
    y_dim = 1
    static_dim = 8

    # Encoder params
    vsn_dim = 64
    lstm_hidden_dim = 64
    ffn_dim = 64
    n_heads = 4

    # ----------- Create Dummy Data -----------
    # Simulated input sequences
    context_x = torch.randn(batch_size, context_len, x_dim, num_contexts)
    context_y = torch.randn(batch_size, context_len, y_dim, num_contexts)
    target_x = torch.randn(batch_size, target_len, x_dim)

    # Static embedding (e.g., asset metadata or ID embedding)
    static_s = torch.randn(batch_size, static_dim)

    # ----------- Initialize Encoder -----------
    encoder = Encoder(
        x_dim=x_dim,
        y_dim=y_dim,
        hidden_dim_list=[64],  # Just placeholder, not used directly anymore
        embedding_dim=ffn_dim,
        self_attention_type='ptmultihead',
        cross_attention_type='ptmultihead',
        n_heads=n_heads,
        static_dim=static_dim,
        vsn_dim=vsn_dim,
        ffn_dim=ffn_dim,
        lstm_hidden_dim=lstm_hidden_dim
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_x = context_x.to(device)
    context_y = context_y.to(device)
    target_x = target_x.to(device)
    static_s = static_s.to(device)
    encoder = encoder.to(device)
    # ----------- Forward Pass -----------
    with torch.no_grad():
        output = encoder(context_x, context_y, target_x, static_s)

    # ----------- Output Check -----------
    print(f"Encoder output shape: {output.shape}")