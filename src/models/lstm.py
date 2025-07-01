import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
	def __init__(self, embedding_dim: int, input_dim: int, d_h: int, dropout: float = 0.5):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=d_h,
			batch_first=True
		)
		self.ffn_h = nn.Sequential(
			nn.Linear(embedding_dim, d_h),
			nn.ELU(),
			nn.Dropout(p=dropout),
			nn.Linear(d_h, d_h)
		)
		self.ffn_c = nn.Sequential(
			nn.Linear(embedding_dim, d_h),
			nn.ELU(),
			nn.Dropout(p=dropout),
			nn.Linear(d_h, d_h)
		)

	def forward(self, x_t_prime_seq: torch.Tensor, ticker_embedding: torch.Tensor, seq_length: torch.Tensor = None):
		"""
		x_t_prime_seq: (batch, seq_len, input_dim)
		ticker_embedding: (batch, embedding_dim)
		seq_length: (batch,) - Optional
		returns h_t_seq: (batch, seq_len, d_h) - The hidden state for every timestep.
		returns h_t: (batch, d_h) - The final hidden state after the last valid timestep.
		"""
		h_0 = self.ffn_h(ticker_embedding).unsqueeze(0)
		c_0 = self.ffn_c(ticker_embedding).unsqueeze(0)

		if seq_length is not None:
			packed_input = pack_padded_sequence(
				x_t_prime_seq, seq_length.cpu(), batch_first=True, enforce_sorted=False
			)
			packed_output, (h_t, c_t) = self.lstm(packed_input, (h_0, c_0))
			h_t_seq, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=x_t_prime_seq.size(1))
		else:
			h_t_seq, (h_t, c_t) = self.lstm(x_t_prime_seq, (h_0, c_0))

		return h_t_seq, h_t.squeeze(0)

if __name__ == "__main__":
	d_h = 64
	embedding_dim = 8
	input_dim = 32
	batch_size = 4
	seq_len = 126
	max_context_len = 21
	min_context_len = 5

	lstm_layer = LSTM(embedding_dim=embedding_dim, input_dim=input_dim, d_h=d_h)
	ticker_embedding = torch.randn(batch_size, embedding_dim)

	print("--- Testing with fixed-length sequences (e.g., targets) ---")
	x_t_prime_seq_fixed = torch.randn(batch_size, seq_len, input_dim)
	h_t_seq_fixed, h_t_fixed = lstm_layer(x_t_prime_seq_fixed, ticker_embedding)
	print("h_t_seq shape:", h_t_seq_fixed.shape)
	print("h_t shape:", h_t_fixed.shape)

	print("\n--- Testing with variable-length sequences (e.g., contexts) ---")
	x_t_prime_seq_padded = torch.randn(batch_size, max_context_len, input_dim)
	seq_length = torch.randint(min_context_len, max_context_len + 1, (batch_size,))
	print("Example seq_length:", seq_length)
	h_t_seq_padded, h_t_padded = lstm_layer(x_t_prime_seq_padded, ticker_embedding, seq_length=seq_length)
	print("h_t_seq shape:", h_t_seq_padded.shape)
	print("h_t shape:", h_t_padded.shape)
