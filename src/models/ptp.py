import torch
import torch.nn as nn

class PTP(nn.Module):
	def __init__(self, input_dim: int, d_h: int, dropout: float = 0.5):
		super().__init__()
		self.ffn = nn.Sequential(
			nn.Linear(input_dim, d_h),
			nn.ELU(),
			nn.Dropout(p=dropout),
			nn.Linear(d_h, 1),
			nn.Tanh()
		)

	def forward(self, decoder_output_seq: torch.Tensor):
		"""
		decoder_output_seq: (batch, seq_len, d_h)
		returns z_seq: (batch, seq_len, 1) - The sequence of positions.
		"""
		z_seq = self.ffn(decoder_output_seq)
		return z_seq

if __name__ == "__main__":
	d_h = 64
	batch_size = 4
	seq_len = 126

	ptp_head = PTP(input_dim=d_h, d_h=d_h)
	decoder_output = torch.randn(batch_size, seq_len, d_h)

	z_seq = ptp_head(decoder_output)
	print("PTP Output (positions) shape:", z_seq.shape)
