import torch
import torch.nn as nn
import numpy as np

class SharpeLoss(nn.Module):
	def __init__(self, warmup_period: int):
		super().__init__()
		self.warmup_period = warmup_period
		self.epsilon = 1e-9
		self.annualization_factor = np.sqrt(252.0)

	def forward(self, z_seq: torch.Tensor, y_seq: torch.Tensor):
		"""
		z_seq: (batch, seq_len, 1) - The sequence of positions taken by the model.
		y_seq: (batch, seq_len) - The sequence of actual next-day returns.
		returns sharpe_loss: scalar - The negative annualized Sharpe ratio.
		"""
		y_seq = y_seq.unsqueeze(-1)
		portfolio_returns = z_seq * y_seq

		portfolio_returns_after_warmup = portfolio_returns[:, self.warmup_period:, :]

		mean_returns = torch.mean(portfolio_returns_after_warmup)
		std_returns = torch.std(portfolio_returns_after_warmup)

		sharpe_ratio = (mean_returns / (std_returns + self.epsilon)) * self.annualization_factor

		return -sharpe_ratio

if __name__ == "__main__":
	seq_len = 126
	batch_size = 4
	warmup = 63

	sharpe_loss_fn = SharpeLoss(warmup_period=warmup)

	positions = torch.randn(batch_size, seq_len, 1)
	returns = torch.randn(batch_size, seq_len)

	loss = sharpe_loss_fn(positions, returns)
	print("Calculated Sharpe Loss:", loss.item())
