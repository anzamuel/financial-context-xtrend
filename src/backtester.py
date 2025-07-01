import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from src.models.xtrend import XTrendModel
from src.dataset import XTrendDataset, FEATURE_COLS, PINNACLE_ASSETS
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import datetime as dt

from src.train import TRAIN_START, TRAIN_END, EMBEDDING_DIM, D_H, N_HEADS, DROPOUT, WARMUP_PERIOD_LEN

DATA_DIR = "dataset/CLCDATA"
TEST_START = dt.datetime(2013, 1, 1)
TEST_START_EXTENDED = TEST_START.replace(year=TEST_START.year - 1)
TEST_END = dt.datetime(2023, 1, 1)
TARGET_VOL = 0.15
VOL_LOOKBACK = 60
TRAIN_STRIDE = 1
BATCH_SIZE = 512
MODEL_PATH = "runs/final_model.pth"

def load_single_asset(symbol):
	filepath = os.path.join(DATA_DIR, f"{symbol}_RAD.CSV")
	if not os.path.exists(filepath):
		print(f"Missing: {filepath}")
		return None
	try:
		df = pd.read_csv(
			filepath,
			header=None,
			names=["Date", "Open", "High", "Low", "Close", "Volume", "OpenInterest"],
			parse_dates=["Date"],
			dayfirst=False,
		)
		df.set_index("Date", inplace=True)
		df = df.loc[TEST_START_EXTENDED:TEST_END]
		if df.empty:
			return None
		return df
	except Exception as e:
		print(f"Error loading {symbol}: {e}")
		return None

def calc_daily_vol(daily_returns):
	return (
		daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
		.std()
		.bfill()
	)

def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
	if not len(daily_vol):
		daily_vol = calc_daily_vol(daily_returns)
	annualised_vol = daily_vol * np.sqrt(252)
	return daily_returns * TARGET_VOL / annualised_vol.shift(1)

def rescale_to_target_vol_by_mean_asset_vol(portfolio_returns):
	return calc_vol_scaled_returns(portfolio_returns)

class MACDStrategy:
	def __init__(self, trend_combinations: List[Tuple[float, float]] = None):
		if trend_combinations is None:
			self.trend_combinations = [(8, 24), (16, 48), (32, 96)]
		else:
			self.trend_combinations = trend_combinations

	@staticmethod
	def calc_signal(srs: pd.Series, short_timescale: int, long_timescale: int) -> pd.Series:
		def _calc_halflife(timescale):
			return np.log(0.5) / np.log(1 - 1 / timescale)
		macd = (
			srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
			- srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
		)
		std_63 = srs.rolling(63).std().bfill()
		q = macd / std_63
		q_norm = q / q.rolling(252).std().bfill()
		return q_norm

	def calc_combined_signal(self, srs: pd.Series) -> pd.Series:
		signals = [self.calc_signal(srs, S, L) for S, L in self.trend_combinations]
		combined = pd.concat(signals, axis=1).mean(axis=1)
		return combined

def compute_macd_returns(assets, macd: MACDStrategy):
	signals = []
	position_returns = []
	for symbol in assets:
		df = load_single_asset(symbol)
		if df is not None:
			price = df["Close"]
			sig = macd.calc_combined_signal(price).rename(symbol)
			ret = price.pct_change().rename(symbol)
			signals.append(sig)
			position_return = calc_vol_scaled_returns(sig.shift(1) * ret)
			position_returns.append(position_return.rename(symbol))

	position_df = pd.concat(position_returns, axis=1).fillna(0)
	signals_df = pd.concat(signals, axis=1).fillna(0)
	mean_returns = position_df.mean(axis=1)
	daily_portfolio_returns = mean_returns.loc[TEST_START:]
	return daily_portfolio_returns, signals_df.loc[TEST_START:]

def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
	returns = srs / srs.shift(day_offset) - 1.0
	return returns

def calc_trend_intermediate_strategy(price: pd.Series, symbol) -> Tuple[pd.Series, pd.Series]:
	daily_returns = price.pct_change().rename(symbol)
	annual_returns = price.pct_change(252).rename(symbol)
	positions = np.sign(annual_returns).shift(1)
	next_day_returns = calc_vol_scaled_returns(positions * daily_returns)
	return next_day_returns, positions

def tsmom_strategy(assets):
	returns = []
	positions = []
	for symbol in assets:
		df = load_single_asset(symbol)
		if df is not None:
			price = df["Close"]
			ret, pos = calc_trend_intermediate_strategy(price, symbol)
			ret = ret.fillna(0)
			pos = pos.fillna(0)
			returns.append(ret)
			positions.append(pos)
	returns_df = pd.concat(returns, axis=1).fillna(0)
	positions_df = pd.concat(positions, axis=1).fillna(0)
	mean_returns = returns_df.mean(axis=1)
	daily_portfolio_returns = mean_returns.loc[TEST_START:]
	return daily_portfolio_returns, positions_df.loc[TEST_START:]

def plot_cumulative_returns(returns_dict, log_scale=True):
	plt.figure(figsize=(12, 6))
	for label, ret in returns_dict.items():
		cumulative = (1 + ret).cumprod()
		plt.plot(cumulative, label=label)
	if log_scale:
		plt.yscale("log")
	plt.xlabel("Date")
	plt.ylabel("Portfolio Value ($)")
	plt.title("Portfolio Performance Comparison (Scaled to 15% Volatility)")
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()

class XTrendMetadataWrapper(Dataset):
	def __init__(self, base_dataset):
		self.base_dataset = base_dataset
	def __len__(self):
		return len(self.base_dataset)
	def __getitem__(self, index):
		return self.base_dataset.get_item_with_metadata(index)

def collate_with_metadata(batch):
	batch_metadata = [item["metadata"] for item in batch]
	batched_data = {}
	skip_keys = {"metadata", "sampled_context_indices"}
	for key in batch[0]:
		if key in skip_keys:
			continue
		value = batch[0][key]
		if isinstance(value, torch.Tensor):
			batched_data[key] = torch.stack([item[key] for item in batch])
	return batched_data, batch_metadata

def backtest_xtrend():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = XTrendModel(
		num_features=len(FEATURE_COLS), num_embeddings=len(PINNACLE_ASSETS),
		embedding_dim=EMBEDDING_DIM, d_h=D_H, n_heads=N_HEADS, dropout=DROPOUT,
		warmup_period_len=WARMUP_PERIOD_LEN
	).to(device)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
	model.eval()
	dataset = XTrendDataset(
		train_start = TRAIN_START, train_end = TRAIN_END,
		eval_start = TEST_START_EXTENDED, eval_end = TEST_END,
		train_stride = TRAIN_STRIDE)
	dataset.set_mode("eval")
	wrapped_dataset = XTrendMetadataWrapper(dataset)
	dataloader = DataLoader(
		wrapped_dataset, batch_size=256,
		shuffle=False, collate_fn=collate_with_metadata
	)
	predictions = []
	for batch_data, batch_metadata in tqdm(dataloader, desc="Evaluating XTrend Model"):
		for key in batch_data:
			batch_data[key] = batch_data[key].to(device)
		_, positions = model.evaluate(batch_data)
		for i in range(len(batch_metadata)):
			prediction = {
				"date": batch_metadata[i]["target_end_date"],
				"ticker": batch_metadata[i]["target_ticker"],
				"pred": float(positions[i, -1, 0].item()),
				"true": float(batch_data["target_y"][i, -1].item()),
			}
			predictions.append(prediction)
	df_results = pd.DataFrame(predictions)
	df_results["strategy_return"] = df_results["pred"] * df_results["true"]

	positions_df = df_results.pivot(columns='ticker', index='date', values='pred').loc[TEST_START:].ffill()

	df_results.set_index("date", inplace=True)
	df_results.index = pd.to_datetime(df_results.index)
	mask = (df_results.index >= TEST_START) & (df_results.index < TEST_END)
	avg_returns = df_results.loc[mask].groupby("date")["strategy_return"].mean().rename("XTrend_Return")
	return avg_returns, positions_df

def calc_sharpe_ratio(returns_series):
	return (returns_series.mean() / returns_series.std()) * np.sqrt(252)

def calc_max_drawdown(returns_series):
	cum_returns = (1 + returns_series).cumprod()
	running_max = cum_returns.cummax()
	drawdown = (cum_returns - running_max) / running_max
	return -drawdown.min()

def calc_turnover(positions_df):
	return positions_df.diff().abs().sum(axis=1).mean()

if __name__ == "__main__":
	results = {}

	all_asset_returns = [load_single_asset(s)["Close"].pct_change().rename(s) for s in PINNACLE_ASSETS if load_single_asset(s) is not None]
	returns_df = pd.concat(all_asset_returns, axis=1).fillna(0)

	long_positions = pd.DataFrame(1.0, index=returns_df.index, columns=returns_df.columns).loc[TEST_START:]
	long_returns_per_asset = returns_df * long_positions.shift(1)
	long_portfolio_returns = long_returns_per_asset.mean(axis=1).loc[TEST_START:]
	results["Long Only"] = (long_portfolio_returns, long_positions)

	macd_returns, macd_positions = compute_macd_returns(PINNACLE_ASSETS, MACDStrategy())
	results["MACD"] = (macd_returns, macd_positions)

	tsmom_returns, tsmom_positions = tsmom_strategy(PINNACLE_ASSETS)
	results["TSMOM"] = (tsmom_returns, tsmom_positions)

	xtrend_returns, xtrend_positions = backtest_xtrend()
	results["XTrend"] = (xtrend_returns, xtrend_positions)

	stats = []
	for name, (returns, positions) in results.items():
		scaled_returns = rescale_to_target_vol_by_mean_asset_vol(returns)
		stats.append({
			"Strategy": name,
			"Sharpe Ratio": calc_sharpe_ratio(scaled_returns),
			"Max Drawdown": f"{calc_max_drawdown(scaled_returns):.2%}",
			"Turnover": f"{calc_turnover(positions):.3f}"
		})
	stats_df = pd.DataFrame(stats).set_index("Strategy")
	print("\n--- Performance Metrics ---")
	print(stats_df)

	returns_dict = {name: rescale_to_target_vol_by_mean_asset_vol(ret) for name, (ret, pos) in results.items()}
	plot_cumulative_returns(returns_dict, log_scale=True)
