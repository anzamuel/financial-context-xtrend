import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch
from src.models.xtrend import XTrendModel
from src.dataset import XTrendDataset, FEATURE_COLS, PINNACLE_ASSETS
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.utils.data import Dataset
import datetime as dt

from src.train import TRAIN_START, TRAIN_END, EVAL_START, EVAL_END, EMBEDDING_DIM, D_H, N_HEADS, DROPOUT, WARMUP_PERIOD_LEN

DATA_DIR = "dataset/CLCDATA"  # Adjust path as needed
TEST_START = EVAL_START
EXTENDED_START_DATE = EVAL_START.replace(year=EVAL_START.year - 1)
TEST_END = EVAL_END
TARGET_VOL = 0.15  # Target annual volatility
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
        df = df.loc[EXTENDED_START_DATE:TEST_END]
        # df = df[df["Close"] > 0]
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None

def compute_portfolio_returns(assets):
    daily_returns = []

    for symbol in assets:
        df = load_single_asset(symbol)
        if df is not None:
            ret = df["Close"].pct_change().rename(symbol)
            daily_returns.append(ret)

    if not daily_returns:
        raise ValueError("No asset data loaded.")

    returns_df = pd.concat(daily_returns, axis=1)
    returns_df = returns_df.fillna(0)
    portfolio_returns = returns_df.mean(axis=1)
    portfolio_returns = portfolio_returns.loc[TEST_START:]
    return portfolio_returns

def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .bfill()
    )

def calc_vol_scaled_returns(daily_returns, daily_vol=pd.Series(None)):
    """calculates volatility scaled returns for annualised VOL_TARGET of 15%
    with input of pandas series daily_returns"""
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
    returns = []
    position_returns = []

    for symbol in assets:
        df = load_single_asset(symbol)
        if df is not None:
            price = df["Close"]
            sig = macd.calc_combined_signal(price).rename(symbol)
            ret = price.pct_change().rename(symbol)
            signals.append(sig)
            returns.append(ret)

            position_return = calc_vol_scaled_returns(sig.shift(1) * ret)
            position_returns.append(position_return.rename(symbol))

    position_df = pd.concat(position_returns, axis=1).fillna(0)

    mean_returns = position_df.mean(axis=1)
    daily_portfolio_returns = mean_returns.loc[TEST_START:]
    return daily_portfolio_returns

def calc_returns(srs: pd.Series, day_offset: int = 1) -> pd.Series:
    returns = srs / srs.shift(day_offset) - 1.0
    return returns

def calc_trend_intermediate_strategy(
    price: pd.Series,  symbol
) -> pd.Series:
    daily_returns = price.pct_change().rename(symbol)
    annual_returns = price.pct_change(252).rename(symbol)

    next_day_returns = calc_vol_scaled_returns(np.sign(annual_returns).shift(1)* daily_returns)

    return next_day_returns

def tsmom_strategy(assets):
    returns = []
    for symbol in assets:
        df = load_single_asset(symbol)
        if df is not None:
            price = df["Close"]
            ret = calc_trend_intermediate_strategy(price, symbol)
            ret = ret.fillna(0)
            ret.name = symbol
            returns.append(ret)

    returns_df = pd.concat(returns, axis=1).fillna(0)

    mean_returns = returns_df.mean(axis=1)

    daily_portfolio_returns = mean_returns.loc[TEST_START:]

    return daily_portfolio_returns



def plot_cumulative_returns(returns_dict, log_scale=True):
    plt.figure(figsize=(12, 6))

    for label, ret in returns_dict.items():
        cumulative = (1 + ret).cumprod()
        plt.plot(cumulative, label=label)

    if log_scale:
        plt.yscale("log")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Strategies (Rescaled to 15% Volatility)")
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
        # Stack only tensors (avoid metadata lists or dates)
        value = batch[0][key]
        if isinstance(value, torch.Tensor):
            batched_data[key] = torch.stack([item[key] for item in batch])

    return batched_data, batch_metadata


def backtest_xtrend():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XTrendModel(
        num_features=len(FEATURE_COLS),
        num_embeddings=len(PINNACLE_ASSETS),
        embedding_dim=EMBEDDING_DIM,
        d_h=D_H,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        warmup_period_len=WARMUP_PERIOD_LEN
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dataset = XTrendDataset(
        train_start = TRAIN_START,
        train_end = TRAIN_END,
        eval_start = TEST_START,
        eval_end = TEST_END,
        train_stride = TRAIN_STRIDE)
    dataset.set_mode("eval")
    wrapped_dataset = XTrendMetadataWrapper(dataset)

    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_with_metadata
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

    df_results = pd.DataFrame(predictions, columns=["date", "ticker", "pred", "true"])
    df_results.to_csv("xtrend_predictions.csv", index=True)
    df_results["strategy_return"] = df_results["pred"] * df_results["true"]

    df_results.set_index("date", inplace=True)
    df_results.index = pd.to_datetime(df_results.index)
    mask = (df_results.index >= TEST_START) & (df_results.index < TEST_END)
    avg_returns = df_results.loc[mask].groupby("date")["strategy_return"].mean().rename("XTrend_Return")
    return avg_returns

def single_backtest_xtrend():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XTrendModel(
        num_features=len(FEATURE_COLS),
        num_embeddings=len(PINNACLE_ASSETS),
        embedding_dim=EMBEDDING_DIM,
        d_h=D_H,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        warmup_period_len=WARMUP_PERIOD_LEN
    ).to(device)
    model.load_state_dict(torch.load("runs\\runs\\model_state_final.pth", map_location=device))
    model.eval()
    dataset = XTrendDataset(train_stride = TRAIN_STRIDE)
    dataset.set_mode("eval")
    predictions = []
    for i in trange(len(dataset), desc="Evaluating XTrend Model"):
        item = dataset.get_item_with_metadata(i)
        item2 = dataset[i]
        for key in item2:
            item2[key] = item2[key].unsqueeze(0).to(device)
        _, positions = model.evaluate(item2)
        prediction = {
            "date": item["metadata"]["target_end_date"],
            "ticker": item["metadata"]["target_ticker"],
            "pred": float(positions[:, -1, 0].item()),
            "true": float(item["target_y"][-1]) if hasattr(item["target_y"][-1], "item") else item["target_y"][-1],
        }
        predictions.append(prediction)

    df_results = pd.DataFrame(predictions, columns=["date", "ticker", "pred", "true"])
    df_results.to_csv("xtrend_predictions.csv", index=True)
    df_results["strategy_return"] = df_results["pred"] * df_results["true"]

    df_results.set_index("date", inplace=True)
    df_results.index = pd.to_datetime(df_results.index)
    mask = (df_results.index >= TEST_START) & (df_results.index < TEST_END)
    avg_returns = df_results.loc[mask].groupby("date")["strategy_return"].mean().rename("XTrend_Return")
    return avg_returns


if __name__ == "__main__":
    # 1) Long-only equal weighted portfolio
    portfolio_returns = compute_portfolio_returns(PINNACLE_ASSETS)
    scaled_long_only = rescale_to_target_vol_by_mean_asset_vol(portfolio_returns)

    # 2) MACD strategy portfolio
    macd = MACDStrategy()
    macd_returns = compute_macd_returns(PINNACLE_ASSETS, macd)
    scaled_macd = rescale_to_target_vol_by_mean_asset_vol(macd_returns)

    # 3) TSMOM strategy portfolio
    tsmom_returns = tsmom_strategy(PINNACLE_ASSETS)
    scaled_tsmom = rescale_to_target_vol_by_mean_asset_vol(tsmom_returns)

    # 4) XTrend model portfolio
    xtrend_returns = backtest_xtrend()
    scaled_xtrend = rescale_to_target_vol_by_mean_asset_vol(xtrend_returns)

    # Plot all cumulative returns together
    plt.figure(figsize=(12, 6))
    plt.plot((1 + scaled_long_only).cumprod(), label="Long-only Equal Weight")
    plt.plot((1 + scaled_macd).cumprod(), label="MACD Strategy")
    plt.plot((1 + scaled_tsmom).cumprod(), label="TSMOM Strategy")
    plt.plot((1 + scaled_xtrend).cumprod(), label="XTrend Model")
    plt.yscale("log")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Portfolio Performance Comparison (Scaled to 15% Volatility)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("backtest_results.png")
