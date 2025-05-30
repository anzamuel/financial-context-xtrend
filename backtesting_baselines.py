import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from empyrical import annual_volatility
from typing import List, Tuple
import torch

# List of all futures symbols
PINNACLE_ASSETS = [
    "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER",
    "ES", "FB", "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX",
    "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP", "TY", "UB",
    "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZI", "ZK",
    "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ"
]

DATA_DIR = "dataset\\CLCDATA"  # Adjust path as needed
START_DATE = "2013-01-01"
EXTENDED_START_DATE = "2012-01-01"
END_DATE = "2023-01-01"
TARGET_VOL = 0.15  # Target annual volatility
VOL_LOOKBACK = 60

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
        df = df.loc[EXTENDED_START_DATE:END_DATE]
        df = df[df["Close"] > 0]
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
    portfolio_returns = portfolio_returns.loc[START_DATE:]
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
    daily_portfolio_returns = mean_returns.loc[START_DATE:]
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

    daily_portfolio_returns = mean_returns.loc[START_DATE:]

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

def backtest_xtrend():
    returns_df = pd.read_csv("predicted_returns.csv").fillna(0)

    # Use trading dates from a real asset (e.g., ES)
    ref_asset = load_single_asset("ES")
    trading_dates = ref_asset.index[ref_asset.index >= START_DATE]
    n_dates = len(trading_dates)

    # Enforce: number of rows in group must match n_dates
    ticker_dfs = []

    for ticker, group in returns_df.groupby("ticker"):
        predicted_returns = group["predicted_return"].values

        # Truncate or pad to exactly n_dates
        if len(predicted_returns) < n_dates:
            predicted_returns = np.pad(predicted_returns, (n_dates - len(predicted_returns), 0), constant_values=0)
        elif len(predicted_returns) > n_dates:
            predicted_returns = predicted_returns[-n_dates:]

        df = pd.DataFrame({
            "date": trading_dates,
            f"ticker_{ticker}": predicted_returns
        })
        ticker_dfs.append(df)

    # Merge all ticker dataframes by date
    merged_df = ticker_dfs[0]
    for df in ticker_dfs[1:]:
        merged_df = merged_df.merge(df, on="date", how="outer")

    merged_df = merged_df.fillna(0).sort_values("date")
    merged_df.set_index("date", inplace=True)

    # Average across tickers
    merged_df["mean_return"] = merged_df.mean(axis=1)

    return merged_df["mean_return"]


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
    print(xtrend_returns.head())
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
    plt.show()
    
