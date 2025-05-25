# %%
#%cd /Users/sammy/Desktop/financial-context-xtrend

# %% imports
import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm

import os
import matplotlib.pyplot as plt
import itertools
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# %% constants
CONTEXT_LBW = 21
MIN_CONTEXT_LEN = 5
MAX_CONTEXT_LEN = 63
CPD_THRESHOLD = 0.95
NUMPY_DTYPE = np.float32
FEATURE_COLS = [
    "norm_daily_return",
    "norm_monthly_return",
    "norm_quarterly_return",
    "norm_biannual_return",
    "norm_annual_return",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96"
]
VALUE_COL = "next_day_norm_return"
TEST_YEAR_START = 2015
PINNACLE_ASSETS = [ "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ" ]
# PINNACLE_ASSETS_TRAIN = ["CC", "DA", "LB", "SB", "ZA", "ZC", "ZF", "ZI", "ZO", "ZR", "ZU", "ZW", "ZZ", "EN", "ES", "MD", "SC", "SP", "XX", "YM", "DT", "FB", "TY", "UB", "US", "AN", "DX", "FN", "JN", "SN"]
# PINNACLE_ASSETS_TEST = ["GI", "JO", "KC", "KW", "NR", "ZG", "ZH", "ZK", "ZL", "ZN", "ZP", "ZT", "CA", "ER", "LX", "NK", "XU", "BN", "CN", "MP"]
# assert set(PINNACLE_ASSETS) == set(PINNACLE_ASSETS_TRAIN + PINNACLE_ASSETS_TEST)
# TEST_END = 2020
# PINNACLE_ASSETS_TRAIN = [ "CC", "DA", "CA", "LB", "SB", "ZA", "ZC", "ZF", ZI ]
def plot_segmented_close_prices(ticker, features_df, regimes):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{sfmath}'

    plt.figure(figsize=(15, 4))
    ax = plt.gca()

    colormap = plt.get_cmap('tab20')
    mpl_colors = [colormap(i) for i in np.linspace(0, 1, colormap.N)]
    color_cycle = itertools.cycle(mpl_colors)

    # Add background spans for M regimes
    m_color = 'lightgray'
    bg_alpha = 0.3

    for regime in regimes:
        start_pos, end_pos, regime_type = regime
        slice_start_pos = max(0, start_pos)
        slice_end_pos = end_pos + 1

        if slice_start_pos >= len(features_df) or slice_start_pos >= slice_end_pos:
            continue
        if slice_end_pos > len(features_df):
            slice_end_pos = len(features_df)

        # Get actual start and end dates from indices
        start_date = features_df.iloc[slice_start_pos]['date']
        end_date = features_df.iloc[slice_end_pos - 1]['date']

        if regime_type == 'M':
            ax.axvspan(start_date, end_date, facecolor=m_color, alpha=bg_alpha, zorder=-1) # zorder         to              putbackground       behind line

    # Plot the close price lines
    for regime in regimes:
        start_pos, end_pos, regime_type = regime

        slice_start_pos = max(0, start_pos)
        slice_end_pos = end_pos + 1

        if slice_start_pos >= len(features_df) or slice_start_pos >= slice_end_pos:
            continue
        if slice_end_pos > len(features_df):
            slice_end_pos = len(features_df)

        regime_data = features_df.iloc[slice_start_pos : slice_end_pos]

        if not regime_data.empty:
            # Use color cycle only for C regimes, or color all and M background highlights
            # Based on request, line color is from cycle, background is for M
            color = next(color_cycle)
            plt.plot(regime_data['date'], regime_data['close'], color=color, linewidth=2.0, alpha=0.8, zorder=1)            #zorder       to put line in front

    plt.xlabel('Date')
    plt.ylabel('Close Price in USD')
    plt.title(rf'\textbf{{CPD Segmented Close Price Over Time for {ticker}}}')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.set_xlim([features_df['date'].min(), features_df['date'].max()])

    # Create custom legend for M regime background
    legend_elements = [
        Patch(facecolor=m_color, alpha=bg_alpha, label=f'contexts with max length ({MAX_CONTEXT_LEN})' )
    ]
    ax.legend(handles=legend_elements, edgecolor='black', loc='upper right', framealpha=1.0, fancybox=False)

    plt.tight_layout()

    plots_dir = "plots/"
    os.makedirs(plots_dir, exist_ok=True)
    file_path = os.path.join(plots_dir, f"{ticker}_prices_cpd_segmented.png")
    plt.savefig(file_path, dpi=600)
    plt.close()


def create_targets_and_contexts(target_len = 126, write_to_disk = False, draw_plots = False):
    targets = {}
    contexts = []
    context_definitions, target_definitions = [], []
    ticker_pbar = tqdm(PINNACLE_ASSETS, dynamic_ncols=True)
    for ticker in ticker_pbar:
        ticker_pbar.set_description(ticker)

        # %% RUN TASKS
        features_df = pd.read_csv(f"dataset/FEATURES/{ticker}.csv", parse_dates=["date"])
        features_df[VALUE_COL] = features_df["norm_daily_return"].shift(-1)
        features_df = features_df[features_df.date <= dt.datetime(TEST_YEAR_START, 1, 1)]
        if features_df.empty:
            tqdm.write(f"Warning: Ticker {ticker} has no data before year {TEST_YEAR_START}. Skipping.")
            continue
        features_df = features_df[features_df.date < dt.datetime(TEST_YEAR_START, 1, 1)]
        features_df = features_df[["date", "close"] + FEATURE_COLS + [VALUE_COL]]
        #features_df.dropna(inplace=True)

        # CREATE TARGETS
        targets_df = features_df.copy()
        targets_df = targets_df[["date"] + FEATURE_COLS + [VALUE_COL]]
        targets[ticker] = targets_df

        for end_idx in range(target_len - 1 , len(targets_df)):
            target_definitions.append({
                "ticker": ticker,
                "end_idx": end_idx,
                "end_date": targets_df["date"][end_idx]
            })

        if write_to_disk:
            base_dir = "dataset_targets"
            os.makedirs(base_dir, exist_ok=True)
            file_name = f"target_{ticker}.parquet"
            file_path = os.path.join(base_dir, file_name)
            targets_df.to_parquet(file_path)

        changepoints_df = pd.read_csv(f"dataset/CPD/{CONTEXT_LBW}/{ticker}.csv", parse_dates=["date"])
        changepoints_df.ffill(inplace=True)
        changepoints_df.bfill(inplace=True)
        #changepoints_df.dropna(inplace=True)

        features_df = features_df.merge(changepoints_df, on="date")
        assert features_df["t"].is_unique
        assert features_df["t"].is_monotonic_increasing
        assert (features_df["t"].diff()[1:] == 1).all()
        features_df = features_df.set_index("t")

        # %%
        min_t = int(features_df.index.min())
        max_t = int(features_df.index.max())

        # %%
        # time series cpd segmentation algorithm
        regimes = []
        t: int = max_t
        t1: int = max_t
        while t >= min_t:
            current_cp_score = features_df.loc[t, "cp_score"]
            if current_cp_score >= CPD_THRESHOLD:
                tc: float = features_df.loc[t, "cp_location"]
                t0: int = round(tc)
                if t1 - t0 >= MIN_CONTEXT_LEN:
                    regimes.insert(0, (t0,t1,"C") if draw_plots else (t0,t1))
                    t = t0 - 1
                    t1 = t0 - 1
                    continue
            if t1 - t == MAX_CONTEXT_LEN:
                t0: int = t
                regimes.insert(0, (t0,t1,"M") if draw_plots else (t0,t1)) #
                t = t0 - 1
                t1 = t0 - 1
                continue
            t = t - 1
            # TODO REMOVE
            if t1 - t > MAX_CONTEXT_LEN:
                print(t1, t)
                assert False

        #print(min_t, max_t, len(regimes), len([_ for _ in regimes if _[1] == 'C']), len([_ for _ in regimes if _[1] ==         'M'))
        #print(changepoints_df["date"], features_df["date"])

        # Validate regimes
        if regimes:
            # Check segment properties
            assert regimes[-1][1] == max_t
            for i, segment in enumerate(regimes):
                start, end = segment[0], segment[1]
                assert start < end, f"Segment {i}: start must be less than end"
                segment_length = end - start
                assert MIN_CONTEXT_LEN <= segment_length <= MAX_CONTEXT_LEN + CONTEXT_LBW, f"Segment {i}: invalid length        {(start, end)}" # the true maximum sequence is 84

            # Check continuity between segments
            for i in range(1, len(regimes)):
                assert regimes[i][0] == regimes[i-1][1] + 1, f"Gap between segments {i-1} and {i}"
        # %%
        #dataset_type = "train" if ticker in PINNACLE_ASSETS_TRAIN else "test"

        if write_to_disk:
            cpd_contexts_base_dir = "dataset_cpd_contexts"
            cpd_contexts_ticker_dir = os.path.join(cpd_contexts_base_dir, ticker)
            os.makedirs(cpd_contexts_ticker_dir, exist_ok=True)

        with tqdm(total=len(regimes), desc=f"{ticker} segments", leave=False) as segment_pbar:
            for i, segment in enumerate(regimes):
                start_idx, end_idx = segment[0], segment[1]
                context_df = features_df.loc[start_idx:end_idx].copy()
                context_df = context_df.reset_index().set_index("date")
                context_df = context_df[FEATURE_COLS + [VALUE_COL]]

                contexts.append(context_df)

                start_date = context_df.index.min()
                end_date = context_df.index.max()

                context_definitions.append({
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                })

                if write_to_disk:
                    file_name = f"context_{i:03d}_{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"
                    file_path = os.path.join(cpd_contexts_ticker_dir, file_name)
                    context_df.to_parquet(file_path, index=True)

                segment_pbar.update(1)

        if draw_plots:
            plot_segmented_close_prices(ticker, features_df, regimes)

    return targets, target_definitions, contexts, context_definitions, PINNACLE_ASSETS, FEATURE_COLS, VALUE_COL

if __name__ == "__main__":
    create_targets_and_contexts(write_to_disk = True, draw_plots = True)