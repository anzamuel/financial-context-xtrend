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
FEATURES = [
    "norm_daily_return",
    "norm_monthly_return",
    "norm_quarterly_return",
    "norm_biannual_return",
    "norm_annual_return",
    "macd_8_24",
    "macd_16_48",
    "macd_32_96"
]
TEST_YEAR_START = 2015
PINNACLE_ASSETS = [ "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ" ]
PINNACLE_ASSETS_TRAIN = ["CC", "DA", "LB", "SB", "ZA", "ZC", "ZF", "ZI", "ZO", "ZR", "ZU", "ZW", "ZZ", "EN", "ES", "MD", "SC", "SP", "XX", "YM", "DT", "FB", "TY", "UB", "US", "AN", "DX", "FN", "JN", "SN"]
PINNACLE_ASSETS_TEST = ["GI", "JO", "KC", "KW", "NR", "ZG", "ZH", "ZK", "ZL", "ZN", "ZP", "ZT", "CA", "ER", "LX", "NK", "XU", "BN", "CN", "MP"]
assert set(PINNACLE_ASSETS) == set(PINNACLE_ASSETS_TRAIN + PINNACLE_ASSETS_TEST)
DRAW_PLOTS = False
# TEST_END = 2020
# PINNACLE_ASSETS_TRAIN = [ "CC", "DA", "CA", "LB", "SB", "ZA", "ZC", "ZF", ZI ]

# contexts = []
ticker_pbar = tqdm(PINNACLE_ASSETS, dynamic_ncols=True)
for ticker in ticker_pbar:
    ticker_pbar.set_description(ticker)

    # %% RUN TASKS
    features_df = pd.read_csv(f"dataset/FEATURES/{ticker}.csv", parse_dates=["date"])
    features_df = features_df[features_df.date < dt.datetime(TEST_YEAR_START, 1, 1)]
    if features_df.empty:
        tqdm.write(f"Warning: Ticker {ticker} has no data before year {TEST_YEAR_START}. Skipping.")
        continue
    features_df["next_day_norm_return"] = features_df["norm_daily_return"].shift(-1)
    features_df = features_df[["date", "close"] + FEATURES + ["next_day_norm_return"]]
    #features_df.dropna(inplace=True)

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
                regimes.insert(0, (t0,t1,"C") if DRAW_PLOTS else (t0,t1))
                t = t0 - 1
                t1 = t0 - 1
                continue
        if t1 - t == MAX_CONTEXT_LEN:
            t0: int = t
            regimes.insert(0, (t0,t1,"M") if DRAW_PLOTS else (t0,t1)) #
            t = t0 - 1
            t1 = t0 - 1
            continue
        t = t - 1
        # TODO REMOVE
        if t1 - t > MAX_CONTEXT_LEN:
            print(t1, t)
            assert False

    #print(min_t, max_t, len(regimes), len([_ for _ in regimes if _[1] == 'C']), len([_ for _ in regimes if _[1] == 'M'))
    #print(changepoints_df["date"], features_df["date"])

    # Validate regimes
    if regimes:
        # Check segment properties
        assert regimes[-1][1] == max_t
        for i, segment in enumerate(regimes):
            start, end = segment[0], segment[1]
            assert start < end, f"Segment {i}: start must be less than end"
            segment_length = end - start
            assert MIN_CONTEXT_LEN <= segment_length <= MAX_CONTEXT_LEN + CONTEXT_LBW, f"Segment {i}: invalid length {(start, end)}" # the true maximum sequence is 84

        # Check continuity between segments
        for i in range(1, len(regimes)):
            assert regimes[i][0] == regimes[i-1][1] + 1, f"Gap between segments {i-1} and {i}"
# %%
    dataset_type = "train" if ticker in PINNACLE_ASSETS_TRAIN else "test"
    base_dir = "cpd_contexts_dataset"
    ticker_dir = os.path.join(base_dir, dataset_type, ticker)
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)
    os.makedirs(ticker_dir, exist_ok=True)
    ticker_metadata = []

    with tqdm(total=len(regimes), desc=f"{ticker} segments", leave=False) as segment_pbar:
        for i, segment in enumerate(regimes):
            start_idx, end_idx = segment[0], segment[1]
            segmented_features_df = features_df.loc[start_idx:end_idx].copy()
            segmented_features_df = segmented_features_df.reset_index().set_index("date")
            segmented_features_df = segmented_features_df[FEATURES + ["next_day_norm_return"]]

            start_date = segmented_features_df.index.min().strftime('%Y%m%d')
            end_date = segmented_features_df.index.max().strftime('%Y%m%d')

            file_name = f"context_{i:03d}_{ticker}_{start_date}_{end_date}.parquet"
            file_path = os.path.join(ticker_dir, file_name)
            segmented_features_df.to_parquet(file_path, index=True)

            segment_pbar.update(1)

    if DRAW_PLOTS:
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
                ax.axvspan(start_date, end_date, facecolor=m_color, alpha=bg_alpha, zorder=-1) # zorder to              putbackground       behind line

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
