# %% imports
import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm

# %% constants
MIN_CONTEXT_LEN = 3
MAX_CONTEXT_LEN = 63
BATCH_SIZE = 32
NUMPY_DTYPE = np.float32
CONTEXT_LBW = 63 # TODO: we could experiment with other lookback windows
CPD_THRESHOLD = 0.95 # threshold for cpd score to be considered a changepoint TODO: we could fine tune this
CPD_BURN_IN = 5 # minimum distance between leftmost changepoints of contexts
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
TEST_END = 2025
PINNACLE_ASSETS = [ "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG",  "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ" ]
# <<<< TODO: DO NOT TOUCH ANYTHING BELOW THIS! I AM WORKING ON IT >>>>
# %% function split_dataframe
def split_dataframe(df, max_series=MAX_CONTEXT_LEN, min_series=MIN_CONTEXT_LEN):
    chunks = list()
    if len(df) % max_series < min_series:
        num_chunks = len(df) // max_series
    else:
        num_chunks = len(df) // max_series + 1
    for i in range(num_chunks):
        if i == num_chunks - 1:
            chunks.append(df.iloc[i * max_series :])
        else:
            chunks.append(df.iloc[i * max_series : (i + 1) * max_series])
    return chunks
# <<<< TODO: DO NOT TOUCH ANYTHING ABOVE THIS! I AM WORKING ON IT >>>>

# %% RUN TASKS
subtasks = []
for ticker in (pbar := tqdm(PINNACLE_ASSETS, dynamic_ncols=True)):
    pbar.set_description(ticker)
# FOR DEBUG:
# while(True):
#     ticker = "AN"
    features_df = pd.read_csv(f"features/{ticker}.csv", parse_dates=["date"])
    features_df["next_day_norm_return"] = features_df["norm_daily_return"].shift(-1)
    features_df = features_df[["date"] + FEATURES + ["next_day_norm_return"]]
    features_df.dropna(inplace=True)
    features_df = features_df[features_df.date < dt.datetime(TEST_YEAR_START, 1, 1)]

    changepoints_df = pd.read_csv(f"dataset/CPD/{CONTEXT_LBW}/{ticker}.csv",parse_dates=["date"])
    changepoints_df.ffill(inplace=True)
    changepoints_df.dropna(inplace=True)

    features_df = features_df.merge(changepoints_df, on="date")
    features_df = features_df.set_index("date")

    boundaries_idx = features_df[(features_df["cp_score"] >= CPD_THRESHOLD)].index
    if len(boundaries_idx) == 0: # no contexts
        features_df["context_num"] = 0
    else:
        last_cp_location = features_df.iloc[0]["cp_location"]
        features_df["context_num"] = np.nan
        features_df.loc[boundaries_idx[0], "context_num"] = 0
        context_num = 1
        for idx in boundaries_idx[1:]:
            current_cp_location = features_df.loc[idx, "cp_location"]
            if current_cp_location - last_cp_location >= CPD_BURN_IN:
                features_df.loc[idx, "context_num"] = context_num
                last_location = current_cp_location
                context_num += 1
        features_df.bfill(inplace=True)
        features_df.fillna(context_num, inplace=True) # fills the last rows with last context_num
        features_df["context_num"] = features_df["context_num"].astype(int)

    features_groupedby_context = features_df.groupby("context_num")
    task_count = features_groupedby_context.size()

# <<<< TODO: DO NOT TOUCH ANYTHING BELOW THIS! I AM WORKING ON IT >>>>
    # TODO: UNDERSTAND
    for context_num, sub_features_df in features_groupedby_context:
        # print(task_num)
        # print(sub_features_df)
        splits = split_dataframe(sub_features_df, MAX_CONTEXT_LEN)
        # print(f"[{task_count[context_num]} {len(splits)}]", end=" ")
        for i in range(len(splits)):
            splits[i] = splits[i].assign(subtask=i).assign(ticker=ticker)
            # No manual volatility scaling - using normalized features directly
            # print(splits[i]["norm_daily_return"].std()*np.sqrt(252))
        subtasks += splits

# FOR DEBUG:
#    break
# %%

all_tasks = pd.concat(subtasks)
unique_tasks = (
    all_tasks[["context_num", "subtask", "ticker"]].drop_duplicates().reset_index(drop=True)
)
unique_tasks["set"] = unique_tasks.index
all_tasks["date"] = all_tasks.index
all_tasks = all_tasks.merge(unique_tasks, on=["context_num", "subtask", "ticker"])
# all_tasks

# train = all_tasks[all_tasks.date < dt.datetime(TEST_YEAR_START, 1, 1)]
train = all_tasks.groupby("set").filter(lambda x: len(x) >= MIN_CONTEXT_LEN)

# %% TODO
def context_all_segments(set_num: int):
    segments = []
    data_whole = train[train["set"] == set_num]
    for length in range(MIN_CONTEXT_LEN, len(data_whole) + 1):
        # Using data with pre-normalized features - no manual scaling needed
        data = data_whole.iloc[0:length].copy()
        date = data["date"].iloc[-1]

        # Create feature array with all 8 features - shape: [seq_len, features]
        x_features = np.zeros((len(data), len(FEATURES)), dtype=NUMPY_DTYPE)
        for i, feature in enumerate(FEATURES):
            x_features[:, i] = data[feature].values.astype(NUMPY_DTYPE)

        context_x = x_features
        context_y = data["next_day_norm_return"].values.reshape(len(data), 1).astype(NUMPY_DTYPE)

        context = (
            context_x,
            context_y,
            set_num,
            len(data),
            date,
        )
        segments.append(context)
    return segments


# %% TODO
segments_and_ticker = train[["set", "ticker"]].drop_duplicates().reset_index(drop=True)

train_data_prepped_all_segments = pd.DataFrame(
    pd.Series(
        pd.Series(segments_and_ticker["set"]).map(context_all_segments).sum()
    ).tolist(),
    columns=["x", "y", "set", "seq_len", "date"],
).merge(segments_and_ticker, on="set")

# %% save to pickle
train_data_prepped_all_segments.to_pickle(f"pinnacle_contexts.pkl")
# <<<< TODO: DO NOT TOUCH ANYTHING ABOVE THIS! I AM WORKING ON IT >>>>
