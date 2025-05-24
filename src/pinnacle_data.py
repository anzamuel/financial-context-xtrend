# %% IMPORTS
import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm



# %% CONSTANTS
MAX_SERIES = 63
MIN_SERIES = 3
BATCH_SIZE = 32

NUMPY_DTYPE = np.float32  # Use float32 for all arrays

LBW_FOR_CONTEXTS = 63  # Changed from 21 to 63 as specified

# Features required from the csv
REQUIRED_FEATURES = [
    "norm_daily_return",    # Already used but was calculated - now import directly
    "norm_monthly_return",  # New feature
    "norm_quarterly_return", # New feature
    "norm_biannual_return", # New feature
    "norm_annual_return",   # New feature
    "macd_8_24",            # New feature
    "macd_16_48",           # New feature
    "macd_32_96"            # New feature
]

TEST_YEAR_START = 2015

TEST_END = 2020

PINNACLE_ASSETS = [
        "AN",
        "BN",
        "CA",
        "CC",
        "CN",
        "DA",
        "DT",
        "DX",
        "EN",
        "ER",
        "ES",
        "FB",
        "FN",
        "GI",
        "JN",
        "JO",
        "KC",
        "KW",
        "LB",
        "LX",
        "MD",
        "MP",
        "NK",
        "NR",
        "SB",
        "SC",
        "SN",
        "SP",
        "TY",
        "UB",
        "US",
        "XU",
        "XX",
        "YM",
        "ZA",
        "ZC",
        "ZF",
        "ZG",
        "ZH",
        "ZI",
        "ZK",
        "ZL",
        "ZN",
        "ZO",
        "ZP",
        "ZR",
        "ZT",
        "ZU",
        "ZW",
        "ZZ",
    ]



# %% FUNC ASSIGN TASKS
def assign_tasks(
    changepoint_data: pd.DataFrame,
    changepoint_threshold: float = 0.995,
    burn_in: int = 5,
) -> pd.DataFrame:

    boundaries = changepoint_data[
        (changepoint_data["cp_score"] >= changepoint_threshold)
    ]

    last_location = boundaries.iloc[0]["cp_location"]
    cp_locations = [last_location]

    data_w_tasks = changepoint_data.copy()
    data_w_tasks["task"] = np.nan

    data_w_tasks.loc[boundaries.index[0], "task"] = 0
    task_number = 1

    for idx, row in boundaries.iloc[1:, :].iterrows():
        # print(idx)
        if row["cp_location"] - last_location >= burn_in:
            data_w_tasks.loc[idx, "task"] = task_number
            # last_change_date = row["date"]
            last_location = row["cp_location"]
            cp_locations.append(last_location)
            task_number += 1

    data_w_tasks = data_w_tasks.bfill().fillna(task_number)
    data_w_tasks["task"] = data_w_tasks["task"].astype(int)

    def boundary_task(l):
        last_before = int(np.floor(l))
        return (
            changepoint_data.reset_index()
            .set_index("t")
            .loc[list(range(last_before - burn_in + 1, last_before + burn_in + 1))]
        )

    cp_loc_srs = pd.Series(cp_locations)
    cp_loc_srs = cp_loc_srs[cp_loc_srs >= changepoint_data["t"].min() + burn_in]
    # print(cp_loc_srs)

    boundary_tasks = cp_loc_srs.map(boundary_task)
    # start_task = data_w_tasks["task"].max() + 1
    # list(range(start_task, start_task + len(boundary_tasks)))

    for i in range(len(boundary_tasks)):
        boundary_tasks.iloc[i] = (
            boundary_tasks.iloc[i]
            .assign(task=(-i - 1))
            .reset_index()
            .set_index("date")[
                ["t", "cp_location", "cp_location_norm", "cp_score", "task"]
            ]
        )

    return pd.concat([data_w_tasks] + boundary_tasks.tolist())



# %% FUNC SPLIT_DATAFRAME
def split_dataframe(df, max_series=MAX_SERIES, min_series=MIN_SERIES):
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



# %% FUNC READ_CHANGEPOINT_RESULTS_AND_FILL_NA
def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .ffill()
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )



# %% FUNC ASSIGN TASKS
def assign_tasks(
    changepoint_data: pd.DataFrame, changepoint_threshold: float = 0.995, burn_in: int = 5
) -> pd.DataFrame:

    boundaries = changepoint_data[
        (changepoint_data["cp_score"] >= changepoint_threshold)
    ]
    if len(boundaries) == 0:
        data_w_tasks = changepoint_data.copy()
        data_w_tasks["task"] = 0
        return data_w_tasks

    last_location = boundaries.iloc[0]["cp_location"]
    data_w_tasks = changepoint_data.copy()
    data_w_tasks["task"] = np.nan

    data_w_tasks.loc[boundaries.index[0], "task"] = 0
    task_number = 1

    for idx, row in boundaries.iloc[1:, :].iterrows():
        # print(idx)
        if row["cp_location"] - last_location >= burn_in:
            data_w_tasks.loc[idx, "task"] = task_number
            # last_change_date = row["date"]
            last_location = row["cp_location"]
            task_number += 1

    data_w_tasks = data_w_tasks.bfill().fillna(task_number)
    data_w_tasks["task"] = data_w_tasks["task"].astype(int)
    return data_w_tasks



# %% RUN TASKS
subtasks = []

progress_bar = tqdm(PINNACLE_ASSETS, dynamic_ncols=True)
for ticker in progress_bar:
    progress_bar.set_description(ticker)
    srs = pd.read_csv(f"dataset/pinnacle_features.csv")
    srs = srs[srs["ticker"]==ticker]

    # Make sure we have all required features
    for feature in REQUIRED_FEATURES:
        if feature not in srs.columns:
            print(f"Warning: Feature {feature} not found in pinnacle_features.csv")

    # Use the normalized daily return shifted by 1 as the target - no manual scaling needed
    srs["next_day_return"] = srs["norm_daily_return"].shift(-1)
    srs = srs.dropna()
    srs["date"] = pd.to_datetime(srs["date"])
    srs = srs[srs.date < dt.datetime(TEST_YEAR_START, 1, 1)]

    changepoint_data = read_changepoint_results_and_fill_na(
        f"dataset/CPD/{LBW_FOR_CONTEXTS}/{ticker}.csv", LBW_FOR_CONTEXTS
    )

    # TODO this is a mess
    # changepoint_data["date"] = changepoint_data.index
    changepoint_data = changepoint_data.merge(srs[["date"] + REQUIRED_FEATURES + ["next_day_return"]], on="date")
    changepoint_data = changepoint_data.set_index("date")
    changepoint_data.index.name = "date"

    # print(changepoint_data)

    tasks = assign_tasks(changepoint_data)

    # get rid of boundary tasks for now
    tasks = tasks[tasks["task"] >= 0]
    task_count = tasks.groupby("task")["t"].count()

    for task_num, task in tasks.groupby("task"):
        # print(task_num)
        splits = split_dataframe(task, MAX_SERIES)
        for i in range(len(splits)):
            splits[i] = splits[i].assign(subtask=i).assign(ticker=ticker)
            # No manual volatility scaling - using normalized features directly
            # print(splits[i]["norm_daily_return"].std()*np.sqrt(252))
        subtasks += splits

all_tasks = pd.concat(subtasks)
unique_tasks = (
    all_tasks[["task", "subtask", "ticker"]].drop_duplicates().reset_index(drop=True)
)
unique_tasks["set"] = unique_tasks.index
all_tasks["date"] = all_tasks.index
all_tasks = all_tasks.merge(unique_tasks, on=["task", "subtask", "ticker"])
# all_tasks

# train = all_tasks[all_tasks.date < dt.datetime(TEST_YEAR_START, 1, 1)]
train = all_tasks.groupby("set").filter(lambda x: len(x) >= MIN_SERIES)

# %% TODO
# def query_set_train_all_segments(set_num: int):
#     segments = []
#     data_whole = train[train["set"] == set_num]
#     for length in range(MIN_SERIES, len(data_whole) + 1):
#         # Using data with pre-normalized features - no manual scaling needed
#         data = data_whole.iloc[0:length].copy()

#         # Create feature array with all 8 features - shape: [seq_len, features]
#         x_features = np.zeros((len(data), len(REQUIRED_FEATURES)), dtype=NUMPY_DTYPE)
#         for i, feature in enumerate(REQUIRED_FEATURES):
#             x_features[:, i] = data[feature].values.astype(NUMPY_DTYPE)

#         context_x = x_features
#         context_y = data["next_day_return"].values.reshape(len(data), 1).astype(NUMPY_DTYPE)
#         target_x = np.array([[float(len(data))]], dtype=NUMPY_DTYPE)
#         target_y = np.array([[data["next_day_return"].iloc[-1]]], dtype=NUMPY_DTYPE)
#         query = (context_x, context_y), target_x
#         segments.append((query, target_y, len(data)))
#     return segments


def context_all_segments(set_num: int):
    segments = []
    data_whole = train[train["set"] == set_num]
    for length in range(MIN_SERIES, len(data_whole) + 1):
        # Using data with pre-normalized features - no manual scaling needed
        data = data_whole.iloc[0:length].copy()
        date = data["date"].iloc[-1]

        # Create feature array with all 8 features - shape: [seq_len, features]
        x_features = np.zeros((len(data), len(REQUIRED_FEATURES)), dtype=NUMPY_DTYPE)
        for i, feature in enumerate(REQUIRED_FEATURES):
            x_features[:, i] = data[feature].values.astype(NUMPY_DTYPE)

        context_x = x_features
        context_y = data["next_day_return"].values.reshape(len(data), 1).astype(NUMPY_DTYPE)

        context = (
            context_x,
            context_y,
            set_num,
            len(data),
            date,
        )
        segments.append(context)
    return segments


segments_and_ticker = train[["set", "ticker"]].drop_duplicates().reset_index(drop=True)

train_data_prepped_all_segments = pd.DataFrame(
    pd.Series(
        pd.Series(segments_and_ticker["set"]).map(context_all_segments).sum()
    ).tolist(),
    columns=["x", "y", "set", "seq_len", "date"],
).merge(segments_and_ticker, on="set")

# Save to pickle without converting to tensors
train_data_prepped_all_segments.to_pickle(f"prepped.pkl")
train_data_prepped_all_segments.to_csv(f"prepped.csv")
