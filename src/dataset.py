import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from tqdm.auto import tqdm
import pandas as pd
import datetime as dt
import multiprocessing as mp

# --- Constants ---
LBW_LEN = 21 # $l_{\matrm{lbw}} = 21$
MIN_CONTEXT_LEN = 5 # $l_{\matrm{min}} = 5$
MAX_CONTEXT_LEN = 21 # $l_c = l_{\matrm{max}} = 21$
CPD_THRESHOLD = 0.90 # $\nu = 0.9$ TODO: fine tune it for each asset!
FEATURE_COLS = [
    "norm_daily_return", "norm_monthly_return", "norm_quarterly_return",
    "norm_biannual_return", "norm_annual_return", "macd_8_24", "macd_16_48",
    "macd_32_96"
]
VALUE_COL = "next_day_norm_return"
PINNACLE_ASSETS = [
    "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB",
    "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR",
    "SB", "SC", "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC",
    "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU",
    "ZW", "ZZ"
]
TARGET_LEN = 126 # $l_t = 126$
CONTEXT_SAMPLE_SIZE = 20 # $\abs{\mathcal{C}} = 20$

# TODO: base every dataframe on polars
def _cpd_segmentation(features_df):
    min_t, max_t = int(features_df.index.min()), int(features_df.index.max())
    regimes = []
    t = t1 = max_t
    while t >= min_t:
        if features_df.loc[t, "cp_score"] >= CPD_THRESHOLD:
            t0 = round(features_df.loc[t, "cp_location"])
            if t1 - t0 >= MIN_CONTEXT_LEN:
                if t1 - t0 <= MAX_CONTEXT_LEN - 1:
                    regimes.insert(0, (t0, t1))
                else:
                    t1_tilde = t1
                    while True:
                        t0_tilde = max(t0, t1_tilde - MAX_CONTEXT_LEN + 1)
                        if t1_tilde - t0_tilde >= MIN_CONTEXT_LEN:
                            regimes.insert(0, (t0_tilde, t1_tilde))
                        if t0_tilde == t0:
                            break
                        t1_tilde = t0_tilde - 1
                t = t1 = t0 - 1
                continue
        if t1 - t == MAX_CONTEXT_LEN - 1:
            t0 = t
            regimes.insert(0, (t0, t1))
            t = t1 = t0 - 1
            continue
        t = t - 1
    return regimes

def _process_ticker_data(args):
    ticker, s_tensor, target_len, train_start, train_end, eval_start, eval_end = args
    warnings = []

    train_x_list, train_y_list, train_s_list, train_start_dates_list, train_end_dates_list = [], [], [], [], []
    eval_x_list, eval_y_list, eval_s_list, eval_start_dates_list, eval_end_dates_list = [], [], [], [], []
    context_s_list, context_x_list, context_xi_list, context_lens_list, context_start_dates_list, context_end_dates_list = [], [], [], [], [], []

    try:
        features_df = pd.read_csv(f"dataset/FEATURES/{ticker}.csv", parse_dates=["date"])
    except FileNotFoundError:
        warnings.append(f"Warning: Features file not found for ticker {ticker}. Skipping.")
        return None, warnings

    features_df[VALUE_COL] = features_df["norm_daily_return"].shift(-1)

    selection_mask_train = (features_df.date >= train_start) & (features_df.date < train_end)
    if not selection_mask_train.any():
        warnings.append(f"Warning: Ticker {ticker} has no data before train end {train_end.strftime('%Y-%m-%d')}. Skipping.")
        return None, warnings
    selection_mask_eval = (features_df.date >= eval_start) & (features_df.date < eval_end)
    selection_mask_train_or_eval = selection_mask_train | selection_mask_eval

    targets_train_df = features_df[selection_mask_train].copy()
    for end_idx in range(target_len - 1, len(targets_train_df)):
        start_idx = end_idx - target_len + 1
        target_slice_df = targets_train_df.iloc[start_idx : end_idx + 1]
        train_x_list.append(torch.tensor(target_slice_df[FEATURE_COLS].values, dtype=torch.float32))
        train_y_list.append(torch.tensor(target_slice_df[VALUE_COL].values, dtype=torch.float32))
        train_s_list.append(s_tensor)
        train_start_dates_list.append(target_slice_df.iloc[0]['date'].toordinal())
        train_end_dates_list.append(target_slice_df.iloc[-1]['date'].toordinal())

    targets_eval_df = features_df[selection_mask_eval].copy()
    for end_idx in range(target_len - 1, len(targets_eval_df)):
        start_idx = end_idx - target_len + 1
        target_slice_df = targets_eval_df.iloc[start_idx : end_idx + 1]
        eval_x_list.append(torch.tensor(target_slice_df[FEATURE_COLS].values, dtype=torch.float32))
        eval_y_list.append(torch.tensor(target_slice_df[VALUE_COL].values, dtype=torch.float32))
        eval_s_list.append(s_tensor)
        eval_start_dates_list.append(target_slice_df.iloc[0]['date'].toordinal())
        eval_end_dates_list.append(target_slice_df.iloc[-1]['date'].toordinal())

    features_df_ctx = features_df[selection_mask_train_or_eval].copy()
    try:
        changepoints_df = pd.read_csv(f"dataset/CPD/{LBW_LEN}/{ticker}.csv", parse_dates=["date"])
        features_df_ctx = features_df_ctx.merge(changepoints_df.ffill().bfill(), on="date")
        if not features_df_ctx.empty and "t" in features_df_ctx.columns:
            features_df_ctx = features_df_ctx.set_index("t")
            regimes = _cpd_segmentation(features_df_ctx)
            for start_idx_seg, end_idx_seg in regimes:
                context_df = features_df_ctx.loc[start_idx_seg:end_idx_seg]
                if context_df.empty:
                    warnings.append(f"Warning: Ticker {ticker} produced an empty context for regime {start_idx_seg} {end_idx_seg}. Skipping.")
                    continue
                x_ctx_tensor = torch.tensor(context_df[FEATURE_COLS].values, dtype=torch.float32)
                xi_ctx_tensor = torch.tensor(context_df[FEATURE_COLS + [VALUE_COL]].values, dtype=torch.float32)
                context_len = x_ctx_tensor.shape[0]
                context_padding = MAX_CONTEXT_LEN - context_len
                x_ctx_padded = F.pad(x_ctx_tensor, (0, 0, 0, context_padding), 'constant', 0.0)
                xi_ctx_padded = F.pad(xi_ctx_tensor, (0, 0, 0, context_padding), 'constant', 0.0)
                context_s_list.append(s_tensor)
                context_x_list.append(x_ctx_padded)
                context_xi_list.append(xi_ctx_padded)
                context_lens_list.append(context_len)
                context_start_dates_list.append(context_df.iloc[0]['date'].toordinal())
                context_end_dates_list.append(context_df.iloc[-1]['date'].toordinal())
    except FileNotFoundError:
        warnings.append(f"Warning: CPD file not found for ticker {ticker}. No contexts will be created.")

    results = {
        "train_x": torch.stack(train_x_list), "train_y": torch.stack(train_y_list),
        "train_s": torch.stack(train_s_list), "eval_x": torch.stack(eval_x_list),
        "train_target_start_dates": torch.tensor(train_start_dates_list, dtype=torch.int32),
        "train_target_end_dates": torch.tensor(train_end_dates_list, dtype=torch.int32),
        "eval_y": torch.stack(eval_y_list), "eval_s": torch.stack(eval_s_list),
        "eval_target_start_dates": torch.tensor(eval_start_dates_list, dtype=torch.int32),
        "eval_target_end_dates": torch.tensor(eval_end_dates_list, dtype=torch.int32),
        "context_s": torch.stack(context_s_list), "context_x": torch.stack(context_x_list),
        "context_xi": torch.stack(context_xi_list),
        "context_lens": torch.tensor(context_lens_list, dtype=torch.long),
        "context_start_dates": torch.tensor(context_start_dates_list, dtype=torch.int32),
        "context_end_dates": torch.tensor(context_end_dates_list, dtype=torch.int32),
    }
    return results, warnings

class XTrendDataset(Dataset):
    def __init__(self,
        train_start: dt.datetime,
        train_end: dt.datetime,
        eval_start: dt.datetime,
        eval_end: dt.datetime,
        target_len: int = TARGET_LEN, context_sample_size: int = CONTEXT_SAMPLE_SIZE, initial_mode: str = 'train', train_stride: int = 1, load_price_data: bool = False
    ):
        self.target_len = target_len
        self.context_sample_size = context_sample_size
        self._output_sampled_indices = False
        self.ticker_to_idx = {ticker: torch.tensor(i, dtype=torch.long) for i, ticker in enumerate(sorted(PINNACLE_ASSETS))}
        self.idx_to_ticker = {i: ticker for i, ticker in enumerate(sorted(PINNACLE_ASSETS))}

        print(f"Info: Training period is from {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"Info: Validation period is from {eval_start.strftime('%Y-%m-%d')} to {eval_end.strftime('%Y-%m-%d')}")
        print(f"Info: Train stride is {train_stride}")

        if load_price_data:
            self.price_data = {}
            print("Info: Loading raw price data")
            for ticker in PINNACLE_ASSETS:
                try:
                    pd.read_csv(
                        f"dataset/CLCDATA/{ticker}_RAD.CSV",
                        header=None,
                        names=["date", "open", "high", "low", "close", "volume", "open_interest"],
                        parse_dates=["date"]
                    )
                except Exception:
                    print(f"Warning: Could not load price data for ticker {ticker}.")

        num_cores = mp.cpu_count()
        print(f"Info: Using {num_cores} cores to load data and create tensors")

        args_list = [
            (ticker, self.ticker_to_idx[ticker], self.target_len, train_start, train_end, eval_start, eval_end)
            for ticker in PINNACLE_ASSETS
        ]
        list_of_results_dicts = []
        all_warnings = []
        with mp.Pool(processes=num_cores) as pool:
            results_iterator = pool.imap_unordered(_process_ticker_data, args_list)
            for res, worker_warnings in tqdm(results_iterator, total=len(args_list), desc="Processing tickers"):
                if res:
                    list_of_results_dicts.append(res)
                if worker_warnings:
                    all_warnings.extend(worker_warnings)

        self.train_x = torch.cat([r["train_x"] for r in list_of_results_dicts], dim=0)
        self.train_y = torch.cat([r["train_y"] for r in list_of_results_dicts], dim=0)
        self.train_s = torch.cat([r["train_s"] for r in list_of_results_dicts], dim=0)
        self.train_target_start_dates = torch.cat([r["train_target_start_dates"] for r in list_of_results_dicts], dim=0)
        self.train_target_end_dates = torch.cat([r["train_target_end_dates"] for r in list_of_results_dicts], dim=0)

        if train_stride > 1:
            self.train_x = self.train_x[::train_stride]
            self.train_y = self.train_y[::train_stride]
            self.train_s = self.train_s[::train_stride]
            self.train_target_start_dates = self.train_target_start_dates[::train_stride]
            self.train_target_end_dates = self.train_target_end_dates[::train_stride]

        self.eval_x = torch.cat([r["eval_x"] for r in list_of_results_dicts], dim=0)
        self.eval_y = torch.cat([r["eval_y"] for r in list_of_results_dicts], dim=0)
        self.eval_s = torch.cat([r["eval_s"] for r in list_of_results_dicts], dim=0)
        self.eval_target_start_dates = torch.cat([r["eval_target_start_dates"] for r in list_of_results_dicts], dim=0)
        self.eval_target_end_dates = torch.cat([r["eval_target_end_dates"] for r in list_of_results_dicts], dim=0)

        context_s = torch.cat([r["context_s"] for r in list_of_results_dicts], dim=0)
        context_x = torch.cat([r["context_x"] for r in list_of_results_dicts], dim=0)
        context_xi = torch.cat([r["context_xi"] for r in list_of_results_dicts], dim=0)
        context_lens = torch.cat([r["context_lens"] for r in list_of_results_dicts], dim=0)
        context_start_dates = torch.cat([r["context_start_dates"] for r in list_of_results_dicts], dim=0)
        context_end_dates = torch.cat([r["context_end_dates"] for r in list_of_results_dicts], dim=0)

        sorted_indices = torch.argsort(context_end_dates)
        self.context_s = context_s[sorted_indices]
        self.context_x = context_x[sorted_indices]
        self.context_xi = context_xi[sorted_indices]
        self.context_lens = context_lens[sorted_indices]
        self.context_start_dates = context_start_dates[sorted_indices]
        self.context_end_dates = context_end_dates[sorted_indices]

        train_end_timestamp = train_end.toordinal()
        self.train_context_upper_bound_idx = int(torch.searchsorted(self.context_end_dates, train_end_timestamp).item())

        if all_warnings:
            for warning in sorted(list(set(all_warnings))):
                print(warning)

        self.num_contexts = len(self.context_x)
        assert self.context_sample_size <= self.num_contexts

        self.mode = initial_mode
        if self.mode not in ['train', 'eval']:
            raise ValueError("initial_mode must be 'train' or 'eval'")

    def set_mode(self, mode: str):
        if mode not in ['train', 'eval']:
            raise ValueError("Mode must be 'train' or 'eval'")
        print(f"Set mode to {mode}.")
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_x)
        else:
            return len(self.eval_x)

    def __getitem__(self, idx):
        if self.mode == 'train':
            target_x = self.train_x[idx]
            target_y = self.train_y[idx]
            target_s = self.train_s[idx]
            sampled_indices = random.SystemRandom().sample(range(self.train_context_upper_bound_idx), self.context_sample_size)
        else: # 'eval' mode
            target_x = self.eval_x[idx]
            target_y = self.eval_y[idx]
            target_s = self.eval_s[idx]
            target_date = self.eval_target_end_dates[idx]
            upper_bound_idx = int(torch.searchsorted(self.context_end_dates, target_date).item())
            sampled_indices = random.SystemRandom().sample(range(upper_bound_idx), self.context_sample_size)

        context_x = self.context_x[sampled_indices]
        context_xi = self.context_xi[sampled_indices]
        context_s = self.context_s[sampled_indices]
        context_lens = self.context_lens[sampled_indices]

        sample = {
            "target_x": target_x, "target_y": target_y, "target_s": target_s,
            "context_x": context_x, "context_xi": context_xi,
            "context_s": context_s, "context_lens": context_lens,
        }
        if not self._output_sampled_indices:
            return sample
        else:
            sample["sampled_context_indices"] = torch.tensor(sampled_indices, dtype=torch.int32)
            return sample

    def get_item_with_metadata(self, idx):
        output_sampled_indices = self._output_sampled_indices
        self._output_sampled_indices = True
        item = self[idx]
        item_metadata = {}

        if self.mode == 'train':
            item_metadata['target_ticker'] = self.idx_to_ticker[int(item['target_s'].item())]
            item_metadata['target_start_date'] = dt.datetime.fromordinal(int(self.train_target_start_dates[idx].item()))
            item_metadata['target_end_date'] = dt.datetime.fromordinal(int(self.train_target_end_dates[idx].item()))
        else: # 'eval' mode
            item_metadata['target_ticker'] = self.idx_to_ticker[int(item['target_s'].item())]
            item_metadata['target_start_date'] = dt.datetime.fromordinal(int(self.eval_target_start_dates[idx].item()))
            item_metadata['target_end_date'] = dt.datetime.fromordinal(int(self.eval_target_end_dates[idx].item()))

        sampled_indices = item["sampled_context_indices"]
        context_s_indices = self.context_s[sampled_indices]

        item_metadata['context_tickers'] = [self.idx_to_ticker[int(s.item())] for s in context_s_indices]
        item_metadata['context_start_dates'] = [dt.datetime.fromordinal(int(ts.item())) for ts in self.context_start_dates[sampled_indices]]
        item_metadata['context_end_dates'] = [dt.datetime.fromordinal(int(ts.item())) for ts in self.context_end_dates[sampled_indices]]

        self._output_sampled_indices = output_sampled_indices

        return {**item, "metadata": item_metadata}

if __name__ == "__main__":
    dataset = XTrendDataset(
        train_start = dt.datetime(2015, 1, 1),
        train_end = dt.datetime(2020, 1, 1),
        eval_start = dt.datetime(2020, 1, 1),
        eval_end = dt.datetime(2021, 1, 1),
        train_stride=1,
        load_price_data=True
    )
    print("\n--- Testing len ---")
    print("Train dataset length:", len(dataset))
    dataset.set_mode('eval')
    print("Validation dataset length:", len(dataset))

    print("\n--- Testing __get_item__ shapes ---")
    dataset.set_mode('train')
    sample = dataset[1]
    for key, value in sample.items():
        print(f"{key} shape:", list(value.shape))

    print("\n--- Testing get_item_with_metadata (Train) ---")
    full_sample_info_train = dataset.get_item_with_metadata(42)
    for key, value in full_sample_info_train["metadata"].items():
        if key in ['context_start_dates', 'context_end_dates', 'context_tickers']:
            print(f"{key}:")
            for i, entry in enumerate(value):
                print(f"  [{i}]: {entry}")
        else:
            print(f"{key}: {value}")

    print("\n--- Testing get_item_with_metadata (Eval) ---")
    dataset.set_mode('eval')
    full_sample_info_eval = dataset.get_item_with_metadata(1)
    for key, value in full_sample_info_eval["metadata"].items():
        if key in ['context_start_dates', 'context_end_dates', 'context_tickers']:
            print(f"{key}:")
            for i, entry in enumerate(value):
                print(f"  [{i}]: {entry}")
        else:
            print(f"{key}: {value}")
