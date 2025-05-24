import pandas as pd
import numpy as np
import datetime as dt
from tqdm.auto import tqdm
import h5py
import torch
import pickle
import os
from pathlib import Path
import json
import gc
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
MIN_CONTEXT_LEN = 3
MAX_CONTEXT_LEN = 63
BATCH_SIZE = 32
NUMPY_DTYPE = np.float32
CONTEXT_LBW = 63
CPD_THRESHOLD = 0.995
CPD_BURN_IN = 5
FEATURES = [
    "norm_daily_return", "norm_monthly_return", "norm_quarterly_return",
    "norm_biannual_return", "norm_annual_return", "macd_8_24", "macd_16_48", "macd_32_96"
]
TEST_YEAR_START = 2015
TEST_END = 2025
PINNACLE_ASSETS = [
    "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", 
    "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", 
    "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZH", 
    "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ"
]

class OptimizedDataProcessor:
    """Optimized data processor with efficient storage and retrieval"""
    
    def __init__(self, data_dir: str = "dataset", cache_dir: str = "cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        (self.cache_dir / "processed").mkdir(exist_ok=True)
        (self.cache_dir / "tensors").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
    def split_dataframe(self, df: pd.DataFrame, max_series: int = MAX_CONTEXT_LEN, 
                       min_series: int = MIN_CONTEXT_LEN) -> List[pd.DataFrame]:
        """Optimized dataframe splitting"""
        chunks = []
        df_len = len(df)
        
        if df_len % max_series < min_series:
            num_chunks = df_len // max_series
        else:
            num_chunks = df_len // max_series + 1
            
        for i in range(num_chunks):
            start_idx = i * max_series
            end_idx = (i + 1) * max_series if i < num_chunks - 1 else df_len
            chunks.append(df.iloc[start_idx:end_idx].copy())
            
        return chunks
    
    def process_single_ticker(self, ticker: str) -> List[pd.DataFrame]:
        """Process a single ticker efficiently"""
        # Load features
        features_path = self.data_dir / "FEATURES" / f"{ticker}.csv"
        features_df = pd.read_csv(features_path, parse_dates=["date"])
        
        # Vectorized operations
        features_df["next_day_norm_return"] = features_df["norm_daily_return"].shift(-1)
        features_df = features_df[["date"] + FEATURES + ["next_day_norm_return"]]
        features_df.dropna(inplace=True)
        features_df = features_df[features_df.date < dt.datetime(TEST_YEAR_START, 1, 1)]
        
        # Load changepoints
        changepoints_path = self.data_dir / "CPD" / str(CONTEXT_LBW) / f"{ticker}.csv"
        changepoints_df = pd.read_csv(changepoints_path, parse_dates=["date"])
        changepoints_df.ffill(inplace=True)
        changepoints_df.dropna(inplace=True)
        
        # Merge efficiently
        features_df = features_df.merge(changepoints_df, on="date")
        features_df.set_index("date", inplace=True)
        
        # Process changepoints vectorized
        boundaries_mask = features_df["cp_score"] >= CPD_THRESHOLD
        boundaries_idx = features_df[boundaries_mask].index
        
        if len(boundaries_idx) == 0:
            features_df["context_num"] = 0
        else:
            features_df["context_num"] = self._assign_context_numbers(
                features_df, boundaries_idx
            )
        
        # Group and split
        subtasks = []
        for context_num, sub_df in features_df.groupby("context_num"):
            splits = self.split_dataframe(sub_df, MAX_CONTEXT_LEN)
            for i, split in enumerate(splits):
                split = split.assign(subtask=i, ticker=ticker)
                subtasks.append(split)
                
        return subtasks
    
    def _assign_context_numbers(self, df: pd.DataFrame, boundaries_idx: pd.Index) -> pd.Series:
        """Efficiently assign context numbers"""
        context_nums = pd.Series(np.nan, index=df.index)
        
        if len(boundaries_idx) > 0:
            last_cp_location = df.iloc[0]["cp_location"]
            context_nums.loc[boundaries_idx[0]] = 0
            context_num = 1
            
            for idx in boundaries_idx[1:]:
                current_cp_location = df.loc[idx, "cp_location"]
                if current_cp_location - last_cp_location >= CPD_BURN_IN:
                    context_nums.loc[idx] = context_num
                    last_cp_location = current_cp_location
                    context_num += 1
            
            context_nums.bfill(inplace=True)
            context_nums.fillna(context_num, inplace=True)
            
        return context_nums.astype(int)
    
    def process_all_tickers_to_hdf5(self, output_file: str = "pinnacle_data.h5") -> None:
        """Process all tickers and save to optimized HDF5 format"""
        output_path = self.cache_dir / output_file
        
        # Process data in chunks to manage memory
        all_subtasks = []
        
        with tqdm(PINNACLE_ASSETS, desc="Processing tickers") as pbar:
            for ticker in pbar:
                pbar.set_description(f"Processing {ticker}")
                subtasks = self.process_single_ticker(ticker)
                all_subtasks.extend(subtasks)
                
                # Periodic memory cleanup
                if len(all_subtasks) % 10 == 0:
                    gc.collect()
        
        # Concatenate all data
        print("Concatenating all tasks...")
        all_tasks = pd.concat(all_subtasks, ignore_index=True)
        
        # Create unique task identifiers
        unique_tasks = (
            all_tasks[["context_num", "subtask", "ticker"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        unique_tasks["set"] = unique_tasks.index
        
        # Merge back
        all_tasks["date"] = all_tasks.index
        all_tasks = all_tasks.merge(unique_tasks, on=["context_num", "subtask", "ticker"])
        
        # Filter by minimum context length
        train = all_tasks.groupby("set").filter(lambda x: len(x) >= MIN_CONTEXT_LEN)
        
        # Save to HDF5 with compression
        print(f"Saving to {output_path}...")
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            metadata_group.attrs['features'] = [s.encode() for s in FEATURES]
            metadata_group.attrs['min_context_len'] = MIN_CONTEXT_LEN
            metadata_group.attrs['max_context_len'] = MAX_CONTEXT_LEN
            metadata_group.attrs['test_year_start'] = TEST_YEAR_START
            
            # Save main dataframe (without large arrays)
            df_cols = ['context_num', 'subtask', 'ticker', 'set', 'date']
            train_meta = train[df_cols].copy()
            
            # Convert dates to strings for HDF5 compatibility
            train_meta['date'] = train_meta['date'].astype(str)
            
            # Save using pandas HDF5 interface for complex data
            train_meta.to_hdf(output_path, key='train_meta', mode='a', complevel=9)
            
        # Process and save tensor data separately
        self._save_tensor_data(train, output_path)
        
        print(f"Data processing complete. Saved to {output_path}")
        return len(train['set'].unique())
    
    def _save_tensor_data(self, train_df: pd.DataFrame, output_path: Path) -> None:
        """Save tensor data efficiently"""
        tensor_cache_dir = self.cache_dir / "tensors"
        
        print("Processing tensor data...")
        segments_and_ticker = train_df[["set", "ticker"]].drop_duplicates()
        
        # Process in batches to manage memory
        batch_size = 50  # Process 50 sets at a time
        set_groups = [segments_and_ticker[i:i+batch_size] 
                     for i in range(0, len(segments_and_ticker), batch_size)]
        
        all_segment_data = []
        
        for batch_idx, batch_df in enumerate(tqdm(set_groups, desc="Processing tensor batches")):
            batch_segments = []
            
            for _, row in batch_df.iterrows():
                set_num = row['set']
                ticker = row['ticker']
                segments = self._context_all_segments(train_df, set_num)
                
                # Add ticker info to each segment
                for seg in segments:
                    batch_segments.append({
                        'x': seg[0], 'y': seg[1], 'set': seg[2], 
                        'seq_len': seg[3], 'date': seg[4], 'ticker': ticker
                    })
            
            # Save batch to separate files to manage memory
            batch_file = tensor_cache_dir / f"batch_{batch_idx}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_segments, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            all_segment_data.extend(batch_segments)
            
            # Memory cleanup
            del batch_segments
            gc.collect()
        
        # Save final combined data
        final_df = pd.DataFrame(all_segment_data)
        final_df.to_pickle(self.cache_dir / "pinnacle_contexts_optimized.pkl")
        
        # Save metadata about batches
        metadata = {
            'num_batches': len(set_groups),
            'total_segments': len(all_segment_data),
            'batch_files': [f"batch_{i}.pkl" for i in range(len(set_groups))]
        }
        
        with open(self.cache_dir / "metadata" / "tensor_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def _context_all_segments(self, train_df: pd.DataFrame, set_num: int) -> List[Tuple]:
        """Create all context segments for a given set"""
        segments = []
        data_whole = train_df[train_df["set"] == set_num].copy()
        
        for length in range(MIN_CONTEXT_LEN, len(data_whole) + 1):
            data = data_whole.iloc[:length].copy()
            date = data["date"].iloc[-1]
            
            # Create feature array efficiently
            x_features = np.zeros((len(data), len(FEATURES)), dtype=NUMPY_DTYPE)
            
            # Vectorized assignment
            for i, feature in enumerate(FEATURES):
                x_features[:, i] = data[feature].values.astype(NUMPY_DTYPE)
            
            context_y = data["next_day_norm_return"].values.reshape(-1, 1).astype(NUMPY_DTYPE)
            
            segments.append((x_features, context_y, set_num, len(data), date))
            
        return segments
    
if __name__ == "__main__":
    processor = OptimizedDataProcessor()
    processor.process_all_tickers_to_hdf5("pinnacle_data.h5")