import torch
from torch.utils.data import Dataset
import random
import numpy as np
from cpd_segmentation import create_targets_and_contexts

class XTrendDataset(Dataset):
    def __init__(self, target_len: int = 126, sample_size_contexts: int = 10, target_warm_up_steps: int = 63):
        self.target_len = target_len
        self.sample_size_contexts = sample_size_contexts
        self.warm_up_steps = target_warm_up_steps

        assert self.target_len > self.warm_up_steps

        # Load and prepare data
        self.targets, self.target_definitions, self.contexts, self.context_definitions, self.assets, self.feature_cols, self.value_col = create_targets_and_contexts(target_len)
        self.num_contexts = len(self.context_definitions)
        assert self.sample_size_contexts <= self.num_contexts

        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(sorted(self.assets))}

        # ✅ Preprocess target tensors
        self.target_tensors = {}  # Dict[ticker] = (x: tensor, y: tensor)
        for ticker, df in self.targets.items():
            x_tensor = torch.tensor(df[self.feature_cols].values, dtype=torch.float32)
            y_tensor = torch.tensor(df[self.value_col].values, dtype=torch.float32)
            self.target_tensors[ticker] = (x_tensor, y_tensor)

        # ✅ Preprocess context tensors: List[(ticker_idx, x_tensor, xi_tensor)]
        self.context_tensors = []
        for i, ctx in enumerate(self.contexts):
            ticker = self.context_definitions[i]['ticker']
            ticker_idx = self.ticker_to_idx[ticker]
            x_tensor = torch.tensor(ctx[self.feature_cols].values, dtype=torch.float32)
            xi_tensor = torch.tensor(ctx[self.feature_cols + [self.value_col]].values, dtype=torch.float32)
            self.context_tensors.append((ticker_idx, x_tensor, xi_tensor))

    def __len__(self):
        return len(self.target_definitions)

    def __getitem__(self, idx):
        target_info = self.target_definitions[idx]
        ticker = target_info["ticker"]
        end_idx = target_info["end_idx"]
        start_idx = end_idx - self.target_len + 1

        target_x_all, target_y_all = self.target_tensors[ticker]
        target_x = target_x_all[start_idx : end_idx + 1]
        target_y = target_y_all[start_idx : end_idx + 1]
        target_s = torch.tensor(self.ticker_to_idx[ticker], dtype=torch.long)

        # Sample contexts without replacement
        sampled_indices = random.sample(range(self.num_contexts), self.sample_size_contexts)

        context_x_list = []
        context_xi_list = []
        context_s_list = []

        for i in sampled_indices:
            ticker_idx, x_tensor, xi_tensor = self.context_tensors[i]
            context_x_list.append(x_tensor)
            context_xi_list.append(xi_tensor)
            context_s_list.append(torch.tensor(ticker_idx, dtype=torch.long))

        return {
            "target_x": target_x,
            "target_y": target_y,
            "target_s": target_s,
            "context_x_list": context_x_list,
            "context_xi_list": context_xi_list,
            "context_s_list": context_s_list
        }
