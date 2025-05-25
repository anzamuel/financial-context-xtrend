import torch
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm.auto import tqdm
from cpd_segmentation import create_targets_and_contexts

class XTrendDataset(Dataset):
    def __init__(self, target_len: int = 126, sample_size_contexts: int = 10, target_warm_up_steps: int = 63, train: bool = True):
        self.target_len = target_len
        self.sample_size_contexts = sample_size_contexts
        self.warm_up_steps = target_warm_up_steps
        self.train = train

        assert self.target_len > self.warm_up_steps

        # load and prepare data
        self.targets_train, self.target_definitions_train, self.contexts, self.context_definitions, self.targets_valid, self.target_definitions_valid, self.assets, self.feature_cols, self.value_col = create_targets_and_contexts(target_len)
        self.num_contexts = len(self.context_definitions)
        assert self.sample_size_contexts <= self.num_contexts

        self.ticker_to_idx = {ticker: torch.tensor(i, dtype=torch.long) for i, ticker in enumerate(sorted(self.assets))}

        # preprocess target tensors
        self.target_tensor_list = []
        for target_info in tqdm(self.target_definitions_train if self.train else self.target_definitions_valid, desc="Loading targets"):
            ticker = target_info["ticker"]
            end_idx = target_info["end_idx"]
            start_idx = end_idx - self.target_len + 1

            # targets_df_map holds the original full DataFrames for each ticker
            target_slice_df = self.targets_train[ticker].iloc[start_idx : end_idx + 1] if self.train else self.targets_valid[ticker].iloc[start_idx : end_idx + 1]

            x_tensor = torch.tensor(target_slice_df[self.feature_cols].values, dtype=torch.float32)
            y_tensor = torch.tensor(target_slice_df[self.value_col].values, dtype=torch.float32)
            s_tensor = self.ticker_to_idx[ticker]
            self.target_tensor_list.append((x_tensor, y_tensor, s_tensor))

        # preprocess context tensors
        self.context_tensors = []
        for i, ctx in tqdm(enumerate(self.contexts), desc="Loading contexts", total=len(self.context_definitions)):
            ticker = self.context_definitions[i]['ticker']
            ticker_idx = self.ticker_to_idx[ticker]
            x_tensor = torch.tensor(ctx[self.feature_cols].values, dtype=torch.float32)
            xi_tensor = torch.tensor(ctx[self.feature_cols + [self.value_col]].values, dtype=torch.float32)
            self.context_tensors.append((ticker_idx, x_tensor, xi_tensor))

    def __len__(self):
        return len(self.target_definitions_train if self.train else self.target_definitions_valid)

    def __getitem__(self, idx):
        target_x, target_y, target_s = self.target_tensor_list[idx]

        # sample contexts without replacement
        sampled_indices = random.sample(range(self.num_contexts), self.sample_size_contexts)

        context_x_list = []
        context_xi_list = []
        context_s_list = []

        for i in sampled_indices:
            ticker_idx, x_tensor, xi_tensor = self.context_tensors[i]
            context_x_list.append(x_tensor)
            context_xi_list.append(xi_tensor)
            context_s_list.append(ticker_idx)

        return {
            "target_x": target_x,
            "target_y": target_y,
            "target_s": target_s,
            "context_x_list": context_x_list,
            "context_xi_list": context_xi_list,
            "context_s_list": context_s_list
        }

if __name__ == "__main__":
    dataset = XTrendDataset()
    print(len(dataset))
    sample = dataset[42]
    print(sample)
    print("target_x shape:", list(sample["target_x"].shape))
    print("target_y shape:", list(sample["target_y"].shape))
    print("target_s shape:", list(sample["target_s"].shape))
    print("content_x shapes:", [list(tensor.shape) for tensor in sample["context_x_list"]])
    print("context_xi shapes:", [list(tensor.shape) for tensor in sample["context_xi_list"]])
    print("context_s shapes:", [list(tensor.shape) for tensor in sample["context_s_list"]])
