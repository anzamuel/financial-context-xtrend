import torch
from torch.utils.data import Dataset
import random
from cpd_segmentation import create_targets_and_contexts

class XTrendDataset(Dataset):
    def __init__(self, target_len: int = 126, sample_size_contexts: int = 20, target_warm_up_steps: int = 63):

        self.target_len = target_len
        self.sample_size_contexts = sample_size_contexts
        self.warm_up_steps = target_warm_up_steps

        assert self.target_len > self.warm_up_steps

        self.targets, self.target_definitions, self.contexts, self.context_definitions, self.assets, self.feature_cols, self.value_col = create_targets_and_contexts(target_len)

        self.num_contexts = len(self.context_definitions)
        assert self.sample_size_contexts <= self.num_contexts

        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(sorted(self.assets))}

    def __len__(self):
        if hasattr(self, 'targets'):
            return len(self.target_definitions)
        return 0

    def __getitem__(self, idx):
        target_info = self.target_definitions[idx]
        target_ticker = target_info["ticker"]
        target_end_idx = target_info["end_idx"]

        targets_df = self.targets[target_ticker]

        start_idx = target_end_idx - self.target_len + 1
        target_slice_df = targets_df.iloc[start_idx : target_end_idx + 1]

        target_features_np = target_slice_df[self.feature_cols].values
        target_fut_returns_np = target_slice_df[self.value_col].values

        target_x = torch.tensor(target_features_np, dtype=torch.float32)
        target_y = torch.tensor(target_fut_returns_np, dtype=torch.float32)
        target_s = torch.tensor(self.ticker_to_idx[target_ticker], dtype=torch.long)

        sampled_indices = random.sample(range(self.num_contexts), self.sample_size_contexts) # sample without replacement

        context_x_list = []
        context_xi_list = []
        context_s_list = []

        for ctx_idx in sampled_indices:
            context_ticker = self.context_definitions[ctx_idx]['ticker']
            context_df = self.contexts[ctx_idx]

            context_features_np = context_df[self.feature_cols].values
            context_x_list.append(torch.tensor(context_features_np, dtype=torch.float32))

            context_xi_np = context_df[self.feature_cols + [self.value_col]].values
            context_xi_list.append(torch.tensor(context_xi_np, dtype=torch.float32))

            context_s_list.append(torch.tensor(self.ticker_to_idx[context_ticker], dtype=torch.long)) # placeholder ticker embedding

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
