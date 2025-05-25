import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

MAX_CONTEXT_LEN_PAPER = 63  # Max segment length from paper (e.g., 3-month)
CONTEXT_LBW_PAPER = 21      # CPD lookback window (from paper's Appendix A) [cite: 381]
FIXED_MAX_CONTEXT_LEN = MAX_CONTEXT_LEN_PAPER + CONTEXT_LBW_PAPER # 63 + 21 = 84

def xtrend_collate_fn(batch):
    target_x_batch = torch.stack([item["target_x"] for item in batch], dim=0)
    target_y_batch = torch.stack([item["target_y"] for item in batch], dim=0)
    target_s_batch = torch.stack([item["target_s"] for item in batch], dim=0)

    context_x_collated = []
    context_xi_collated = []
    context_s_collated = []

    context_lens_batch = []

    for item in batch:
        # process contexts
        context_lens = []
        context_x_padded_list = []
        context_xi_padded_list = []
        for context_x_tensor, context_xi_tensor in zip(item["context_x_list"], item["context_xi_list"]):
            context_len = context_x_tensor.shape[0]
            context_lens.append(context_len)
            context_padding = FIXED_MAX_CONTEXT_LEN - context_len

            context_x_padded_tensor = F.pad(context_x_tensor, (0, 0, 0, context_padding), mode='constant', value=0.0)
            context_x_padded_list.append(context_x_padded_tensor)

            context_xi_padded_tensor = F.pad(context_xi_tensor, (0, 0, 0, context_padding), mode='constant', value=0.0)
            context_xi_padded_list.append(context_xi_padded_tensor)

        context_x_collated.append(torch.stack(context_x_padded_list, dim=0))
        context_xi_collated.append(torch.stack(context_xi_padded_list, dim=0))

        context_lens_batch.append(context_lens)
        # process context embeddings
        context_s_collated.append(torch.stack(item["context_s_list"], dim=0))

    # stack along batch dimension
    context_x_collated_batch = torch.stack(context_x_collated, dim=0)
    context_xi_collated_batch = torch.stack(context_xi_collated, dim=0)
    context_s_collated_batch = torch.stack(context_s_collated, dim=0)
    actual_context_lens_collated = torch.tensor(context_lens_batch, dtype=torch.long)

    return {
        "target_x": target_x_batch, # (batch_size, target_len, x_dim_target)
        "target_y": target_y_batch, # (batch_size, target_len) - actual returns for loss
        "target_s": target_s_batch, # (batch_size,) - target ticker indices
        "context_x_padded": context_x_collated_batch, # (batch_size, num_ctx, FIXED_MAX_CONTEXT_LEN, x_dim_ctx_keys)
        "context_xi_padded": context_xi_collated_batch, # (batch_size, num_ctx, FIXED_MAX_CONTEXT_LEN, xi_dim_ctx_values)
        "context_s_padded": context_s_collated_batch, # (batch_size, num_ctx) - context ticker indices
        "actual_context_lens": actual_context_lens_collated # (batch_size, num_ctx)
    }

if __name__ == "__main__":
    dataset = XTrendDataset()
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=xtrend_collate_fn, shuffle=True)

    sample = next(iter(dataloader))
    for key, value in sample.items():
        print(f"{key} shape: {list(value.shape)}")
