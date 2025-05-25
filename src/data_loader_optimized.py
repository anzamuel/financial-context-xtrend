import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

MAX_CONTEXT_LEN_PAPER = 63  # Max segment length from paper (e.g., 3-month)
CONTEXT_LBW_PAPER = 21      # CPD lookback window (from paper's Appendix A) [cite: 381]
FIXED_MAX_CONTEXT_LEN = MAX_CONTEXT_LEN_PAPER + CONTEXT_LBW_PAPER # 63 + 21 = 84


def xtrend_collate_fn(batch):
    """
    Ultra-optimized version using vectorized operations where possible
    """
    batch_size = len(batch)
    
    # Stack simple tensors
    target_x_batch = torch.stack([item["target_x"] for item in batch], dim=0)
    target_y_batch = torch.stack([item["target_y"] for item in batch], dim=0)
    target_s_batch = torch.stack([item["target_s"] for item in batch], dim=0)
    
    # Collect all context data for vectorized processing
    all_context_x = []
    all_context_xi = []
    all_context_s = []
    all_context_lens = []
    
    for item in batch:
        ctx_x_list = item["context_x_list"]
        ctx_xi_list = item["context_xi_list"]
        ctx_s_list = item["context_s_list"]
        
        # Collect lengths for this batch item
        lens = [ctx.shape[0] for ctx in ctx_x_list]
        all_context_lens.append(lens)
        
        all_context_x.extend(ctx_x_list)
        all_context_xi.extend(ctx_xi_list)
        all_context_s.extend(ctx_s_list)
    
    # Get dimensions
    num_contexts = len(batch[0]["context_x_list"])
    x_dim = all_context_x[0].shape[1]
    xi_dim = all_context_xi[0].shape[1]
    
    # Vectorized padding using pad_sequence (more efficient for variable lengths)
    
    # Pad all contexts at once
    context_x_padded_flat = pad_sequence(
        all_context_x, 
        batch_first=True, 
        padding_value=0.0
    )
    context_xi_padded_flat = pad_sequence(
        all_context_xi, 
        batch_first=True, 
        padding_value=0.0
    )
    
    # Ensure fixed max length
    if context_x_padded_flat.shape[1] < FIXED_MAX_CONTEXT_LEN:
        pad_amount = FIXED_MAX_CONTEXT_LEN - context_x_padded_flat.shape[1]
        context_x_padded_flat = F.pad(context_x_padded_flat, (0, 0, 0, pad_amount))
        context_xi_padded_flat = F.pad(context_xi_padded_flat, (0, 0, 0, pad_amount))
    elif context_x_padded_flat.shape[1] > FIXED_MAX_CONTEXT_LEN:
        context_x_padded_flat = context_x_padded_flat[:, :FIXED_MAX_CONTEXT_LEN]
        context_xi_padded_flat = context_xi_padded_flat[:, :FIXED_MAX_CONTEXT_LEN]
    
    # Reshape to batch format
    total_contexts = batch_size * num_contexts
    context_x_collated = context_x_padded_flat.view(batch_size, num_contexts, FIXED_MAX_CONTEXT_LEN, x_dim)
    context_xi_collated = context_xi_padded_flat.view(batch_size, num_contexts, FIXED_MAX_CONTEXT_LEN, xi_dim)
    
    # Handle context embeddings and lengths
    context_s_collated = torch.stack([
        torch.stack(item["context_s_list"], dim=0) for item in batch
    ], dim=0)
    
    context_lens_batch = torch.tensor(all_context_lens, dtype=torch.long)
    
    return {
        "target_x": target_x_batch,
        "target_y": target_y_batch,
        "target_s": target_s_batch, 
        "context_x_padded": context_x_collated,
        "context_xi_padded": context_xi_collated,
        "context_s_padded": context_s_collated,
        "actual_context_lens": context_lens_batch
    }
