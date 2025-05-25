import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from models.xtrend import XTrendModel
from data_optimized import XTrendDataset
from data_valid import XTrendDatasetValid
from torch.utils.data import DataLoader
from data_loader_optimized import xtrend_collate_fn

TESTING = False
VAL_DIVERSIFIED_SHARPE = True

BATCH_SIZE =  128
# BATCH_SIZE = 252
NUM_CONTEXT = 10
ITERATIONS = 200
N = 50
LR = 1e-3

VALID_START_YEAR = 2005  # valid 2015-2018

TEST_START_YEAR = 2010  # valid 2015-2018


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = XTrendDataset()
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = xtrend_collate_fn, pin_memory = True, shuffle=True)
dataset_valid = XTrendDatasetValid()
dataloader_valid  = DataLoader(dataset_valid, batch_size = BATCH_SIZE, collate_fn = xtrend_collate_fn, pin_memory = True, shuffle=True)

hidden_dim = 32
model = XTrendModel(
    x_dim=8,
    y_dim=1,
    static_dim=8,
    encoder_hidden_dim=hidden_dim,
    vsn_dim=hidden_dim,
    ffn_dim=hidden_dim,
    lstm_hidden_dim=hidden_dim,
    n_heads=4,
    sharpe_dim=1,
    mle_dim=1,
    self_attention_type="ptmultihead",
    cross_attention_type="ptmultihead",
    dropout=0.3
).to(device)

# Set up the optimizer and train step
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_sharpe = 0
epoch_sharpes = []
for it in range(ITERATIONS):
    print(f"Iteration {it}")

    train_sharpes = [] 
    train_mle_losses = []
    pbar = tqdm(dataloader, desc=f"Iteration {it} | Train Sharpe: N/A", leave=True)
    for batch in pbar:
        target_x = batch["target_x"].to(device)
        target_y = batch["target_y"].to(device)
        target_s = batch["target_s"].to(device)
        context_x_list = batch["context_x_padded"].to(device)
        context_xi_list = batch["context_xi_padded"].to(device)
        context_s_list = batch["context_s_padded"].to(device)
        total_loss, sharpe_loss, mle_loss = model.training_step(
            (target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list), optimizer, alpha=1.0
        )
        
        current_sharpe = -sharpe_loss
        train_sharpes.append(current_sharpe)
        train_mle_losses.append(mle_loss)
        avg_sharpe = np.mean(train_sharpes)
        pbar.set_description(f"Iteration {it} | Train Sharpe: {avg_sharpe:.4f} | Current: {current_sharpe:.4f}")
        if TESTING:
            break

    print("Train Sharpe: ", np.mean(train_sharpes))
    print("Train MLE Loss: ", np.mean(train_mle_losses))

    valid_sharpes = []
    valid_mle_losses = []
    all_returns = []
    
    val_pbar = tqdm(dataloader_valid, desc=f"Validation | Valid Sharpe: N/A", leave=True)
    for batch in val_pbar:
        target_x = batch["target_x"].to(device)
        target_y = batch["target_y"].to(device)
        target_s = batch["target_s"].to(device)
        context_x_list = batch["context_x_padded"].to(device)
        context_xi_list = batch["context_xi_padded"].to(device)
        context_s_list = batch["context_s_padded"].to(device)

        sharpe_loss = model.evaluate(
            (target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list), optimizer, alpha=1.0
        )
        captured_returns = sharpe_loss
        if captured_returns.shape[0] == BATCH_SIZE:
            valid_sharpes.append(
                (
                    torch.mean(captured_returns)
                    / (torch.std(captured_returns) + 1e-9)
                    * np.sqrt(252.0)
                )
                .detach()
                .item()
            )
        captured_returns = sharpe_loss[:, -1, 0].tolist()
        all_returns += captured_returns

        if VAL_DIVERSIFIED_SHARPE and all_returns:
            diversified = np.array(all_returns)
            val_sharpe = diversified.mean() * np.sqrt(252) / (diversified.std() + 1e-9)
            print("Div Valid Sharpe: ", val_sharpe)
            print("Valid Single Sharpe: ", np.mean(valid_sharpes))
