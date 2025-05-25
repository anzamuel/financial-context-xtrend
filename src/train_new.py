import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.xtrend import XTrendModel
from data import XTrendDataset
from torch.utils.data import DataLoader
from data_loader import xtrend_collate_fn

TESTING = False
VAL_DIVERSIFIED_SHARPE = True

BATCH_SIZE = 512
NUM_CONTEXT = 10
ITERATIONS = 200
N = 50
LR = 1e-3

VALID_START_YEAR = 2015  # valid 2015-2020

TEST_START_YEAR = 2020  # valid 2020-2025

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = XTrendDataset()
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = xtrend_collate_fn, shuffle=True)
dataset_valid = XTrendDataset(train = False)
dataloader_valid  = DataLoader(dataset_valid, batch_size = BATCH_SIZE, collate_fn = xtrend_collate_fn, shuffle=True)

PINNACLE_ASSETS = [ "AN", "BN", "CA", "CC", "CN", "DA", "DT", "DX", "EN", "ER", "ES", "FB", "FN", "GI", "JN", "JO", "KC", "KW", "LB", "LX", "MD", "MP", "NK", "NR", "SB", "SC", "SN", "SP", "TY", "UB", "US", "XU", "XX", "YM", "ZA", "ZC", "ZF", "ZG", "ZH", "ZI", "ZK", "ZL", "ZN", "ZO", "ZP", "ZR", "ZT", "ZU", "ZW", "ZZ" ]

ticker2id = {ticker: idx for idx, ticker in enumerate(PINNACLE_ASSETS)}

hidden_dim = 64
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

    val_pbar = tqdm(dataloader_valid, desc=f"Iteration {it} | Valid Sharpe: N/A", leave=True)
    for batch in val_pbar:
        target_x = batch["target_x"].to(device)
        target_y = batch["target_y"].to(device)
        target_s = batch["target_s"].to(device)
        context_x_list = batch["context_x_padded"].to(device)
        context_xi_list = batch["context_xi_padded"].to(device)
        context_s_list = batch["context_s_padded"].to(device)

        sharpe_loss = model.evaluate(
            (target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list)
        )
        captured_returns = sharpe_loss
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
        all_returns += zip(target_s, captured_returns)
        pbar.set_description(f"Iteration {it} | Valid Sharpe: {valid_sharpes[-1]:.4f}")

    if VAL_DIVERSIFIED_SHARPE:
        diversified = (
            pd.DataFrame(all_returns, columns=["ticker", "captured_returns"])
            .groupby("ticker")["captured_returns"]
            .sum()
        )
        val_sharpe = diversified.mean() * np.sqrt(252) / diversified.std()
        # print(valid_sharpes)
        print("Div Valid Sharpe: ", val_sharpe)
        print("Valid Single Sharpe: ", np.mean(valid_sharpes))
