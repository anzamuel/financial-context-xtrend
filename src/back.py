

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.xtrend import XTrendModel
from data_valid import XTrendDatasetValid
from data_loader_optimized import xtrend_collate_fn
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters (must match training)
hidden_dim = 64

# Load model
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

model.load_state_dict(torch.load("src\\runs\\runs\\model_state_final.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Validation dataset
dataset_valid = XTrendDatasetValid()
dataloader_valid  = DataLoader(dataset_valid, batch_size=128, collate_fn=xtrend_collate_fn, shuffle=False)

all_returns = []

with torch.no_grad():
    for batch in tqdm(dataloader_valid, desc="Backtesting"):
        target_x = batch["target_x"].to(device)
        target_y = batch["target_y"].to(device)
        target_s = batch["target_s"].to(device)
        context_x_list = batch["context_x_padded"].to(device)
        context_xi_list = batch["context_xi_padded"].to(device)
        context_s_list = batch["context_s_padded"].to(device)

        predicted_returns = model.evaluate(
            (target_x, target_y, target_s, context_x_list, context_xi_list, context_s_list)
        )

        # Use last timestep return prediction
        predicted_returns = predicted_returns[:, -1, 0].tolist()
        tickers = batch["target_s"].tolist()

        all_returns += list(zip(tickers, predicted_returns))
    
    results_df = pd.DataFrame(all_returns, columns=["ticker", "predicted_return"])
    results_df.to_csv("predicted_returns.csv", index=False)
