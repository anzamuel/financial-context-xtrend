import torch
import numpy as np
import os
from tqdm import tqdm
from models.xtrend import XTrendModel
from dataset import XTrendDataset
from torch.utils.data import DataLoader
from dataloader import xtrend_collate_fn

TESTING = False
VAL_DIVERSIFIED_SHARPE = True
BATCH_SIZE = 512
ITERATIONS = 200
N = 50
LR = 1e-3

results_dir = 'runs'
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = XTrendDataset()
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = xtrend_collate_fn, shuffle=True)

hidden_dim = 64
# TODO: IMPLEMENT WARMUP STEPS (!!)
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

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_sharpe = 0
all_train_sharpes = []
all_train_mle_losses = []

try:
    for it in range(ITERATIONS):
        # TRAININING
        model.train()
        train_sharpes = []
        train_mle_losses = []
        pbar = tqdm(dataloader, desc=f"Iteration {it} | Train Sharpe: N/A", leave=True)
        for batch in pbar:
            target_x = batch["target_x"].to(device)
            target_y = batch["target_y"].to(device)
            target_s = batch["target_s"].to(device)
            context_x = batch["context_x"].to(device)
            context_xi = batch["context_xi"].to(device)
            context_s = batch["context_s"].to(device) ## CONTEXT LENS??
            total_loss, sharpe_loss, mle_loss = model.training_step(
                (target_x, target_y, target_s, context_x, context_xi, context_s), optimizer, alpha=0 #1.0
            )
            current_sharpe_value = -sharpe_loss
            mle_loss_value = mle_loss
            train_sharpes.append(current_sharpe_value)
            train_mle_losses.append(mle_loss_value)
            avg_sharpe = np.mean(train_sharpes)
            pbar.set_description(f"Iteration {it} | Train Sharpe: {avg_sharpe:.4f} | Current: {current_sharpe_value:.4f}")
            if TESTING:
                break # Break the batch loop for testing

        avg_train_sharpe = np.mean(train_sharpes)
        avg_train_mle_loss = np.mean(train_mle_losses)
        print("Train Sharpe: ", avg_train_sharpe)
        print("Train MLE Loss: ", avg_train_mle_loss)
        all_train_sharpes.append(avg_train_sharpe)
        all_train_mle_losses.append(avg_train_mle_loss)


        print("\nTraining complete.")
        # Save final metrics and model state
        final_metrics_filename = os.path.join(results_dir, 'metrics_final.npz')
        final_model_filename = os.path.join(results_dir, 'model_state_final.pth')

        np.savez(final_metrics_filename, train_sharpes=all_train_sharpes, train_mle_losses=all_train_mle_losses)
        torch.save(model.state_dict(), final_model_filename)
        print(f"Saved final metrics to {final_metrics_filename}")
        print(f"Saved final model state to {final_model_filename}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    # Save metrics and model state at interruption point
    interrupted_metrics_filename = os.path.join(results_dir, 'metrics_interrupted.npz')
    interrupted_model_filename = os.path.join(results_dir, 'model_state_interrupted.pth')

    # Note: all_train_sharpes and all_train_mle_losses will contain data
    # up to the last completed iteration + potentially some data from the
    # current, interrupted iteration if the append happened before the interrupt.
    np.savez(interrupted_metrics_filename, train_sharpes=all_train_sharpes, train_mle_losses=all_train_mle_losses)
    torch.save(model.state_dict(), interrupted_model_filename)
    print(f"Saved interrupted metrics to {interrupted_metrics_filename}")
    print(f"Saved interrupted model state to {interrupted_model_filename}")
