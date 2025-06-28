import torch
import numpy as np
import os
from tqdm import tqdm
from models.xtrend import XTrendModel
from dataset import XTrendDataset
from torch.utils.data import DataLoader

TRAIN_STRIDE = 1
BATCH_SIZE = 512
ITERATIONS = 10_000
LR = 1e-3 # 1e-3

results_dir = 'runs'
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = XTrendDataset(train_stride = TRAIN_STRIDE)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, pin_memory=True)

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
    dropout=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_sharpe = 0
all_train_sharpes = []
all_train_mle_losses = []

try:
    for it in range(1, ITERATIONS + 1):
        print("---")
        dataset.set_mode("train")
        train_sharpes = []
        train_pbar = tqdm(dataloader, desc=f"TRAIN {it} | Sharpe: N/A", leave=True)
        for train_batch in train_pbar:
            train_batch = {k: v.to(device) for k, v in train_batch.items()}
            train_sharpe_loss, _, _ = model.training_step(train_batch, optimizer, alpha=0) # mle_loss and total loss ignored
            current_train_sharpe = -train_sharpe_loss.numpy(force=True)
            train_sharpes.append(current_train_sharpe)
            avg_train_sharpe = np.mean(train_sharpes) # use torch instead of numpy
            train_pbar.set_description(f"TRAIN {it} | Sharpe: {avg_train_sharpe:7.4f} | Current: {current_train_sharpe:7.4f}")

        dataset.set_mode("eval")
        eval_sharpes = []
        eval_pbar = tqdm(dataloader, desc=f"EVAL {it} | Sharpe: N/A", leave=True)
        for eval_batch in eval_pbar:
            eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
            eval_sharpe_loss, _ = model.evaluate(eval_batch) # TODO: look at captured positions
            current_eval_sharpe = -eval_sharpe_loss.numpy(force=True)
            eval_sharpes.append(current_eval_sharpe)
            avg_eval_sharpe = np.mean(eval_sharpes) # use torch instead of numpy
            eval_pbar.set_description(f"EVAL {it} | Sharpe: {avg_eval_sharpe:7.4f} | Current: {current_eval_sharpe:7.4f}")

    print("---")
    print("Training complete.")

except KeyboardInterrupt:
    print("---")
    print("Training interrupted by user.")
    interrupted_metrics_filename = os.path.join(results_dir, 'metrics_interrupted.npz')
    interrupted_model_filename = os.path.join(results_dir, 'model_state_interrupted.pth')
    np.savez(interrupted_metrics_filename, train_sharpes=all_train_sharpes, train_mle_losses=all_train_mle_losses)
    torch.save(model.state_dict(), interrupted_model_filename)
    print(f"Saved interrupted metrics to {interrupted_metrics_filename}")
    print(f"Saved interrupted model state to {interrupted_model_filename}")
