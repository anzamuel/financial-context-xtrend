import torch
import numpy as np
import os
import datetime as dt
from tqdm import tqdm
from src.models.xtrend import XTrendModel
from src.dataset import XTrendDataset, FEATURE_COLS, PINNACLE_ASSETS
from torch.utils.data import DataLoader, SubsetRandomSampler
import copy

# --- Training Hyperparameters ---
TRAIN_STRIDE = 1
TRAIN_START = dt.datetime(2013, 1, 1)
TRAIN_END = dt.datetime(2018, 1, 1)
EVAL_START = dt.datetime(2016, 1, 1)
EVAL_END = dt.datetime(2018, 1, 1)
BATCH_SIZE = 512
ITERATIONS = 200
LR = 1e-3
TRAIN_SUBSET_FRACTION = 0.2
EARLY_STOPPING_PATIENCE = 10
D_H = 64
EMBEDDING_DIM = 4
N_HEADS = 4
DROPOUT = 0.5
WARMUP_PERIOD_LEN = 63

def training_loop():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = XTrendDataset(
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        train_stride=TRAIN_STRIDE
    )

    model = XTrendModel(
        num_features=len(FEATURE_COLS),
        num_embeddings=len(PINNACLE_ASSETS),
        embedding_dim=EMBEDDING_DIM,
        d_h=D_H,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        warmup_period_len=WARMUP_PERIOD_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_sharpe = -np.inf
    best_model_state = None
    patience_counter = 0

    all_train_sharpes = []
    all_valid_sharpes = []

    results_dir = 'runs'
    os.makedirs(results_dir, exist_ok=True)

    try:
        for it in range(1, ITERATIONS + 1):
            print(f"--- Iteration {it}/{ITERATIONS} ---")

            dataset.set_mode("train")
            num_train_samples = int(len(dataset) * TRAIN_SUBSET_FRACTION)
            train_indices = np.random.choice(len(dataset), num_train_samples, replace=False)
            train_sampler = SubsetRandomSampler(train_indices)
            train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)

            train_sharpes = []
            train_pbar = tqdm(train_dataloader, desc=f"TRAIN {it} | Sharpe: N/A", leave=False)
            for train_batch in train_pbar:
                train_batch = {k: v.to(device) for k, v in train_batch.items() if isinstance(v, torch.Tensor)}
                loss = model.training_step(train_batch, optimizer)
                current_train_sharpe = -loss.item()
                train_sharpes.append(current_train_sharpe)
                avg_train_sharpe = np.mean(train_sharpes)
                train_pbar.set_description(f"TRAIN {it} | Sharpe: {avg_train_sharpe:7.4f}")
            all_train_sharpes.append(avg_train_sharpe)

            dataset.set_mode("eval")
            eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            eval_sharpes = []
            avg_eval_sharpe = 0

            for run_idx in range(1):
                eval_pbar = tqdm(eval_dataloader, desc=f"EVAL Run {run_idx+1}/10 | Sharpe: {avg_eval_sharpe:7.4f}", leave=False)
                for eval_batch in eval_pbar:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items() if isinstance(v, torch.Tensor)}
                    eval_sharpe_loss, _ = model.evaluate(eval_batch)
                    current_eval_sharpe = -eval_sharpe_loss.item()
                    eval_sharpes.append(current_eval_sharpe)

                    avg_eval_sharpe = np.nanmean(eval_sharpes)
                    eval_pbar.set_description(f"EVAL Run {run_idx+1}/10 | Sharpe: {avg_eval_sharpe:7.4f}")

            avg_epoch_eval_sharpe = np.mean(eval_sharpes)
            all_valid_sharpes.append(avg_epoch_eval_sharpe)

            if avg_epoch_eval_sharpe > best_valid_sharpe:
                best_valid_sharpe = avg_epoch_eval_sharpe
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"New best validation Sharpe: {best_valid_sharpe:.4f}")
            else:
                patience_counter += 1
                print(f"Validation Sharpe did not improve: {avg_epoch_eval_sharpe:.4f}")
                print(f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        if best_model_state:
            print(f"\nBest validation Sharpe was: {best_valid_sharpe:.4f}")
            user_input = input("Save best model and metrics? (y/n): ").lower()
            if user_input == 'y':
                print("Loading best model state...")
                model.load_state_dict(best_model_state)
                final_metrics_filename = os.path.join(results_dir, 'final_metrics.npz')
                final_model_filename = os.path.join(results_dir, 'final_model.pth')
                np.savez(final_metrics_filename, train_sharpes=all_train_sharpes, valid_sharpes=all_valid_sharpes)
                torch.save(model.state_dict(), final_model_filename)
                print(f"Saved final metrics to {final_metrics_filename}")
                print(f"Saved final model state to {final_model_filename}")
            else:
                print("Save operation cancelled by user.")
        else:
            print("\nNo best model to save. Training did not complete a full validation cycle.")

if __name__ == "__main__":
    training_loop()
