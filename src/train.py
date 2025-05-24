import torch
import pandas as pd 
import numpy as np
from prep_data import prepare_batches
from tqdm import tqdm
from models.xtrend import XTrendModel

TESTING = False
# TRAIN_LAST_OUTPUT = 0  # more items in sequence is neccessary!
TRAIN_LAST_OUTPUT = 1
LATENT_PATH = True
VAL_DIVERSIFIED_SHARPE = True

# BATCH_SIZE = 64
BATCH_SIZE = 252
# NUM_CONTEXT = 10
NUM_CONTEXT = 10
ITERATIONS = 200
# MIN_SEQ_LEN = 5
# MIN_SEQ_LEN = 5

# REMOVE_FIRST_N_OUTPUTS = 3
REMOVE_FIRST_N_OUTPUTS = 0

N = 50

# LR = 0.001
LR = 1e-3

# LEN_CONTEXT = 6
# BATCH_SIZE = 256

# EMBEDDING_DIM = 32
EMBEDDING_DIM = 64

ENCODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]
DECODER_OUTPUT_SIZES = [EMBEDDING_DIM, EMBEDDING_DIM]

LATENT_DIM = 4
# LATENT_DIM = 4
USE_X_ATTENTION = True
USE_SELF_ATTENTION = True

VALID_START_YEAR = 2005  # valid 2015-208

TEST_START_YEAR = 2010  # valid 2015-208


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


all_segments = pd.read_pickle("prepped.pkl")
all_segments['x'] = all_segments['x'].apply(lambda item: torch.from_numpy(item).unsqueeze(0))
all_segments['y'] = all_segments['y'].apply(lambda item: torch.from_numpy(item).unsqueeze(0))


all_segments.index = all_segments.groupby(["seq_len", "ticker"]).cumcount()
all_segments["day"] = all_segments.groupby("date").ngroup()

all_segments_day = all_segments.set_index("day")
all_segments_day_dict = {}
for s in all_segments_day.seq_len.unique():
    all_segments_day_dict[s] = all_segments_day[all_segments_day["seq_len"] == s]

targets = all_segments[all_segments.index > NUM_CONTEXT]
train_contexts = targets[targets.date.dt.year < VALID_START_YEAR]
valid_contexts = targets[
    (targets.date.dt.year >= VALID_START_YEAR)
    & (targets.date.dt.year < TEST_START_YEAR)
]

all_tickers = all_segments['ticker'].unique()
ticker2id = {ticker: idx for idx, ticker in enumerate(all_tickers)}

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
    mle_dim=hidden_dim,
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
    print("Prepping")
    batches = prepare_batches(train_contexts, all_segments_day_dict, BATCH_SIZE, NUM_CONTEXT)
    print(len(batches))
    print("Ready")
    pbar = tqdm(batches, desc=f"Iteration {it} | Train Sharpe: N/A", leave=True)
    for seq_len, x_context, y_context, x_target, y_target, _, tickers in pbar:
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        
        static_s = torch.tensor([ticker2id[ticker] for ticker in tickers], device=device)
        total_loss, sharpe_loss, mle_loss = model.training_step(
            (x_context, y_context, x_target, y_target, static_s), optimizer, alpha=1.0
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
    targ = valid_contexts if not TESTING else train_contexts
    val_batches = prepare_batches(targ, all_segments_day_dict, BATCH_SIZE, NUM_CONTEXT)
    val_pbar = tqdm(val_batches, desc=f"Validation | Valid Sharpe: N/A", leave=True)
    for (
        seq_len,
        x_context,
        y_context,
        x_target,
        y_target,
        dates,
        tickers,
    ) in val_pbar:
        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)
        static_s = torch.tensor([ticker2id[ticker] for ticker in tickers], device=device)

        sharpe_loss = model.evaluate(
            (x_context, y_context, x_target, y_target, static_s), alpha=1.0
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
        all_returns += zip(dates, tickers, captured_returns)

    if VAL_DIVERSIFIED_SHARPE and all_returns:
        diversified = (
            pd.DataFrame(all_returns, columns=["date", "ticker", "captured_returns"])
            .groupby("date")["captured_returns"]
            .sum()
        )
        val_sharpe = diversified.mean() * np.sqrt(252) / diversified.std()
        print("Div Valid Sharpe: ", val_sharpe)
        print("Valid Single Sharpe: ", np.mean(valid_sharpes))