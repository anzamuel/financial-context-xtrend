from tqdm import tqdm
import random
import torch


def prepare_batches(targets, all_segments_day_dict, batch_size,num_context, keep_partial_batches=False):
    shuffled = targets.sample(frac=1)
    shuffled["batch"] = shuffled.groupby("seq_len").cumcount() // batch_size
    grouped = shuffled.groupby(["seq_len", "batch"])
    if not keep_partial_batches:
        shuffled = grouped.filter(lambda x: x["batch"].count() == batch_size)
    shuffled.index = shuffled.groupby(["seq_len", "batch"]).ngroup()
    num_batches = shuffled.index.max()
    order = list(range(num_batches))
    random.shuffle(order)

    shuffled["contexts"] = shuffled.apply(
        lambda row: all_segments_day_dict[row.seq_len]
        .loc[lambda df: df.index < row.day][["x", "y"]].sample(n=num_context),
        axis=1,
    )

    shuffled["context_x"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["x"].tolist(), dim=3)
    )
    shuffled["context_y"] = shuffled["contexts"].map(
        lambda c: torch.stack(c["y"].tolist(), dim=3)
    )

    batches = []
    for i in tqdm(order):
        batch = shuffled.loc[[i]]
        batches.append(
            (
                batch["seq_len"].iloc[0],
                torch.cat(batch["context_x"].tolist()),
                torch.cat(batch["context_y"].tolist()),
                torch.cat(batch["x"].tolist()),
                torch.cat(batch["y"].tolist()),
                batch["date"].tolist(),
                batch["ticker"].tolist(),
            )
        )

    return batches