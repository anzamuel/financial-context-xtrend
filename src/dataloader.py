from torch.utils.data import DataLoader, default_collate
from dataset import XTrendDataset
import time
from tqdm.auto import tqdm

def xtrend_collate_fn(batch):
    batched_item = default_collate(batch)
    return {
        "target_x": batched_item["target_x"],
        "target_y": batched_item["target_y"],
        "target_s": batched_item["target_s"],
        "context_x_padded": batched_item["context_x"],
        "context_xi_padded": batched_item["context_xi"],
        "context_s_padded": batched_item["context_s"],
        "actual_context_lens": batched_item["context_lens"]
    }

if __name__ == "__main__":
    dataset = XTrendDataset()

    batch_size = 512
    num_iterations = 1000

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=xtrend_collate_fn,
        shuffle=True
    )

    start_time = time.time()
    loader_iter = iter(dataloader)

    sample = {}
    for _ in tqdm(range(num_iterations), desc="Benchmarking Dataloader"):
        try:
            sample = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            sample = next(loader_iter)

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_iterations

    print(f"Total time for {num_iterations} batches: {total_time:.4f} seconds")
    print(f"Average time per batch: {avg_time_per_batch * 1000:.4f} ms")

    print("\nShape of a single sample batch for verification:")
    for key, value in sample.items():
        print(f"{key} shape: {list(value.shape)}")
