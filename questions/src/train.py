
"""Training and evaluation for all four variants (Q4-Q6).

CLI:  python src/train.py --variant {temporal,fixed,adaptive,hybrid}
All variants share the encoder, head, splits, seed, optimiser and epoch budget.
"""
import argparse
import copy
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (INPUT_LENGTH, HORIZON, WindowDataset, fit_normaliser,
                  load_processed, split_chronologically)
from model import build_model, AdaptiveAdjacency, HybridAdjacency
from utils import set_seed


@dataclass
class Config:
    seed: int = 42
    hidden_dim: int = 32
    embedding_dim: int = 10
    top_k: int = 20
    learn_alpha: bool = True
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 20
    patience: int = 5
    train_stride: int = 1            # use >1 to subsample training windows on slow CPUs
    processed_dir: str = "../data/processed"
    results_dir: str = "../results"
    checkpoint_dir: str = "../checkpoints"
    prediction_sensors: tuple = (0, 100, 300)    # sensors saved for pred-vs-actual plots


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def make_loaders(config):
    flow, timestamps, adjacency, meta = load_processed(config.processed_dir)
    splits = split_chronologically(flow, timestamps)
    mean, std = fit_normaliser(splits["train"].raw_flow)          # TRAIN ONLY

    generator = torch.Generator().manual_seed(config.seed)
    datasets = {
        name: WindowDataset(split, mean, std,
                            stride=config.train_stride if name == "train" else 1)
        for name, split in splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size,
                         shuffle=(name == "train"), generator=generator if name == "train" else None)
        for name, dataset in datasets.items()
    }
    return loaders, mean, std, adjacency, splits


# --------------------------------------------------------------------------
# Loss and metrics (observed entries only)
# --------------------------------------------------------------------------
def masked_mae_loss(prediction, target, mask):
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model, loader, mean, std, device, sensors_to_keep=None):
    """MAE / RMSE in ORIGINAL flow units (vehicles per 15 min), observed entries only."""
    model.eval()
    abs_error_sum = torch.zeros(HORIZON, dtype=torch.float64)
    sq_error_sum = torch.zeros(HORIZON, dtype=torch.float64)
    count = torch.zeros(HORIZON, dtype=torch.float64)
    kept_predictions, kept_targets = [], []

    for history, target, mask in loader:
        history, target, mask = history.to(device), target.to(device), mask.to(device)
        prediction = model(history) * std + mean
        target_original = target * std + mean
        error = (prediction - target_original) * mask                   # (B, H, N)
        abs_error_sum += error.abs().sum(dim=(0, 2)).double().cpu()
        sq_error_sum += (error ** 2).sum(dim=(0, 2)).double().cpu()
        count += mask.sum(dim=(0, 2)).double().cpu()
        if sensors_to_keep is not None:
            kept_predictions.append(prediction[:, :, sensors_to_keep].cpu())
            kept_targets.append(target_original[:, :, sensors_to_keep].cpu())

    metrics = {
        "mae": float(abs_error_sum.sum() / count.sum()),
        "rmse": float(torch.sqrt(sq_error_sum.sum() / count.sum())),
        "mae_per_horizon": (abs_error_sum / count).tolist(),
        "rmse_per_horizon": torch.sqrt(sq_error_sum / count).tolist(),
    }
    saved = None
    if sensors_to_keep is not None:
        saved = (torch.cat(kept_predictions).numpy(), torch.cat(kept_targets).numpy())
    return metrics, saved


# --------------------------------------------------------------------------
# One experiment
# --------------------------------------------------------------------------
def run_experiment(variant, config=None, prepared=None):
    """Train one variant and return a results dict (also written to results_dir)."""
    config = config or Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, mean, std, adjacency, _ = prepared or make_loaders(config)

    model = build_model(variant, adjacency, INPUT_LENGTH, HORIZON, config.hidden_dim,
                        config.embedding_dim, config.top_k, config.learn_alpha).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_val_mae, best_epoch, best_state = float("inf"), -1, None
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        running_loss, num_batches = 0.0, 0
        for history, target, mask in loaders["train"]:
            history, target, mask = history.to(device), target.to(device), mask.to(device)
            optimiser.zero_grad()
            loss = masked_mae_loss(model(history), target, mask)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            num_batches += 1

        val_metrics, _ = evaluate(model, loaders["val"], mean, std, device)
        print(f"[{variant}] epoch {epoch + 1:02d} "
              f"train_loss(norm)={running_loss / num_batches:.4f} "
              f"val_MAE={val_metrics['mae']:.3f} val_RMSE={val_metrics['rmse']:.3f}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae, best_epoch = val_metrics["mae"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    sensors = list(config.prediction_sensors)
    val_metrics, _ = evaluate(model, loaders["val"], mean, std, device)
    test_metrics, saved = evaluate(model, loaders["test"], mean, std, device, sensors)

    results = {
        "variant": variant, "best_epoch": best_epoch, "epochs_run": epoch + 1,
        "training_seconds": training_seconds,
        "seconds_per_epoch": training_seconds / (epoch + 1),
        "num_parameters": num_parameters, "device": str(device),
        "val": val_metrics, "test": test_metrics, "config": asdict(config),
    }

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{variant}.json").write_text(json.dumps(results, indent=2))
    np.savez(results_dir / f"{variant}_test_predictions.npz",
             prediction=saved[0], target=saved[1], sensors=np.array(sensors))

    provider = model.adjacency_provider
    if isinstance(provider, (AdaptiveAdjacency, HybridAdjacency)):
        with torch.no_grad():
            learned = (provider.adaptive() if isinstance(provider, HybridAdjacency)
                       else provider()).cpu().numpy()
        np.save(results_dir / f"{variant}_learned_adjacency.npy", learned)
        if isinstance(provider, HybridAdjacency):
            results["final_alpha"] = float(provider.alpha.item())
            (results_dir / f"{variant}.json").write_text(json.dumps(results, indent=2))

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_state, Path(config.checkpoint_dir) / f"{variant}.pt")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["temporal", "fixed", "adaptive", "hybrid"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()
    config = Config(seed=args.seed, max_epochs=args.max_epochs,
                    train_stride=args.train_stride, processed_dir=args.processed_dir,
                    results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir)
    results = run_experiment(args.variant, config)
    print(json.dumps({k: results[k] for k in ("variant", "training_seconds")}, indent=2))
    print("test MAE/RMSE:", results["test"]["mae"], results["test"]["rmse"])


if __name__ == "__main__":
    main()
