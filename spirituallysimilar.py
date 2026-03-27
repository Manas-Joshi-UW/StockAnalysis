"""
Train a stock-price autoencoder on sliding 50-day windows and derive stock
similarity from the learned latent space.

For each parquet in price_history/, the script builds overlapping windows:
days 1-50, 2-51, 3-52, ...

Each 50-day window is normalized relative to its first price so the model learns
shape rather than nominal price level. The autoencoder architecture is:
50 -> 32 -> 16 -> 8 -> 16 -> 32 -> 50

Outputs:
- stock_autoencoder.pt
- stock_autoencoder_metrics.json
- stock_embeddings.json
- autoencoder_similar_stocks.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PRICE_HISTORY_DIR = "price_history"
WINDOW_SIZE = 50
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
LATENT_DIM = 8
TOP_K = 10

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_SEED = 42

MODEL_OUTPUT = "stock_autoencoder.pt"
METRICS_OUTPUT = "stock_autoencoder_metrics.json"
EMBEDDINGS_OUTPUT = "stock_embeddings.json"
SIMILARITY_OUTPUT = "autoencoder_similar_stocks.json"
CORRELATION_EPS = 1e-8


@dataclass
class StockRecord:
    ticker: str
    series: np.ndarray
    train_starts: np.ndarray
    val_starts: np.ndarray
    test_starts: np.ndarray

    @property
    def total_windows(self) -> int:
        return len(self.series) - WINDOW_SIZE + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stock-price autoencoder.")
    parser.add_argument("--price-dir", default=PRICE_HISTORY_DIR, help="Directory containing parquet files.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--max-windows-per-stock",
        type=int,
        default=None,
        help="Optional cap on windows sampled from each ticker to reduce runtime.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Number of nearest neighbors to save for each ticker.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("cpu", "cuda"),
        help="Force training device. Defaults to cuda when available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_window(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    base = float(window[0])
    if not np.isfinite(base) or abs(base) < 1e-8:
        base = 1.0
    normalized = (window / base) - 1.0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32, copy=False)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.droplevel(-1)
    return df


def load_price_series(path: Path) -> Optional[np.ndarray]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"[{path.stem}] Failed to read parquet: {exc}")
        return None

    df = _flatten_columns(df)
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        print(f"[{path.stem}] Missing Close/Adj Close column")
        return None

    series = pd.to_numeric(df[close_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(series) < WINDOW_SIZE:
        return None
    return series


def build_split_starts(window_count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(window_count, dtype=np.int32)
    if window_count == 1:
        return starts, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    scores = rng.random(window_count)
    train = starts[scores < TRAIN_RATIO]
    val = starts[(scores >= TRAIN_RATIO) & (scores < TRAIN_RATIO + VAL_RATIO)]
    test = starts[scores >= TRAIN_RATIO + VAL_RATIO]

    if len(train) == 0:
        train = starts[:1]
        remaining = starts[1:]
        val = remaining[:0]
        test = remaining

    if len(val) == 0 and window_count >= 3:
        candidate = train[-1]
        train = train[:-1]
        val = np.asarray([candidate], dtype=np.int32)
        if len(train) == 0:
            train = test[:1]
            test = test[1:]

    if len(test) == 0 and window_count >= 2:
        if len(train) > 1:
            candidate = train[-1]
            train = train[:-1]
        elif len(val) > 0:
            candidate = val[-1]
            val = val[:-1]
        else:
            candidate = starts[-1]
        test = np.asarray([candidate], dtype=np.int32)

    return np.sort(train), np.sort(val), np.sort(test)


def maybe_downsample(starts: np.ndarray, max_windows: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if max_windows is None or len(starts) <= max_windows:
        return starts
    choice = rng.choice(starts, size=max_windows, replace=False)
    return np.sort(choice.astype(np.int32, copy=False))


def load_stock_records(
    price_dir: Path,
    seed: int,
    max_windows_per_stock: Optional[int] = None,
) -> List[StockRecord]:
    rng = np.random.default_rng(seed)
    records: List[StockRecord] = []

    for path in sorted(price_dir.glob("*.parquet")):
        series = load_price_series(path)
        if series is None:
            continue

        window_count = len(series) - WINDOW_SIZE + 1
        starts = np.arange(window_count, dtype=np.int32)
        starts = maybe_downsample(starts, max_windows_per_stock, rng)
        if len(starts) == 0:
            continue

        split_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        train_starts, val_starts, test_starts = build_split_starts(len(starts), split_rng)

        records.append(
            StockRecord(
                ticker=path.stem,
                series=series,
                train_starts=starts[train_starts],
                val_starts=starts[val_starts],
                test_starts=starts[test_starts],
            )
        )

    return records


class SlidingWindowDataset(Dataset):
    def __init__(self, records: Sequence[StockRecord], split: str):
        self.window_size = WINDOW_SIZE
        self.series: List[np.ndarray] = []
        self.starts: List[np.ndarray] = []

        for record in records:
            split_starts = getattr(record, f"{split}_starts")
            if len(split_starts) == 0:
                continue
            self.series.append(record.series)
            self.starts.append(split_starts)

        self.lengths = np.asarray([len(s) for s in self.starts], dtype=np.int64)
        self.cumulative_lengths = np.cumsum(self.lengths) if len(self.lengths) else np.asarray([], dtype=np.int64)

    def __len__(self) -> int:
        if len(self.cumulative_lengths) == 0:
            return 0
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        stock_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        prev_total = 0 if stock_idx == 0 else int(self.cumulative_lengths[stock_idx - 1])
        local_idx = idx - prev_total

        start = int(self.starts[stock_idx][local_idx])
        window = self.series[stock_idx][start : start + self.window_size]
        features = normalize_window(window)
        return torch.from_numpy(features)


class StockAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 16),
            nn.SiLU(),
            nn.Linear(16, 32),
            nn.SiLU(),
            nn.Linear(32, WINDOW_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def correlation_reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Pearson correlation is cosine similarity after mean-centering each window.
    reconstructed_centered = reconstructed - reconstructed.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    correlation = F.cosine_similarity(
        reconstructed_centered,
        target_centered,
        dim=1,
        eps=CORRELATION_EPS,
    )
    return (1.0 - correlation).mean()


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_epoch(
    model: StockAutoencoder,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = correlation_reconstruction_loss(reconstructed, batch)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = batch.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return math.nan
    return total_loss / total_samples


def train_model(
    model: StockAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=device) if len(val_loader.dataset) else math.nan

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{epochs} | train_corr_loss={train_loss:.6f} | val_corr_loss={val_loss:.6f}")

        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_model(model: StockAutoencoder, loader: DataLoader, device: torch.device) -> float:
    return run_epoch(model, loader, optimizer=None, device=device)


def encode_windows(
    model: StockAutoencoder,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0, LATENT_DIM), dtype=np.float32)

    model.eval()
    encoded_batches: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            latent = model.encode(batch).cpu().numpy()
            encoded_batches.append(latent.astype(np.float32, copy=False))

    return np.vstack(encoded_batches)


def compute_stock_embeddings(
    model: StockAutoencoder,
    records: Sequence[StockRecord],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    embeddings: Dict[str, Dict[str, object]] = {}

    for record in records:
        all_starts = np.arange(record.total_windows, dtype=np.int32)
        windows = np.asarray(
            [normalize_window(record.series[start : start + WINDOW_SIZE]) for start in all_starts],
            dtype=np.float32,
        )

        model.eval()
        latent = encode_windows(model, windows, batch_size=batch_size, device=device)

        with torch.no_grad():
            reconstructed = []
            for start_idx in range(0, len(windows), batch_size):
                batch = torch.from_numpy(windows[start_idx : start_idx + batch_size]).to(device)
                reconstructed.append(model(batch).cpu())
            reconstructed_tensor = torch.cat(reconstructed, dim=0) if reconstructed else torch.empty((0, WINDOW_SIZE))
            error = (
                correlation_reconstruction_loss(reconstructed_tensor, torch.from_numpy(windows)).item()
                if len(windows)
                else math.nan
            )

        mean_embedding = latent.mean(axis=0) if len(latent) else np.zeros(LATENT_DIM, dtype=np.float32)
        embeddings[record.ticker] = {
            "num_windows": int(record.total_windows),
            "reconstruction_loss": float(error),
            "embedding": [float(x) for x in mean_embedding],
        }

    return embeddings


def compute_nearest_neighbors(
    embeddings: Dict[str, Dict[str, object]],
    top_k: int,
) -> Dict[str, List[Dict[str, float]]]:
    tickers = list(embeddings.keys())
    matrix = np.asarray([embeddings[ticker]["embedding"] for ticker in tickers], dtype=np.float32)
    neighbors: Dict[str, List[Dict[str, float]]] = {}

    for idx, ticker in enumerate(tickers):
        deltas = matrix - matrix[idx]
        distances = np.linalg.norm(deltas, axis=1)
        order = np.argsort(distances)

        nearest: List[Dict[str, float]] = []
        for neighbor_idx in order:
            if neighbor_idx == idx:
                continue
            nearest.append(
                {
                    "ticker": tickers[neighbor_idx],
                    "distance": float(distances[neighbor_idx]),
                }
            )
            if len(nearest) == top_k:
                break

        neighbors[ticker] = nearest

    return neighbors


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def choose_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device is not None:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def summarize_records(records: Sequence[StockRecord]) -> Dict[str, int]:
    return {
        "num_tickers": len(records),
        "total_train_windows": int(sum(len(record.train_starts) for record in records)),
        "total_val_windows": int(sum(len(record.val_starts) for record in records)),
        "total_test_windows": int(sum(len(record.test_starts) for record in records)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    price_dir = base_dir / args.price_dir
    if not price_dir.is_dir():
        raise SystemExit(f"Directory not found: {price_dir}")

    records = load_stock_records(
        price_dir=price_dir,
        seed=args.seed,
        max_windows_per_stock=args.max_windows_per_stock,
    )
    if not records:
        raise SystemExit(f"No eligible parquet files with at least {WINDOW_SIZE} rows found in {price_dir}")

    dataset_summary = summarize_records(records)
    print(
        "Loaded "
        f"{dataset_summary['num_tickers']} tickers | "
        f"train windows={dataset_summary['total_train_windows']:,} | "
        f"val windows={dataset_summary['total_val_windows']:,} | "
        f"test windows={dataset_summary['total_test_windows']:,}"
    )

    train_dataset = SlidingWindowDataset(records, split="train")
    val_dataset = SlidingWindowDataset(records, split="val")
    test_dataset = SlidingWindowDataset(records, split="test")

    if len(train_dataset) == 0:
        raise SystemExit("Training dataset is empty after window generation.")

    train_loader = build_loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = build_loader(test_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    device = choose_device(args.device)
    print(f"Using device: {device}")

    model = StockAutoencoder().to(device)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )

    test_loss = evaluate_model(model, test_loader, device=device) if len(test_dataset) else math.nan
    print(f"Test correlation reconstruction loss: {test_loss:.6f}")

    checkpoint_path = base_dir / MODEL_OUTPUT
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "window_size": WINDOW_SIZE,
            "latent_dim": LATENT_DIM,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
        },
        checkpoint_path,
    )

    embeddings = compute_stock_embeddings(model, records, batch_size=args.batch_size, device=device)
    neighbors = compute_nearest_neighbors(embeddings, top_k=args.top_k)

    metrics_payload = {
        "history": history,
        "test_loss": test_loss,
        "loss_name": "1 - pearson_correlation",
        "dataset_summary": dataset_summary,
        "window_size": WINDOW_SIZE,
        "architecture": [50, 32, 16, 8, 16, 32, 50],
        "normalization": "window / first_price - 1",
        "outputs": {
            "checkpoint": MODEL_OUTPUT,
            "embeddings": EMBEDDINGS_OUTPUT,
            "neighbors": SIMILARITY_OUTPUT,
        },
    }

    save_json(base_dir / METRICS_OUTPUT, metrics_payload)
    save_json(base_dir / EMBEDDINGS_OUTPUT, embeddings)
    save_json(base_dir / SIMILARITY_OUTPUT, neighbors)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {base_dir / METRICS_OUTPUT}")
    print(f"Saved embeddings to {base_dir / EMBEDDINGS_OUTPUT}")
    print(f"Saved similarities to {base_dir / SIMILARITY_OUTPUT}")


if __name__ == "__main__":
    main()
