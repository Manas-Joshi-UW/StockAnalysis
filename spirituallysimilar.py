"""
Find stocks similar to each other via DTW on z-normalized price series.
Reads parquet files from price_history/, computes pairwise DTW, saves top-10
per stock to a JSON file. Uses GPU (CuPy CUDA kernel) when available, else CPU.

Optional GPU: pip install cupy-cuda12x (use Python 3.9–3.12 for prebuilt wheels
on Windows; Python 3.8 often has no wheel and triggers a build that needs
Visual C++ and CUDA dev setup).
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PRICE_HISTORY_DIR = "price_history"
OUTPUT_JSON = "dtw_similar_stocks.json"
TOP_K = 10
DTW_WINDOW_FRAC = 0.1
# Skip stocks with less than 3 years of trading days (~756)
MIN_TRADING_DAYS = 252 * 8
# Max series length that fits in GPU shared memory (2 rows of doubles; 48KB => max 3070).
# Above this we use CPU so no truncation.
GPU_MAX_LEN_LIMIT = 3070

_GPU_AVAILABLE = False
_cp = None
_dtw_kernel = None


def _init_gpu():
    global _GPU_AVAILABLE, _cp, _dtw_kernel
    if _dtw_kernel is not None:
        return
    try:
        import cupy as cp  # noqa: F401
        # One thread per pair; each block runs one DTW in shared mem
        kernel_src = r"""
        extern "C" __global__
        void dtw_kernel(
            const double* series,
            const int* lengths,
            int max_len,
            const int* pair_i,
            const int* pair_j,
            int num_pairs,
            double* out
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_pairs) return;
            int i = pair_i[idx];
            int j = pair_j[idx];
            int len_i = lengths[i];
            int len_j = lengths[j];
            if (len_i == 0 || len_j == 0) { out[idx] = 1e308; return; }
            int w = max(1, (int)(max_len * 0.1));
            extern __shared__ double smem[];
            double* row_prev = smem;
            double* row_curr = smem + (max_len + 1);
            const double* s1 = series + i * max_len;
            const double* s2 = series + j * max_len;
            for (int k = 0; k <= len_j; ++k) row_prev[k] = 1e308;
            row_prev[0] = 0.0;
            for (int ii = 1; ii <= len_i; ++ii) {
                int j_lo = max(1, ii - w);
                int j_hi = min(len_j + 1, ii + w + 1);
                for (int k = 0; k <= len_j; ++k) row_curr[k] = 1e308;
                for (int jj = j_lo; jj < j_hi; ++jj) {
                    double c = (s1[ii-1] - s2[jj-1]) * (s1[ii-1] - s2[jj-1]);
                    double v = fmin(fmin(row_prev[jj], row_prev[jj-1]), row_curr[jj-1]);
                    row_curr[jj] = c + v;
                }
                double* t = row_prev; row_prev = row_curr; row_curr = t;
            }
            double v = row_prev[len_j];
            out[idx] = (v >= 1e307) ? 1e308 : sqrt(v);
        }
        """
        _dtw_kernel = cp.RawKernel(kernel_src, "dtw_kernel")  # type: ignore[attr-defined]
        _cp = cp
        _GPU_AVAILABLE = True
    except Exception:
        _cp = None
        _dtw_kernel = None
        _GPU_AVAILABLE = False


def z_normalize(series: np.ndarray) -> np.ndarray:
    out = np.asarray(series, dtype=np.float64)
    out = out[~np.isnan(out)]
    if len(out) < 2:
        return np.asarray(series, dtype=np.float64)
    mean, std = out.mean(), out.std()
    if std <= 0:
        return np.zeros_like(series, dtype=np.float64)
    return (np.asarray(series, dtype=np.float64) - mean) / std


def dtw_distance_cpu(s1: np.ndarray, s2: np.ndarray, window_frac: float = DTW_WINDOW_FRAC) -> float:
    """DTW distance with Sakoe-Chiba band (CPU)."""
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return np.inf
    w = max(1, int(window_frac * max(n, m)))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m + 1, i + w + 1)
        for j in range(j_lo, j_hi):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def _dtw_worker(pairs_chunk: List[Tuple[int, int]], series_list: List[np.ndarray]) -> List[Tuple[int, int, float]]:
    """Compute DTW for a chunk of (i,j) pairs. Module-level for multiprocessing."""
    out = []
    for i, j in pairs_chunk:
        d = dtw_distance_cpu(series_list[i], series_list[j])
        out.append((i, j, d))
    return out


def _dtw_cpu_matrix(tickers: List[str], series_list: List[np.ndarray], n: int) -> np.ndarray:
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    chunk_size = max(1, (len(pairs) + n_workers - 1) // n_workers)
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    matrix = np.full((n, n), np.inf)
    for i in range(n):
        matrix[i, i] = np.inf
    with Pool(n_workers) as pool:
        for result in pool.starmap(
            _dtw_worker,
            [(c, series_list) for c in chunks],
            chunksize=1,
        ):
            for i, j, d in result:
                matrix[i, j] = matrix[j, i] = d
    return matrix


def _dtw_gpu_matrix(
    tickers: List[str],
    series_list: List[np.ndarray],
    n: int,
    max_len: int,
) -> np.ndarray:
    cp = _cp
    # Pad to (n, max_len); use full series (no truncation), zero-pad if shorter
    series_np = np.zeros((n, max_len), dtype=np.float64)
    lengths = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = series_list[i]
        L = len(s)
        lengths[i] = L
        series_np[i, :L] = s
    series_gpu = cp.asarray(series_np)
    lengths_gpu = cp.asarray(lengths)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pairs)
    pair_i = np.array([p[0] for p in pairs], dtype=np.int32)
    pair_j = np.array([p[1] for p in pairs], dtype=np.int32)
    pair_i_gpu = cp.asarray(pair_i)
    pair_j_gpu = cp.asarray(pair_j)
    out_gpu = cp.empty(num_pairs, dtype=cp.float64)
    smem = 2 * (max_len + 1) * 8  # bytes for two rows
    threads = 256
    blocks = (num_pairs + threads - 1) // threads
    _dtw_kernel((blocks,), (threads,), (  # type: ignore[union-attr]
        series_gpu,
        lengths_gpu,
        np.int32(max_len),
        pair_i_gpu,
        pair_j_gpu,
        np.int32(num_pairs),
        out_gpu,
    ), shared_mem=smem)
    distances = cp.asnumpy(out_gpu)
    matrix = np.full((n, n), np.inf)
    for i in range(n):
        matrix[i, i] = np.inf
    for k, (i, j) in enumerate(pairs):
        matrix[i, j] = matrix[j, i] = float(distances[k])
    return matrix


def load_series_from_parquet(path: str) -> Optional[np.ndarray]:
    try:
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        col = "Close" if "Close" in df.columns else "Adj Close"
        if col not in df.columns:
            return None
        return df[col].values
    except Exception:
        return None


def main():
    _init_gpu()
    base = Path(__file__).resolve().parent
    price_dir = base / PRICE_HISTORY_DIR
    if not price_dir.is_dir():
        raise SystemExit(f"Directory not found: {price_dir}")

    tickers = []
    series_list = []
    for f in sorted(price_dir.glob("*.parquet")):
        ticker = f.stem
        raw = load_series_from_parquet(str(f))
        if raw is None or len(raw) < MIN_TRADING_DAYS:
            print(f"Skipping {ticker} with {len(raw)} trading days (less than {MIN_TRADING_DAYS})")
            continue
        zn = z_normalize(raw)
        tickers.append(ticker)
        series_list.append(zn)

    n = len(tickers)
    if n == 0:
        raise SystemExit("No valid series loaded from price_history/")

    actual_max_len = max(len(s) for s in series_list)
    use_gpu = _GPU_AVAILABLE and actual_max_len <= GPU_MAX_LEN_LIMIT

    if use_gpu:
        print(f"Loaded {n} tickers (max series length {actual_max_len}). Computing pairwise DTW on GPU (full length)...")
        try:
            matrix = _dtw_gpu_matrix(tickers, series_list, n, actual_max_len)
        except Exception as e:
            print(f"GPU failed ({e}), falling back to CPU...")
            matrix = _dtw_cpu_matrix(tickers, series_list, n)
    else:
        if actual_max_len > GPU_MAX_LEN_LIMIT:
            print(f"Loaded {n} tickers (max length {actual_max_len} > {GPU_MAX_LEN_LIMIT}). Using CPU so no truncation.")
        else:
            print(f"Loaded {n} tickers. Computing pairwise DTW on CPU (window_frac={DTW_WINDOW_FRAC})...")
        matrix = _dtw_cpu_matrix(tickers, series_list, n)

    out = {}
    for i, ticker in enumerate(tickers):
        idx = np.argsort(matrix[i])[:TOP_K]
        out[ticker] = [tickers[j] for j in idx]

    out_path = base / OUTPUT_JSON
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote top-{TOP_K} similar tickers for {n} stocks to {out_path}")


if __name__ == "__main__":
    main()
