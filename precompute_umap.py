"""
Precompute the UMAP projection from stock_embeddings.json and save it to
umap_projection.json so that interface.py doesn't have to run UMAP at startup.

Run this script whenever stock_embeddings.json is regenerated:
    python precompute_umap.py
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np
import umap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "stock_embeddings.json")
LISTING_CSV_PATH = os.path.join(BASE_DIR, "nyse_nasdaq_listings.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "umap_projection.json")


def _load_listing_name_map() -> Dict[str, str]:
    if not os.path.isfile(LISTING_CSV_PATH):
        return {}
    result: Dict[str, str] = {}
    try:
        with open(LISTING_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = str(row.get("Symbol", "")).strip().upper()
                name = str(row.get("Security Name", "")).strip()
                if symbol and name and symbol not in result:
                    result[symbol] = name
    except Exception as e:
        print(f"Warning: could not load listing name map: {e}")
    return result


def _load_embeddings() -> Dict[str, dict]:
    if not os.path.isfile(EMBEDDINGS_PATH):
        raise SystemExit(f"stock_embeddings.json not found at {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result: Dict[str, dict] = {}
    for symbol, payload in raw.items():
        symbol = str(symbol).strip().upper()
        if not symbol or not isinstance(payload, dict):
            continue
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        try:
            vector = [float(v) for v in embedding]
        except (TypeError, ValueError):
            continue
        result[symbol] = {
            "embedding": vector,
            "num_windows": int(payload.get("num_windows") or 0),
            "reconstruction_score": float(payload.get("reconstruction_loss", float("nan"))),
        }
    return result


def main() -> None:
    print("Loading embeddings...")
    embeddings = _load_embeddings()
    listing_names = _load_listing_name_map()

    rows = []
    vectors = []
    expected_vector_size = 0

    for symbol in sorted(embeddings):
        payload = embeddings[symbol]
        vector = payload["embedding"]
        if expected_vector_size == 0:
            expected_vector_size = len(vector)
        if len(vector) != expected_vector_size:
            continue
        rows.append(
            {
                "ticker": symbol,
                "company_name": listing_names.get(symbol, ""),
                "num_windows": payload["num_windows"],
                "reconstruction_score": payload["reconstruction_score"],
            }
        )
        vectors.append(vector)

    if not vectors:
        raise SystemExit("No valid embedding vectors found.")

    print(f"Running UMAP on {len(vectors)} stocks (embedding dim={expected_vector_size})...")
    matrix = np.array(vectors, dtype=float)
    projected = np.zeros((matrix.shape[0], 3), dtype=float)

    if matrix.shape[0] > 3:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=max(2, min(15, matrix.shape[0] - 1)),
            min_dist=0.1,
            metric="cosine",
            n_jobs=1,
            random_state=42,
        )
        projected = reducer.fit_transform(matrix)
    elif matrix.shape[0] > 1:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        component_count = min(3, centered.shape[0], centered.shape[1])
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected[:, :component_count] = centered @ vt[:component_count].T

    for i, row in enumerate(rows):
        row["x"] = float(projected[i, 0])
        row["y"] = float(projected[i, 1])
        row["z"] = float(projected[i, 2])

    output = {
        "vector_size": expected_vector_size,
        "axis_titles": ["UMAP 1", "UMAP 2", "UMAP 3"],
        "rows": rows,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved projection for {len(rows)} stocks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
