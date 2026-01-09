import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import HDBSCAN

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import cfg
from src.statistics import stats
from src.cluster.preprocess import get_preprocessor


def count_clusters_noise(labels: np.ndarray) -> tuple[int, int]:
    labels = np.asarray(labels)
    noise = int(np.sum(labels < 0))
    n_clusters = int(np.unique(labels[labels >= 0]).size)
    return n_clusters, noise


def search_hdbscan_params() -> None:
    print("Searching for HDBSCAN parameters that yield exactly 2 clusters with minimal noise...")

    data = pd.read_csv(cfg.paths["data"])
    stats.init(data)

    target_col = cfg.dataset["label"]
    X = data.drop(columns=[target_col])

    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    min_cluster_size_range = range(2, 51)
    min_samples_range = [None, *range(1, 21)]

    total = len(min_cluster_size_range) * len(min_samples_range)
    best_noise = None
    best = []
    n_ok = 0

    grid = product(min_cluster_size_range, min_samples_range)
    for min_cluster_size, min_samples in tqdm(grid, total=total):
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_jobs=-1,
        )
        labels = model.fit_predict(X_processed)
        n_clusters, noise = count_clusters_noise(labels)
        if n_clusters != 2:
            continue

        n_ok += 1
        row = {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "noise": noise,
        }
        if best_noise is None or noise < best_noise:
            best_noise = noise
            best = [row]
            continue
        if noise == best_noise:
            best.append(row)

    if best_noise is None:
        print("No parameters found that yield exactly 2 clusters.")
        return

    print(f"\nFound {n_ok} configurations yielding 2 clusters.")
    print(f"Best (min noise={best_noise}) has {len(best)} configuration(s):")
    print("-" * 60)
    print(f"{'min_cluster_size':<16} | {'min_samples':<11} | {'Noise'}")
    print("-" * 60)

    key = lambda r: (r["min_cluster_size"], -1 if r["min_samples"] is None else r["min_samples"])
    for r in sorted(best, key=key):
        print(f"{r['min_cluster_size']:<16} | {str(r['min_samples']):<11} | {r['noise']}")


if __name__ == "__main__":
    search_hdbscan_params()


