import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import OPTICS

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


def search_optics_params() -> None:
    print("Searching for OPTICS parameters that yield exactly 2 clusters with minimal noise...")

    data = pd.read_csv(cfg.paths["data"])
    stats.init(data)

    target_col = cfg.dataset["label"]
    X = data.drop(columns=[target_col])

    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    min_samples_range = range(3, 11)
    xi_range = np.round(np.arange(0.01, 0.21, 0.5), 2)
    min_cluster_size_range = [None, *range(2, 21)]

    total = len(min_samples_range) * len(xi_range) * len(min_cluster_size_range)
    best_noise = None
    best = []
    n_ok = 0

    grid = product(min_samples_range, xi_range, min_cluster_size_range)
    for min_samples, xi, min_cluster_size in tqdm(grid, total=total):
        model = OPTICS(
            min_samples=min_samples,
            cluster_method="xi",
            xi=float(xi),
            min_cluster_size=min_cluster_size,
            n_jobs=-1,
        )
        labels = model.fit_predict(X_processed)
        n_clusters, noise = count_clusters_noise(labels)
        if n_clusters != 2:
            continue

        n_ok += 1
        row = {
            "min_samples": min_samples,
            "xi": float(xi),
            "min_cluster_size": min_cluster_size,
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
    print("-" * 76)
    print(f"{'min_samples':<11} | {'xi':<6} | {'min_cluster_size':<16} | {'Noise'}")
    print("-" * 76)

    key = lambda r: (r["noise"], r["min_samples"], r["xi"], str(r["min_cluster_size"]))
    for r in sorted(best, key=key):
        xi_s = f"{r['xi']:.2f}"
        mcs_s = str(r["min_cluster_size"])
        print(f"{r['min_samples']:<11} | {xi_s:<6} | {mcs_s:<16} | {r['noise']}")


if __name__ == "__main__":
    search_optics_params()


