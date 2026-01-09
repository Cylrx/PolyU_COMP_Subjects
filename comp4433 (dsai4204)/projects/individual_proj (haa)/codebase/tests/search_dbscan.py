import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import cfg
from src.statistics import stats
from src.cluster.preprocess import get_preprocessor

def search_dbscan_params():
    print("Searching for DBSCAN parameters that yield exactly 2 clusters...")

    # Load data
    data = pd.read_csv(cfg.paths['data'])
    stats.init(data)

    target_col = cfg.dataset['label']
    X = data.drop(columns=[target_col])
    
    # Preprocessing
    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Search space
    eps_range = np.arange(0.5, 2.5, 0.01)
    min_samples_range = range(2, 50)
    
    results = []

    for eps in tqdm(eps_range):
        for min_samples in min_samples_range:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = model.fit_predict(X_processed)

            # Count clusters (excluding noise -1)
            labels = set(y_pred)
            if -1 in labels:
                labels.remove(-1)
            n_clusters = len(labels)
            n_noise = np.sum(y_pred == -1)

            if n_clusters == 2:
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'noise': n_noise
                })

    if not results:
        print("No parameters found that yield exactly 2 clusters.")
        return

    # Sort by noise level (descending)
    results.sort(key=lambda x: x['noise'], reverse=True)

    print(f"\nFound {len(results)} configurations yielding 2 clusters (sorted by noise descending):")
    print("-" * 60)
    print(f"{'eps':<10} | {'min_samples':<15} | {'Noise Points'}")
    print("-" * 60)
    
    for res in results:
        print(f"{res['eps']:<10.2f} | {res['min_samples']:<15} | {res['noise']}")

if __name__ == "__main__":
    search_dbscan_params()
