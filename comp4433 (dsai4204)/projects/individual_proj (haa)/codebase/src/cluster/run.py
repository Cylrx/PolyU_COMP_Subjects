from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures

from src.config import cfg
from src.statistics import stats
from src.utils import abs_path, print_error

from .embedding import make_embedder, tabpfn_predict_proba_y1
from .pipeline import (
    Cluster,
    Embed,
    InitStats,
    LoadCsv,
    Matrix,
    Pipe,
    Space,
    SplitXY,
    Transform,
)
from .preprocess import get_preprocessor
from .report import generate_color_verification, generate_report

_MODEL_REGISTRY: dict[str, Any] = {
    "kmeans": KMeans,
    "dbscan": DBSCAN,
    "hdbscan": HDBSCAN,
    "agglomerative": AgglomerativeClustering,
    "ward": AgglomerativeClustering,
    "gmm": GaussianMixture,
    "spectral": SpectralClustering,
    "optics": OPTICS,
}


def _make_cluster_model(
    model_name: str,
    models_cfg: Mapping[str, Any],
    *,
    cfg_name: str,
) -> tuple[Any, Mapping[str, Any]]:
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")
    if model_name not in models_cfg:
        raise ValueError(f"Model config for '{model_name}' not found in {cfg_name}")
    params = models_cfg[model_name]
    return _MODEL_REGISTRY[model_name](**params), params


def make_cluster_model(model_name: str) -> tuple[Any, Mapping[str, Any]]:
    return _make_cluster_model(model_name, cfg.models, cfg_name="cfg.models")


def make_intra_cluster_model(model_name: str) -> tuple[Any, Mapping[str, Any]]:
    return _make_cluster_model(model_name, cfg.models_intra, cfg_name="cfg.models_intra")


def get_direct_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "ACC": accuracy_score(y_true, y_pred),
    }


def best_match_binary(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """
    Binary-only label alignment: pick {pred, flipped(pred)} that maximizes ACC.
    Noise labels (<0) are preserved.
    """
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {y.shape[0]} vs {p.shape[0]}")

    core = p[p >= 0]
    uniq = np.unique(core)
    if uniq.size != 2 or not np.array_equal(np.sort(uniq), np.array([0, 1])):
        return p

    flipped = p.copy()
    mask = flipped >= 0
    flipped[mask] = 1 - flipped[mask]
    return flipped if accuracy_score(y, flipped) > accuracy_score(y, p) else p


def _cluster_counts(y_pred: np.ndarray) -> dict[str, int]:
    y_pred = np.asarray(y_pred).reshape(-1)
    noise = int(np.sum(y_pred < 0))
    n_clusters = int(np.unique(y_pred[y_pred >= 0]).size)
    return {"Clusters": n_clusters, "Noise": noise}


def get_intra_metrics(_: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
    return _cluster_counts(y_pred)


def no_align(_: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    return np.asarray(y_pred).reshape(-1)


def _load_ensemble_logits(base: Matrix, *, proba_col: str = "p_y1_mean") -> np.ndarray:
    """
    Loads ensemble probabilities from cfg.paths["proba"] and returns logits L, shape (n, 1).

    We sort by 'idx' when present to ensure stable row alignment, and fail-fast on any
    mismatch with the clustering dataset.
    """

    def _sort_by_idx(df: pd.DataFrame) -> pd.DataFrame:
        if "idx" not in df.columns:
            return df.reset_index(drop=True)

        if df["idx"].isnull().any():
            raise ValueError("proba.csv contains NaN in column 'idx'")
        if not df["idx"].is_unique:
            raise ValueError("proba.csv column 'idx' must be unique")

        out = df.sort_values("idx").reset_index(drop=True)
        idx = out["idx"].to_numpy()
        if not np.array_equal(idx, np.arange(idx.shape[0])):
            raise ValueError("proba.csv column 'idx' must equal 0..N-1 after sorting")
        return out

    path = abs_path(cfg.paths["proba"])
    df = _sort_by_idx(pd.read_csv(path))
    if proba_col not in df.columns:
        raise ValueError(f"Missing probability column '{proba_col}' in {path}")

    p = df[proba_col].to_numpy(dtype=float).reshape(-1)
    n = int(np.asarray(base.X).shape[0])
    if p.shape[0] != n:
        raise ValueError(f"proba rows {p.shape[0]} != dataset rows {n}")

    if "y_true" in df.columns:
        y_true = df["y_true"].to_numpy().reshape(-1)
        y = np.asarray(base.y).reshape(-1)
        if y_true.shape[0] != y.shape[0] or not np.array_equal(y_true.astype(int), y.astype(int)):
            raise ValueError("proba.csv 'y_true' does not match dataset labels; check row alignment")

    if not np.all(np.isfinite(p)):
        raise ValueError("Probability column contains non-finite values")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("Probabilities must be strictly within (0, 1) to compute logits")

    logits = np.log(p) - np.log1p(-p)
    if not np.all(np.isfinite(logits)):
        raise ValueError("Computed logits contain non-finite values")

    return logits.reshape(-1, 1)


def _residualize(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Removes the best predictor of X from logits L using either polynomial or RBF regression.
    
    Method selection and parameters are read from cfg.cluster_orthogonalize:
    - Polynomial: regresses against [L, L^2, ..., L^order]
    - RBF: kernel ridge regression with automatic or grid-searched hyperparameters
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    L = np.asarray(L)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    if L.ndim != 2 or L.shape[1] != 1:
        raise ValueError(f"L must be shape (n, 1), got {L.shape}")
    if L.shape[0] != X.shape[0]:
        raise ValueError(f"Row mismatch: L has {L.shape[0]} rows but X has {X.shape[0]} rows")

    ortho_cfg = cfg.cluster_orthogonalize
    
    if ortho_cfg.get("use_rbf", False):
        rbf_cfg = ortho_cfg["rbf"]
        regressor = KernelRidge(
            kernel="rbf",
            gamma=rbf_cfg["gamma"],
            alpha=rbf_cfg["alpha"]
        )
        
        if rbf_cfg.get("grid_search", False):
            param_grid = {
                "gamma": rbf_cfg["grid_gamma"],
                "alpha": rbf_cfg["grid_alpha"]
            }
            regressor = GridSearchCV(
                regressor,
                param_grid,
                cv=rbf_cfg["cv_folds"],
                scoring="neg_mean_squared_error"
            )
        
        regressor.fit(L, X)
        return X - regressor.predict(L)
    
    # Polynomial path
    order = int(ortho_cfg["order"])
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    
    poly = PolynomialFeatures(degree=order, include_bias=False)
    L_poly = poly.fit_transform(L)
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(L_poly, X)
    return X - lr.predict(L_poly)


def _make_ortho_space(space: Space, L: np.ndarray) -> Space:
    return Space(
        X=_residualize(space.X, L),
        y=space.y,
        embedding=f"{space.embedding}_ortho",
        X_raw=space.X_raw,
    )


def _build_base_matrix() -> Matrix:
    """Loads and transforms data once."""
    pipe: Pipe = Pipe(
        LoadCsv(cfg.paths["data"]),
        InitStats(stats.init),
        SplitXY(cfg.dataset["label"]),
        Transform(get_preprocessor),
    )
    return pipe.run(None)


def _embed(base: Matrix, embedding: str) -> Space:
    """Applies embedding to base matrix."""
    params = cfg.cluster_embedding_params.get(embedding, {})
    embedder = make_embedder(embedding, params=params, seed=cfg.seed)
    return Embed(embedder)(base)


def _require_binary_y(y: pd.Series) -> None:
    uniq = set(np.unique(np.asarray(y).reshape(-1)))
    if uniq != {0, 1}:
        raise ValueError(f"Expected binary labels {{0,1}}, got {sorted(uniq)}")


def _iter_intra_spaces(space: Space) -> list[tuple[int, np.ndarray, Space]]:
    _require_binary_y(space.y)
    y = np.asarray(space.y).reshape(-1)

    out: list[tuple[int, np.ndarray, Space]] = []
    for cls in (0, 1):
        mask = y == cls
        if not bool(np.any(mask)):
            raise ValueError(f"Intra-class subset is empty for y=={cls}")
        
        sub = Space(
            X=np.asarray(space.X)[mask],
            y=space.y[mask].reset_index(drop=True),
            embedding=f"{space.embedding}_intra_y{cls}",
            X_raw=space.X_raw.loc[mask].reset_index(drop=True)
        )
        out.append((cls, mask, sub))
    return out


def _run_clustering(space: Space, *, intra: bool, binary_only: bool) -> None:
    model_factory = make_intra_cluster_model if intra else make_cluster_model
    metrics_fn = get_intra_metrics if intra else get_direct_metrics
    align_fn = no_align if intra else best_match_binary

    out = Cluster(
        cfg.cluster_models,
        model_factory,
        metrics_fn,
        align_fn,
        verbose=cfg.verbose,
    )(space)
    generate_report(out.X, out.y_preds, tag=out.embedding, binary_only=binary_only)


def _run_one_space(space: Space, p_y1: np.ndarray) -> None:
    _run_clustering(space, intra=False, binary_only=True)
    
    for cls, mask, sub in _iter_intra_spaces(space):
        # 1. Color-coding Verification (features + p_y1)
        # For y=0 subset, we show P(y=0) = 1 - P(y=1)
        # For y=1 subset, we show P(y=1)
        if cls == 0:
            proba_val = 1 - p_y1[mask]
            proba_lbl = "TabPFN P(y=0)"
        else:
            proba_val = p_y1[mask]
            proba_lbl = "TabPFN P(y=1)"

        generate_color_verification(
            sub.X, 
            sub.X_raw, 
            proba_val, 
            tag=sub.embedding,
            proba_label=proba_lbl,
        )
        
        # 2. Intra-clustering
        _run_clustering(sub, intra=True, binary_only=False)

def _run_one_embedding(base: Matrix, embedding: str, p_y1: np.ndarray) -> Space:
    space = _embed(base, embedding)
    _run_one_space(space, p_y1)
    return space


def main() -> None:
    # 1. Build base matrix once
    base = _build_base_matrix()
    
    # 2. Compute TabPFN P(y=1) once
    # We use this for all embeddings to have consistent "model confidence" coloring
    try:
        p_y1 = tabpfn_predict_proba_y1(base.X_raw, base.y)
    except Exception as e:
        print_error(f"Failed to compute TabPFN probabilities: {e}")
        raise

    logits = None
    if bool(cfg.cluster_orthogonalize["enable"]):
        try:
            logits = _load_ensemble_logits(base)
        except Exception as e:
            print_error(f"Failed to load ensemble logits for orthogonal variant: {e}")
            raise

    for embedding in cfg.cluster_embeddings:
        try:
            space = _run_one_embedding(base, str(embedding), p_y1)
            if logits is not None:
                _run_one_space(_make_ortho_space(space, logits), p_y1)
        except Exception as e:
            if "tabpfn" in str(e).lower():
                print_error("Failed to run TabPFN embeddings. Do you have access to TabPFN v2.5?")
                print_error(f"Read error message below for more details:\n{e}\n")
                return
            raise


if __name__ == "__main__":
    main()
