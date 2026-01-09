from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from src.config import cfg
from src.utils import get_categorical_indices


class Embedder(Protocol):
    name: str

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NoEmbedding:
    name: str = "none"

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(X)


@dataclass(frozen=True)
class TabPFNEmbedder:
    model_kwargs: Mapping[str, Any]
    n_fold: int = 0
    reduce: str = "mean"  # {"mean", "first"}
    data_source: str = "train"  # {"train", "test"}
    train_size: float | int = 1.0  # 1.0 => use full data as "train"
    name: str = "tabpfn"

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = _as_2d(X, "X")
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"y length mismatch: {y.shape[0]} vs X has {X.shape[0]} rows")

        # Local import: keep 'none' path lightweight.
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.embedding import TabPFNEmbedding

        X_train, y_train = _pick_train_split(
            X,
            y,
            train_size=self.train_size,
            seed=int(dict(self.model_kwargs).get("random_state", 0)),
        )

        clf = TabPFNClassifier(**dict(self.model_kwargs))
        extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=int(self.n_fold))
        E = extractor.get_embeddings(X_train, y_train, X, data_source=str(self.data_source))
        Z = _reduce_embeddings(E, self.reduce)
        if Z.shape[0] != X.shape[0]:
            raise ValueError(
                f"Embeddings rows {Z.shape[0]} != X rows {X.shape[0]}. "
                f"Hint: if you set train_size<1, use data_source='test' to embed full X."
            )
        return Z


def make_embedder(name: str, *, params: Mapping[str, Any] | None, seed: int) -> Embedder:
    match str(name).lower():
        case "none" | "identity":
            return NoEmbedding()
        case "tabpfn":
            p = dict(params or {})
            model = _tabpfn_model_kwargs(seed, p.pop("model", None))
            return TabPFNEmbedder(
                model_kwargs=model,
                n_fold=int(p.pop("n_fold", 0)),
                reduce=str(p.pop("reduce", "mean")),
                data_source=str(p.pop("data_source", "train")),
                train_size=p.pop("train_size", 1.0),
            )
        case _:
            raise ValueError(f"Unknown embedding: {name}")


def _tabpfn_model_kwargs(seed: int, overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_estimators": 1,
        "random_state": int(seed),
        "device": "auto",
        "categorical_features_indices": get_categorical_indices(),
    }
    if overrides is None:
        return base
    if not isinstance(overrides, Mapping):
        raise TypeError("cluster_embedding_params.tabpfn.model must be a mapping")
    return base | dict(overrides)


def _as_2d(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    if X.shape[0] <= 0:
        raise ValueError(f"{name} is empty")
    return X


def _reduce_embeddings(E: np.ndarray, how: str) -> np.ndarray:
    E = np.asarray(E)
    if E.ndim == 2:
        return E
    if E.ndim != 3:
        raise ValueError(f"Unexpected embeddings shape {E.shape}; expected (n, d) or (m, n, d)")

    match str(how).lower():
        case "mean" | "avg":
            return E.mean(axis=0)
        case "first" | "0":
            return E[0]
        case _:
            raise ValueError(f"Unknown reduce='{how}'. Use 'mean' or 'first'.")


def _pick_train_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_size: float | int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if train_size == 1 or train_size == 1.0:
        return X, y

    from sklearn.model_selection import train_test_split

    X_tr, _, y_tr, _ = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=int(seed),
        stratify=y,
    )
    return np.asarray(X_tr), np.asarray(y_tr).reshape(-1)


def tabpfn_predict_proba_y1(X_raw: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Computes TabPFN P(y=1) for the entire dataset (one-shot).
    Uses portable settings from cfg.cluster_embedding_params.tabpfn.model.
    """
    from tabpfn import TabPFNClassifier

    X = np.asarray(X_raw)
    y_in = np.asarray(y).reshape(-1)

    if X.shape[0] != y_in.shape[0]:
        raise ValueError(f"X_raw rows {X.shape[0]} != y rows {y_in.shape[0]}")

    # 1. Get portable kwargs
    # We reuse the same model params used for embeddings (device, n_estimators, seed)
    # but we don't need the embedding-specific params (n_fold, reduce, etc.)
    emb_cfg = cfg.cluster_embedding_params.get("tabpfn", {})
    model_params = _tabpfn_model_kwargs(cfg.seed, emb_cfg.get("model"))

    # 2. Fit & Predict
    clf = TabPFNClassifier(**model_params)
    clf.fit(X, y_in)

    # 3. Extract P(y=1)
    # predict_proba returns (N, 2) for binary classification
    probs = clf.predict_proba(X)

    if probs.shape[1] < 2:
        raise ValueError(f"TabPFN output shape {probs.shape} implies <2 classes?")

    p_y1 = probs[:, 1]

    if np.any(np.isnan(p_y1)):
        raise ValueError("TabPFN predicted NaNs in probability output")

    return p_y1
