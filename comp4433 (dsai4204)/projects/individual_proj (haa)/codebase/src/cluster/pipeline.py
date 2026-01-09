from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import abs_path, print_good, print_info

from .embedding import Embedder


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame


@dataclass(frozen=True)
class XY:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class Matrix:
    X: np.ndarray
    y: pd.Series
    X_raw: pd.DataFrame


@dataclass(frozen=True)
class Space:
    X: np.ndarray
    y: pd.Series
    embedding: str
    X_raw: pd.DataFrame


@dataclass(frozen=True)
class ClusterOutputs:
    X: np.ndarray
    y: pd.Series
    embedding: str
    y_preds: dict[str, pd.Series | np.ndarray]
    metrics: dict[str, dict[str, float | int]]


class Pipe:
    def __init__(self, *filters: Callable[[Any], Any]):
        self._filters = list(filters)

    def run(self, x: Any) -> Any:
        for f in self._filters:
            x = f(x)
        return x


class LoadCsv:
    def __init__(self, path: str):
        self._path = path

    def __call__(self, _: None) -> Dataset:
        df = pd.read_csv(abs_path(self._path))
        return Dataset(df)


class InitStats:
    def __init__(self, init_fn: Callable[[pd.DataFrame], None]):
        self._init_fn = init_fn

    def __call__(self, data: Dataset) -> Dataset:
        self._init_fn(data.df)
        return data


class SplitXY:
    def __init__(self, label_col: str):
        self._label_col = label_col

    def __call__(self, data: Dataset) -> XY:
        df = data.df.reset_index(drop=True)
        if self._label_col not in df.columns:
            raise ValueError(f"Label column '{self._label_col}' not found in dataset")
        return XY(df.drop(columns=[self._label_col]), df[self._label_col])


class Transform:
    def __init__(self, preprocessor_factory: Callable[[], Any]):
        self._factory = preprocessor_factory

    def __call__(self, xy: XY) -> Matrix:
        pre = self._factory()
        X = np.asarray(pre.fit_transform(xy.X))
        return Matrix(X=X, y=xy.y, X_raw=xy.X.reset_index(drop=True))


class Embed:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def __call__(self, m: Matrix) -> Space:
        X = np.asarray(m.X)
        y = np.asarray(m.y)
        Z = self._embedder.fit_transform(X, y)
        return Space(X=np.asarray(Z), y=m.y, embedding=self._embedder.name, X_raw=m.X_raw)


class Cluster:
    def __init__(
        self,
        model_names: Sequence[str],
        model_factory: Callable[[str], tuple[Any, Mapping[str, Any]]],
        metrics_fn: Callable[[pd.Series, np.ndarray], Mapping[str, float]],
        align_fn: Callable[[pd.Series, np.ndarray], np.ndarray],
        *,
        verbose: bool,
    ):
        self._model_names = list(model_names)
        self._model_factory = model_factory
        self._metrics_fn = metrics_fn
        self._align_fn = align_fn
        self._verbose = bool(verbose)

    def __call__(self, space: Space) -> ClusterOutputs:
        y_preds: dict[str, pd.Series | np.ndarray] = {"ground_truth": space.y}
        metrics: dict[str, dict[str, float | int]] = {}

        print_info(f"Embedding: {space.embedding}")
        for name in self._model_names:
            model, params = self._model_factory(name)
            y_raw = _fit_predict(model, space.X)
            y_pred = self._align_fn(space.y, y_raw)

            y_preds[name] = np.asarray(y_pred).reshape(-1)
            base = dict(self._metrics_fn(space.y, np.asarray(y_pred)))
            extra = _cluster_stats(y_pred) if self._verbose and _needs_cluster_stats(params) else {}
            out = base | extra
            metrics[name] = out
            _print_model_summary(name, out)

        return ClusterOutputs(
            X=np.asarray(space.X),
            y=space.y,
            embedding=space.embedding,
            y_preds=y_preds,
            metrics=metrics,
        )


class Report:
    def __init__(self, report_fn: Callable[..., None]):
        self._report_fn = report_fn

    def __call__(self, out: ClusterOutputs) -> ClusterOutputs:
        self._report_fn(out.X, out.y_preds, tag=out.embedding)
        return out


def _fit_predict(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "fit_predict"):
        return np.asarray(model.fit_predict(X))
    if not hasattr(model, "fit") or not hasattr(model, "predict"):
        raise TypeError(f"{type(model).__name__} must implement fit_predict or fit+predict")
    model.fit(X)
    return np.asarray(model.predict(X))


def _needs_cluster_stats(params: Mapping[str, Any]) -> bool:
    return ("n_clusters" not in params) and ("n_components" not in params)


def _cluster_stats(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels).reshape(-1)
    noise = int(np.sum(labels < 0))
    n_clusters = int(np.unique(labels[labels >= 0]).size)
    return {"Clusters": n_clusters, "Noise": noise}


def _print_model_summary(model_name: str, stats: Mapping[str, float | int]) -> None:
    print_good(f"Model: {model_name}")
    for k, v in stats.items():
        print(f"\t{k}: {_fmt(v)}")


def _fmt(v: object) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)
