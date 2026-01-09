import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT, R_CATEGORY, R_BOOL
from autogluon.features.generators.abstract import AbstractFeatureGenerator


logger = logging.getLogger(__name__)


class KNNFeatureGenerator(AbstractFeatureGenerator):
    """
    Fill missing values using KNN on a mixed numeric+categorical space.
    - Numeric: min-max to [0,1], missing for distance uses column median (fit-time).
    - Categorical: one-hot per column (+ sentinel '__nan__'), scaled by 0.5 so each original column contributes in [0,1].
    - Distance: L1 (Manhattan).
    - Imputation:
        * Numeric -> (weighted) mean over neighbors that have non-null values, fallback to column median.
        * Categorical -> (weighted) mode over neighbors that have non-null values, fallback to column mode.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",  # 'uniform' | 'distance'
        metric: str = "manhattan",
        inplace: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.inplace = inplace

        # Fit artifacts
        self._numeric_features: List[str] = []
        self._categorical_features: List[str] = []
        self._raw_types: Dict[str, str] = {}
        self._num_min: Dict[str, float] = {}
        self._num_max: Dict[str, float] = {}
        self._num_median: Dict[str, float] = {}
        self._cat_categories: Dict[str, List] = {}
        self._cat_mode: Dict[str, object] = {}
        self._sentinel = "__nan__"

        self._X_train: DataFrame = None
        self._train_matrix: np.ndarray = None
        self._nn: NearestNeighbors = None

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        # Use all columns; impute only missing entries, keep names/dtypes.
        return dict()

    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        self._raw_types = self.feature_metadata_in.type_map_raw
        self._numeric_features = [
            f for f in self.features_in if self.feature_metadata_in.get_feature_type_raw(f) in (R_INT, R_FLOAT)
        ]
        self._categorical_features = [
            f
            for f in self.features_in
            if self.feature_metadata_in.get_feature_type_raw(f) in (R_OBJECT, R_CATEGORY, R_BOOL)
        ]

        # Numeric stats
        for col in self._numeric_features:
            s = X[col]
            self._num_min[col] = float(np.nanmin(s.values)) if s.size else 0.0
            self._num_max[col] = float(np.nanmax(s.values)) if s.size else 1.0
            self._num_median[col] = float(np.nanmedian(s.values)) if s.size else 0.0

        # Categorical categories and mode (exclude NaN for mode)
        for col in self._categorical_features:
            col_vals = X[col]
            if str(col_vals.dtype) == "category":
                cats = list(col_vals.cat.categories)
            else:
                cats = list(pd.Series(col_vals.unique()).dropna().tolist())
            # Ensure deterministic order
            try:
                cats.sort()
            except Exception:
                pass
            # Include sentinel for NaN/unknown
            if self._sentinel not in cats:
                cats.append(self._sentinel)
            self._cat_categories[col] = cats
            # Mode excluding NaN; fallback to sentinel if none
            mode_series = col_vals.dropna()
            self._cat_mode[col] = mode_series.mode().iloc[0] if not mode_series.empty else self._sentinel

        # Train matrix and NN index
        self._X_train = X[self.features_in].copy()
        self._train_matrix = self._to_metric_matrix(self._X_train, is_fit=True)
        n_neighbors = min(self.n_neighbors + 1, len(self._train_matrix))  # +1 to drop self-neighbor
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric, algorithm="auto")
        self._nn.fit(self._train_matrix)

        # Impute training data
        X_imputed = self._impute_dataframe(self._X_train, use_training_neighbors=True)
        return X_imputed, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._impute_dataframe(X, use_training_neighbors=False)

    # ---- Helpers ----
    def _to_metric_matrix(self, X: DataFrame, is_fit: bool) -> np.ndarray:
        parts: List[np.ndarray] = []

        if self._numeric_features:
            num = X[self._numeric_features].astype(float)
            # Fill for distance with column median
            for col in self._numeric_features:
                v = num[col].values
                v = np.where(np.isnan(v), self._num_median[col], v)
                num[col] = v
            # Min-max to [0,1], constant -> 0
            num_arr = num.to_numpy(copy=False)
            for j, col in enumerate(self._numeric_features):
                mn, mx = self._num_min[col], self._num_max[col]
                denom = mx - mn
                if denom > 0:
                    num_arr[:, j] = (num_arr[:, j] - mn) / denom
                else:
                    num_arr[:, j] = 0.0
            parts.append(num_arr.astype(np.float64, copy=False))

        if self._categorical_features:
            cat_blocks: List[np.ndarray] = []
            for col in self._categorical_features:
                cats = self._cat_categories[col]
                # Build one-hot including sentinel, each col contributes max 1 after 0.5 scaling
                col_vals = X[col]
                # Map values (unknown -> sentinel)
                vals = col_vals.copy()
                if str(vals.dtype) == "category":
                    vals = vals.astype(object)
                vals = vals.where(~vals.isna(), self._sentinel)
                # Unknown to sentinel
                mask_unknown = ~vals.isin(cats)
                if mask_unknown.any():
                    vals.loc[mask_unknown] = self._sentinel
                # One-hot
                oh = np.zeros((len(vals), len(cats)), dtype=np.float64)
                cat_index = {c: i for i, c in enumerate(cats)}
                # Vectorized placement via map to indices
                idx = vals.map(cat_index).to_numpy()
                row_idx = np.arange(len(vals))
                oh[row_idx, idx] = 1.0
                # scale by 0.5 so per-original-column contribution is in [0,1]
                oh *= 0.5
                cat_blocks.append(oh)
            parts.append(np.concatenate(cat_blocks, axis=1) if len(cat_blocks) > 1 else cat_blocks[0])

        if not parts:
            return np.zeros((len(X), 0), dtype=np.float64)
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    def _kneighbors(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dists, inds = self._nn.kneighbors(M, return_distance=True)
        # Drop first neighbor (self) when using training X
        if dists.shape[1] > self.n_neighbors:
            dists, inds = dists[:, 1:], inds[:, 1:]
        return dists, inds

    def _neighbor_weights(self, dists: np.ndarray) -> np.ndarray:
        if self.weights == "uniform":
            w = np.ones_like(dists)
        elif self.weights == "distance":
            w = 1.0 / (dists + 1e-12)
        else:
            raise ValueError(f"Invalid weights: {self.weights}")
        # Normalize per-row
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0.0] = 1.0
        return w / w_sum

    def _impute_dataframe(self, X: DataFrame, use_training_neighbors: bool) -> DataFrame:
        X_in = X if self.inplace else X.copy()
        if self._train_matrix is None:
            return X_in

        M = self._to_metric_matrix(X_in, is_fit=False)
        dists, inds = self._kneighbors(M)
        w = self._neighbor_weights(dists)

        # Access to neighbor rows from training data
        neigh_X = self._X_train

        # Numeric columns
        for col in self._numeric_features:
            col_vals = X_in[col].to_numpy()
            nan_mask = np.isnan(col_vals)
            if not nan_mask.any():
                continue
            # Gather neighbor values
            neigh_vals = neigh_X[col].to_numpy()[inds]  # shape (n_rows, K)
            valid = ~np.isnan(neigh_vals)
            # Weighted mean per row over valid neighbors
            weighted = (neigh_vals * w) * valid
            denom = (w * valid).sum(axis=1)
            denom[denom == 0.0] = np.nan
            filled = np.nansum(weighted, axis=1) / denom
            # Fallback to column median where still nan
            filled = np.where(np.isnan(filled), self._num_median[col], filled)
            # Cast back to int if needed
            if self._raw_types.get(col) == R_INT:
                filled = np.rint(filled).astype(neigh_X[col].dtype, copy=False)
            col_vals[nan_mask] = filled[nan_mask]
            X_in[col] = col_vals

        # Categorical columns
        for col in self._categorical_features:
            s = X_in[col]
            nan_mask = s.isna().to_numpy()
            if not nan_mask.any():
                continue
            neigh_vals = neigh_X[col].to_numpy()[inds]
            # For each row, compute weighted vote ignoring nan
            out = s.to_numpy(dtype=object)
            for i in np.where(nan_mask)[0]:
                vals = neigh_vals[i]
                ww = w[i]
                mask = pd.notna(vals)
                if not mask.any():
                    out[i] = self._cat_mode[col]
                    continue
                vals = vals[mask]
                ww = ww[mask]
                # Weighted mode
                uniq, inv = np.unique(vals, return_inverse=True)
                scores = np.zeros(len(uniq), dtype=np.float64)
                np.add.at(scores, inv, ww)
                out[i] = uniq[np.argmax(scores)]
            X_in[col] = out
            # If original dtype was categorical, try restore category dtype the safest way
            if str(self._X_train[col].dtype) == "category":
                cats = self._cat_categories[col]
                # Ensure sentinel not present in final categorical dtype
                if self._sentinel in cats:
                    cats_no_sentinel = [c for c in cats if c != self._sentinel]
                else:
                    cats_no_sentinel = cats
                X_in[col] = pd.Categorical(X_in[col], categories=cats_no_sentinel)

        return X_in

    def _more_tags(self):
        return {"feature_interactions": False}


