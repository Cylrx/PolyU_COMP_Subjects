import logging
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PowerTransformer

from autogluon.common.features.types import R_INT, R_FLOAT
from autogluon.features.generators.abstract import AbstractFeatureGenerator


logger = logging.getLogger(__name__)


class YeoJohnsonFeatureGenerator(AbstractFeatureGenerator):
    """
    Apply Yeo-Johnson power transform to numeric columns.
    - Operates only on raw numeric columns (R_INT, R_FLOAT).
    - Skips columns that look discrete (low cardinality) or constant.
    - NaN values are kept as NaN (fit/transform are applied on non-null entries only).
    - Does not standardize after transform (standardize=False) to avoid altering scale beyond power transform.
    """

    def __init__(
        self,
        inplace: bool = False,
        unique_count_threshold: int = 20,
        unique_ratio_threshold: float = 0.05,
        standardize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inplace = inplace
        self.unique_count_threshold = unique_count_threshold
        self.unique_ratio_threshold = unique_ratio_threshold
        self.standardize = standardize

        self._numeric_features: List[str] = []
        self._col_to_transformer: Dict[str, PowerTransformer] = {}


    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        # Accept all columns; operate on numeric internally
        return dict()
    
    
    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        self._numeric_features = self.feature_metadata_in.get_features(valid_raw_types=[R_INT, R_FLOAT])
        X_out = X if self.inplace else X.copy()

        for col in self._numeric_features:
            s = pd.to_numeric(X_out[col], errors="coerce")
            valid_mask = s.notna()
            n_valid = int(valid_mask.sum())
            if n_valid <= 1:
                # Not enough data to fit transformer
                continue

            s_valid = s[valid_mask]
            n_unique = int(s_valid.nunique(dropna=True))
            if n_unique <= 1:
                # Constant column: skip
                continue

            if self._looks_discrete(n_unique=n_unique, n_valid=n_valid):
                # Likely a discrete/categorical-coded numeric; skip transform
                continue

            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=self.standardize)
                transformed = pt.fit_transform(s_valid.to_numpy().reshape(-1, 1)).reshape(-1)
            except Exception as e:
                logger.warning(f"YeoJohnson transform failed for column '{col}': {e}")
                continue

            # Build final column values in a float array, then assign once to avoid dtype warnings
            new_vals = s.to_numpy(dtype=np.float64, copy=True)
            new_vals[valid_mask.to_numpy()] = transformed
            # Ensure dtype via DataFrame.astype to avoid incompatible-dtype warnings on setitem
            if str(X_out.dtypes[col]) != "float64":
                X_out = X_out.astype({col: np.float64}, copy=False)
            X_out.iloc[:, X_out.columns.get_loc(col)] = new_vals
            self._col_to_transformer[col] = pt

        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        X_out = X if self.inplace else X.copy()
        for col, pt in self._col_to_transformer.items():
            if col not in X_out.columns:
                continue
            s = pd.to_numeric(X_out[col], errors="coerce")
            valid_mask = s.notna()
            if not valid_mask.any():
                continue
            try:
                transformed = pt.transform(s[valid_mask].to_numpy().reshape(-1, 1)).reshape(-1)
            except Exception as e:
                logger.warning(f"YeoJohnson transform (inference) failed for column '{col}': {e}")
                continue
            new_vals = s.to_numpy(dtype=np.float64, copy=True)
            new_vals[valid_mask.to_numpy()] = transformed
            if str(X_out.dtypes[col]) != "float64":
                X_out = X_out.astype({col: np.float64}, copy=False)
            X_out.iloc[:, X_out.columns.get_loc(col)] = new_vals
        return X_out

    def _looks_discrete(self, n_unique: int, n_valid: int) -> bool:
        # Heuristic: treat as discrete if absolute cardinality small OR ratio small
        ratio = (n_unique / n_valid) if n_valid > 0 else 0.0
        return (n_unique <= self.unique_count_threshold) or (ratio <= self.unique_ratio_threshold)

    def _more_tags(self):
        return {"feature_interactions": False}


class ZeroCenterFeatureGenerator(AbstractFeatureGenerator):
    """
    Subtract per-column mean (computed on non-null entries) from numeric columns.
    - Operates only on raw numeric columns (R_INT, R_FLOAT).
    - Skips columns that look discrete (low cardinality) or constant.
    - NaN values are kept as NaN.
    """

    def __init__(
        self,
        inplace: bool = False,
        unique_count_threshold: int = 20,
        unique_ratio_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inplace = inplace
        self.unique_count_threshold = unique_count_threshold
        self.unique_ratio_threshold = unique_ratio_threshold

        self._numeric_features: List[str] = []
        self._col_to_mean: Dict[str, float] = {}

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        # Accept all columns; operate on numeric internally
        return dict()

    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        self._numeric_features = self.feature_metadata_in.get_features(valid_raw_types=[R_INT, R_FLOAT])
        X_out = X if self.inplace else X.copy()

        for col in self._numeric_features:
            s = pd.to_numeric(X_out[col], errors="coerce")
            valid_mask = s.notna()
            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                continue
            s_valid = s[valid_mask]
            n_unique = int(s_valid.nunique(dropna=True))
            if n_unique <= 1:
                # Constant column; subtracting mean would yield zeros; skip to avoid creating zero-only columns early.
                continue
            if self._looks_discrete(n_unique=n_unique, n_valid=n_valid):
                # Likely discrete; skip normalization
                continue
            mu = float(s_valid.mean())
            self._col_to_mean[col] = mu
            # Build final float array and assign once
            new_vals = s.to_numpy(dtype=np.float64, copy=True)
            new_vals[valid_mask.to_numpy()] = (s.loc[valid_mask] - mu).to_numpy()
            if str(X_out.dtypes[col]) != "float64":
                X_out = X_out.astype({col: np.float64}, copy=False)
            X_out.iloc[:, X_out.columns.get_loc(col)] = new_vals

        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        X_out = X if self.inplace else X.copy()
        for col, mu in self._col_to_mean.items():
            if col not in X_out.columns:
                continue
            s = pd.to_numeric(X_out[col], errors="coerce")
            valid_mask = s.notna()
            if not valid_mask.any():
                continue
            new_vals = s.to_numpy(dtype=np.float64, copy=True)
            new_vals[valid_mask.to_numpy()] = (s.loc[valid_mask] - mu).to_numpy()
            if str(X_out.dtypes[col]) != "float64":
                X_out = X_out.astype({col: np.float64}, copy=False)
            X_out.iloc[:, X_out.columns.get_loc(col)] = new_vals
        return X_out

    def _looks_discrete(self, n_unique: int, n_valid: int) -> bool:
        ratio = (n_unique / n_valid) if n_valid > 0 else 0.0
        return (n_unique <= self.unique_count_threshold) or (ratio <= self.unique_ratio_threshold)

    def _more_tags(self):
        return {"feature_interactions": False}
