import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from autogluon.common.features.types import R_CATEGORY
from autogluon.features.generators.abstract import AbstractFeatureGenerator


logger = logging.getLogger(__name__)


class OneHotFeatureGenerator(AbstractFeatureGenerator):
    """
    Convert categorical columns (raw dtype 'category') into multiple boolean columns via one-hot encoding.
    - Column names: {prefix}__{col}__{cat}
    - Values are bool; NaN maps to all False unless dummy_na=True.
    """

    def __init__(
        self,
        drop_original: bool = True,
        dummy_na: bool = False,
        drop_first: bool = False,
        prefix: str = "oh",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.drop_original = drop_original
        self.dummy_na = dummy_na
        self.drop_first = drop_first
        self.prefix = prefix

        self._cat_cols: List[str] = []
        self._col_to_cats: Dict[str, List] = {}
        self._col_to_newcols: Dict[str, List[str]] = {}

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=[R_CATEGORY])

    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        self._cat_cols = list(self.features_in)
        X_out = self._generate(X)
        return X_out, dict()

    def _transform(self, X: DataFrame) -> DataFrame:
        return self._generate(X)

    def _generate(self, X: DataFrame) -> DataFrame:
        parts: List[DataFrame] = []
        for col in self._cat_cols:
            s = X[col]
            # categories
            if not self._col_to_cats:
                pass
            if col not in self._col_to_cats:
                cats = list(s.cat.categories)
                if self.drop_first and len(cats) > 0:
                    cats = cats[1:]
                self._col_to_cats[col] = cats
                newcols = [self._colname(col, c) for c in cats]
                if self.dummy_na:
                    newcols.append(self._colname(col, "__nan__"))
                self._col_to_newcols[col] = newcols
            cats = self._col_to_cats[col]
            newcols = self._col_to_newcols[col]

            # boolean one-hot
            block = DataFrame(index=X.index)
            if cats:
                for c in cats:
                    block[self._colname(col, c)] = (s == c).fillna(False).astype(bool)
            if self.dummy_na:
                block[self._colname(col, "__nan__")] = s.isna()
            parts.append(block)

        X_out = pd.concat(parts, axis=1) if parts else DataFrame(index=X.index)
        if not self.drop_original:
            X_out = pd.concat([X_out, X[self._cat_cols]], axis=1)
        return X_out

    def _colname(self, col: str, cat) -> str:
        return f"{self.prefix}__{col}__{str(cat)}"

    def _more_tags(self):
        return {"feature_interactions": False}


