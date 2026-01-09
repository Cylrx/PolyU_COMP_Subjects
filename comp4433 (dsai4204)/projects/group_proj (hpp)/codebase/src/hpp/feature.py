import pandas as pd
from autogluon.features.generators import (
    PipelineFeatureGenerator,
    IdentityFeatureGenerator,
    CategoryFeatureGenerator,
    DatetimeFeatureGenerator,
    FillNaFeatureGenerator,
    DropUniqueFeatureGenerator,
)
from autogluon.common.features.types import (
    R_OBJECT, R_CATEGORY, R_DATETIME,
    R_INT, R_FLOAT,
)
from src.hpp.generators import (
    KNNFeatureGenerator,
    OneHotFeatureGenerator,
    YeoJohnsonFeatureGenerator,
    ZeroCenterFeatureGenerator,
    CustomizedFeatureGenerator
)
from typing import Optional

from src.hpp.config import cfg


class TogglePipeline(PipelineFeatureGenerator):
    def __init__(
        self,
        # True=feature engineering, False=as-is
        categorical_engineer=True,
        yeo_johnson_engineer=False,
        zero_center_engineer=False,
        datetime_engineer=True,
        fill_na_engineer=True,
        drop_unique_engineer=True,
        knn_impute_engineer=False,
        onehot_engineer=False,
        customized_engineer=False,
        knn_kwargs=None,
        onehot_kwargs=None,
        **kwargs,
    ):
        self.categorical_engineer = categorical_engineer
        self.datetime_engineer = datetime_engineer
        self.fill_na_engineer = fill_na_engineer
        self.yeo_johnson_engineer = yeo_johnson_engineer
        self.zero_center_engineer = zero_center_engineer
        self.drop_unique_engineer = drop_unique_engineer
        self.knn_impute_engineer = knn_impute_engineer
        self.onehot_engineer = onehot_engineer
        self.customized_engineer = customized_engineer
        self.knn_kwargs = {} if knn_kwargs is None else knn_kwargs
        self.onehot_kwargs = {} if onehot_kwargs is None else onehot_kwargs
        pre_g, mid_g, post_g = self._build_groups()
        # Note:
        # - pre_g runs sequentially before generators
        # - generators is a list of groups; each group runs in parallel, groups run sequentially
        # - post_g runs sequentially after generators
        generators = [mid_g]  # single-stage main group
        super().__init__(generators=generators, pre_generators=pre_g, post_generators=post_g, **kwargs)

    def _build_groups(self):
        pre_g, mid_g, post_g = [], [], []

        # ---- pre stage ----
        # Order: Yeo-Johnson -> Zero-Center -> KNN -> FillNa
        # if customized feature is enabled, the order will be changed to:
        # Customized -> Yeo-Johnson -> Zero-Center 
        
        if self.customized_engineer :
            pre_g.append(CustomizedFeatureGenerator(inplace=True))  # fill na and new feature creation        
        if self.yeo_johnson_engineer:
            pre_g.append(YeoJohnsonFeatureGenerator(inplace=True))
        if self.zero_center_engineer:
            pre_g.append(ZeroCenterFeatureGenerator(inplace=True))
        if self.knn_impute_engineer:
            pre_g.append(KNNFeatureGenerator(**self.knn_kwargs))
        if self.fill_na_engineer:
            pre_g.append(FillNaFeatureGenerator(inplace=True))


        # ---- main stage (single group run in parallel, outputs concatenated) ----
        if self.categorical_engineer:
            mid_g.append(CategoryFeatureGenerator())
        else:
            mid_g.append(IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_OBJECT, R_CATEGORY])))

        if self.datetime_engineer:
            mid_g.append(DatetimeFeatureGenerator())
        else:
            mid_g.append(IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_DATETIME])))

        # Always carry numeric features forward
        mid_g.append(IdentityFeatureGenerator(infer_features_in_args=dict(valid_raw_types=[R_INT, R_FLOAT])))

        # ---- post stage ----
        if self.onehot_engineer:
            post_g.append(OneHotFeatureGenerator(**self.onehot_kwargs))
        if self.drop_unique_engineer:
            post_g.append(DropUniqueFeatureGenerator())
        return pre_g, mid_g, post_g


def feature_engineering(train_data: pd.DataFrame, test_data: pd.DataFrame, overlays: Optional[list[str]] = None, is_eval: bool = False):
    """
    Args: 
        train_data: Training set dataframe, must contain label columns
        test_data: Test set / Evaluation set dataframe. The latter should contain label columns
        overlays: list[str]
        is_eval: whether `test_data` is an holdout set (contain label) or test set (don't contain label).
    Returns:
        train_out: pd.DataFrame
        test_out: pd.DataFrame
    """

    t, u = len(train_data), len(test_data)
    diff = [c for c in train_data.columns if c not in test_data.columns]

    assert len(diff) == (1 if not is_eval else 0)
    label = diff[0]
    assert label in train_data.columns and label not in test_data.columns
    X = pd.concat([train_data.drop(columns=[label]), test_data], ignore_index=True)
    n = t + u
    assert len(X) == n

    exp = cfg.exp(overlays)
    X = TogglePipeline(**exp['feat']).fit_transform(X)

    assert len(X) == n and list(X.index) == list(range(n))
    train_out = X.iloc[:t].copy().reset_index(drop=True)
    test_out = X.iloc[t:].copy().reset_index(drop=True)
    train_out[label] = train_data[label].to_numpy()

    if is_eval: 
        test_out[label] = test_data[label].to_numpy()
        return train_out, test_out
    
    assert label in train_out.columns and label not in test_out.columns
    return train_out, test_out