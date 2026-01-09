import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import NearestNeighbors

from autogluon.common.features.types import R_INT, R_FLOAT, R_OBJECT, R_CATEGORY, R_BOOL
from autogluon.features.generators.abstract import AbstractFeatureGenerator


logger = logging.getLogger(__name__)

class CustomizedFeatureGenerator(AbstractFeatureGenerator):
    """
    Customized feature generator for house price prediction.
    - Handles missing values systematically for different feature types
    - Creates new features through combinations and transformations
    - Operates on both categorical and numerical features
    - Preserves data types and handles edge cases
    """

    def __init__(self, inplace: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.inplace = inplace

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        # Accept all columns by default
        return dict()

    def _handle_missing_values(self, X: DataFrame) -> DataFrame:
        """Systematically handle missing values for different feature types"""
        
        # --- Categorical Features ---
        # 'NA' means 'No such facility', fill with 'None'
        na_means_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                        'BsmtFinType2', 'MasVnrType']
        for col in na_means_none:
            if col in X.columns:
                X[col] = X[col].fillna('None')

        # --- Numeric Features ---
        # Missing garage/basement related features mean no garage/basement
        zero_fill = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                    'BsmtHalfBath', 'MasVnrArea']
        for col in zero_fill:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # LotFrontage: Use neighborhood median
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )

        # --- Mode Fill ---
        # For categorical features with few missing values
        mode_fill = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
                    'Exterior2nd', 'SaleType', 'Functional']
        for col in mode_fill:
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])

        return X

    def _create_features(self, X: DataFrame) -> DataFrame:
        """Create new derived features to enhance model performance"""
        
        # --- Area Features ---
        if all(col in X.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        
        if all(col in X.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
            X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath']) +
                                  X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))
        
        porch_cols = ['OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF']
        if all(col in X.columns for col in porch_cols):
            X['Total_Porch_SF'] = sum(X[col] for col in porch_cols)

        # --- Time Features ---
        if all(col in X.columns for col in ['YrSold', 'YearBuilt']):
            X['Age'] = X['YrSold'] - X['YearBuilt']
            if 'YearRemodAdd' in X.columns:
                X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
            X['IsNew'] = (X['YearBuilt'] == X['YrSold']).astype(int)

        # --- Quality and Area Interactions ---
        if 'OverallQual' in X.columns:
            if 'GrLivArea' in X.columns:
                X['OverallQual_x_GrLivArea'] = X['OverallQual'] * X['GrLivArea']
            if 'TotalSF' in X.columns:
                X['OverallQual_x_TotalSF'] = X['OverallQual'] * X['TotalSF']

        # --- Binary Features ---
        binary_features = {
            'HasPool': 'PoolArea',
            'Has2ndFlr': '2ndFlrSF',
            'HasGarage': 'GarageArea',
            'HasBsmt': 'TotalBsmtSF',
            'HasFireplace': 'Fireplaces'
        }
        for new_col, source_col in binary_features.items():
            if source_col in X.columns:
                X[new_col] = (X[source_col] > 0).astype(int)

        # --- Seasonal Features ---
        if 'MoSold' in X.columns:
            def map_season(month):
                if month in [12, 1, 2]: return 'Winter'
                elif month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                else: return 'Autumn'
            X['SeasonSold'] = X['MoSold'].apply(map_season)

        # --- Polynomial Features ---
        poly_cols = ['OverallQual', 'GrLivArea', 'TotalSF', 'GarageArea', 'Age']
        for col in poly_cols:
            if col in X.columns:
                X[f'{col}_sq'] = X[col] ** 2
                X[f'{col}_cub'] = X[col] ** 3

        return X

    def _fit_transform(self, X: DataFrame, **kwargs) -> Tuple[DataFrame, dict]:
        """Fit and transform the data by handling missing values and creating features"""
        X_out = X if self.inplace else X.copy()
        
        # Step 1: Handle missing values
        X_out = self._handle_missing_values(X_out)
        
        # Step 2: Create new features
        X_out = self._create_features(X_out)
        
        return X_out, self.feature_metadata_in.type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        """Transform new data using the same feature engineering steps"""
        X_out = X if self.inplace else X.copy()
        
        # Apply the same transformations as in fit_transform
        X_out = self._handle_missing_values(X_out)
        X_out = self._create_features(X_out)
        
        return X_out

    def _more_tags(self):
        return {
            "feature_interactions": True,  # We create interaction features
            "handles_missing": True,       # We explicitly handle missing values
        }
