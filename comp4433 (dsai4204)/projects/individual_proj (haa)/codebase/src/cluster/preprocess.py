from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src.config import cfg
from src.statistics import stats

def get_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer for clustering (OneHot + MinMax).
    Uses global categories to ensure consistency.
    """
    features = cfg.dataset['features']
    cat_cols = [col for col, dtype in features.items() if dtype == 'cat']
    num_cols = [col for col, dtype in features.items() if dtype in ('int', 'num')]

    # Retrieve global categories to enforce consistent schema
    # The order of categories lists must match the order of cat_cols
    global_categories = [stats.get(col)['cats'] for col in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_cols),
            ("cat", OneHotEncoder(categories=global_categories, sparse_output=False), cat_cols),
        ],
        verbose_feature_names_out=False,
        remainder='drop'
    )

