from typing import List, Optional, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn import set_config

from src.config import cfg
from src.statistics import stats

    # Configure sklearn to output pandas DataFrames
    # set_config(transform_output="pandas")

def get_preprocessor(model_name: str) -> Optional[ColumnTransformer]:
    """
    Returns a ColumnTransformer tailored to the specific model requirements.
    
    CRITICAL: We inject GLOBAL category lists into OneHotEncoder.
    This prevents the "Missing Category in Split" problem where a CV fold 
    might miss a rare category, leading to inconsistent feature dimensions.
    """
    features = cfg.dataset['features']
    cat_cols = [col for col, dtype in features.items() if dtype == 'cat']
    num_cols = [col for col, dtype in features.items() if dtype in ('int', 'num')]

    # Retrieve global categories to enforce consistent schema
    # The order of categories lists must match the order of cat_cols
    global_categories = [stats.get(col)['cats'] for col in cat_cols]

    # Common transformers
    # handle_unknown='ignore' is still good practice for truly unseen data (e.g. inference)
    # but 'categories' ensures we cover the known global support.
    ohe = OneHotEncoder(
        categories=global_categories, 
        sparse_output=False, 
        handle_unknown='ignore'
    )
    
    minmax = MinMaxScaler()
    passthrough = FunctionTransformer(lambda x: x, feature_names_out="one-to-one")

    match model_name:
        case "logreg" | "knn":
            transformers = [
                ("num", minmax, num_cols),
                ("cat", ohe, cat_cols),
            ]
        case "xgboost" | "rf":
            return None
        case "tabpfn" | "limix":
            # TabPFN handles preprocessing internally and efficiently
            return None
        case _:
            raise ValueError(f"Model {model_name} not supported in preprocessing")

    return ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
        remainder='drop'  # Drop any columns not explicitly handled
    )
