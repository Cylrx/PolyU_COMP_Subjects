import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from typing import Tuple, Optional

from sklearn.metrics import root_mean_squared_log_error
from src.hpp.kaggle import write_submission
import pickle
import json

from src.hpp.config import cfg
from src.hpp.utils import *
from src.hpp.feature import TogglePipeline

def split_data(
    data: pd.DataFrame, 
    eval_ratio: float = 0.2,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and eval sets.
    
    Args:
        data: Input DataFrame to split
        eval_ratio: Proportion of data for eval set (0 < eval_ratio < 1)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, eval_data) DataFrames
    """
    assert 0 < eval_ratio < 1
    assert isinstance(data, pd.DataFrame)

    n = len(data)
    n_eval = int(n * eval_ratio)
    if n <= 0: 
        print_error("Cannot split empty DataFrame")
        raise ValueError()
    if n - n_eval < 0: 
        print_error(f"eval_ratio too large: {n - n_eval} train samples")
        raise ValueError()
    if n_eval < 0: 
        print_error(f"eval_ratio too small: {n_eval} eval samples")
        raise ValueError()
    
    n = len(data)
    n_eval = int(n * eval_ratio)
    
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(n)
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    train_data: pd.DataFrame = data.iloc[train_idx].reset_index(drop=True)
    eval_data: pd.DataFrame = data.iloc[eval_idx].reset_index(drop=True)
    return train_data, eval_data

def log1p_label(df: pd.DataFrame, label: str, reverse: bool) -> pd.DataFrame:
    df = df.copy()
    if reverse: df[label] = np.expm1(df[label].astype(float))
    else: df[label] = np.log1p(df[label].astype(float))
    return df


def modeling(train_data: pd.DataFrame, test_data: pd.DataFrame, run_uid: str, overlays: Optional[list[str]] = None): 
    assert isinstance(train_data, pd.DataFrame), "invalid format"
    assert isinstance(test_data, pd.DataFrame), "invalid format"

    exp = cfg.exp(overlays)
    label: str = exp["init"]["label"]

    if exp["other"]["log1p"]:
        train_data = log1p_label(train_data, label, reverse=False)

    tr_dataset = TabularDataset(train_data)
    te_dataset = TabularDataset(test_data)
    assert label not in te_dataset.columns
    assert label in tr_dataset.columns

    print_info(f"Training model...")
    model_save_path = abs_path(cfg.paths["model"]) / run_uid
    ensure_dir(model_save_path)
    predictor = TabularPredictor(
        path=model_save_path,
        **exp["init"],
    ).fit(
        train_data=tr_dataset,
        **exp["fit"],
    )
    
    print_info(f"Fitted Models:\n{predictor.model_names()}")
    print_good(f"Best Model: {predictor.model_best}")

    print_info(f"Constructing Leaderboard on test data...")
    print(predictor.leaderboard(te_dataset))
    print_info(f"Predicting on test data...")

    preds = predictor.predict(te_dataset)
    if exp["other"]["log1p"]:
        preds = np.expm1(preds.astype(float))
    
    predictor.unpersist() # free mem
    return preds


def load_model(eval_data: pd.DataFrame, run_uid: str):
    model_save_path = abs_path(cfg.paths["model"]) / run_uid
    predictor = TabularPredictor.load(model_save_path)
    print_info(f"Current best model: {predictor.model_best}")
    exp = cfg.exp()
    label = exp["init"]["label"]
    if exp["other"]["log1p"]:
        eval_data = log1p_label(eval_data, label, reverse=False)
    te_dataset = TabularDataset(eval_data)


    # Feature Importance via SHAP
    try: 
        print_info(f"Computing Feature Importance...")
        fi = predictor.feature_importance(te_dataset)
        print(fi)
    except Exception as e: 
        print_error(f"Failed to compute Feature Importance: {e}")
        print_error("Probably CUDA OOM if using TFMs.")
        print_error("Requires at least an RTX5090")
        print_warn(f"Skipping feature importance computation...")


    # Leaderboard on eval data for choosing best model (for Dynamic Stacking)
    print_info(f"Computing Leaderboard...")
    lb = predictor.leaderboard(te_dataset)
    best_eval_model = lb.loc[lb["score_test"].idxmax(), "model"]
    print_good(f"Best Eval Model: {best_eval_model}")

    if input("Do you wish to use and save the best eval model? (y/n): ").lower().strip() == "y":
        predictor.set_model_best(best_eval_model)
        print_info(f"Best model is now: {predictor.model_best}")
        name = input("Enter save name for the best eval model: ")
        preds = predictor.predict(te_dataset.drop(columns=[label]), model=best_eval_model)
        if exp["other"]["log1p"]:
            preds = np.expm1(preds.astype(float))
        write_submission(preds, name, subfolder="best_eval_model")
        predictor.save()
