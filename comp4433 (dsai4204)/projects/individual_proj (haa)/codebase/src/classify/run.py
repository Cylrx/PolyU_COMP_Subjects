from typing import Dict, Optional, Tuple, Union, List, Any
from dataclasses import dataclass
import os
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier, TabPFNRegressor

from .preprocess import get_preprocessor
from src.config import cfg
from src.stability import perturb
from src.statistics import stats
from .report import generate_report, plot_calibration_curves
from .metrics.ece_kde import ece_kde_score, get_calibration_estimate
from src.utils import (
    print_info,
    print_good,
    print_warn,
    print_error,
    save_json,
    save_df,
    abs_path,
    get_categorical_indices
)

seed = cfg.seed
vb = cfg.verbose

if "limix" in cfg.classify_models:
    from .limix import LimiXClassifier

def get_model(model_name: str) -> Any:
    if model_name not in cfg.models:
        raise ValueError(f"Model config for {model_name} not found in cfg.models")

    params = dict(cfg.models[model_name])

    if model_name == "tabpfn":
        params["categorical_features_indices"] = get_categorical_indices()

    match model_name:
        case "logreg": return LogisticRegression(**params)
        case "xgboost": return XGBClassifier(**params)
        case "tabpfn": return TabPFNClassifier(**params)
        case "rf": return RandomForestClassifier(**params)
        case "knn": return KNeighborsClassifier(**params)
        case "limix" : return LimiXClassifier(**params)
        case _: raise ValueError(f"Model {model_name} not supported")

def get_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

def avg_metrics(metrics: List[dict]) -> dict:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {k: np.mean([m[k] for m in metrics]) for k in keys}

def eval_with_perturb(X: pd.DataFrame, y: pd.Series, model: Any) -> List[dict]: 
    # the change of the metrics w.r.t. the perturbation level
    metrics: List[dict] = [] 
    n_reps = cfg.perturb['repeat']
    X_tiled = pd.concat([X] * n_reps, ignore_index=True)

    for level in range(cfg.perturb_levels):
        rng = np.random.default_rng(cfg.seed + level)
        
        # vectorized repetition
        # Pipeline handles preprocessing internally
        y_pred_all = model.predict(perturb(X_tiled, level, rng))
        y_pred_mat = y_pred_all.reshape(n_reps, len(X))
        reps = [get_metrics(y, row) for row in y_pred_mat]
        
        metrics.append(avg_metrics(reps))

    return metrics


def eval_calibr_with_perturb(X: pd.DataFrame, y: pd.Series, model: Any) -> List[dict]: 
    """
    Similar to `eval_with_perturb` except we use the `predict` 
    method of `ProbabilisticWrapper` (`predict_proba` under the hood)
    Returns the ECE_KDE and Brier .
    """
    metrics: List[dict] = []
    n_reps = cfg.perturb['repeat']
    X_tiled = pd.concat([X] * n_reps, ignore_index=True)

    for level in range(cfg.perturb_levels):
        rng = np.random.default_rng(cfg.seed + level)

        # Vectorized repetition with probability scores
        # We take [:, 1] for the positive class probability in binary classification
        y_prob_all = model.predict_proba(perturb(X_tiled, level, rng))[:, 1]
        y_prob_mat = y_prob_all.reshape(n_reps, len(X))
        
        reps = [
            {
                "ece_kde": ece_kde_score(y, row),
                "brier": brier_score_loss(y, row)
            }
            for row in y_prob_mat
        ]
        
        metrics.append(avg_metrics(reps))

    return metrics


def eval_calibr_curve(X: pd.DataFrame, y: pd.Series, model: Any) -> List[Tuple[float, float]]:
    """
    Evaluates the calibration curve for the given model.
    Uses ECE_KDE approach instead of binning.
    Returns:
        List[Tuple[float, float]]
            - (accuracy, confidence)
    """
    y_prob = model.predict_proba(X)[:, 1]
    est_acc, conf = get_calibration_estimate(y, y_prob)
    
    # Sort by confidence for plotting
    curve = sorted(zip(est_acc, conf), key=lambda x: x[1])
    return curve

# -------------------------------------------------------------------------
# Reusable Helpers
# -------------------------------------------------------------------------

def _split_xy(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits data into X and y based on config."""
    target_col = cfg.dataset['label']
    assert target_col in data.columns
    df = data.reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def _iter_folds(X: pd.DataFrame, y: pd.Series, random_state: int = cfg.seed):
    """Yields (train_idx, val_idx) for stratified K-fold."""
    kf = StratifiedKFold(n_splits=cfg.k_folds, shuffle=True, random_state=random_state)
    return kf.split(X, y)

def _fit_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Creates and fits a pipeline/model. Checks for predict_proba."""
    preprocessor = get_preprocessor(model_name)
    model = get_model(model_name) if preprocessor is None else Pipeline([
        ('prep', preprocessor),
        ('clf', get_model(model_name))
    ])
    model.fit(X_train, y_train)
    
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(f"Model {model_name} does not have predict_proba")
        
    return model

# -------------------------------------------------------------------------
# Cross Validation
# -------------------------------------------------------------------------

def cross_validation(data: pd.DataFrame) -> None:
    """
    Executes k-fold cross-validation.
    """
    X, y = _split_xy(data)
    
    print_info(f"{cfg.k_folds}-fold CV starts")

    # [folds] -> {model_name: [(accuracy, confidence)]}
    k_fold_calibr_curve: List[Dict[str, List[Tuple[float, float]]]] = [] 
    k_fold_metrics: List[dict] = []
    k_fold_calibrs: List[dict] = []

    for fold, (train_idx, val_idx) in enumerate(_iter_folds(X, y), start=1):
        print_info(f"Fold {fold}/{cfg.k_folds} starts")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

        metrics = {}
        calibrs = {}
        calibr_curve = {}
        for model_name in cfg.classify_models:
            print_info(f"\tFold {fold} | model {model_name}") if vb else 0
            
            model = _fit_model(model_name, X_train, y_train)
            
            print_info(f"\tFold {fold} | model {model_name} fitted") if vb else 0
            print_info(f"\t\tFold {fold} | model {model_name} evaluating metrics") if vb else 0
            metrics[model_name] = eval_with_perturb(X_val, y_val, model)
            print_info(f"\t\tFold {fold} | model {model_name} evaluating calibrs") if vb else 0
            calibrs[model_name] = eval_calibr_with_perturb(X_val, y_val, model)
            print_info(f"\t\tFold {fold} | model {model_name} evaluating calibr curve") if vb else 0
            calibr_curve[model_name] = eval_calibr_curve(X_val, y_val, model)
        
        k_fold_metrics.append(metrics)
        k_fold_calibrs.append(calibrs)
        k_fold_calibr_curve.append(calibr_curve)
            
        print_good(f"Fold {fold}/{cfg.k_folds} Done.")
    
    generate_report(k_fold_metrics, k_fold_calibrs, k_fold_calibr_curve)

# -------------------------------------------------------------------------
# Ensembling
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class OofSplit:
    round_idx: int
    fold_idx: int # 1-based
    val_idx: np.ndarray
    y_true: np.ndarray

def _predict_split_probas(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    model_names: List[str]
) -> Dict[str, np.ndarray]:
    """Fits models and predicts probabilities for a single split."""
    probas = {}
    for model_name in model_names:
        model = _fit_model(model_name, X_train, y_train)
        probas[model_name] = model.predict_proba(X_val)[:, 1]
    return probas

def collect_oof_probs(
    data: pd.DataFrame, 
    model_names: List[str]
) -> Tuple[List[OofSplit], Dict[str, List[np.ndarray]]]:
    """
    Collects Out-Of-Fold probabilities for N rounds of K-fold CV.
    Returns:
        splits: Metadata and ground truth for each split.
        probs_by_model: Dict mapping model name to List of prob arrays (aligned with splits).
    """
    X, y = _split_xy(data)
    N = cfg.ensembling['weight_grid']['N']
    K = cfg.ensembling['weight_grid']['K']
    
    print_info(f"Ensembling: Collecting OOF probs ({N} rounds * {K} folds)")
    
    splits: List[OofSplit] = []
    probs_by_model: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}
    
    for round_idx in range(N):
        print_info(f"Ensembling Round {round_idx + 1}/{N}")
        
        # Use seed + round_idx for varying folds
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=cfg.seed + round_idx)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]
            
            # 1. Record Split Metadata
            split = OofSplit(
                round_idx=round_idx,
                fold_idx=fold,
                val_idx=val_idx,
                y_true=y_val.to_numpy()
            )
            splits.append(split)
            
            # 2. Get Probs for all models
            probas = _predict_split_probas(X_train, y_train, X_val, model_names)
            
            for m in model_names:
                probs_by_model[m].append(probas[m])
                
            # Verify shapes
            current_len = len(y_val)
            for m in model_names:
                assert len(probs_by_model[m][-1]) == current_len
                
    return splits, probs_by_model

def weight_grid(step: float) -> List[Tuple[float, float, float]]:
    """Generates a grid of weights (w1, w2, w3) summing to 1."""
    n = int(round(1.0 / step))
    assert np.isclose(step * n, 1.0), "Step size must divide 1.0"
    
    grid = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            w1 = i / n
            w2 = j / n
            w3 = (n - i - j) / n
            grid.append((w1, w2, w3))
    return grid

def _grid_search_weights(
    splits: List[OofSplit],
    probs_by_model: Dict[str, List[np.ndarray]],
    models: List[str],
    step: float
) -> Tuple[Tuple[float, float, float], float, Dict[str, float]]:
    """Performs grid search to find best weights minimizing ECE."""
    weights = weight_grid(step)
    best_ece = float('inf')
    best_weights = None
    y_true_all = [s.y_true for s in splits]

    # Pre-calculate base model ECEs
    base_ece = {}
    for m in models:
        ece_list = [ece_kde_score(y, p) for y, p in zip(y_true_all, probs_by_model[m])]
        base_ece[m] = np.mean(ece_list)
        print_info(f"Base model {m} mean OOF ECE: {base_ece[m]:.4f}")
        
    print_info(f"Grid searching {len(weights)} weight combinations...")
    
    # Pre-fetch arrays to avoid dict lookup in loop
    p1_all = probs_by_model[models[0]]
    p2_all = probs_by_model[models[1]]
    p3_all = probs_by_model[models[2]]
    n_samples = len(y_true_all)

    for w1, w2, w3 in weights:
        ece_sum = 0.0
        for i in range(n_samples):
            p_ens = w1 * p1_all[i] + w2 * p2_all[i] + w3 * p3_all[i]
            ece_sum += ece_kde_score(y_true_all[i], p_ens)
            
        avg_ece = ece_sum / n_samples
        if avg_ece < best_ece:
            best_ece = avg_ece
            best_weights = (w1, w2, w3)

    return best_weights, best_ece, base_ece

def _ensemble_split_probas(
    splits: List[OofSplit],
    probs_by_model: Dict[str, List[np.ndarray]],
    models: List[str],
    weights: Tuple[float, float, float]
) -> List[np.ndarray]:
    """Computes weighted ensemble probabilities for each split."""
    w1, w2, w3 = weights
    p1_all = probs_by_model[models[0]]
    p2_all = probs_by_model[models[1]]
    p3_all = probs_by_model[models[2]]
    
    return [
        w1 * p1 + w2 * p2 + w3 * p3
        for p1, p2, p3 in zip(p1_all, p2_all, p3_all)
    ]

def _build_oof_matrix(
    splits: List[OofSplit],
    ensemble_probas: List[np.ndarray],
    n_samples: int,
    n_rounds: int
) -> np.ndarray:
    """
    Constructs (N_rounds, n_samples) OOF probability matrix.
    Raises error if any sample is missing in any round.
    """
    matrix = np.full((n_rounds, n_samples), np.nan)
    
    for split, prob in zip(splits, ensemble_probas):
        matrix[split.round_idx, split.val_idx] = prob
        
    if np.isnan(matrix).any():
        raise RuntimeError("OOF matrix contains NaNs! Some samples were not covered in some rounds.")
        
    return matrix

def _export_oof_proba_csv(
    matrix: np.ndarray,
    y_full: pd.Series,
    path: str
) -> None:
    """Exports OOF probabilities to CSV in wide format."""
    df = pd.DataFrame()
    df['idx'] = np.arange(len(y_full))
    df['y_true'] = y_full.values
    
    # Add per-round columns
    for r in range(matrix.shape[0]):
        df[f'p_y1_r{r}'] = matrix[r]
        
    # Add stats
    df['p_y1_mean'] = np.mean(matrix, axis=0)
    df['p_y1_std'] = np.std(matrix, axis=0)
    
    save_df(df, abs_path(path))
    print_good(f"Saved OOF probabilities to {path}")

def ensembling(data: pd.DataFrame) -> None:
    print_info("Starting Ensembling...")
    
    # 1. Config Validation
    models = cfg.ensembling['models']
    step = cfg.ensembling['weight_grid']['step']
    assert len(models) == 3, "Exactly 3 models required for ensembling"
    
    # 2. Collect OOF Probs
    splits, probs_by_model = collect_oof_probs(data, models)
    
    # 3. Grid Search
    best_weights, best_ece, base_ece = _grid_search_weights(splits, probs_by_model, models, step)
    
    print_good(f"Best Weights: {dict(zip(models, best_weights))}")
    print_good(f"Best Ensemble ECE: {best_ece:.4f}")
    
    # 4. Save Results
    result = {
        "models": models,
        "step": step,
        "best_weights": best_weights,
        "best_ece_kde": best_ece,
        "base_ece_kde": base_ece
    }
    save_json(result, abs_path(cfg.paths['output']) / "classify/ensembling_best.json")
    
    # 5. Plot Calibration Curve for Ensemble
    ensemble_probas = _ensemble_split_probas(splits, probs_by_model, models, best_weights)
    ensemble_fold_curves = []
    
    for split, p_ens in zip(splits, ensemble_probas):
        est_acc, conf = get_calibration_estimate(split.y_true, p_ens)
        curve = sorted(zip(est_acc, conf), key=lambda x: x[1])
        ensemble_fold_curves.append({"Ensemble": curve})
        
    plot_calibration_curves(ensemble_fold_curves, filename="ensemble_calibration_curve.png")

    # 6. Export OOF Probabilities
    # Note: data here is the full dataset (data), not split yet. 
    # But collect_oof_probs used _split_xy(data) internally which drops index/target
    # So we need to match that.
    _, y_full = _split_xy(data)
    
    oof_matrix = _build_oof_matrix(
        splits, 
        ensemble_probas, 
        n_samples=len(y_full),
        n_rounds=cfg.ensembling['weight_grid']['N']
    )
    
    _export_oof_proba_csv(oof_matrix, y_full, cfg.paths['proba'])

def main(): 
    data = pd.read_csv(cfg.paths['data'])
    stats.init(data)
    try: 
        cross_validation(data)
        ensembling(data)
    except Exception as e:
        if "tabpfn" in str(e).lower():
            print_error("Failed to run TabFPN. Do you have access to TabPFN v2.5?")
            print_error(f"Read error message below for more details:\n{e}\n")
            exit(0)
        raise e

if __name__ == "__main__":
    main()
