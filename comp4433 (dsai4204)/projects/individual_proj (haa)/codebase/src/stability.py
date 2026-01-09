import numpy as np
import pandas as pd
from .config import cfg
from .statistics import stats

def perturb(X: pd.DataFrame, level: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Applies structure-aware perturbation using GLOBAL statistics.
    Ensures that validation sets are perturbed consistently with the training manifold.
    """
    alpha = level / max(1, cfg.perturb_levels - 1)
    if alpha <= 0: return X.copy()

    X_out = X.copy()

    for col, dtype in cfg.dataset['features'].items():
        match dtype:
            case 'num': _perturb_continuous(X_out, col, alpha, rng)
            case 'int': _perturb_integer(X_out, col, alpha, rng)
            case 'cat': _perturb_categorical(X_out, col, alpha, rng)
            
    return X_out


def _add_noise(df: pd.DataFrame, col: str, alpha: float, rng: np.random.Generator) -> pd.Series:
    """Core noise generation using GLOBAL standard deviation."""
    s = stats.get(col)
    if s['std'] == 0: return df[col]
    
    scale = alpha * cfg.perturb['max_noise_std_ratio'] * s['std']
    return df[col] + rng.normal(0, scale, size=len(df))


def _perturb_continuous(df: pd.DataFrame, col: str, alpha: float, rng: np.random.Generator):
    """Perturbs floats, clipped to GLOBAL min/max."""
    s = stats.get(col)
    df[col] = _add_noise(df, col, alpha, rng).clip(s['min'], s['max'])


def _perturb_integer(df: pd.DataFrame, col: str, alpha: float, rng: np.random.Generator):
    """Perturbs integers, rounded and clipped to GLOBAL min/max."""
    s = stats.get(col)
    df[col] = _add_noise(df, col, alpha, rng).round().clip(s['min'], s['max']).astype(df[col].dtype)


def _perturb_categorical(df: pd.DataFrame, col: str, alpha: float, rng: np.random.Generator):
    """Perturbs categories using GLOBAL marginal distribution."""
    prob = alpha * cfg.perturb['max_flip_prob']
    mask = rng.random(len(df)) < prob
    
    if not mask.any(): return
    
    s = stats.get(col)
    df.loc[mask, col] = rng.choice(s['cats'], size=mask.sum(), p=s['probs'])
