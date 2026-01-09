from typing import Dict, Any
import pandas as pd
from .config import cfg

class GlobalStats:
    """
    Singleton repository for global dataset statistics.
    
    CRITICAL: This prevents the 'Local Statistics Trap'.
    Perturbations must be scaled against the GLOBAL population distribution,
    not the local batch statistics, to ensure consistent and valid stress testing.
    """
    _instance = None
    _stats: Dict[str, Any] = {}
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def init(cls, df: pd.DataFrame) -> None:
        """Computes and freezes statistics from the full dataset."""
        if cls._initialized: return

        # Fail-fast: Schema validation
        if missing := set(cfg.dataset['features']) - set(df.columns):
            raise ValueError(f"Global stats init failed. Missing columns: {missing}")

        for col, dtype in cfg.dataset['features'].items():
            s = df[col]
            if dtype in ('num', 'int'):
                cls._stats[col] = {
                    'std': s.std(),
                    'min': s.min(),
                    'max': s.max()
                }
            elif dtype == 'cat':
                # Pre-calculate marginal probabilities for weighted sampling
                # sort_index() is CRITICAL: OneHotEncoder fails if numerical categories are unsorted
                counts = s.value_counts(normalize=True).sort_index()
                cls._stats[col] = {
                    'cats': counts.index.values,
                    'probs': counts.values
                }
        
        cls._initialized = True

    @classmethod
    def get(cls, col: str) -> Dict[str, Any]:
        if not cls._initialized:
            raise RuntimeError("Global stats not initialized. Call stats.init(df) first.")
        return cls._stats[col]

# Singleton instance
stats = GlobalStats()
