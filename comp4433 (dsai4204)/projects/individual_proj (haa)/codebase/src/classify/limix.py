import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Add LimiX directory to sys.path to allow its internal imports (inference, utils, etc.) to resolve correctly.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
limix_path = os.path.join(project_root, 'LimiX')
if limix_path not in sys.path:
    sys.path.append(limix_path)

from typing import Optional
from src.config import cfg
from src.utils import print_warn, print_error
from LimiX.inference.predictor import LimiXPredictor

os.makedirs("./cache", exist_ok=True)
if not os.path.exists("./cache/LimiX-16M.ckpt"):
    from huggingface_hub import hf_hub_download
    model_file = hf_hub_download(repo_id="stableai-org/LimiX-16M", filename="LimiX-16M.ckpt", local_dir="./cache")


__all__ = ["LimiXClassifier"]


class LimiXClassifier:
    def __init__(self, device: str, model_path: str, inference_config: str):
        self.device = device
        self.model_path = model_path
        self.inference_config = inference_config
        self.model = LimiXPredictor(
            device=torch.device(device), 
            model_path=model_path, 
            inference_config=inference_config
        )
        self.X_train: Optional[pd.DataFrame] = None
        self.Y_train: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.X_train is not None or self.Y_train is not None: 
            print_warn("LimiXClassifier is already fitted, skipping fit.")

        self.X_train = X
        self.Y_train = y

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.X_train is None or self.Y_train is None:
            print_error("LimiXClassifier is not fitted. Call fit() first.")
            exit(1)
        return self.model.predict(self.X_train, self.Y_train, X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns class labels for X.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
