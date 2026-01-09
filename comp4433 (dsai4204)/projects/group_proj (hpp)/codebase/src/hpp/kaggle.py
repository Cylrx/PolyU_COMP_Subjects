import pandas as pd
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime

from src.hpp.config import cfg
from src.hpp.utils import abs_path, save_df, print_good

def get_run_uid(overlays: Iterable[str]) -> str: 
    overlays_str: str = "-".join(overlays)
    return f"{overlays_str}_{datetime.now().strftime('%d_%H%M')}"

def write_submission(preds: pd.Series, run_uid: str, subfolder: Optional[str] = None): 
    submission_path: Path = abs_path(cfg.paths["submission_template"])
    output_path: Path = abs_path(cfg.paths["submission_output"])
    submission_df = pd.read_csv(submission_path)
    submission_df["SalePrice"] = preds
    
    if subfolder is not None:
        output_path = output_path / subfolder
    output_path = output_path / f"submission__{run_uid}.csv"
    save_df(submission_df, output_path)
    print_good(f"Submission written to {output_path}")