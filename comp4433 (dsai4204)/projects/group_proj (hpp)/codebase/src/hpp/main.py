import pandas as pd
from pathlib import Path
from typing import Tuple

from src.hpp.modeling import modeling, load_model
from src.hpp.config import cfg
from src.hpp.utils import abs_path, print_info, print_error, print_good
from src.hpp.feature import feature_engineering
from src.hpp.grid import grid_search
from src.hpp.kaggle import write_submission, get_run_uid

def load_hpp_data() -> Tuple[pd.DataFrame, pd.DataFrame]: 
    train_path: Path = abs_path(cfg.paths["train_data"])
    test_path: Path = abs_path(cfg.paths["test_data"])
    train_df: pd.DataFrame = pd.read_csv(train_path)
    test_df: pd.DataFrame = pd.read_csv(test_path)
    unwanted_labels: list[str] = cfg.dataset["unwanted_labels"]
    train_df = train_df.drop(columns=unwanted_labels)
    test_df = test_df.drop(columns=unwanted_labels)

    return train_df, test_df


def load_eval_data() -> pd.DataFrame:
    eval_path: Path = abs_path(cfg.paths["eval_data"])
    try: 
        eval_df: pd.DataFrame = pd.read_csv(eval_path)
    except FileNotFoundError as e: 
        print_error("Evaluation data not found!")
        print_error("Please manually create a holdout set and save it to the evaluation data path!")
        raise e

    unwanted_labels: list[str] = cfg.dataset["unwanted_labels"]
    eval_df = eval_df.drop(columns=unwanted_labels)
    return eval_df


def main(train_data: pd.DataFrame, test_data: pd.DataFrame): 

    if cfg.grid_mode: 
        grid_search(train_data, test_data)
    else:
        overlays: list[str] = cfg.experiments["run_overlays"]
        if input("Wish to load and evaluate an existing model on holdout? (y/n): ").lower().strip() == "y":
            print_info("Loading Holdout Data for evaluation...")
            eval_data = load_eval_data()
            _, eval_data = feature_engineering(train_data, eval_data, overlays, is_eval=True)
            model_path = abs_path(cfg.paths["model"])
            model_list = [d for d in model_path.iterdir() if d.is_dir()]
            print(f"Available models:\n {'\n'.join([f'\t{i}. {d.name}' for i, d in enumerate(model_list)])}")
            run_idx = int(input("Enter run number: "))
            run_uid = model_list[run_idx].name
            load_model(eval_data, run_uid)
        else:

            print_info("Feature Engineering...")
            train_data, test_data = feature_engineering(train_data, test_data, overlays, is_eval=False)
            print_good("Feature Engineering Completed.")
            run_uid: str = get_run_uid(cfg.experiments["run_overlays"])
            preds = modeling(train_data, test_data, run_uid, overlays)
            write_submission(preds, run_uid)
