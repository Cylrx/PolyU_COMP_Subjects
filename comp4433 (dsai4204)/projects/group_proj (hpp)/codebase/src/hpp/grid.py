import os
import re
import time
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime

from src.hpp.config import cfg
from src.hpp.feature import feature_engineering
from src.hpp.modeling import modeling
from src.hpp.kaggle import write_submission, get_run_uid
from src.hpp.utils import print_good, print_info, abs_path, ensure_dir

def find_existing_grids(submission_dir: Path) -> List[Path]:
    grids = [d for d in submission_dir.iterdir() if d.is_dir() and d.name.startswith('grid_')]
    return sorted(grids)


def load_grid_info(grid_dir: Path) -> Dict:
    info_path = grid_dir / 'grid_info.yaml'
    assert info_path.exists(), f"grid_info.yaml not found in {grid_dir}"
    with open(info_path, 'r') as f:
        return yaml.safe_load(f)


def write_grid_info(grid_dir: Path, grid_id: str, grid_dict: Dict) -> None:
    ensure_dir(grid_dir)
    info_path = grid_dir / 'grid_info.yaml'
    info = {'grid_id': grid_id, 'run_grid': grid_dict}
    with open(info_path, 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


def validate_grid_match(saved_grid: Dict, current_grid: Dict) -> None:
    assert 'run_grid' in saved_grid, "Invalid grid_info.yaml: missing run_grid"
    saved = saved_grid['run_grid']
    assert saved == current_grid, f"Grid mismatch: saved {saved} != current {current_grid}"


def parse_submission_overlays(filename: str, categories: List[str]) -> List[str]:
    name = filename.replace('submission__', '').replace('.csv', '')
    name = re.sub(r'_\d{2}_\d{4}$', '', name)
    overlays = name.split('-')
    assert len(overlays) == len(categories), f"Overlay count mismatch: {len(overlays)} != {len(categories)}"
    return overlays


def get_completed_combinations(grid_dir: Path, categories: List[str]) -> Set[str]:
    completed = set()
    for f in grid_dir.glob('submission__*.csv'):
        overlays = parse_submission_overlays(f.name, categories)
        sig = json.dumps(cfg.exp(overlays), sort_keys=True)
        completed.add(sig)
    return completed


def prompt_continue_grid(grids: List[Path]) -> Optional[Path]:
    if not grids:
        return None
    print_info(f"Found {len(grids)} existing grid search(es):")
    for i, g in enumerate(grids, 1):
        print_info(f"\t{i}. {g.name}")
    print_info(f"\t{len(grids) + 1}. Start new grid search")
    choice = input("Select option: ").strip()
    idx = int(choice) - 1
    assert 0 <= idx <= len(grids), f"Invalid choice: {choice}"
    return grids[idx] if idx < len(grids) else None


def grid_search(train_data: pd.DataFrame, test_data: pd.DataFrame):
    grid_dict: Dict[str, List[str]] = cfg.experiments["run_grid"]
    submission_dir = abs_path(cfg.paths["submission_output"])
    
    categories: List[str] = list(grid_dict.keys())
    grid: List[List[str]] = [vals for vals in grid_dict.values()]
    choice: List[int] = [0] * len(categories)
    
    existing_grids = find_existing_grids(submission_dir)
    selected_grid = prompt_continue_grid(existing_grids) if existing_grids else None
    
    if selected_grid:
        saved_info = load_grid_info(selected_grid)
        validate_grid_match(saved_info, grid_dict)
        grid_id = saved_info['grid_id']
        completed = get_completed_combinations(selected_grid, categories)
        print_info(f"Continuing grid search: {grid_id}")
        print_info(f"Already completed: {len(completed)} combinations")
    else:
        grid_id = f"grid_{datetime.now().strftime('%d%H%M')}"
        grid_dir = submission_dir / grid_id
        write_grid_info(grid_dir, grid_id, grid_dict)
        completed = set()
        print_info(f"Starting new grid search: {grid_id}")

    done = False
    seen = set(); n_all = 0; n_run = 0; n_skip = 0
    while not done: 
        overlays: list[str] = [grid[i][choice[i]] for i in range(len(categories))]
        n_all += 1
        sig = json.dumps(cfg.exp(overlays), sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            if sig in completed:
                n_skip += 1
                print_info(f"Skip completed: {'-'.join(overlays)}")
            else:
                n_run += 1
                run_one_experiment(train_data, test_data, overlays, grid_id)
        else:
            print_info(f"Skip duplicate: {'-'.join(overlays)}")

        done = True
        for i in range(len(categories)): 
            if choice[i] + 1 < len(grid[i]):
                done = False
                choice[i] += 1
                break
            choice[i] = 0
    print_good(f"Grid combos: {n_all}, unique runs: {n_run}, skipped: {n_skip}")


def run_one_experiment(train_data: pd.DataFrame, test_data: pd.DataFrame, overlays: list[str], grid_id: str):
    run_uid: str = get_run_uid(overlays)
    print_info(f"Running Experiment: {run_uid}...")
    time_start: float = time.time()
    train_data, test_data = feature_engineering(train_data, test_data, overlays, is_eval=False)
    preds = modeling(train_data, test_data, run_uid, overlays)
    write_submission(preds, run_uid, subfolder=grid_id)
    time_end: float = time.time()

    print_good(f"Run Completed: {run_uid}")
    print_good(f"Time Elapsed: {time_end - time_start:.1f} seconds", end='\n\n')