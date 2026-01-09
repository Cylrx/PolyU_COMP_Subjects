import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

from src.config import cfg
from src.utils import save_fig, save_df, abs_path, print_good

plt.rcParams.update(cfg.plots['rcParams'])
plt.rcParams['axes.prop_cycle'] = cycler(color=cfg.plots['colors'])

out_dir = abs_path(cfg.paths['output']) / "classify"

def _aggregate_k_fold_metrics(k_fold_data: List[dict]) -> pd.DataFrame:
    """
    Flattens the nested k_fold_data and computes mean/std grouped by model and level.
    Returns a DataFrame with MultiIndex columns (metric, stat).
    """
    # Flatten: Unpack the nested list-dict structure into a list of flat records
    flat_data = [
        {"model": model, "level": lvl, **metrics}
        for fold in k_fold_data
        for model, experiments in fold.items()
        for lvl, metrics in enumerate(experiments)
    ]

    # Aggregate: Compute mean and std for every metric across folds
    df = pd.DataFrame(flat_data)
    report_df = df.groupby(["model", "level"]).agg(["mean", "std"])
    return report_df

def _to_wide_mean_std_df(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the aggregated report DataFrame (MultiIndex columns) to a wide format
    suitable for CSV export.
    Columns will be: model, level, metric1_mean, metric1_std, metric2_mean, ...
    """
    # Create a copy to avoid modifying the original
    df = report_df.copy()
    
    # Flatten columns from (metric, stat) to metric_stat
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    
    # Reset index to make model and level columns
    df = df.reset_index()
    
    return df

def process_and_plot_metrics(
    k_fold_data: List[dict], 
    suffix: str = "",
    precision: int = 4
) -> None:
    """
    Helper function to process metrics data (flatten -> groupby) and generate plots.
    """
    if not k_fold_data:
        return

    report_df = _aggregate_k_fold_metrics(k_fold_data)

    print(f"\n=== Aggregated Report {suffix} (Mean ± Std) ===\n")
    with pd.option_context('display.precision', precision):
        print(report_df)

    # Export to CSV
    summary_df = _to_wide_mean_std_df(report_df)
    csv_path = out_dir / f"perturb_groupby_metrics{suffix}.csv"
    save_df(summary_df, csv_path)
    print_good(f"Saved aggregated metrics to {csv_path}")

    plot_perturb_groupby_metrics(report_df, suffix)
    plot_perturb_groupby_models(report_df, suffix)


def generate_report(
    k_fold_metrics: List[dict], 
    k_fold_calibrs: List[dict],
    k_fold_calibr_curve: List[dict],
    precision: int = 4
) -> None:
    """
    Args: 
        k_fold_metrics: List[dict]
            - Structure: [Fold] -> {Model: [PerturbationLevel -> {Metric: Value}]}
        k_fold_calibrs: List[dict]
            - Similar structure to k_fold_metrics except uses calibration metrics
        k_fold_calibr_curve: List[dict]
            - Structure: [Fold] -> {Model: List[Tuple[accuracy, confidence]]}
        precision: int
            - Decimal precision for the printed dataframe.
    """
    
    process_and_plot_metrics(k_fold_metrics, suffix="_perfs", precision=precision)
    process_and_plot_metrics(k_fold_calibrs, suffix="_calibrs", precision=precision)
    plot_calibration_curves(k_fold_calibr_curve)


def plot_perturb_groupby_metrics(report_df: pd.DataFrame, suffix: str = "") -> None:
    """
    Generates a single plot with subplots for each metric.
    Each subplot shows curves for all models.
    """
    metrics = report_df.columns.levels[0].tolist()
    n_metrics = len(metrics)
    stdev_factor = cfg.plots['err_band_stdev_factor']
    
    # Determine grid layout
    cols = min(3, n_metrics)
    rows = math.ceil(n_metrics / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    axes_flat = [axes] if rows * cols == 1 else axes.flatten()
    
    models = report_df.index.levels[0].tolist()

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        
        for model in models:
            # Extract mean and std for the specific model and metric
            subset = report_df.loc[model][metric]
            
            ax.plot(subset.index, subset['mean'], label=model)
            ax.fill_between(
                subset.index,
                subset['mean'] - stdev_factor * subset['std'],
                subset['mean'] + stdev_factor * subset['std'],
                alpha=cfg.plots['err_band_alpha']
            )

        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel('Score')
        
        if idx == 0:
            ax.legend()

    # Turn off unused subplots
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].axis('off')

    out_path = out_dir / f"perturb_groupby_metrics{suffix}.png"
    save_fig(fig, out_path)
    print_good(f"Saved metrics plot to {out_path}")


def plot_perturb_groupby_models(report_df: pd.DataFrame, suffix: str = "") -> None:
    """
    Generates a single plot with subplots for each model.
    Each subplot shows normalized curves for all metrics.
    """
    models = report_df.index.levels[0].tolist()
    
    # Filter out ignored models based on config
    ignore_list = cfg.plots.get('perturn_groupby_models', {}).get('ignore_list', [])
    if ignore_list:
        models = [m for m in models if m not in ignore_list]

    metrics = report_df.columns.levels[0].tolist()
    
    # -------------------------------------------------------------------------
    # Pre-computation: Global Scale Unification
    # -------------------------------------------------------------------------
    
    LOWER_IS_BETTER = {'ece_kde', 'brier', 'log_loss'}

    def get_normalized_curve(model: str, metric: str) -> pd.Series:
        subset = report_df.loc[model][metric]['mean']
        start_val = subset.iloc[0]
        
        # Determine direction
        if metric in LOWER_IS_BETTER:
            # Lower is better: Increase in metric => Decrease in Score
            # Use inverse ratio: start / current
            # Handle division by zero (if current error is 0)
            # Add epsilon to denominator to avoid inf
            return start_val / (subset + 1e-9)
        else:
            # Higher is better: Decrease in metric => Decrease in Score
            # Use direct ratio: current / start
            return subset / start_val if abs(start_val) > 1e-9 else subset

    all_curves = [get_normalized_curve(m, met) for m in models for met in metrics]
    
    # Filter out curves that are entirely NaN or empty
    valid_curves = [c for c in all_curves if not c.empty and not c.isna().all()]
    
    if not valid_curves:
        y_limit_min = 0.0
        y_limit_max = 1.01
    else:
        global_min = min(curve.min() for curve in valid_curves)
        global_max = max(curve.max() for curve in valid_curves)
        
        if pd.isna(global_min):
            global_min = 0.0
        
        if pd.isna(global_max):
             global_max = 1.0
             
        y_limit_min = global_min - 0.01
        y_limit_max = global_max + 0.01

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    
    n_models = len(models)
    cols = min(3, n_models)
    rows = math.ceil(n_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True)
    axes_flat = [axes] if rows * cols == 1 else axes.flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        
        for metric in metrics:
            curve = get_normalized_curve(model, metric)
            ax.plot(curve.index, curve, label=metric.replace('_', ' ').title())

        ax.set_title(model.upper())
        ax.set_xlabel('Perturbation Level')
        ax.set_ylabel('-Normalized Score')
        ax.set_ylim(bottom=y_limit_min, top=y_limit_max)
        
        if idx == 0:
            ax.legend()

    # Turn off unused subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].axis('off')

    out_path = out_dir / f"perturb_groupby_models{suffix}.png"
    save_fig(fig, out_path)
    print_good(f"Saved models plot to {out_path}")


def plot_calibration_curves(k_fold_calibr_curve: List[dict], filename: str = "calibration_curves.png") -> None:
    """
    Plots the calibration curves for each model, aggregated over K folds.
    Generates a single plot with subplots for each model.
    """
    if not k_fold_calibr_curve:
        return

    # Assuming all folds have the same models
    models = list(k_fold_calibr_curve[0].keys())
    n_models = len(models)
    
    # Determine grid layout
    cols = min(3, n_models)
    rows = math.ceil(n_models / cols)
    
    stdev_factor = cfg.plots['err_band_stdev_factor']
    x_grid = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), constrained_layout=True)
    axes_flat = [axes] if rows * cols == 1 else axes.flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

        interpolated_curves = []
        
        for fold_data in k_fold_calibr_curve:
            curve = fold_data.get(model)
            if not curve:
                continue
            
            # Unzip (accuracy, confidence)
            # The curve is sorted by confidence from eval_calibr_curve
            # But just in case, we sort again to be safe for interpolation
            curve_sorted = sorted(curve, key=lambda x: x[1])
            accs, confs = zip(*curve_sorted)
            
            # Interpolate onto fixed grid
            interp_acc = np.interp(x_grid, confs, accs)
            interpolated_curves.append(interp_acc)
        
        if interpolated_curves:
            arr = np.array(interpolated_curves)
            mean_curve = np.mean(arr, axis=0)
            std_curve = np.std(arr, axis=0)
            
            ax.plot(x_grid, mean_curve, label=model)
            ax.fill_between(
                x_grid,
                mean_curve - stdev_factor * std_curve,
                mean_curve + stdev_factor * std_curve,
                alpha=cfg.plots['err_band_alpha']
            )

        ax.set_title(model.upper())
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy (Estimated)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    # Turn off unused subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].axis('off')

    out_path = out_dir / filename
    save_fig(fig, out_path)
    print_good(f"Saved calibration curves to {out_path}")
