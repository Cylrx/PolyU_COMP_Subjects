from __future__ import annotations

import ast
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from src.config import cfg
from src.utils import abs_path, ensure_dir, print_good, print_warn, print_error, save_fig

# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------

plt.rcParams.update(cfg.plots["rcParams"])
# We do not strictly use the cycle here because we need fixed mapping for 3 targets.
# However, we will pick colors from cfg.plots["colors"].

TARGET_PREFERENCE = ["binned_proba", "logits", "binary"]
ALGORITHM_ORDER = ["subgroup", "rulefit", "arm"]  # Fixed order as requested


def _get_luminance(hex_color: str) -> float:
    """Calculates relative luminance of a color."""
    rgb = to_rgb(hex_color)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _order_by_preference(items: List[str], preference: List[str]) -> List[str]:
    """
    Orders `items` by `preference`, appending any unknown items (sorted) at the end.
    """
    s = set(items)
    ordered = [x for x in preference if x in s]
    rest = sorted([x for x in items if x not in set(preference)])
    return ordered + rest


def _get_target_colors(targets: List[str]) -> Dict[str, str]:
    """
    Returns a mapping {target: hex_color}.
    Logic:
    - Pick N colors from cfg.plots['colors'] (fallback if insufficient).
    - Sort colors by luminance (Darkest -> Lightest).
    - Assign colors to targets ordered by TARGET_PREFERENCE so that:
      binned_proba is the darkest among present targets, binary the lightest.
    """
    if not targets:
        return {}

    n = len(targets)
    available = list(cfg.plots.get("colors", []))
    if len(available) < n:
        # Fallback palette (keep it deterministic; no matplotlib colormap dependency).
        fallback = ["#000000", "#555555", "#AAAAAA", "#1f77b4", "#ff7f0e", "#2ca02c"]
        available = (available + fallback)[:n]
    else:
        available = available[:n]

    # Darkest -> Lightest
    colors_sorted = sorted(available, key=_get_luminance)
    targets_sorted = _order_by_preference(targets, TARGET_PREFERENCE)
    return {t: c for t, c in zip(targets_sorted, colors_sorted)}


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def parse_report(report_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    config: Dict[str, Any] = {}
    overall_lines: List[str] = []
    bymodel_lines: List[str] = []

    # State machine for parsing
    section = "header"  # header, overall, bymodel

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("algorithms:"):
            val = stripped.split(":", 1)[1].strip()
            config["algorithms"] = ast.literal_eval(val)
        elif stripped.startswith("targets:"):
            val = stripped.split(":", 1)[1].strip()
            config["targets"] = ast.literal_eval(val)
        elif stripped.startswith("models:"):
            val = stripped.split(":", 1)[1].strip()
            config["models"] = ast.literal_eval(val)
        elif stripped.startswith("rounds_per_model:"):
            val = stripped.split(":", 1)[1].strip()
            try:
                config["rounds_per_model"] = int(val)
            except ValueError:
                config["rounds_per_model"] = val

        elif stripped.startswith("Overall (aggregated"):
            section = "overall"
            continue
        elif stripped.startswith("By model:"):
            section = "bymodel"
            continue
        elif stripped.startswith("Errors:"):
            section = "errors"
            continue
        elif stripped.startswith("="):
            continue

        # Accumulate table lines
        if section == "overall":
            overall_lines.append(line)
        elif section == "bymodel":
            bymodel_lines.append(line)

    # Parse DataFrames
    # Use whitespace separator
    if overall_lines:
        overall_df = pd.read_csv(StringIO("\n".join(overall_lines)), sep=r"\s+")
    else:
        overall_df = pd.DataFrame()

    if bymodel_lines:
        bymodel_df = pd.read_csv(StringIO("\n".join(bymodel_lines)), sep=r"\s+")
    else:
        bymodel_df = pd.DataFrame()

    return config, overall_df, bymodel_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _plot_cluster_group(
    ax: plt.Axes,
    df: pd.DataFrame,
    algorithms: List[str],
    targets: List[str],
    target_colors: Dict[str, str],
    show_legend: bool = False,
    title: str = "",
):
    """
    Plots grouped bars on a single axes.
    Groups = Algorithms (Subgroup, RuleFit, ARM)
    Bars within group = Targets (binned_proba, logits, binary)
    """
    # X configuration
    n_groups = len(algorithms)
    group_width = 0.8  # Total width allocated for one group of bars
    indices = np.arange(n_groups)
    
    # Bars configuration
    if not targets:
        return
    n_bars = len(targets)
    bar_width = group_width / n_bars
    
    # To center the group of bars on the tick
    # The center of the group is at index.
    # The offsets should range from -(total_width/2) + bar_width/2 to +(total_width/2) - bar_width/2
    total_bar_block_width = n_bars * bar_width
    start_offset = -total_bar_block_width / 2 + bar_width / 2

    for i, tgt in enumerate(targets):
        offset = start_offset + i * bar_width
        color = target_colors.get(tgt, "#333333")
        
        # Collect data for this target across algorithms
        means = []
        stds = []
        for alg in algorithms:
            row = df[(df["algorithm"] == alg) & (df["target"] == tgt)]
            if not row.empty:
                means.append(row.iloc[0]["mean"])
                # std can be NaN when count==1
                std_val = row.iloc[0]["std"]
                stds.append(float(std_val) if pd.notnull(std_val) else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
        
        # Plot bars
        # x positions
        x_pos = indices + offset
        
        ax.bar(
            x_pos, 
            means, 
            yerr=stds, 
            width=bar_width, 
            color=color, 
            edgecolor="black",  # Black border
            linewidth=1.0,      # Slightly thicker border for visibility
            label=tgt if show_legend else "",
            capsize=3,
            error_kw={"ecolor": "black", "elinewidth": 1.0}
        )

    # Axis formatting
    ax.set_xticks(indices)
    ax.set_xticklabels([a.capitalize() for a in algorithms])
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score (0-10)")
    ax.set_title(title)
    
    # Grid (vertical off, horizontal on)
    ax.grid(visible=False, axis='x')
    ax.grid(visible=True, axis='y', alpha=0.3)


def plot_overall(
    df: pd.DataFrame, 
    out_path: Path, 
    algorithms: List[str], 
    targets: List[str],
    target_colors: Dict[str, str]
):
    """Generates the Overall chart (single plot)."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    
    _plot_cluster_group(
        ax, 
        df, 
        algorithms, 
        targets,
        target_colors, 
        show_legend=True, 
        title="Overall LLM Blind Evaluation"
    )
    
    # Place legend outside or best location
    # Since we have only 3 items, upper right or outside is fine.
    # Let's put it upper right inside if space permits, or outside top.
    # Given ylim=10 and potential high scores, outside might be safer, 
    # but let's try 'best' with frameon=True for visibility over grid.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    
    save_fig(fig, out_path)
    print_good(f"Saved Overall chart to {out_path}")


def plot_by_model(
    df: pd.DataFrame, 
    out_path: Path, 
    models: List[str], 
    algorithms: List[str], 
    targets: List[str],
    target_colors: Dict[str, str]
):
    """Generates the By-Model chart (1 row x N columns)."""
    n_models = len(models)
    
    # Figure sizing: Width depends on N models. 
    # Base width per subplot ~4 inches.
    fig_width = 4.0 * n_models
    fig_height = 4.5
    
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=n_models, 
        figsize=(fig_width, fig_height), 
        constrained_layout=True,
        sharey=True
    )
    
    if n_models == 1:
        axes = [axes]
    
    # Plot each model
    for i, model in enumerate(models):
        ax = axes[i]
        model_df = df[df["model"] == model]
        
        # Put the legend inside the rightmost subplot (top-right corner),
        # instead of a figure-level legend above the entire grid.
        show_legend = (i == n_models - 1)
        _plot_cluster_group(
            ax, 
            model_df, 
            algorithms, 
            targets,
            target_colors, 
            show_legend=show_legend, 
            title=model
        )

        if show_legend:
            ax.legend(loc="upper right", frameon=False)
        
        # Remove y label for non-first plots to clean up
        if i > 0:
            ax.set_ylabel("")

    save_fig(fig, out_path)
    print_good(f"Saved By-Model chart to {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_plot():
    report_path = abs_path(cfg.paths["output"]) / "mining" / "llm_blind_eval_report.txt"
    out_dir = report_path.parent
    ensure_dir(out_dir)

    try:
        config, overall_df, bymodel_df = parse_report(report_path)
    except Exception as e:
        print_error(f"Failed to parse report at {report_path}: {e}")
        return

    # Extract configured items or use defaults (report config is the source of truth).
    algorithms = list(config.get("algorithms", ALGORITHM_ORDER))
    models = list(config.get("models", []))

    # Determine which targets exist (handle arbitrary subset of {binned_proba, logits, binary}).
    configured_targets = list(config.get("targets", TARGET_PREFERENCE))
    present_targets: set[str] = set()
    if not overall_df.empty and "target" in overall_df.columns:
        present_targets |= set(overall_df["target"].dropna().astype(str).unique().tolist())
    if not bymodel_df.empty and "target" in bymodel_df.columns:
        present_targets |= set(bymodel_df["target"].dropna().astype(str).unique().tolist())

    if present_targets:
        targets = _order_by_preference([t for t in configured_targets if t in present_targets], TARGET_PREFERENCE)
        # If report has targets not listed in config for any reason, still plot them.
        unknown = [t for t in sorted(present_targets) if t not in targets]
        targets = targets + unknown
    else:
        targets = _order_by_preference(configured_targets, TARGET_PREFERENCE)
    
    # Filter algorithms to ensure we only plot what we know/expect if needed,
    # or just use what's in the report. Using report's algorithms ensures consistency.
    # However, user asked for fixed order: Subgroup -> RuleFit -> ARM.
    # We should sort the report's algorithms by this order if they exist.
    present_algs = set(overall_df["algorithm"].unique()) | set(bymodel_df["algorithm"].unique())
    sorted_algs = [a for a in ALGORITHM_ORDER if a in present_algs]
    # If there are other algorithms not in our fixed list, append them?
    others = [a for a in present_algs if a not in ALGORITHM_ORDER]
    final_algs = sorted_algs + sorted(others)

    if not final_algs:
        print_warn("No algorithms found in report data.")
        return

    # Colors (derived from actual targets plotted)
    target_colors = _get_target_colors(targets)

    # Plot Overall
    if not overall_df.empty:
        plot_overall(
            overall_df, 
            out_dir / "llm_blind_eval_overall_barchart.png", 
            final_algs, 
            targets,
            target_colors
        )
    else:
        print_warn("Overall results dataframe is empty.")

    # Plot By Model
    if not bymodel_df.empty and models:
        plot_by_model(
            bymodel_df, 
            out_dir / "llm_blind_eval_by_model_barchart.png", 
            models, 
            final_algs, 
            targets,
            target_colors
        )
    else:
        print_warn("By-model results dataframe is empty or no models found.")


def main():
    run_plot()


if __name__ == "__main__":
    main()

