from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.stats import gaussian_kde

from src.config import cfg
from src.statistics import stats
from src.utils import abs_path, print_good, save_fig


plt.rcParams.update(cfg.plots["rcParams"])
plt.rcParams["axes.prop_cycle"] = cycler(color=cfg.plots["colors"])


def _eda_cfg() -> Mapping[str, Any]:
    try:
        eda = cfg.plots["eda"]
    except Exception as e:
        raise KeyError("Missing cfg.plots['eda'] in src/config.yaml") from e

    if not isinstance(eda, Mapping):
        raise TypeError("cfg.plots['eda'] must be a mapping")
    return eda


def _require_key(m: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in m:
        raise KeyError(f"Missing key '{key}' in {where}")
    return m[key]


def _label_palette(eda: Mapping[str, Any]) -> tuple[list[int], dict[int, str], dict[int, str]]:
    label = _require_key(eda, "label", where="cfg.plots['eda']")
    if not isinstance(label, Mapping):
        raise TypeError("cfg.plots['eda'].label must be a mapping")

    values = _require_key(label, "values", where="cfg.plots['eda'].label")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("cfg.plots['eda'].label.values must be a list")

    labels = [int(v) for v in values]
    if len(set(labels)) != len(labels):
        raise ValueError("cfg.plots['eda'].label.values must be unique")

    idx = _require_key(label, "color_index", where="cfg.plots['eda'].label")
    leg = _require_key(label, "legend", where="cfg.plots['eda'].label")
    if not isinstance(idx, Mapping) or not isinstance(leg, Mapping):
        raise TypeError("cfg.plots['eda'].label.color_index and .legend must be mappings")

    palette = list(cfg.plots["colors"])
    if not palette:
        raise ValueError("cfg.plots['colors'] must be non-empty")

    colors: dict[int, str] = {}
    legend: dict[int, str] = {}

    for lab in labels:
        if lab not in idx:
            raise KeyError(f"Missing label {lab} in cfg.plots['eda'].label.color_index")
        if lab not in leg:
            raise KeyError(f"Missing label {lab} in cfg.plots['eda'].label.legend")

        ci = int(idx[lab])
        if ci < 0 or ci >= len(palette):
            raise ValueError(
                f"Color index for label {lab} out of range: {ci} (len(plots.colors)={len(palette)})"
            )

        colors[lab] = str(palette[ci])
        legend[lab] = str(leg[lab])

    return labels, colors, legend


def _fig_dpi() -> int:
    rc = cfg.plots["rcParams"]
    if not isinstance(rc, Mapping):
        raise TypeError("cfg.plots['rcParams'] must be a mapping")
    if "figure.dpi" not in rc:
        raise KeyError("Missing cfg.plots['rcParams']['figure.dpi']")
    return int(rc["figure.dpi"])


def _make_grid(
    n_items: int,
    n_cols: int,
    *,
    col_width: float,
    row_height: float,
    constrained_layout: bool,
) -> tuple[plt.Figure, list[plt.Axes]]:
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if n_cols <= 0:
        raise ValueError("n_cols must be positive")

    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_width * n_cols, row_height * n_rows),
        constrained_layout=bool(constrained_layout),
        squeeze=False,
    )

    axes_flat = list(axes.flatten())
    for ax in axes_flat[n_items:]:
        ax.axis("off")

    return fig, axes_flat[:n_items]


def _feature_groups() -> tuple[list[str], list[str]]:
    features = cfg.dataset["features"]
    if not isinstance(features, Mapping):
        raise TypeError("cfg.dataset['features'] must be a mapping")

    cat = [k for k, t in features.items() if str(t) == "cat"]
    num = [k for k, t in features.items() if str(t) in {"int", "num"}]
    return cat, num


def _validate_schema(df: pd.DataFrame, *, label_col: str, features: Sequence[str]) -> None:
    missing = {label_col, *features} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")


def _sorted_categories(s: pd.Series, *, sort_categories: bool) -> list[Any]:
    cats = pd.Index(s.dropna().unique())
    cats = cats.sort_values() if bool(sort_categories) else cats
    return list(cats)


def _categorical_categories(
    df: pd.DataFrame,
    cat_features: Sequence[str],
    *,
    sort_categories: bool,
) -> tuple[dict[str, list[Any]], int]:
    cats: dict[str, list[Any]] = {
        f: _sorted_categories(df[f], sort_categories=sort_categories) for f in cat_features
    }
    max_bins = max((len(v) for v in cats.values()), default=0)
    return cats, int(max_bins)


def _plot_categorical(
    df: pd.DataFrame,
    *,
    label_col: str,
    features: Sequence[str],
    eda: Mapping[str, Any],
    labels: Sequence[int],
    label_colors: Mapping[int, str],
    label_legend: Mapping[int, str],
) -> plt.Figure:
    cfg_cat = _require_key(eda, "categorical", where="cfg.plots['eda']")
    if not isinstance(cfg_cat, Mapping):
        raise TypeError("cfg.plots['eda'].categorical must be a mapping")

    layout = _require_key(eda, "layout", where="cfg.plots['eda']")
    if not isinstance(layout, Mapping):
        raise TypeError("cfg.plots['eda'].layout must be a mapping")

    figsize = _require_key(layout, "figsize", where="cfg.plots['eda'].layout")
    if not isinstance(figsize, Mapping):
        raise TypeError("cfg.plots['eda'].layout.figsize must be a mapping")

    n_cols = int(_require_key(cfg_cat, "n_cols", where="cfg.plots['eda'].categorical"))
    col_w = float(_require_key(figsize, "col_width", where="cfg.plots['eda'].layout.figsize"))
    row_h = float(_require_key(figsize, "row_height", where="cfg.plots['eda'].layout.figsize"))
    constrained = bool(_require_key(layout, "constrained_layout", where="cfg.plots['eda'].layout"))

    style = _require_key(eda, "style", where="cfg.plots['eda']")
    if not isinstance(style, Mapping):
        raise TypeError("cfg.plots['eda'].style must be a mapping")

    edgecolor = str(_require_key(style, "bin_edgecolor", where="cfg.plots['eda'].style"))
    linewidth = float(_require_key(style, "bin_linewidth", where="cfg.plots['eda'].style"))

    normalize = str(_require_key(cfg_cat, "normalize", where="cfg.plots['eda'].categorical")).strip()
    if normalize not in {"count", "within_label"}:
        raise ValueError("cfg.plots['eda'].categorical.normalize must be 'count' or 'within_label'")

    sort_categories = bool(_require_key(cfg_cat, "sort_categories", where="cfg.plots['eda'].categorical"))
    cats_map, max_bins = _categorical_categories(df, features, sort_categories=sort_categories)
    if max_bins <= 0:
        raise ValueError("No categorical bins found")

    axis_cfg = _require_key(cfg_cat, "axis", where="cfg.plots['eda'].categorical")
    bar_cfg = _require_key(cfg_cat, "bar", where="cfg.plots['eda'].categorical")
    xtick_cfg = _require_key(cfg_cat, "xtick", where="cfg.plots['eda'].categorical")
    legend_cfg = _require_key(cfg_cat, "legend", where="cfg.plots['eda'].categorical")
    titles_cfg = _require_key(cfg_cat, "titles", where="cfg.plots['eda'].categorical")
    if not all(isinstance(x, Mapping) for x in (axis_cfg, bar_cfg, xtick_cfg, legend_cfg, titles_cfg)):
        raise TypeError("categorical.axis/bar/xtick/legend/titles must all be mappings")

    axis_pad = float(_require_key(axis_cfg, "pad", where="cfg.plots['eda'].categorical.axis"))
    width_ratio = float(_require_key(bar_cfg, "width_ratio", where="cfg.plots['eda'].categorical.bar"))
    alpha = float(_require_key(bar_cfg, "alpha", where="cfg.plots['eda'].categorical.bar"))

    rot = float(_require_key(xtick_cfg, "rotation", where="cfg.plots['eda'].categorical.xtick"))
    ha = str(_require_key(xtick_cfg, "ha", where="cfg.plots['eda'].categorical.xtick"))

    show_legend = bool(_require_key(legend_cfg, "enable", where="cfg.plots['eda'].categorical.legend"))
    only_first = bool(
        _require_key(legend_cfg, "only_first_subplot", where="cfg.plots['eda'].categorical.legend")
    )

    fig_title = str(_require_key(titles_cfg, "figure", where="cfg.plots['eda'].categorical.titles"))
    subplot_fmt = str(
        _require_key(titles_cfg, "subplot_fmt", where="cfg.plots['eda'].categorical.titles")
    )

    fig, axes = _make_grid(
        len(features),
        n_cols,
        col_width=col_w,
        row_height=row_h,
        constrained_layout=constrained,
    )

    n_labels = len(labels)
    if n_labels <= 0:
        raise ValueError("No labels configured")

    group_w = float(width_ratio)
    bar_w = group_w / n_labels
    offsets = (np.arange(n_labels) - (n_labels - 1) / 2.0) * bar_w

    for i, (ax, feat) in enumerate(zip(axes, features)):
        cats = cats_map[feat]
        x = np.arange(len(cats), dtype=float)

        for j, lab in enumerate(labels):
            s = df.loc[df[label_col] == lab, feat]
            counts = s.value_counts(dropna=True).reindex(cats, fill_value=0).astype(float).to_numpy()
            if normalize == "within_label":
                denom = float(np.sum(counts))
                counts = counts / denom if denom > 0 else counts

            ax.bar(
                x + float(offsets[j]),
                counts,
                width=bar_w,
                color=label_colors[int(lab)],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=label_legend[int(lab)],
            )

        ax.set_title(subplot_fmt.format(feature=feat))
        ax.set_xlim(-axis_pad, (max_bins - 1) + axis_pad)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats], rotation=rot, ha=ha)

        if show_legend and (not only_first or i == 0):
            ax.legend()

    fig.suptitle(fig_title)
    return fig


def _hist_edges(values: np.ndarray, bins: object) -> np.ndarray:
    if isinstance(bins, (str, bytes)):
        return np.asarray(np.histogram_bin_edges(values, bins=str(bins)), dtype=float)
    if isinstance(bins, (int, np.integer)):
        return np.asarray(np.histogram_bin_edges(values, bins=int(bins)), dtype=float)
    raise TypeError("cfg.plots['eda'].numeric.bins.method must be a string or an integer")


def _numeric_bin_edges(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    bins_cfg: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], int]:
    method = _require_key(bins_cfg, "method", where="cfg.plots['eda'].numeric.bins")
    min_bins = int(_require_key(bins_cfg, "min_bins", where="cfg.plots['eda'].numeric.bins"))
    max_bins = int(_require_key(bins_cfg, "max_bins", where="cfg.plots['eda'].numeric.bins"))

    if min_bins <= 0 or max_bins <= 0 or min_bins > max_bins:
        raise ValueError("numeric.bins.min_bins/max_bins must be positive with min_bins <= max_bins")

    edges_map: dict[str, np.ndarray] = {}
    for feat in features:
        v = df[feat].to_numpy(dtype=float)
        edges = _hist_edges(v, method)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"Invalid histogram edges for feature '{feat}': shape={edges.shape}")

        n_bins = int(edges.size - 1)
        clamped = int(np.clip(n_bins, min_bins, max_bins))
        if clamped != n_bins:
            edges = _hist_edges(v, clamped)

        edges_map[feat] = np.asarray(edges, dtype=float)

    max_n_bins = max((e.size - 1 for e in edges_map.values()), default=0)
    return edges_map, int(max_n_bins)


def _plot_numeric(
    df: pd.DataFrame,
    *,
    label_col: str,
    features: Sequence[str],
    eda: Mapping[str, Any],
    labels: Sequence[int],
    label_colors: Mapping[int, str],
    label_legend: Mapping[int, str],
) -> plt.Figure:
    cfg_num = _require_key(eda, "numeric", where="cfg.plots['eda']")
    if not isinstance(cfg_num, Mapping):
        raise TypeError("cfg.plots['eda'].numeric must be a mapping")

    layout = _require_key(eda, "layout", where="cfg.plots['eda']")
    if not isinstance(layout, Mapping):
        raise TypeError("cfg.plots['eda'].layout must be a mapping")

    figsize = _require_key(layout, "figsize", where="cfg.plots['eda'].layout")
    if not isinstance(figsize, Mapping):
        raise TypeError("cfg.plots['eda'].layout.figsize must be a mapping")

    n_cols = int(_require_key(cfg_num, "n_cols", where="cfg.plots['eda'].numeric"))
    col_w = float(_require_key(figsize, "col_width", where="cfg.plots['eda'].layout.figsize"))
    row_h = float(_require_key(figsize, "row_height", where="cfg.plots['eda'].layout.figsize"))
    constrained = bool(_require_key(layout, "constrained_layout", where="cfg.plots['eda'].layout"))

    style = _require_key(eda, "style", where="cfg.plots['eda']")
    if not isinstance(style, Mapping):
        raise TypeError("cfg.plots['eda'].style must be a mapping")

    edgecolor = str(_require_key(style, "bin_edgecolor", where="cfg.plots['eda'].style"))
    linewidth = float(_require_key(style, "bin_linewidth", where="cfg.plots['eda'].style"))

    normalize = str(_require_key(cfg_num, "normalize", where="cfg.plots['eda'].numeric")).strip()
    if normalize not in {"density", "count"}:
        raise ValueError("cfg.plots['eda'].numeric.normalize must be 'density' or 'count'")

    bins_cfg = _require_key(cfg_num, "bins", where="cfg.plots['eda'].numeric")
    hist_cfg = _require_key(cfg_num, "hist", where="cfg.plots['eda'].numeric")
    kde_cfg = _require_key(cfg_num, "kde", where="cfg.plots['eda'].numeric")
    x_cfg = _require_key(cfg_num, "x", where="cfg.plots['eda'].numeric")
    legend_cfg = _require_key(cfg_num, "legend", where="cfg.plots['eda'].numeric")
    titles_cfg = _require_key(cfg_num, "titles", where="cfg.plots['eda'].numeric")

    if not all(isinstance(x, Mapping) for x in (bins_cfg, hist_cfg, kde_cfg, x_cfg, legend_cfg, titles_cfg)):
        raise TypeError("numeric.bins/hist/kde/x/legend/titles must all be mappings")

    edges_map, max_n_bins = _numeric_bin_edges(df, features, bins_cfg=bins_cfg)
    if max_n_bins <= 0:
        raise ValueError("No numeric bins found")

    width_ratio = float(_require_key(hist_cfg, "width_ratio", where="cfg.plots['eda'].numeric.hist"))
    alpha = float(_require_key(hist_cfg, "alpha", where="cfg.plots['eda'].numeric.hist"))

    kde_enable = bool(_require_key(kde_cfg, "enable", where="cfg.plots['eda'].numeric.kde"))
    bw_method = _require_key(kde_cfg, "bw_method", where="cfg.plots['eda'].numeric.kde")
    grid_size = int(_require_key(kde_cfg, "grid_size", where="cfg.plots['eda'].numeric.kde"))
    kde_lw = float(_require_key(kde_cfg, "linewidth", where="cfg.plots['eda'].numeric.kde"))
    kde_alpha = float(_require_key(kde_cfg, "alpha", where="cfg.plots['eda'].numeric.kde"))

    pad_ratio = float(_require_key(x_cfg, "pad_ratio", where="cfg.plots['eda'].numeric.x"))

    show_legend = bool(_require_key(legend_cfg, "enable", where="cfg.plots['eda'].numeric.legend"))
    only_first = bool(_require_key(legend_cfg, "only_first_subplot", where="cfg.plots['eda'].numeric.legend"))

    fig_title = str(_require_key(titles_cfg, "figure", where="cfg.plots['eda'].numeric.titles"))
    subplot_fmt = str(_require_key(titles_cfg, "subplot_fmt", where="cfg.plots['eda'].numeric.titles"))

    fig, axes = _make_grid(
        len(features),
        n_cols,
        col_width=col_w,
        row_height=row_h,
        constrained_layout=constrained,
    )

    y = df[label_col].to_numpy().reshape(-1)
    if y.shape[0] != len(df):
        raise ValueError("label_col length mismatch")

    density = normalize == "density"

    for i, (ax, feat) in enumerate(zip(axes, features)):
        edges = edges_map[feat]
        x0, x1 = float(edges[0]), float(edges[-1])
        x_range = float(x1 - x0)
        if not np.isfinite(x_range) or x_range <= 0:
            raise ValueError(f"Invalid x-range for feature '{feat}': {x0}..{x1}")

        pad = float(pad_ratio) * x_range
        ax.set_xlim(x0 - pad, x1 + pad)

        centers = 0.5 * (edges[1:] + edges[:-1])
        bar_w = (x_range / max_n_bins) * float(width_ratio)

        for lab in labels:
            vals = df.loc[y == lab, feat].to_numpy(dtype=float)
            hist, _ = np.histogram(vals, bins=edges, density=density)

            ax.bar(
                centers,
                hist,
                width=bar_w,
                align="center",
                color=label_colors[int(lab)],
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=label_legend[int(lab)],
            )

            if kde_enable:
                if vals.size < 2 or np.unique(vals).size < 2:
                    continue

                kde = gaussian_kde(vals, bw_method=bw_method)
                x_grid = np.linspace(x0, x1, int(grid_size))
                y_grid = kde(x_grid)

                if not density:
                    bin_w = float(np.mean(np.diff(edges)))
                    y_grid = y_grid * float(vals.size) * bin_w

                ax.plot(
                    x_grid,
                    y_grid,
                    color=label_colors[int(lab)],
                    linewidth=kde_lw,
                    alpha=kde_alpha,
                )

        ax.set_title(subplot_fmt.format(feature=feat))
        if show_legend and (not only_first or i == 0):
            ax.legend()

    fig.suptitle(fig_title)
    return fig


def _save(fig: plt.Figure, *, eda: Mapping[str, Any], filename_key: str) -> None:
    out_dir = str(_require_key(eda, "out_dir", where="cfg.plots['eda']"))
    files = _require_key(eda, "files", where="cfg.plots['eda']")
    if not isinstance(files, Mapping):
        raise TypeError("cfg.plots['eda'].files must be a mapping")
    if filename_key not in files:
        raise KeyError(f"Missing cfg.plots['eda'].files['{filename_key}']")

    out_path = abs_path(out_dir) / str(files[filename_key])
    save_fig(fig, out_path, dpi=_fig_dpi())
    print_good(f"Saved EDA plot to {out_path}")


def main() -> None:
    eda = _eda_cfg()

    key = str(_require_key(eda, "data_path_key", where="cfg.plots['eda']"))
    if key not in cfg.paths:
        raise KeyError(f"cfg.plots['eda'].data_path_key='{key}' not found in cfg.paths")

    df = pd.read_csv(abs_path(cfg.paths[key]))
    stats.init(df)

    label_col = str(cfg.dataset["label"])
    cat_features, num_features = _feature_groups()
    _validate_schema(df, label_col=label_col, features=[*cat_features, *num_features])

    labels, label_colors, label_legend = _label_palette(eda)

    if cat_features:
        fig = _plot_categorical(
            df,
            label_col=label_col,
            features=cat_features,
            eda=eda,
            labels=labels,
            label_colors=label_colors,
            label_legend=label_legend,
        )
        _save(fig, eda=eda, filename_key="categorical")

    if num_features:
        fig = _plot_numeric(
            df,
            label_col=label_col,
            features=num_features,
            eda=eda,
            labels=labels,
            label_colors=label_colors,
            label_legend=label_legend,
        )
        _save(fig, eda=eda, filename_key="numeric")


if __name__ == "__main__":
    main()
