from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from cycler import cycler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from src.config import cfg
from src.utils import abs_path, save_fig, print_good

plt.rcParams.update(cfg.plots["rcParams"])
plt.rcParams["axes.prop_cycle"] = cycler(color=cfg.plots["colors"])

out_dir = abs_path(cfg.paths["output"]) / "cluster"


@dataclass(frozen=True)
class DRMethod:
    name: str
    label: str
    params: Mapping[str, Any]


_DR_REGISTRY: dict[str, Any] = {"pca": PCA, "umap": UMAP, "tsne": TSNE}


def generate_report(
    X: np.ndarray,
    y_preds: Dict[str, pd.Series | np.ndarray],
    *,
    tag: str = "none",
    binary_only: bool = True,
) -> None:
    """
    Args:
        X: Feature/embedding matrix used for clustering, shape (n_samples, n_features)
        y_preds: {name: labels}
            - Must contain "ground_truth"
            - If binary_only=True, only labels 0/1 are colored and others are shown in gray.
            - If binary_only=False, all non-negative labels are colored (cycled), noise (<0) is gray.
        tag: Report suffix. "none" keeps the legacy filename.
    """
    _validate_inputs(X, y_preds)
    dim_reduction_and_plot(X, y_preds, tag=tag, binary_only=binary_only)



def generate_color_verification(
    X: np.ndarray,
    X_raw: pd.DataFrame,
    p_y1: np.ndarray,
    *,
    tag: str,
    proba_label: str = "TabPFN P(y=1)",
) -> None:
    """
    Generates a KxN grid (K dim-reduction rows) where each column is colored by a
    feature (from X_raw) or TabPFN probability.
    """
    X = np.asarray(X)
    p_y1 = np.asarray(p_y1).reshape(-1)
    if X.shape[0] != p_y1.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != p_y1 rows {p_y1.shape[0]}")
    if X.shape[0] != len(X_raw):
        raise ValueError(f"X rows {X.shape[0]} != X_raw rows {len(X_raw)}")

    # 1. Prepare values to plot
    # Dict[name, (values, kind)]
    # kind: "cat" or "cont"
    to_plot: dict[str, tuple[np.ndarray, str]] = {}
    
    # Features from config order
    features_cfg = cfg.dataset["features"]
    for name, dtype in features_cfg.items():
        if name not in X_raw.columns:
            continue
        vals = X_raw[name].to_numpy()
        kind = "cat" if dtype == "cat" else "cont"
        to_plot[name] = (vals, kind)
        
    # TabPFN probability
    to_plot[proba_label] = (p_y1, "cont")
    
    # 2. Compute embeddings
    methods = _dr_methods_from_cfg()
    embeddings = _compute_embeddings(X, methods)
    
    # 3. Plot
    fig = _plot_cv_grid(to_plot, _dr_rows(methods, embeddings))
    
    suffix = "" if str(tag).lower() in {"", "none"} else f"_{tag}"
    out_path = out_dir / f"color_verification{suffix}.png"
    save_fig(fig, out_path)
    print_good(f"Saved color verification plot to {out_path}")


def dim_reduction_and_plot(
    X: np.ndarray,
    y_preds: Mapping[str, pd.Series | np.ndarray],
    *,
    tag: str,
    binary_only: bool,
) -> None:
    """
    Plot: K rows x (N+1) cols
        - Rows: dim-reduction methods from cfg.plots.cluster_report.dim_reduction.methods
        - Col 1: ground truth, remaining cols: clustering assignments
    """
    keys = _ordered_keys(y_preds)
    methods = _dr_methods_from_cfg()
    embeddings = _compute_embeddings(X, methods)
    fig = _plot_grid(keys, y_preds, _dr_rows(methods, embeddings), binary_only=binary_only)

    suffix = "" if str(tag).lower() in {"", "none"} else f"_{tag}"
    out_path = out_dir / f"cluster_comparison{suffix}.png"
    save_fig(fig, out_path)
    print_good(f"Saved cluster comparison plot to {out_path}")


def _validate_inputs(X: np.ndarray, y_preds: Mapping[str, object]) -> None:
    if "ground_truth" not in y_preds:
        raise ValueError('y_preds must contain key "ground_truth"')

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("X is empty")

    colors = cfg.plots["colors"]
    if len(colors) < 2:
        raise ValueError("cfg.plots['colors'] must contain at least 2 colors for labels 0/1")

    for k, v in y_preds.items():
        a = _to_1d_array(v)
        if a.shape[0] != n:
            raise ValueError(f"Label length mismatch for '{k}': {a.shape[0]} vs X has {n} samples")


def _ordered_keys(y_preds: Mapping[str, object]) -> list[str]:
    gt = "ground_truth"
    primary = [m for m in cfg.cluster_models if m in y_preds and m != gt]
    seen = {gt, *primary}
    extras = [k for k in y_preds.keys() if k not in seen]
    return [gt, *primary, *extras]


def _dr_methods_from_cfg() -> list[DRMethod]:
    try:
        methods = cfg.plots["cluster_report"]["dim_reduction"]["methods"]
    except Exception as e:  # fail-fast with clearer message
        raise KeyError("Missing cfg.plots.cluster_report.dim_reduction.methods") from e

    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise TypeError("cfg.plots.cluster_report.dim_reduction.methods must be a list")
    if not methods:
        raise ValueError("cfg.plots.cluster_report.dim_reduction.methods must be non-empty")

    out: list[DRMethod] = []
    for i, raw in enumerate(methods):
        if not isinstance(raw, Mapping):
            raise TypeError(f"methods[{i}] must be a mapping")

        name = str(raw.get("name", "")).strip().lower()
        label = str(raw.get("label", "")).strip()
        params = raw.get("params", None)
        if not name:
            raise ValueError(f"methods[{i}].name is required")
        if name not in _DR_REGISTRY:
            raise ValueError(f"Unknown DR method '{name}'. Supported: {sorted(_DR_REGISTRY)}")
        if not label:
            raise ValueError(f"methods[{i}].label is required")
        if not isinstance(params, Mapping):
            raise TypeError(f"methods[{i}].params must be a mapping (can be empty)")

        out.append(DRMethod(name=name, label=label, params=dict(params)))

    labels = [m.label for m in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate DR labels in config: {labels}")

    return out


def _make_reducer(method: DRMethod) -> Any:
    ctor = _DR_REGISTRY.get(str(method.name).lower())
    if ctor is None:
        raise ValueError(f"Unknown DR method: {method.name}")
    return ctor(**dict(method.params))


def _compute_embeddings(X: np.ndarray, methods: Sequence[DRMethod]) -> Dict[str, np.ndarray]:
    X = np.asarray(X)
    return {m.label: np.asarray(_make_reducer(m).fit_transform(X)) for m in methods}


def _dr_rows(methods: Sequence[DRMethod], embeddings: Mapping[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    return [(m.label, np.asarray(embeddings[m.label])) for m in methods]


def _plot_grid(
    keys: Sequence[str],
    y_preds: Mapping[str, pd.Series | np.ndarray],
    rows: Sequence[tuple[str, np.ndarray]],
    *,
    binary_only: bool,
) -> plt.Figure:
    n_cols = len(keys)
    plot_cfg = cfg.plots["cluster_report"]
    col_w = plot_cfg["figsize"]["col_width"]
    row_h = plot_cfg["figsize"]["row_height"]
    n_rows = len(rows)
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_w * n_cols, row_h * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for r, (row_name, emb) in enumerate(rows):
        _plot_row(
            axes[r],
            emb,
            keys,
            y_preds,
            row_name=row_name,
            show_titles=(r == 0),
            binary_only=binary_only,
        )

    return fig


def _plot_cv_grid(
    to_plot: Mapping[str, tuple[np.ndarray, str]],
    rows: Sequence[tuple[str, np.ndarray]],
) -> plt.Figure:
    keys = list(to_plot.keys())
    n_cols = len(keys)
    plot_cfg = cfg.plots["cluster_report"]
    col_w = plot_cfg["figsize"]["col_width"]
    row_h = plot_cfg["figsize"]["row_height"]
    n_rows = len(rows)
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(col_w * n_cols, row_h * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    cmap = _gradient_cmap()

    for r, (row_name, emb) in enumerate(rows):
        emb_2d = np.asarray(emb)
        xlim, ylim = _axis_limits(emb_2d)
        
        for c, key in enumerate(keys):
            ax = axes[r, c]
            vals, kind = to_plot[key]
            
            colors = _values_to_colors(vals, kind, cmap)
            
            ax.scatter(
                emb_2d[:, 0],
                emb_2d[:, 1],
                c=colors,
                **plot_cfg["scatter"],
            )
            ax.set(xlim=xlim, ylim=ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            
            if r == 0:
                ax.set_title(key)
        
        axes[r, 0].set_ylabel(row_name)

    return fig

def _plot_row(
    axes_row: Sequence[plt.Axes],
    emb_2d: np.ndarray,
    keys: Sequence[str],
    y_preds: Mapping[str, pd.Series | np.ndarray],
    *,
    row_name: str,
    show_titles: bool,
    binary_only: bool,
) -> None:
    emb_2d = np.asarray(emb_2d)
    if emb_2d.ndim != 2 or emb_2d.shape[1] != 2:
        raise ValueError(f"{row_name} embedding must be (n, 2), got {emb_2d.shape}")

    plot_cfg = cfg.plots["cluster_report"]
    scatter_kwargs = plot_cfg["scatter"]

    xlim, ylim = _axis_limits(emb_2d)
    for ax, key in zip(axes_row, keys):
        labels = _to_1d_array(y_preds[key])
        ax.scatter(
            emb_2d[:, 0],
            emb_2d[:, 1],
            c=_label_colors(labels, binary_only=(True if key == "ground_truth" else binary_only)),
            **scatter_kwargs,
        )
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_titles:
            ax.set_title(_display_name(key))

    axes_row[0].set_ylabel(row_name)


def _axis_limits(emb_2d: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x = emb_2d[:, 0]
    y = emb_2d[:, 1]
    
    pad_factor = cfg.plots["cluster_report"]["layout"]["pad_factor"]
    
    x_pad = pad_factor * (float(x.max()) - float(x.min()) + 1e-12)
    y_pad = pad_factor * (float(y.max()) - float(y.min()) + 1e-12)
    return (float(x.min()) - x_pad, float(x.max()) + x_pad), (float(y.min()) - y_pad, float(y.max()) + y_pad)


def _to_1d_array(labels: object) -> np.ndarray:
    if isinstance(labels, pd.Series):
        labels = labels.to_numpy()
    return np.asarray(labels).reshape(-1)


def _label_colors(labels: np.ndarray, *, binary_only: bool) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    palette = list(cfg.plots["colors"])
    other = cfg.plots["cluster_report"]["colors"]["other"]

    if binary_only:
        if len(palette) < 2:
            raise ValueError("cfg.plots['colors'] must contain at least 2 colors for labels 0/1")
        color_0, color_1 = palette[0], palette[1]
        out = np.full(labels.shape[0], other, dtype=object)
        out[labels == 0] = color_0
        out[labels == 1] = color_1
        return out

    if len(palette) < 1:
        raise ValueError("cfg.plots['colors'] must contain at least 1 color for multi-cluster plots")
    out = np.full(labels.shape[0], other, dtype=object)
    core = labels[labels >= 0]
    for i, lab in enumerate(np.unique(core)):
        out[labels == lab] = palette[i % len(palette)]
    return out


def _gradient_cmap() -> LinearSegmentedColormap:
    """Creates a continuous colormap interpolated from cfg.plots['colors']."""
    colors = cfg.plots["colors"]
    if len(colors) < 2:
        # Fallback if not enough colors
        return plt.get_cmap("viridis")
    return LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)


def _values_to_colors(
    values: np.ndarray,
    kind: str,
    cmap: LinearSegmentedColormap,
) -> np.ndarray:
    """
    Maps values to colors.
    kind="cat": discrete mapping from cfg.plots["colors"].
    kind="cont": continuous mapping using cmap.
    """
    palette = list(cfg.plots["colors"])
    other = cfg.plots["cluster_report"]["colors"]["other"]
    
    if kind == "cat":
        # Handle categoricals
        # Sort unique values to ensure stable coloring
        uniques = np.unique(values)
        out = np.full(len(values), other, dtype=object)
        
        for i, val in enumerate(uniques):
            out[values == val] = palette[i % len(palette)]
        return out
        
    else: # kind == "cont"
        # Handle continuous
        # Normalize to [0, 1]
        vals = np.asarray(values, dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        
        if np.isnan(vmin) or np.isnan(vmax) or (vmin == vmax):
            # All same or all nan
            return np.full(len(values), other, dtype=object)
            
        norm = Normalize(vmin=vmin, vmax=vmax)
        # Apply colormap -> (N, 4) RGBA
        return cmap(norm(vals))


def _display_name(key: str) -> str:
    return "Ground Truth" if key == "ground_truth" else key.replace("_", " ").upper()
