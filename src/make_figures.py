"""Figure generation for the POLCA Net assignment (arXiv:2410.07289).

The script reads the artefacts produced by :mod:`src.evaluate` (``results/
metrics.csv``, ``results/ablation.csv``, ``results/latents.npz`` and
``results/loss_history.json``) and writes the eight figures of the report
into ``results/figures/``:

1. ``reconstruction_grid.png``   -- original / PCA / AE / POLCA reconstructions
2. ``latent_orthogonality.png``  -- cosine-similarity heat map
3. ``variance_ordering.png``     -- per-dimension latent variance
4. ``latent_scatter.png``        -- 2-D latent embedding coloured by class
5. ``classification_metrics.png``-- accuracy per method per classifier
6. ``ablation.png``              -- ablation study
7. ``loss_curves.png``           -- POLCA training curves
8. ``dataset_summary.png``       -- reconstruction metrics across datasets

Usage::

    python src/make_figures.py --out-dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.polca.utils import ensure_dir  # noqa: E402

__all__ = [
    "apply_style",
    "figure_reconstruction_grid",
    "figure_latent_orthogonality",
    "figure_variance_ordering",
    "figure_latent_scatter",
    "figure_classification_metrics",
    "figure_ablation",
    "figure_loss_curves",
    "figure_dataset_summary",
    "make_all_figures",
    "build_parser",
    "main",
]

#: Colour cycle used consistently across every figure.
PALETTE = [
    "#3B6EA5",
    "#D1495B",
    "#3F8F5C",
    "#E9A03B",
    "#7A5C9E",
    "#4C9F9E",
    "#B5651D",
    "#8C8C8C",
]

#: Display names of the methods.
METHOD_LABELS = {
    "pca": "PCA",
    "lda": "LDA",
    "ae": "Autoencoder",
    "polca": "POLCA Net",
    "polca_sup": "POLCA Net + $L_{class}$",
}

#: Display names of the ablation variants.
ABLATION_LABELS = {
    "polca_full": "Full POLCA",
    "polca_no_ort": "w/o $L_{ort}$",
    "polca_no_com": "w/o $L_{com}$",
    "polca_no_var": "w/o $L_{var}$",
}

#: Order in which the methods are shown.
METHOD_ORDER = ["pca", "lda", "ae", "polca", "polca_sup"]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def apply_style() -> None:
    """Apply a clean, publication-like matplotlib style."""
    available = {
        str(getattr(entry, "name", "")).lower()
        for entry in matplotlib.font_manager.fontManager.ttflist
    }
    family = "serif"
    if any("times" in name for name in available):
        family = "Times New Roman"
    elif any("dejavu serif" in name for name in available):
        family = "DejaVu Serif"

    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _fig(nrows: int = 1, ncols: int = 1, figsize: Optional[tuple] = None, **kwargs):
    """Create a figure with the project's default figure size."""
    if figsize is None:
        figsize = (4.2 * ncols, 3.4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def _save(fig, path: Path) -> Path:
    """Save a figure with a tight layout."""
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _method_color(method: str) -> str:
    """Return the colour associated with a method name."""
    if method in METHOD_ORDER:
        return PALETTE[METHOD_ORDER.index(method) % len(PALETTE)]
    return PALETTE[hash(method) % len(PALETTE)]


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _load_metrics(out_dir: Path) -> Optional[pd.DataFrame]:
    """Load ``metrics.csv`` (returns ``None`` when it does not exist)."""
    path = out_dir / "metrics.csv"
    if not path.exists():
        print(f"  ! {path} not found -- skipping the figures that need it")
        return None
    return pd.read_csv(path)


def _load_latents(out_dir: Path) -> Dict[str, Any]:
    """Load the cached latent codes (``latents.npz``)."""
    path = out_dir / "latents.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_loss_history(out_dir: Path) -> Dict[str, Any]:
    """Load the cached training histories (``loss_history.json``)."""
    path = out_dir / "loss_history.json"
    if not path.exists():
        return {}
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_images(dataset: str, n: int = 6):
    """Load a handful of images of a dataset (returns images and labels)."""
    from src.polca.data import get_dataset

    bundle = get_dataset(dataset, train=True)
    images = np.asarray(bundle["images"])
    labels = np.asarray(bundle["y"])
    return images[:n], labels[:n]


def _to_display_image(flat: np.ndarray, shape: tuple) -> np.ndarray:
    """Reshape a flat feature vector into a 2-D image in ``[0, 1]``."""
    arr = np.asarray(flat, dtype=np.float64).reshape(shape)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-12:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 1. Reconstruction grid
# --------------------------------------------------------------------------- #
def figure_reconstruction_grid(
    out_dir: Path,
    fig_dir: Path,
    datasets: Sequence[str] = ("digits", "mnist", "fashion"),
    n_samples: int = 6,
) -> Optional[Path]:
    """Original vs. PCA / Autoencoder / POLCA reconstructions."""
    latents = _load_latents(out_dir)
    if not latents:
        print("  ! no latents.npz -- skipping reconstruction_grid.png")
        return None

    datasets = [d for d in datasets if any(k.startswith(f"{d}__") for k in latents)]
    if not datasets:
        print("  ! no matching latent entries -- skipping reconstruction_grid.png")
        return None

    # which methods have a cached reconstruction for this dataset?
    available: List[str] = []
    for k in latents:
        if not k.endswith("__recon_test"):
            continue
        parts = k.split("__")
        if parts[0] in datasets and parts[1] not in available:
            available.append(parts[1])
    methods = [m for m in ("pca", "ae", "polca") if m in available] or available[:3]
    if not methods:
        print("  ! no cached reconstructions -- skipping reconstruction_grid.png")
        return None

    # --- the actual grid: rows = (dataset, method), cols = samples ------- #
    cols = n_samples
    n_rows = len(datasets) * (1 + len(methods))
    fig, axes = _fig(n_rows, cols, figsize=(1.25 * cols, 1.3 * n_rows))

    row = 0
    for dataset in datasets:
        images, _ = _dataset_images(dataset, n=n_samples)
        shape = (8, 8) if dataset == "digits" else (28, 28)

        for method in ["original"] + methods:
            key = None
            if method != "original":
                for k in sorted(latents):
                    if (
                        k.startswith(f"{dataset}__{method}__")
                        and k.endswith("__recon_test")
                    ):
                        key = k
                        break
            recon = latents[key] if key is not None else None

            for c in range(cols):
                ax = axes[row, c]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if method == "original":
                    if c < images.shape[0]:
                        ax.imshow(images[c].astype(np.float64) / 255.0, cmap="gray", vmin=0, vmax=1)
                    else:
                        ax.axis("off")
                elif recon is not None and c < recon.shape[0]:
                    ax.imshow(_to_display_image(recon[c], shape), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.axis("off")
                if c == 0:
                    label = (
                        "Original"
                        if method == "original"
                        else METHOD_LABELS.get(method, method)
                    )
                    ax.text(
                        -0.18,
                        0.5,
                        f"{dataset}\n{label}",
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        fontsize=8,
                    )
            row += 1

    return _save(fig, fig_dir / "reconstruction_grid.png")


# --------------------------------------------------------------------------- #
# 2. Latent orthogonality heat map
# --------------------------------------------------------------------------- #
def figure_latent_orthogonality(
    out_dir: Path,
    fig_dir: Path,
    dataset: str = "digits",
    latent_dim: int = 8,
    seed: int = 5,
    datasets: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """Heat map of the latent cosine-similarity matrix (PCA vs. POLCA).

    When several datasets are given, one row of panels is produced per
    dataset so that the orthogonality property can be checked on all of
    them rather than on one.
    """
    from src.metrics import cosine_similarity_matrix

    latents = _load_latents(out_dir)
    if not latents:
        print("  ! no latents.npz -- skipping latent_orthogonality.png")
        return None

    target_datasets = list(datasets) if datasets else [dataset]
    rows: List[Tuple[str, List[Tuple[str, np.ndarray]]]] = []
    for ds in target_datasets:
        panels: List[Tuple[str, np.ndarray]] = []
        for method in ("pca", "polca"):
            key = f"{ds}__{method}__d{latent_dim}__s{seed}__z_test"
            if key not in latents:
                candidates = [
                    k
                    for k in latents
                    if k.startswith(f"{ds}__{method}__") and k.endswith("__z_test")
                ]
                if not candidates:
                    continue
                key = sorted(candidates)[0]
            z = latents[key]
            panels.append((METHOD_LABELS.get(method, method), cosine_similarity_matrix(z)))

        if panels:
            rows.append((ds, panels))

    if not rows:
        print("  ! no usable latent entries -- skipping latent_orthogonality.png")
        return None

    n_cols = max(len(panels) for _, panels in rows)
    fig, axes = _fig(len(rows), n_cols, figsize=(4.2 * n_cols, 3.6 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (ds, panels) in enumerate(rows):
        for c, (title, sim) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{title} latent space ({ds})")
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel("Latent dimension")
            n = sim.shape[0]
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.grid(False)
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j,
                        i,
                        f"{sim[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(sim[i, j]) > 0.55 else "black",
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")
        # hide the unused panels of a short row
        for c in range(len(panels), n_cols):
            axes[r][c].axis("off")

    fig.suptitle(
        "Latent orthogonality: cosine-similarity matrix (paper convention)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, fig_dir / "latent_orthogonality.png")


# --------------------------------------------------------------------------- #
# 3. Variance ordering
# --------------------------------------------------------------------------- #
def figure_variance_ordering(
    out_dir: Path,
    fig_dir: Path,
    dataset: str = "digits",
    latent_dim: int = 8,
    seed: int = 5,
    datasets: Optional[Sequence[str]] = None,
) -> Optional[Path]:
    """PCA explained-variance curve vs. the POLCA latent variances.

    When several datasets are given, one row of panels is produced per
    dataset.
    """
    from src.metrics import latent_variance_per_dim

    latents = _load_latents(out_dir)
    if not latents:
        print("  ! no latents.npz -- skipping variance_ordering.png")
        return None

    target_datasets = list(datasets) if datasets else [dataset]

    fig, axes = _fig(len(target_datasets), 2, figsize=(9.0, 3.5 * len(target_datasets)))
    axes = np.atleast_2d(axes)

    for r, ds in enumerate(target_datasets):
        pca_key = f"{ds}__pca__d{latent_dim}__s{seed}__z_test"
        polca_key = f"{ds}__polca__d{latent_dim}__s{seed}__z_test"
        for key in (pca_key, polca_key):
            if key not in latents:
                candidates = [
                    k
                    for k in latents
                    if k.startswith(f"{ds}__{key.split('__')[1]}__") and k.endswith("__z_test")
                ]
                if candidates:
                    if key == pca_key:
                        pca_key = sorted(candidates)[0]
                    else:
                        polca_key = sorted(candidates)[0]

        # --- PCA explained variance -------------------------------------- #
        ax = axes[r][0]
        ev_key = None
        for k in latents:
            if k.startswith(f"{ds}__pca__") and k.endswith("__explained_variance_ratio"):
                ev_key = k
                break
        if ev_key is not None:
            ev = np.asarray(latents[ev_key], dtype=np.float64)
            idx = np.arange(1, ev.size + 1)
            ax.bar(idx, ev, color=PALETTE[0], width=0.62, label="PCA explained variance")
            ax.plot(idx, np.cumsum(ev), color=PALETTE[1], marker="o", label="Cumulative")
            ax.set_xlabel("Principal component")
            ax.set_ylabel("Explained variance ratio")
            ax.set_title(f"PCA variance spectrum ({ds})")
            ax.set_xticks(idx)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No PCA explained variance available",
                    ha="center", va="center")
            ax.set_axis_off()

        # --- POLCA latent variances ------------------------------------- #
        ax = axes[r][1]
        if polca_key in latents:
            var = np.asarray(latent_variance_per_dim(latents[polca_key]), dtype=np.float64)
            idx = np.arange(1, var.size + 1)
            ax.bar(idx, var, color=PALETTE[2], width=0.62, label="POLCA latent variance")
            ax.plot(idx, var, color=PALETTE[3], marker="s", linestyle="--", label="Ordered")
            ax.set_xlabel("Latent component")
            ax.set_ylabel("Variance")
            ax.set_title(f"POLCA latent variances ({ds})")
            ax.set_xticks(idx)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No POLCA latent codes available", ha="center", va="center")
            ax.set_axis_off()

    fig.suptitle("Variance ordering of the latent space", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, fig_dir / "variance_ordering.png")


# --------------------------------------------------------------------------- #
# 4. Latent scatter
# --------------------------------------------------------------------------- #
def figure_latent_scatter(
    out_dir: Path,
    fig_dir: Path,
    datasets: Sequence[str] = ("digits", "mnist", "fashion"),
    seed: int = 5,
    max_points: int = 1500,
) -> Optional[Path]:
    """2-D scatter of the first two latent dimensions, coloured by class."""
    from src.polca.data import get_dataset

    latents = _load_latents(out_dir)
    if not latents:
        print("  ! no latents.npz -- skipping latent_scatter.png")
        return None

    datasets = [d for d in datasets if any(k.startswith(f"{d}__") for k in latents)]
    if not datasets:
        print("  ! no matching latent entries -- skipping latent_scatter.png")
        return None

    methods = ("pca", "polca")
    fig, axes = _fig(len(datasets), len(methods), figsize=(4.3 * len(methods), 3.5 * len(datasets)))
    axes = np.atleast_2d(axes)

    cmap = plt.get_cmap("tab10")
    for r, dataset in enumerate(datasets):
        bundle = get_dataset(dataset, train=True)
        y_all = np.asarray(bundle["y"]).reshape(-1)
        for c, method in enumerate(methods):
            ax = axes[r, c]
            key = None
            for k in sorted(latents):
                if k.startswith(f"{dataset}__{method}__") and k.endswith("__z_test"):
                    key = k
                    break
            if key is None:
                ax.text(0.5, 0.5, f"No {method} latent codes", ha="center", va="center")
                ax.set_axis_off()
                continue
            z = latents[key]
            z_test = z
            # the labels of the test split (cached during the evaluation)
            y = None
            y_key = key.replace("__z_test", "__y_test")
            if y_key in latents:
                y = np.asarray(latents[y_key]).reshape(-1)
            else:
                from src.polca.data import train_test_split_idx

                y_all = np.asarray(bundle["y"]).reshape(-1)
                _, test_idx = train_test_split_idx(len(y_all), 0.2, seed)
                y = y_all[test_idx] if len(test_idx) == len(z_test) else y_all[: len(z_test)]

            if z_test.shape[1] < 2:
                ax.text(0.5, 0.5, "Latent dimension < 2", ha="center", va="center")
                ax.set_axis_off()
                continue

            sel = np.arange(z_test.shape[0])[:max_points]
            for cls in np.unique(y[sel]):
                mask = sel[y[sel] == cls]
                ax.scatter(
                    z_test[mask, 0],
                    z_test[mask, 1],
                    s=7,
                    alpha=0.75,
                    color=cmap(int(cls) % 10),
                    edgecolors="none",
                    label=str(int(cls)),
                )
            ax.set_xlabel("Latent dim 1")
            ax.set_ylabel("Latent dim 2")
            ax.set_title(f"{METHOD_LABELS.get(method, method)} -- {dataset}")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, ncol=2, markerscale=1.4, title="class", title_fontsize=8)

    fig.suptitle("Latent space projection (first two components)", fontsize=11)
    return _save(fig, fig_dir / "latent_scatter.png")


# --------------------------------------------------------------------------- #
# 5. Classification metrics
# --------------------------------------------------------------------------- #
def figure_classification_metrics(
    out_dir: Path,
    fig_dir: Path,
    datasets: Sequence[str] = ("digits", "mnist", "fashion"),
) -> Optional[Path]:
    """Grouped bar chart of the accuracy of every linear classifier."""
    df = _load_metrics(out_dir)
    if df is None:
        return None

    df = df[df["classifier"].astype(str) != "n/a"]
    classifiers = [c for c in ["Perceptron", "Ridge", "LogisticRegression", "LinearSVM"] if c in set(df["classifier"])]
    if not classifiers:
        classifiers = sorted(set(df["classifier"]))
    if not classifiers:
        print("  ! no classification rows -- skipping classification_metrics.png")
        return None

    datasets = [d for d in datasets if d in set(df["dataset"])] or sorted(set(df["dataset"]))
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]

    fig, axes = _fig(len(datasets), 1, figsize=(10.0, 3.4 * len(datasets)))
    axes = np.atleast_1d(axes)

    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(classifiers))

    for ax, dataset in zip(axes, datasets):
        sub = df[df["dataset"] == dataset]
        for i, method in enumerate(methods):
            means, errors = [], []
            for clf in classifiers:
                vals = sub[(sub["method"] == method) & (sub["classifier"] == clf)]["accuracy"]
                vals = vals.dropna()
                means.append(float(vals.mean()) if len(vals) else 0.0)
                errors.append(float(vals.std(ddof=0)) if len(vals) else 0.0)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                means,
                width=width,
                yerr=errors,
                capsize=2,
                color=_method_color(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{dataset}")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8, ncol=len(methods))
    axes[0].set_title("Downstream classification accuracy on the latent space")
    fig.suptitle(
        "Linear classifiers trained on the learned latent representations",
        fontsize=11,
        y=1.0,
    )
    return _save(fig, fig_dir / "classification_metrics.png")


# --------------------------------------------------------------------------- #
# 6. Ablation
# --------------------------------------------------------------------------- #
def figure_ablation(out_dir: Path, fig_dir: Path, datasets: Sequence[str] = ()) -> Optional[Path]:
    """Bar chart of the ablation study."""
    path = out_dir / "ablation.csv"
    if not path.exists():
        print("  ! ablation.csv not found -- skipping ablation.png")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print("  ! ablation.csv is empty -- skipping ablation.png")
        return None

    datasets = [d for d in datasets if d in set(df["dataset"])] or sorted(set(df["dataset"]))
    variants = [v for v in ABLATION_LABELS if v in set(df["variant"])]
    if not variants:
        variants = sorted(set(df["variant"]))

    metrics = [
        ("test_nrmse", "NRMSE (lower is better)", False),
        ("test_psnr", "PSNR (dB, higher is better)", True),
        ("test_ssim", "SSIM (higher is better)", True),
        ("orthogonality_error", "Orthogonality error (lower is better)", False),
        ("variance_ordering_score", "Variance ordering score", True),
    ]
    metrics = [(k, lab, up) for k, lab, up in metrics if k in df.columns]

    fig, axes = _fig(len(datasets), len(metrics), figsize=(3.3 * len(metrics), 3.2 * len(datasets)))
    axes = np.atleast_2d(axes)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)

    x = np.arange(len(variants))
    width = 0.8 / max(len(datasets), 1)

    for r, dataset in enumerate(datasets):
        sub = df[df["dataset"] == dataset]
        for c, (key, label, _) in enumerate(metrics):
            ax = axes[r, c]
            for i, variant in enumerate(variants):
                vals = sub[sub["variant"] == variant][key].dropna()
                value = float(vals.mean()) if len(vals) else 0.0
                ax.bar(
                    x + i * width - 0.4 + width / 2,
                    value,
                    width=width,
                    color=PALETTE[i % len(PALETTE)],
                    label=ABLATION_LABELS.get(variant, variant),
                )
            ax.set_xticks(x)
            ax.set_xticklabels([ABLATION_LABELS.get(v, v) for v in variants], fontsize=8)
            ax.set_title(f"{dataset}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_xlabel("")
            if r == 0 and c == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Ablation of the POLCA loss terms", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ensure_dir(fig_dir)
    path_out = fig_dir / "ablation.png"
    fig.savefig(path_out)
    plt.close(fig)
    print(f"  wrote {path_out}")
    return path_out


# --------------------------------------------------------------------------- #
# 7. Loss curves
# --------------------------------------------------------------------------- #
def figure_loss_curves(out_dir: Path, fig_dir: Path, max_panels: int = 3) -> Optional[Path]:
    """Training curves of the POLCA objective (total + components)."""
    history = _load_loss_history(out_dir)
    if not history:
        print("  ! loss_history.json not found -- skipping loss_curves.png")
        return None

    keys = sorted(history.keys())
    # prefer the unsupervised POLCA runs
    keys.sort(key=lambda k: (0 if "__polca__" in k else 1, k))
    keys = keys[:max_panels]
    if not keys:
        return None

    fig, axes = _fig(1, len(keys), figsize=(4.2 * len(keys), 3.5))
    axes = np.atleast_1d(axes)
    components = [
        ("total", "Total", PALETTE[0]),
        ("rec", "$L_{rec}$", PALETTE[1]),
        ("ort", r"$\alpha L_{ort}$", PALETTE[2]),
        ("com", r"$\beta L_{com}$", PALETTE[3]),
        ("var", r"$\gamma L_{var}$", PALETTE[4]),
        ("class", "$L_{class}$", PALETTE[5]),
    ]

    for ax, key in zip(axes, keys):
        hist = history[key]
        epochs = np.arange(1, len(hist.get("total", [])) + 1)
        for comp, label, color in components:
            values = hist.get(comp)
            if not values:
                continue
            if comp == "total":
                ax.plot(epochs, values, color=color, linewidth=2.0, label=label)
            else:
                ax.plot(epochs, values, color=color, linewidth=1.2, linestyle="--", label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.set_title(key.replace("__", " / "), fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle("POLCA Net training loss components", fontsize=11)
    return _save(fig, fig_dir / "loss_curves.png")


# --------------------------------------------------------------------------- #
# 8. Dataset summary
# --------------------------------------------------------------------------- #
def figure_dataset_summary(out_dir: Path, fig_dir: Path) -> Optional[Path]:
    """Reconstruction metrics across every dataset and method."""
    df = _load_metrics(out_dir)
    if df is None:
        return None

    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    datasets = sorted(set(df["dataset"]))
    if not methods or not datasets:
        return None

    metrics = [
        ("test_nrmse", "NRMSE"),
        ("test_psnr", "PSNR (dB)"),
        ("test_ssim", "SSIM"),
    ]
    fig, axes = _fig(1, len(metrics), figsize=(4.0 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)

    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)

    for ax, (key, label) in zip(axes, metrics):
        for i, method in enumerate(methods):
            means, errors = [], []
            for dataset in datasets:
                vals = df[(df["dataset"] == dataset) & (df["method"] == method)][key].dropna()
                means.append(float(vals.mean()) if len(vals) else 0.0)
                errors.append(float(vals.std(ddof=0)) if len(vals) else 0.0)
            ax.bar(
                x + i * width - 0.4 + width / 2,
                means,
                width=width,
                yerr=errors,
                capsize=2,
                color=_method_color(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Reconstruction quality per dataset and method", fontsize=11)
    return _save(fig, fig_dir / "dataset_summary.png")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def make_all_figures(
    out_dir: str = "results",
    datasets: Sequence[str] = ("digits", "mnist", "fashion"),
    latent_dim: int = 8,
    seed: int = 5,
) -> List[Path]:
    """Generate every figure of the report into ``out_dir/figures``."""
    out_path = Path(out_dir)
    fig_dir = ensure_dir(out_path / "figures")
    apply_style()

    print(f"Generating figures in {fig_dir}")
    produced: List[Path] = []

    for func, kwargs in [
        (figure_reconstruction_grid, {"datasets": datasets}),
        (figure_latent_orthogonality, {"dataset": datasets[0], "latent_dim": latent_dim, "seed": seed, "datasets": datasets}),
        (figure_variance_ordering, {"dataset": datasets[0], "latent_dim": latent_dim, "seed": seed, "datasets": datasets}),
        (figure_latent_scatter, {"datasets": datasets, "seed": seed}),
        (figure_classification_metrics, {"datasets": datasets}),
        (figure_ablation, {"datasets": datasets}),
        (figure_loss_curves, {}),
        (figure_dataset_summary, {}),
    ]:
        try:
            path = func(out_path, fig_dir, **kwargs)
        except Exception as exc:  # pragma: no cover - keep going on failures
            print(f"  ! {func.__name__} failed: {exc}")
            path = None
        if path is not None:
            produced.append(path)

    print(f"Generated {len(produced)} figure(s).")
    return produced


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser of the figure script."""
    parser = argparse.ArgumentParser(
        description="Generate all the figures of the POLCA Net assignment.",
    )
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--datasets", nargs="+", default=["digits", "mnist", "fashion"]
    )
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python src/make_figures.py``."""
    args = build_parser().parse_args(argv)
    make_all_figures(
        out_dir=args.out_dir,
        datasets=tuple(args.datasets),
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
