"""Full experimental evaluation of POLCA Net (arXiv:2410.07289).

The script trains / fits every method on every requested dataset and writes

* ``results/metrics.csv``  -- one row per (dataset, method, seed)
* ``results/metrics.json`` -- the aggregated (mean / std) metrics
* ``results/summary.md``   -- a human-readable table mirroring Table 1 of
  the paper (NRMSE, PSNR, SSIM, Accuracy, F1)
* ``results/ablation.csv`` -- the ablation study (each POLCA term removed)
* ``results/latents.npz``  -- cached latent codes / reconstructions used by
  :mod:`src.make_figures`

Command line usage::

    python src/evaluate.py --datasets digits --seeds 5 --epochs 3 --out-dir results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src import metrics as M  # noqa: E402
from src.baselines import LDABaseline, PCABaseline  # noqa: E402
from src.polca.data import get_dataset, train_test_split_idx  # noqa: E402
from src.polca.utils import ensure_dir, get_device, save_csv, save_json, set_seed  # noqa: E402
from src.train import train_plain_ae, train_polca  # noqa: E402

__all__ = [
    "run_full_evaluation",
    "METHODS",
    "ABLATIONS",
    "build_parser",
    "main",
]

#: Methods evaluated by default.
METHODS = ("pca", "lda", "ae", "polca", "polca_sup")

#: The ablation variants (each disables one POLCA loss term).
ABLATIONS = {
    "polca_full": {"use_ort": True, "use_com": True, "use_var": True},
    "polca_no_ort": {"use_ort": False, "use_com": True, "use_var": True},
    "polca_no_com": {"use_ort": True, "use_com": False, "use_var": True},
    "polca_no_var": {"use_ort": True, "use_com": True, "use_var": False},
}

#: Human readable names used in the summary table.
PRETTY_NAMES = {
    "pca": "PCA",
    "lda": "LDA",
    "ae": "Autoencoder",
    "polca": "POLCA Net",
    "polca_sup": "POLCA Net + L_class",
    "polca_full": "POLCA Net (full)",
    "polca_no_ort": "w/o L_ort",
    "polca_no_com": "w/o L_com",
    "polca_no_var": "w/o L_var",
}

#: Column order of ``results/metrics.csv``.
CSV_COLUMNS = [
    "dataset",
    "method",
    "seed",
    "latent_dim",
    "num_latent_dims",
    "train_nrmse",
    "train_psnr",
    "train_ssim",
    "test_nrmse",
    "test_psnr",
    "test_ssim",
    "orthogonality_error",
    "variance_ordering_score",
    "classifier",
    "accuracy",
    "f1",
    "train_time_sec",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensors / arrays to a float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _reconstruction_metrics(
    x: np.ndarray,
    recon: np.ndarray,
    image_shape: Optional[tuple],
) -> Dict[str, float]:
    """NRMSE / PSNR / SSIM between the originals and the reconstructions."""
    x = _to_numpy(x)
    recon = _to_numpy(recon).reshape(x.shape)
    return {
        "nrmse": float(M.nrmse(x, recon)),
        "psnr": float(M.psnr(x, recon)),
        "ssim": float(M.ssim(x, recon, data_range=1.0, image_shape=image_shape)),
    }


def _classification_block(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Run the four linear classifiers on the latent space."""
    try:
        return M.classification_report_from_latent(z_train, y_train, z_test, y_test, seed=seed)
    except Exception as exc:  # pragma: no cover - degenerate latents
        print(f"    ! classification failed: {exc}")
        return {}


def _flatten_classification(
    report: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    """Turn the classifier report into flat CSV rows."""
    rows = []
    for clf_name, scores in report.items():
        rows.append(
            {
                "classifier": clf_name,
                "accuracy": float(scores.get("accuracy", float("nan"))),
                "f1": float(scores.get("f1_macro", scores.get("f1", float("nan")))),
            }
        )
    return rows


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean and (population) standard deviation of a list of numbers."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# Per-method evaluation
# --------------------------------------------------------------------------- #
def _evaluate_latents(
    method: str,
    z_train: np.ndarray,
    z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    """Latent-space diagnostics + downstream classification for one method."""
    report = _classification_block(z_train, y_train, z_test, y_test, seed)
    block: Dict[str, Any] = {
        "orthogonality_error": float(M.orthogonality_error(z_test)),
        "variance_ordering_score": float(M.variance_ordering_score(z_test)),
        "latent_variances": [float(v) for v in M.latent_variance_per_dim(z_test)],
        "classification": report,
    }
    return block


def _run_method(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    image_shape: Optional[tuple],
    latent_dim: int,
    hidden_dims: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device: Optional[str],
    seed: int,
    linear_decoder: bool,
    alpha: float,
    beta: float,
    gamma: float,
    out_dir: Optional[Path] = None,
    keep_artifacts: bool = False,
    max_updates: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit one method and return its metrics, latents and reconstructions."""
    result: Dict[str, Any] = {
        "method": method,
        "train_time_sec": 0.0,
        "latent_dim": int(latent_dim),
        "num_latent_dims": None,
        "history": None,
    }
    artifacts: Dict[str, Any] = {}

    if method == "pca":
        model = PCABaseline(n_components=latent_dim, random_state=seed)
        model.fit(X_train)
        z_train = _to_numpy(model.latent(X_train))
        z_test = _to_numpy(model.latent(X_test))
        recon_train = _to_numpy(model.reconstruct(X_train))
        recon_test = _to_numpy(model.reconstruct(X_test))
        result["num_latent_dims"] = int(z_train.shape[1])
        result["explained_variance_ratio"] = [
            float(v) for v in model.pca.explained_variance_ratio_
        ]

    elif method == "lda":
        model = LDABaseline(n_components=latent_dim, random_state=seed)
        model.fit(X_train, y_train)
        z_train = _to_numpy(model.latent(X_train))
        z_test = _to_numpy(model.latent(X_test))
        recon_train = _to_numpy(model.reconstruct(X_train))
        recon_test = _to_numpy(model.reconstruct(X_test))
        # LDA is limited to n_classes - 1 discriminant dimensions.
        result["num_latent_dims"] = int(z_train.shape[1])

    elif method in ("ae", "ae_sup"):
        supervised = method == "ae_sup"
        out = train_plain_ae(
            X_train,
            y_train if supervised else None,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            seed=seed,
            supervised=supervised,
            linear_decoder=linear_decoder,
            log_every=max(epochs, 1),
            num_classes=int(y_train.max()) + 1,
            verbose=False,
            max_updates=max_updates,
        )
        model = out["model"]
        result["train_time_sec"] = float(out["train_time"])
        result["history"] = out["history"]
        z_train, recon_train = _infer(model, X_train)
        z_test, recon_test = _infer(model, X_test)
        result["num_latent_dims"] = int(z_train.shape[1])

    elif method in ("polca", "polca_sup"):
        supervised = method == "polca_sup"
        out = train_polca(
            X_train,
            y_train if supervised else None,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            seed=seed,
            supervised=supervised,
            linear_decoder=linear_decoder,
            log_every=max(epochs, 1),
            num_classes=int(y_train.max()) + 1,
            verbose=False,
            max_updates=max_updates,
        )
        model = out["model"]
        result["train_time_sec"] = float(out["train_time"])
        result["history"] = out["history"]
        z_train, recon_train = _infer(model, X_train)
        z_test, recon_test = _infer(model, X_test)
        result["num_latent_dims"] = int(z_train.shape[1])

    else:
        raise ValueError(f"Unknown method {method!r}")

    train_metrics = _reconstruction_metrics(X_train, recon_train, image_shape)
    test_metrics = _reconstruction_metrics(X_test, recon_test, image_shape)
    result["train_metrics"] = train_metrics
    result["test_metrics"] = test_metrics
    result.update(_evaluate_latents(method, z_train, z_test, y_train, y_test, seed))

    if keep_artifacts:
        artifacts = {
            "z_test": z_test,
            "recon_test": recon_test,
            "z_train": z_train,
            "recon_train": recon_train,
        }

    result["_artifacts"] = artifacts
    return result


def _infer(model, X: np.ndarray):
    """Encode / decode ``X`` with a torch model (no grad)."""
    import torch

    try:
        dev = next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        dev = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        recon, latent = model(torch.as_tensor(np.asarray(X, dtype=np.float32), device=dev))
    return _to_numpy(latent), _to_numpy(recon)


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def _run_ablation(
    dataset: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
    image_shape: Optional[tuple],
    latent_dim: int,
    hidden_dims: Sequence[int],
    epochs: int,
    lr: float,
    batch_size: int,
    device: Optional[str],
    seed: int,
    linear_decoder: bool,
    alpha: float,
    beta: float,
    gamma: float,
    max_updates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Train POLCA with each of the three regularisers ablated."""
    rows: List[Dict[str, Any]] = []
    for variant, flags in ABLATIONS.items():
        out = train_polca(
            X_train,
            None,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            seed=seed,
            supervised=False,
            use_ort=flags["use_ort"],
            use_com=flags["use_com"],
            use_var=flags["use_var"],
            linear_decoder=linear_decoder,
            log_every=max(epochs, 1),
            verbose=False,
            max_updates=max_updates,
        )
        model = out["model"]
        z_train, recon_train = _infer(model, X_train)
        z_test, recon_test = _infer(model, X_test)
        train_metrics = _reconstruction_metrics(X_train, recon_train, image_shape)
        test_metrics = _reconstruction_metrics(X_test, recon_test, image_shape)
        report = _classification_block(z_train, y_train, z_test, y_test, seed)
        acc = [v["accuracy"] for v in report.values()] or [float("nan")]
        f1 = [v.get("f1_macro", float("nan")) for v in report.values()] or [float("nan")]

        row = {
            "dataset": dataset,
            "variant": variant,
            "seed": int(seed),
            "latent_dim": int(latent_dim),
            "use_ort": flags["use_ort"],
            "use_com": flags["use_com"],
            "use_var": flags["use_var"],
            "alpha": alpha if flags["use_ort"] else 0.0,
            "beta": beta if flags["use_com"] else 0.0,
            "gamma": gamma if flags["use_var"] else 0.0,
            "train_nrmse": train_metrics["nrmse"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
            "test_nrmse": test_metrics["nrmse"],
            "test_psnr": test_metrics["psnr"],
            "test_ssim": test_metrics["ssim"],
            "orthogonality_error": float(M.orthogonality_error(z_test)),
            "variance_ordering_score": float(M.variance_ordering_score(z_test)),
            "accuracy_mean": float(np.mean(acc)),
            "f1_mean": float(np.mean(f1)),
            "train_time_sec": float(out["train_time"]),
        }
        rows.append(row)
        print(
            f"  ablation {variant:<12} nrmse={row['test_nrmse']:.4f} "
            f"psnr={row['test_psnr']:.2f} ssim={row['test_ssim']:.4f} "
            f"ort={row['orthogonality_error']:.4f} "
            f"vorder={row['variance_ordering_score']:.4f}",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def _write_summary(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    methods: Sequence[str],
    notes: Sequence[str],
) -> None:
    """Write ``results/summary.md`` (Table 1 style)."""
    lines: List[str] = []
    lines.append("# POLCA Net -- Experimental Results")
    lines.append("")
    lines.append(
        "Reconstruction metrics on the test split (NRMSE / PSNR / SSIM) and "
        "downstream classification quality (Accuracy / F1, macro-averaged over "
        "Perceptron, Ridge, Logistic Regression and Linear SVM)."
    )
    lines.append("")

    for dataset in datasets:
        lines.append(f"## Dataset: {dataset}")
        lines.append("")
        lines.append(
            "| Method | Latent dims | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for method in methods:
            sel = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not sel:
                continue
            nrmse = _mean_std([r["test_nrmse"] for r in sel])
            psnr = _mean_std([r["test_psnr"] for r in sel])
            ssim = _mean_std([r["test_ssim"] for r in sel])
            acc = _mean_std([r["accuracy"] for r in sel])
            f1 = _mean_std([r["f1"] for r in sel])
            orth = _mean_std([r["orthogonality_error"] for r in sel])
            vorder = _mean_std([r["variance_ordering_score"] for r in sel])
            dims = sel[0].get("num_latent_dims")
            dims_txt = f"{dims}" if dims else "-"
            lines.append(
                f"| {PRETTY_NAMES.get(method, method)} | {dims_txt} | "
                f"{nrmse['mean']:.4f} | {psnr['mean']:.2f} | {ssim['mean']:.4f} | "
                f"{acc['mean']:.4f} | {f1['mean']:.4f} | "
                f"{orth['mean']:.4f} | {vorder['mean']:.4f} |"
            )
        lines.append("")

    if ablation_rows:
        lines.append("## Ablation study (POLCA Net, unsupervised)")
        lines.append("")
        lines.append(
            "| Variant | Dataset | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for variant in ABLATIONS:
            for dataset in datasets:
                sel = [
                    r
                    for r in ablation_rows
                    if r["dataset"] == dataset and r["variant"] == variant
                ]
                if not sel:
                    continue
                lines.append(
                    f"| {PRETTY_NAMES.get(variant, variant)} | {dataset} | "
                    f"{np.mean([r['test_nrmse'] for r in sel]):.4f} | "
                    f"{np.mean([r['test_psnr'] for r in sel]):.2f} | "
                    f"{np.mean([r['test_ssim'] for r in sel]):.4f} | "
                    f"{np.mean([r['accuracy_mean'] for r in sel]):.4f} | "
                    f"{np.mean([r['f1_mean'] for r in sel]):.4f} | "
                    f"{np.mean([r['orthogonality_error'] for r in sel]):.4f} | "
                    f"{np.mean([r['variance_ordering_score'] for r in sel]):.4f} |"
                )
        lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def run_full_evaluation(
    datasets: Sequence[str] = ("digits", "mnist", "fashion"),
    latent_dims: Sequence[int] = (8,),
    epochs: int = 100,
    seeds: Sequence[int] = (5,),
    methods: Sequence[str] = METHODS,
    out_dir: str = "results",
    hidden_dims: Sequence[int] = (64, 32),
    lr: float = 1e-3,
    batch_size: int = 64,
    alpha: float = 1e-2,
    beta: float = 1e-2,
    gamma: float = 1e-7,
    linear_decoder: bool = False,
    device: Optional[str] = None,
    test_size: float = 0.2,
    run_ablation: bool = True,
    save_latents: bool = True,
    max_samples: Optional[int] = None,
    max_updates: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete benchmark and write the result files.

    Parameters
    ----------
    datasets : sequence of str
        Dataset names (``'digits'``, ``'mnist'``, ``'fashion'``).
    latent_dims : sequence of int
        Latent dimensions to evaluate.
    epochs : int
        Number of training epochs for the neural methods.
    max_updates : int, optional
        Fixed budget of gradient updates (the paper uses 20 000).  When set it
        overrides ``epochs``, making the training budget independent of the
        dataset size.
    seeds : sequence of int
        Seeds (the paper uses ``5``).
    methods : sequence of str
        Methods to evaluate -- see :data:`METHODS`.
    out_dir : str, default ``"results"``
        Output directory.
    hidden_dims : sequence of int
        Encoder hidden widths.
    lr, batch_size, alpha, beta, gamma : float
        Hyper-parameters (paper defaults).
    linear_decoder : bool
        Use the linear decoder variant.
    device : str, optional
        ``"cpu"`` / ``"cuda"`` / ``"auto"``.
    test_size : float
        Fraction of the data held out for testing.
    run_ablation : bool
        Also run the ablation study.
    save_latents : bool
        Cache the latent codes in ``latents.npz`` for the figures.
    max_samples : int, optional
        Cap the number of samples per dataset (useful for smoke tests).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{'rows': [...], 'ablation_rows': [...], 'out_dir': str}``.
    """
    out_path = ensure_dir(Path(out_dir))
    dev = device or get_device()
    rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    latent_store: Dict[str, Any] = {}
    history_store: Dict[str, Any] = {}

    for dataset in datasets:
        print(f"\n=== dataset: {dataset} ===", flush=True)
        bundle = get_dataset(dataset, train=True)
        X = np.asarray(bundle["X"], dtype=np.float32)
        y = np.asarray(bundle["y"], dtype=np.int64)
        image_shape = tuple(bundle["image_shape"])
        input_dim = int(bundle["input_dim"])

        if max_samples is not None and X.shape[0] > max_samples:
            keep = np.arange(max_samples)
            X = X[keep]
            y = y[keep]
            notes.append(f"{dataset}: only the first {max_samples} samples were used.")

        for latent_dim in latent_dims:
            for seed in seeds:
                set_seed(int(seed))
                train_idx, test_idx = train_test_split_idx(X.shape[0], test_size, seed)
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                for method in methods:
                    t0 = time.time()
                    print(
                        f"  method={method:<10} latent_dim={latent_dim} seed={seed} ...",
                        flush=True,
                    )
                    try:
                        res = _run_method(
                            method,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            input_dim,
                            image_shape,
                            int(latent_dim),
                            hidden_dims,
                            int(epochs),
                            float(lr),
                            int(batch_size),
                            dev,
                            int(seed),
                            bool(linear_decoder),
                            float(alpha),
                            float(beta),
                            float(gamma),
                            out_dir=out_path,
                            keep_artifacts=True,
                            max_updates=max_updates,
                        )
                    except Exception as exc:
                        print(f"    ! method {method} failed: {exc}")
                        continue

                    clf_rows = _flatten_classification(res["classification"])
                    if not clf_rows:
                        # keep the row even when the classifiers could not run
                        clf_rows = [{"classifier": "n/a", "accuracy": float("nan"), "f1": float("nan")}]
                    for clf_row in clf_rows:
                        row = {
                            "dataset": dataset,
                            "method": method,
                            "seed": int(seed),
                            "latent_dim": int(latent_dim),
                            "num_latent_dims": res.get("num_latent_dims"),
                            "train_nrmse": res["train_metrics"]["nrmse"],
                            "train_psnr": res["train_metrics"]["psnr"],
                            "train_ssim": res["train_metrics"]["ssim"],
                            "test_nrmse": res["test_metrics"]["nrmse"],
                            "test_psnr": res["test_metrics"]["psnr"],
                            "test_ssim": res["test_metrics"]["ssim"],
                            "orthogonality_error": res["orthogonality_error"],
                            "variance_ordering_score": res["variance_ordering_score"],
                            "classifier": clf_row["classifier"],
                            "accuracy": clf_row["accuracy"],
                            "f1": clf_row["f1"],
                            "train_time_sec": res["train_time_sec"],
                        }
                        rows.append(row)

                    key = f"{dataset}__{method}__d{latent_dim}__s{seed}"
                    artifacts = res.get("_artifacts") or {}
                    if save_latents and artifacts:
                        latent_store[f"{key}__z_test"] = artifacts["z_test"]
                        latent_store[f"{key}__recon_test"] = artifacts["recon_test"]
                        latent_store[f"{key}__z_train"] = artifacts["z_train"]
                        latent_store[f"{key}__y_test"] = np.asarray(
                            y_test, dtype=np.int64
                        )
                    if res.get("history") is not None:
                        history_store[key] = res["history"]
                    if res.get("explained_variance_ratio"):
                        latent_store[f"{key}__explained_variance_ratio"] = np.asarray(
                            res["explained_variance_ratio"], dtype=np.float32
                        )

                    print(
                        f"    nrmse={res['test_metrics']['nrmse']:.4f} "
                        f"psnr={res['test_metrics']['psnr']:.2f} "
                        f"ssim={res['test_metrics']['ssim']:.4f} "
                        f"({time.time() - t0:.1f}s)",
                        flush=True,
                    )

            if run_ablation:
                print("  -- ablation --", flush=True)
                ablation_rows.extend(
                    _run_ablation(
                        dataset,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        input_dim,
                        image_shape,
                        int(latent_dim),
                        hidden_dims,
                        int(epochs),
                        float(lr),
                        int(batch_size),
                        dev,
                        int(seeds[0]),
                        bool(linear_decoder),
                        float(alpha),
                        float(beta),
                        float(gamma),
                        max_updates=max_updates,
                    )
                )

    # ---- latent dimension caveat ---------------------------------------- #
    dims_per_method = {}
    for row in rows:
        dims_per_method.setdefault(row["method"], set()).add(row.get("num_latent_dims"))
    if any(len(v) > 1 for v in dims_per_method.values()) or any(
        v and list(v)[0] != list(latent_dims)[0] for v in dims_per_method.values()
    ):
        notes.append(
            "The latent dimensionality differs between methods: LDA is limited to "
            "`n_classes - 1` discriminant dimensions, so its latent space is "
            "smaller than the one of the other methods. The classification "
            "comparison is therefore made in differently sized latent spaces."
        )

    # ---- write the artefacts --------------------------------------------- #
    save_csv(rows, out_path / "metrics.csv", fieldnames=CSV_COLUMNS)
    save_csv(
        ablation_rows,
        out_path / "ablation.csv",
        fieldnames=list(ablation_rows[0].keys()) if ablation_rows else None,
    )

    aggregated: Dict[str, Any] = {}
    for dataset in datasets:
        for method in methods:
            sel = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not sel:
                continue
            aggregated.setdefault(dataset, {})[method] = {
                "test_nrmse": _mean_std([r["test_nrmse"] for r in sel]),
                "test_psnr": _mean_std([r["test_psnr"] for r in sel]),
                "test_ssim": _mean_std([r["test_ssim"] for r in sel]),
                "train_nrmse": _mean_std([r["train_nrmse"] for r in sel]),
                "train_psnr": _mean_std([r["train_psnr"] for r in sel]),
                "train_ssim": _mean_std([r["train_ssim"] for r in sel]),
                "accuracy": _mean_std([r["accuracy"] for r in sel]),
                "f1": _mean_std([r["f1"] for r in sel]),
                "orthogonality_error": _mean_std([r["orthogonality_error"] for r in sel]),
                "variance_ordering_score": _mean_std([r["variance_ordering_score"] for r in sel]),
                "num_latent_dims": sorted({r.get("num_latent_dims") for r in sel}, key=lambda v: (v is None, v)),
                "n_seeds": len({r["seed"] for r in sel}),
            }
    save_json(
        {
            "config": {
                "datasets": list(datasets),
                "latent_dims": [int(d) for d in latent_dims],
                "epochs": int(epochs),
                "max_updates": (int(max_updates) if max_updates is not None else None),
                "seeds": [int(s) for s in seeds],
                "methods": list(methods),
                "hidden_dims": [int(h) for h in hidden_dims],
                "lr": float(lr),
                "batch_size": int(batch_size),
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "linear_decoder": bool(linear_decoder),
                "device": str(dev),
                "test_size": float(test_size),
            },
            "aggregated": aggregated,
            "rows": rows,
            "ablation_rows": ablation_rows,
            "notes": notes,
        },
        out_path / "metrics.json",
    )

    _write_summary(
        out_path / "summary.md",
        rows,
        ablation_rows,
        datasets,
        methods,
        notes,
    )

    if save_latents:
        np.savez_compressed(out_path / "latents.npz", **latent_store)
    if history_store:
        save_json(history_store, out_path / "loss_history.json")

    print(f"\nWrote {out_path/'metrics.csv'} ({len(rows)} rows)", flush=True)
    print(f"Wrote {out_path/'metrics.json'}", flush=True)
    print(f"Wrote {out_path/'summary.md'}", flush=True)
    if ablation_rows:
        print(f"Wrote {out_path/'ablation.csv'} ({len(ablation_rows)} rows)", flush=True)

    return {"rows": rows, "ablation_rows": ablation_rows, "out_dir": str(out_path), "notes": notes}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser of the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run the full POLCA Net evaluation and write the result tables.",
    )
    parser.add_argument("--datasets", nargs="+", default=list(("digits", "mnist", "fashion")))
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[8])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 32])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=1e-7)
    parser.add_argument("--linear-decoder", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help=(
            "Fixed budget of gradient updates (the paper uses 20000). "
            "Overrides --epochs when set."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python src/evaluate.py``."""
    args = build_parser().parse_args(argv)
    run_full_evaluation(
        datasets=args.datasets,
        latent_dims=args.latent_dims,
        epochs=args.epochs,
        seeds=args.seeds,
        methods=args.methods,
        out_dir=args.out_dir,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        linear_decoder=args.linear_decoder,
        device=args.device,
        test_size=args.test_size,
        run_ablation=not args.no_ablation,
        max_samples=args.max_samples,
        max_updates=args.max_updates,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
