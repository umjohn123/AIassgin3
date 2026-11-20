"""Training entry point for POLCA Net (arXiv:2410.07289).

This module implements the optimisation loop of the POLCA Net autoencoder
together with the ablation switches that allow each regulariser of the
composite objective to be switched off individually:

.. math::

    L_{polca} = L_{rec} + L_{class}
                 + \\alpha L_{ort} + \\beta L_{com} + \\gamma L_{var}

Command line usage::

    python src/train.py --dataset digits --latent-dim 8 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# Make ``src`` importable no matter how the script is launched.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.polca.losses import polca_loss  # noqa: E402
from src.polca.model import Autoencoder, POLCANet  # noqa: E402
from src.polca.utils import ensure_dir, get_device, set_seed  # noqa: E402

__all__ = [
    "train_polca",
    "train_plain_ae",
    "build_parser",
    "main",
]

#: Component keys tracked for every training run.
COMPONENT_KEYS = ("total", "rec", "class", "ort", "com", "var")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float_array(x: Any) -> np.ndarray:
    """Convert anything array-like into a ``float32`` numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _resolve_device(device: Optional[str]) -> torch.device:
    """Return a :class:`torch.device` from a string (or auto-detect)."""
    if device is None or str(device).lower() == "auto":
        return torch.device(get_device())
    return torch.device(str(device))


def _make_loader(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> DataLoader:
    """Build a shuffled :class:`~torch.utils.data.DataLoader`."""
    tensors = [torch.as_tensor(X, dtype=torch.float32, device=device)]
    if y is not None:
        tensors.append(torch.as_tensor(np.asarray(y).reshape(-1), dtype=torch.long, device=device))
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=int(max(1, batch_size)),
        shuffle=bool(shuffle),
        drop_last=False,
    )


def _component_value(out: Any, name: str) -> float:
    """Extract a scalar component from the output of :func:`polca_loss`.

    The loss module exposes the classification term as ``cls`` (Python
    keyword), therefore the mapping below translates the friendly names
    used by this script onto the attribute names of ``POLCALossOutput``.
    """
    if out is None:
        return 0.0
    attr = {"class": "cls"}.get(name, name)
    value = getattr(out, attr, None)
    if value is None and isinstance(out, Mapping):
        value = out.get(attr, out.get(name))
    if value is None:
        return 0.0
    try:
        return float(value.detach().item()) if hasattr(value, "detach") else float(value)
    except Exception:  # pragma: no cover - defensive
        return float(value)


def _history_dict() -> Dict[str, list]:
    """Return an empty history container with the expected keys."""
    return {key: [] for key in COMPONENT_KEYS}


def _save_run_artifacts(result: Mapping[str, Any], out_dir: Path, tag: str) -> None:
    """Persist the training history and the configuration of a run."""
    ensure_dir(out_dir)
    history = result.get("history", {})
    with (out_dir / f"history_{tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (out_dir / f"config_{tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(result.get("config", {}), handle, indent=2, default=str)


# --------------------------------------------------------------------------- #
# POLCA training
# --------------------------------------------------------------------------- #
def train_polca(
    X_train,
    y_train=None,
    input_dim: Optional[int] = None,
    latent_dim: int = 8,
    hidden_dims: Sequence[int] = (64, 32),
    alpha: float = 1e-2,
    beta: float = 1e-2,
    gamma: float = 1e-7,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    seed: int = 5,
    supervised: bool = False,
    use_ort: bool = True,
    use_com: bool = True,
    use_var: bool = True,
    linear_decoder: bool = False,
    max_updates: Optional[int] = None,
    log_every: int = 50,
    out_dir: Optional[str] = None,
    num_classes: Optional[int] = None,
    activation: str = "relu",
    tag: str = "polca",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a POLCA Net model.

    Parameters
    ----------
    X_train : array-like
        ``(N, input_dim)`` training inputs in ``[0, 1]``.
    y_train : array-like, optional
        ``(N,)`` labels.  Required when ``supervised=True``.
    input_dim : int, optional
        Dimension of the flattened input.  Inferred from ``X_train`` when
        omitted.
    latent_dim : int, default 8
        Dimension ``n`` of the latent space.
    hidden_dims : sequence of int, default ``(64, 32)``
        Hidden widths of the encoder.
    alpha, beta, gamma : float
        Weights of ``L_ort``, ``L_com`` and ``L_var`` (paper: ``1e-2``,
        ``1e-2``, ``1e-7``).
    epochs : int, default 100
        Number of passes over the training data.  Ignored when
        ``max_updates`` is set.
    max_updates : int, optional
        Maximum number of gradient updates (optimiser steps).  The paper
        trains for a fixed budget of 20 000 gradient updates (Table 2), which
        is reproduced by setting ``max_updates=20000``.  When ``None`` the
        loop is epoch-based.
    lr : float, default 1e-3
        Adam learning rate.
    batch_size : int, default 64
        Mini-batch size (the paper uses 64).
    device : str, optional
        ``"cpu"`` / ``"cuda"`` / ``"auto"``.
    seed : int, default 5
        Random seed.
    supervised : bool, default False
        When ``True`` the cross-entropy term ``L_class`` is enabled.
    use_ort, use_com, use_var : bool, default True
        Ablation switches.  Disabling one sets the corresponding weight to
        zero.
    linear_decoder : bool, default False
        Use the paper's optional linear decoder.
    log_every : int, default 50
        Print the running losses every ``log_every`` epochs (``0`` disables
        the progress output).
    out_dir : str, optional
        Directory where the history / config JSON files are written.
    num_classes : int, optional
        Number of classes of the classification head (inferred from
        ``y_train`` when omitted).
    activation : str, default ``"relu"``
        Hidden activation of the autoencoder.
    tag : str, default ``"polca"``
        Name used for the artefact files.
    verbose : bool, default True
        Print the progress lines.

    Returns
    -------
    dict
        ``{'model': POLCANet, 'history': {...}, 'config': {...},
        'train_time': float, 'final': {...}}``.
    """
    x = _as_float_array(X_train)
    if x.ndim != 2:
        raise ValueError(f"X_train must be 2-D, got shape {x.shape}")
    n_samples, dim = x.shape
    if input_dim is None:
        input_dim = dim
    input_dim = int(input_dim)
    if input_dim != dim:
        raise ValueError(f"input_dim={input_dim} does not match X_train dim {dim}")

    # ---- ablation: switching a term off sets its weight to zero -------- #
    alpha_eff = float(alpha) if use_ort else 0.0
    beta_eff = float(beta) if use_com else 0.0
    gamma_eff = float(gamma) if use_var else 0.0

    set_seed(int(seed))
    dev = _resolve_device(device)

    model = POLCANet(
        input_dim=input_dim,
        latent_dim=int(latent_dim),
        hidden_dims=tuple(int(h) for h in hidden_dims),
        linear_decoder=bool(linear_decoder),
        activation=activation,
    ).to(dev)

    labels = None
    classifier = None
    if supervised:
        if y_train is None:
            raise ValueError("supervised=True requires y_train")
        labels = np.asarray(y_train).reshape(-1).astype(np.int64)
        if num_classes is None:
            num_classes = int(labels.max()) + 1
        num_classes = int(num_classes)
        classifier = nn.Linear(int(latent_dim), num_classes).to(dev)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None

    loader = _make_loader(x, labels, batch_size, dev, shuffle=True)

    parameters = list(model.parameters())
    if classifier is not None:
        parameters += list(classifier.parameters())
    optimiser = torch.optim.Adam(parameters, lr=float(lr))

    history = _history_dict()
    start_time = time.time()

    # ------------------------------------------------------------------ #
    # Training loop
    #
    # Two protocols are supported:
    #   * epoch based  -- ``epochs`` passes over the training set;
    #   * update based -- a fixed budget of gradient updates, which is what
    #     the paper uses (20 000 gradient updates, Table 2).  This makes the
    #     training budget independent of the dataset size, so the same
    #     number of updates is used for every dataset and can be compared
    #     directly with the published results.
    # ------------------------------------------------------------------ #
    steps_per_epoch = max(len(loader), 1)
    if max_updates is not None:
        total_epochs = int(math.ceil(int(max_updates) / steps_per_epoch))
    else:
        total_epochs = int(epochs)

    updates_done = 0
    update_budget = int(max_updates) if max_updates is not None else None

    for epoch in range(total_epochs):
        model.train()
        if classifier is not None:
            classifier.train()
        sums = {key: 0.0 for key in COMPONENT_KEYS}
        count = 0
        for batch in loader:
            if update_budget is not None and updates_done >= update_budget:
                break
            if labels is None:
                xb = batch[0]
                yb = None
            else:
                xb, yb = batch[0], batch[1]

            optimiser.zero_grad(set_to_none=True)
            recon, latent = model(xb)
            out = polca_loss(
                recon,
                latent,
                xb,
                labels=yb,
                class_criterion=criterion,
                alpha=alpha_eff,
                beta=beta_eff,
                gamma=gamma_eff,
                classifier=classifier,
                return_components=True,
            )
            loss = out.total
            loss.backward()
            optimiser.step()
            updates_done += 1

            bs = int(xb.shape[0])
            count += bs
            for key in COMPONENT_KEYS:
                sums[key] += _component_value(out, key) * bs

        for key in COMPONENT_KEYS:
            history[key].append(sums[key] / max(count, 1))

        if log_every and verbose and (
            (epoch + 1) % int(log_every) == 0 or epoch == 0 or (epoch + 1) == total_epochs
        ):
            print(
                f"[POLCA{'-sup' if supervised else ''}] epoch {epoch + 1}/{total_epochs} "
                f"updates={updates_done} "
                f"total={history['total'][-1]:.6f} rec={history['rec'][-1]:.6f} "
                f"ort={history['ort'][-1]:.6f} com={history['com'][-1]:.6f} "
                f"var={history['var'][-1]:.6f}"
                + (f" cls={history['class'][-1]:.6f}" if supervised else ""),
                flush=True,
            )

    train_time = time.time() - start_time
    model.eval()
    if classifier is not None:
        classifier.eval()

    config = {
        "method": "polca_sup" if supervised else "polca",
        "input_dim": input_dim,
        "latent_dim": int(latent_dim),
        "hidden_dims": [int(h) for h in hidden_dims],
        "alpha": alpha_eff,
        "beta": beta_eff,
        "gamma": gamma_eff,
        "use_ort": bool(use_ort),
        "use_com": bool(use_com),
        "use_var": bool(use_var),
        "supervised": bool(supervised),
        "epochs": int(epochs),
        "max_updates": (int(max_updates) if max_updates is not None else None),
        "gradient_updates": int(updates_done),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "batch_size_used": int(min(batch_size, max(n_samples, 1))),
        "seed": int(seed),
        "device": str(dev),
        "linear_decoder": bool(linear_decoder),
        "activation": activation,
        "n_samples": int(n_samples),
        "train_time_sec": float(train_time),
    }

    result: Dict[str, Any] = {
        "model": model,
        "classifier": classifier,
        "history": history,
        "config": config,
        "train_time": float(train_time),
        "final": {key: (history[key][-1] if history[key] else 0.0) for key in COMPONENT_KEYS},
    }

    if out_dir:
        _save_run_artifacts(result, ensure_dir(Path(out_dir)), tag)

    return result


# --------------------------------------------------------------------------- #
# Plain autoencoder baseline
# --------------------------------------------------------------------------- #
def train_plain_ae(
    X_train,
    y_train=None,
    input_dim: Optional[int] = None,
    latent_dim: int = 8,
    hidden_dims: Sequence[int] = (64, 32),
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: Optional[str] = None,
    seed: int = 5,
    supervised: bool = False,
    linear_decoder: bool = False,
    log_every: int = 50,
    out_dir: Optional[str] = None,
    num_classes: Optional[int] = None,
    activation: str = "relu",
    tag: str = "ae",
    verbose: bool = True,
    max_updates: Optional[int] = None,
) -> Dict[str, Any]:
    """Train the plain autoencoder baseline (``L_rec`` only).

    The returned dictionary has the same structure as the one of
    :func:`train_polca` so that the evaluation code can treat both models
    uniformly.  The ``ort`` / ``com`` / ``var`` components are always zero.
    """
    x = _as_float_array(X_train)
    if x.ndim != 2:
        raise ValueError(f"X_train must be 2-D, got shape {x.shape}")
    n_samples, dim = x.shape
    if input_dim is None:
        input_dim = dim
    input_dim = int(input_dim)

    set_seed(int(seed))
    dev = _resolve_device(device)

    model = Autoencoder(
        input_dim=input_dim,
        latent_dim=int(latent_dim),
        hidden_dims=tuple(int(h) for h in hidden_dims),
        linear_decoder=bool(linear_decoder),
        activation=activation,
    ).to(dev)

    labels = None
    classifier = None
    criterion = None
    if supervised:
        if y_train is None:
            raise ValueError("supervised=True requires y_train")
        labels = np.asarray(y_train).reshape(-1).astype(np.int64)
        if num_classes is None:
            num_classes = int(labels.max()) + 1
        num_classes = int(num_classes)
        classifier = nn.Linear(int(latent_dim), num_classes).to(dev)
        criterion = nn.CrossEntropyLoss()

    loader = _make_loader(x, labels, batch_size, dev, shuffle=True)

    parameters = list(model.parameters())
    if classifier is not None:
        parameters += list(classifier.parameters())
    optimiser = torch.optim.Adam(parameters, lr=float(lr))

    history = _history_dict()
    start_time = time.time()

    # See :func:`train_polca` for the rationale behind the two protocols.
    steps_per_epoch = max(len(loader), 1)
    if max_updates is not None:
        total_epochs = int(math.ceil(int(max_updates) / steps_per_epoch))
    else:
        total_epochs = int(epochs)
    updates_done = 0
    update_budget = int(max_updates) if max_updates is not None else None

    for epoch in range(total_epochs):
        model.train()
        if classifier is not None:
            classifier.train()
        sums = {key: 0.0 for key in COMPONENT_KEYS}
        count = 0
        for batch in loader:
            if update_budget is not None and updates_done >= update_budget:
                break
            if labels is None:
                xb = batch[0]
                yb = None
            else:
                xb, yb = batch[0], batch[1]

            optimiser.zero_grad(set_to_none=True)
            recon, latent = model(xb)
            rec = nn.functional.mse_loss(recon, xb)
            cls = torch.zeros((), device=dev)
            if supervised:
                cls = criterion(classifier(latent), yb)
            loss = rec + cls
            loss.backward()
            optimiser.step()
            updates_done += 1

            bs = int(xb.shape[0])
            count += bs
            sums["total"] += float(loss.detach().item()) * bs
            sums["rec"] += float(rec.detach().item()) * bs
            sums["class"] += float(cls.detach().item()) * bs

        for key in COMPONENT_KEYS:
            history[key].append(sums[key] / max(count, 1))

        if log_every and verbose and (
            (epoch + 1) % int(log_every) == 0 or epoch == 0 or (epoch + 1) == total_epochs
        ):
            print(
                f"[PlainAE] epoch {epoch + 1}/{total_epochs} updates={updates_done} "
                f"total={history['total'][-1]:.6f} rec={history['rec'][-1]:.6f}",
                flush=True,
            )

    train_time = time.time() - start_time
    model.eval()
    if classifier is not None:
        classifier.eval()

    config = {
        "method": "ae_sup" if supervised else "ae",
        "input_dim": input_dim,
        "latent_dim": int(latent_dim),
        "hidden_dims": [int(h) for h in hidden_dims],
        "epochs": int(epochs),
        "max_updates": (int(max_updates) if max_updates is not None else None),
        "gradient_updates": int(updates_done),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "device": str(dev),
        "linear_decoder": bool(linear_decoder),
        "supervised": bool(supervised),
        "n_samples": int(n_samples),
        "train_time_sec": float(train_time),
    }

    result: Dict[str, Any] = {
        "model": model,
        "classifier": classifier,
        "history": history,
        "config": config,
        "train_time": float(train_time),
        "final": {key: (history[key][-1] if history[key] else 0.0) for key in COMPONENT_KEYS},
    }

    if out_dir:
        _save_run_artifacts(result, ensure_dir(Path(out_dir)), tag)

    return result


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the ``argparse`` parser of the training script."""
    parser = argparse.ArgumentParser(
        description="Train POLCA Net (arXiv:2410.07289) on one of the assignment datasets.",
    )
    parser.add_argument("--dataset", default="digits", choices=("digits", "mnist", "fashion"))
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[64, 32])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", default=None, help="cpu / cuda / auto")
    parser.add_argument("--linear-decoder", action="store_true")
    parser.add_argument("--supervised", action="store_true", help="Enable the L_class term.")
    parser.add_argument("--no-ort", action="store_true", help="Ablate L_ort.")
    parser.add_argument("--no-com", action="store_true", help="Ablate L_com.")
    parser.add_argument("--no-var", action="store_true", help="Ablate L_var.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Stop after this many gradient updates (the paper uses 20000).",
    )
    parser.add_argument("--out-dir", default=None, help="Where to write history/config JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``python src/train.py``."""
    args = build_parser().parse_args(argv)

    from src.polca.data import get_dataset, train_test_split_idx

    bundle = get_dataset(args.dataset, train=True)
    X = np.asarray(bundle["X"], dtype=np.float32)
    y = np.asarray(bundle["y"], dtype=np.int64)
    train_idx, _ = train_test_split_idx(X.shape[0], 0.2, args.seed)

    result = train_polca(
        X[train_idx],
        y[train_idx],
        input_dim=int(bundle["input_dim"]),
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        supervised=args.supervised,
        use_ort=not args.no_ort,
        use_com=not args.no_com,
        use_var=not args.no_var,
        linear_decoder=args.linear_decoder,
        max_updates=args.max_updates,
        log_every=args.log_every,
        out_dir=args.out_dir,
        num_classes=int(bundle["num_classes"]),
        tag=f"{args.dataset}_polca_seed{args.seed}",
    )
    print(
        f"Trained POLCA on {args.dataset}: {result['train_time']:.1f}s, "
        f"final total loss = {result['final']['total']:.6f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
