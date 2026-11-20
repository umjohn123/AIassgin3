"""Merge the per-dataset result files into the canonical ``results/`` layout.

``src/evaluate.py`` writes one bundle per invocation.  ``run_all.py`` runs
one process per dataset so that the three datasets are evaluated in
parallel, which leaves the results split across ``results/<dataset>/``.
This script merges those bundles back into a single set of files:

    results/metrics.csv
    results/metrics.json
    results/ablation.csv
    results/summary.md
    results/latents.npz
    results/loss_history.json

The merged ``metrics.csv`` keeps exactly the same column layout as the
per-dataset files, so every downstream consumer (figures, report) works
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.polca.utils import ensure_dir  # noqa: E402

DATASETS = ("digits", "mnist", "fashion")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge(results_dir: str = "results", datasets=DATASETS) -> Dict[str, Any]:
    """Merge the per-dataset bundles into ``results/``."""
    root = ensure_dir(Path(results_dir))

    frames: List[pd.DataFrame] = []
    ablation_frames: List[pd.DataFrame] = []
    metrics_json: Dict[str, Any] = {}
    latents: Dict[str, np.ndarray] = {}
    losses: Dict[str, Any] = {}

    for ds in datasets:
        sub = root / ds
        frame = _read_csv(sub / "metrics.csv")
        if not frame.empty:
            frames.append(frame)
        ablation = _read_csv(sub / "ablation.csv")
        if not ablation.empty:
            ablation_frames.append(ablation)
        payload = _read_json(sub / "metrics.json")
        if payload:
            metrics_json[ds] = payload
        npz = sub / "latents.npz"
        if npz.exists():
            with np.load(npz) as bundle:
                for key in bundle.files:
                    # The per-dataset bundles already carry the dataset name
                    # inside their keys (e.g. ``digits__pca__d8__s5__z_test``),
                    # so no extra prefix is added here -- otherwise the figure
                    # code would not find the entries it looks for.
                    latents[key] = bundle[key]
        history = _read_json(sub / "loss_history.json")
        if history:
            losses.update(history)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(root / "metrics.csv", index=False)
    if ablation_frames:
        pd.concat(ablation_frames, ignore_index=True).to_csv(
            root / "ablation.csv", index=False
        )
    if metrics_json:
        with (root / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics_json, handle, indent=2)
    if latents:
        np.savez_compressed(root / "latents.npz", **latents)
    if losses:
        with (root / "loss_history.json").open("w", encoding="utf-8") as handle:
            json.dump(losses, handle, indent=2)

    print(f"merged metrics.csv rows: {len(frames) and sum(len(f) for f in frames)}")
    return {"datasets": list(datasets), "root": str(root)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-dataset POLCA results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args(argv)
    merge(args.results_dir, args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
