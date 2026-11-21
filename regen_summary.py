"""Regenerate results/summary.md from the merged metrics/ablation tables.

This is the lightweight counterpart of ``merge_results.py``: it does not
touch ``latents.npz`` or any other heavy artefact, it only rebuilds the
human-readable summary table from the CSV files that are already merged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluate import METHODS, _write_summary  # noqa: E402

DATASETS = ("digits", "mnist", "fashion")


def main() -> int:
    root = _REPO_ROOT / "results"
    metrics = pd.read_csv(root / "metrics.csv")
    ablation = pd.read_csv(root / "ablation.csv")

    _write_summary(
        root / "summary.md",
        metrics.to_dict(orient="records"),
        ablation.to_dict(orient="records"),
        list(DATASETS),
        list(METHODS),
        notes=[
            "This file aggregates the three datasets; the per-dataset "
            "bundles under results/<dataset>/ carry the same numbers.",
        ],
    )
    print(f"wrote {root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
