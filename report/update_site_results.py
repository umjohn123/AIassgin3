"""Regenerate the results table inside ``docs/index.html`` from ``results/``.

The project site must satisfy the same rule as the report: *no number that
is not present in* ``results/``.  Rather than hand-editing the HTML every
time the experiments are re-run, this script rewrites the table between the
``<!-- BEGIN RESULTS TABLE -->`` / ``<!-- END RESULTS TABLE -->`` markers
with the measured values, and leaves an em dash for anything that is
missing.

Usage::

    python report/update_site_results.py
    python report/update_site_results.py --results-dir results --site docs/index.html
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.polca.utils import ensure_dir  # noqa: E402

BEGIN_MARKER = "<!-- BEGIN RESULTS TABLE -->"
END_MARKER = "<!-- END RESULTS TABLE -->"

#: Display name of every method, in the order in which they are listed.
METHOD_ORDER: Tuple[Tuple[str, str], ...] = (
    ("pca", "PCA"),
    ("lda", "LDA"),
    ("ae", "Autoencoder"),
    ("polca", "POLCA-Net"),
    ("polca_sup", "POLCA-Net + <code>L_class</code>"),
)

#: Display name of every dataset.
DATASET_ORDER: Tuple[Tuple[str, str], ...] = (
    ("digits", "digits"),
    ("mnist", "MNIST"),
    ("fashion", "FashionMNIST"),
)

DASH = "&mdash;"


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """Read ``metrics.csv`` into a list of dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: Sequence[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return (sum(clean) / len(clean)) if clean else None


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float | None]]:
    """Aggregate the long-format metric rows into ``(dataset, method) -> stats``.

    The long format stores one row per classifier, so the reconstruction
    metrics are averaged over the classifiers as well as over the seeds (they
    are identical across classifiers); the accuracy and F1 are the mean over
    the classifiers and over the seeds.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("method", "")))
        grouped[key].append(row)

    stats: Dict[Tuple[str, str], Dict[str, float | None]] = {}
    for key, group in grouped.items():
        stats[key] = {
            "nrmse": _mean([_to_float(r.get("test_nrmse")) for r in group]),
            "psnr": _mean([_to_float(r.get("test_psnr")) for r in group]),
            "ssim": _mean([_to_float(r.get("test_ssim")) for r in group]),
            "accuracy": _mean([_to_float(r.get("accuracy")) for r in group]),
            "f1": _mean([_to_float(r.get("f1")) for r in group]),
            "seeds": len({r.get("seed") for r in group}),
        }
    return stats


def _cell(value: float | None, fmt: str) -> str:
    if value is None:
        return f'<td class="num">{DASH}</td>'
    return f'<td class="num">{format(value, fmt)}</td>'


def build_table(stats: Dict[Tuple[str, str], Dict[str, float | None]]) -> str:
    """Build the HTML of the results table."""
    lines: List[str] = []
    lines.append(BEGIN_MARKER)
    lines.append("    <table>")
    lines.append(
        "      <caption>Reconstruction and classification metrics on the test "
        "split (mean over the seeds and over the four linear classifiers). "
        "Every value is read from <code>results/metrics.csv</code>.</caption>"
    )
    lines.append("      <thead>")
    lines.append("        <tr>")
    lines.append("          <th>Method</th>")
    lines.append("          <th>NRMSE &darr;</th>")
    lines.append("          <th>PSNR &uarr;</th>")
    lines.append("          <th>SSIM &uarr;</th>")
    lines.append("          <th>Accuracy &uarr;</th>")
    lines.append("          <th>F1 &uarr;</th>")
    lines.append("        </tr>")
    lines.append("      </thead>")
    lines.append("      <tbody>")

    for dataset, dataset_label in DATASET_ORDER:
        present = [m for m, _ in METHOD_ORDER if (dataset, m) in stats]
        if not present:
            continue
        lines.append(f"        <tr class=\"rowhead\"><th colspan=\"6\">"
                     f"{dataset_label}</th></tr>")
        for method, method_label in METHOD_ORDER:
            entry = stats.get((dataset, method))
            if entry is None:
                continue
            strong = method in ("polca", "polca_sup")
            label = f"<strong>{method_label}</strong>" if strong else method_label
            cells = (
                _cell(entry["nrmse"], ".4f")
                + _cell(entry["psnr"], ".2f")
                + _cell(entry["ssim"], ".4f")
                + _cell(entry["accuracy"], ".4f")
                + _cell(entry["f1"], ".4f")
            )
            lines.append(f"        <tr><td>{label}</td>{cells}</tr>")

    lines.append("      </tbody>")
    lines.append("    </table>")
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_site(results_dir: Path, site_path: Path) -> bool:
    """Rewrite the results table in the site.  Returns True when it changed."""
    rows = load_rows(Path(results_dir) / "metrics.csv")
    if not rows:
        print("  ! no metrics.csv -- the site table was left untouched")
        return False

    stats = aggregate(rows)
    table = build_table(stats)

    html = Path(site_path).read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    if not pattern.search(html):
        print(f"  ! the markers {BEGIN_MARKER} / {END_MARKER} were not found "
              f"in {site_path}")
        return False

    new_html = pattern.sub(lambda _m: table, html)
    if new_html == html:
        print("  the site table was already up to date")
        return False

    Path(site_path).write_text(new_html, encoding="utf-8")
    print(f"  updated the results table in {site_path} "
          f"({len(stats)} dataset/method combinations)")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the results table of docs/index.html from results/.",
    )
    parser.add_argument("--results-dir", default=str(_REPO_ROOT / "results"))
    parser.add_argument("--site", default=str(_REPO_ROOT / "docs" / "index.html"))
    args = parser.parse_args(argv)

    ensure_dir(Path(args.results_dir))
    update_site(Path(args.results_dir), Path(args.site))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
