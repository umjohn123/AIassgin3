"""Launch the benchmark for the two large datasets (mnist, fashion).

The three datasets are evaluated with two concurrent processes: running all
three at once exhausted the available RAM on this 16 GB machine (the LDA and
autoencoder baselines allocate several float64 copies of the 48000x784
design matrix), so the large datasets are processed two at a time while
``digits`` runs on its own.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ("mnist", "fashion")
SEEDS = ("5", "6", "7")
MAX_UPDATES = "10000"

procs = []
for ds in DATASETS:
    out = ROOT / "results" / ds
    out.mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "results" / f"run_{ds}.log", "w", encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "src" / "evaluate.py"),
        "--datasets",
        ds,
        "--latent-dims",
        "8",
        "--seeds",
        *SEEDS,
        "--max-updates",
        MAX_UPDATES,
        "--out-dir",
        str(out),
    ]
    print("launching:", " ".join(cmd), flush=True)
    procs.append((ds, subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT), log))

failed = []
for ds, proc, log in procs:
    rc = proc.wait()
    log.close()
    print(f"{ds} finished rc={rc}", flush=True)
    if rc != 0:
        failed.append(ds)

print("FAILED:", failed if failed else "none", flush=True)
sys.exit(1 if failed else 0)
