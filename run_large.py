"""Launch the benchmark for the two large datasets (mnist, fashion).

The three datasets are evaluated with two concurrent processes: running all
three at once exhausted the available RAM on this 16 GB machine (the LDA and
autoencoder baselines allocate several float64 copies of the 48000x784
design matrix), so the large datasets are processed two at a time while
``digits`` runs on its own.

The child processes are started in *detached* mode so that they survive the
death of the parent shell (the previous run was killed together with its
terminal and lost everything after the fourth method).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ("mnist", "fashion")
SEEDS = ("5", "6", "7")
MAX_UPDATES = "10000"

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

procs = []
for ds in DATASETS:
    out = ROOT / "results" / ds
    out.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "results" / f"run_{ds}.log"
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
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED,
        close_fds=True,
    )
    handle.close()
    print(f"  {ds}: pid={proc.pid} log={log_path}", flush=True)
    procs.append((ds, proc))

print("all datasets launched in detached mode", flush=True)
