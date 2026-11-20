"""Dependency-free smoke test of the POLCA Net pipeline.

The test uses plain ``assert`` statements (no pytest required) and can be
launched directly::

    python tests/test_pipeline.py

It checks that

* the POLCA loss stays finite and non-negative during a short training,
* the ablation switches really change the composite loss,
* ``get_dataset('digits')`` returns the expected shapes,
* the orthogonality error of a random orthonormal set is (numerically) zero,
* a tiny end-to-end evaluation writes the result files.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.metrics import orthogonality_error  # noqa: E402
from src.polca.data import get_dataset, train_test_split_idx  # noqa: E402
from src.polca.losses import polca_loss  # noqa: E402
from src.polca.model import POLCANet  # noqa: E402
from src.train import train_plain_ae, train_polca  # noqa: E402

__all__ = [
    "test_polca_training_losses",
    "test_ablation_flags_change_loss",
    "test_digits_dataset_shapes",
    "test_orthogonality_of_orthonormal_set",
    "test_end_to_end_evaluation",
    "main",
]


# --------------------------------------------------------------------------- #
# 1. Losses stay finite and non-negative during training
# --------------------------------------------------------------------------- #
def test_polca_training_losses() -> None:
    """A short POLCA training must produce finite, non-negative losses."""
    rng = np.random.RandomState(0)
    X = rng.rand(64, 32).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.int64)

    out = train_polca(
        X,
        y,
        input_dim=32,
        latent_dim=4,
        hidden_dims=(16,),
        epochs=3,
        lr=1e-3,
        batch_size=16,
        device="cpu",
        seed=5,
        supervised=True,
        log_every=0,
        verbose=False,
    )

    history = out["history"]
    assert set(["total", "rec", "class", "ort", "com", "var"]).issubset(history.keys())
    for key in ("total", "rec", "class", "ort", "com", "var"):
        assert len(history[key]) == 3, f"history[{key!r}] should have 3 entries"
        for value in history[key]:
            assert np.isfinite(value), f"{key} is not finite: {value}"
            assert value >= -1e-8, f"{key} is negative: {value}"
    assert out["train_time"] > 0.0
    assert isinstance(out["model"], POLCANet)
    assert out["config"]["alpha"] == 1e-2
    assert out["config"]["beta"] == 1e-2
    assert out["config"]["gamma"] == 1e-7
    print("  ok: POLCA training produces finite non-negative losses")


# --------------------------------------------------------------------------- #
# 2. The ablation switches really change the loss
# --------------------------------------------------------------------------- #
def test_ablation_flags_change_loss() -> None:
    """Switching a term off must change the composite loss."""
    torch.manual_seed(0)
    model = POLCANet(16, latent_dim=4, hidden_dims=(8,)).eval()
    x = torch.rand(32, 16)
    with torch.no_grad():
        recon, latent = model(x)

    full = polca_loss(recon, latent, x, alpha=1e-2, beta=1e-2, gamma=1e-7, return_components=True)
    no_ort = polca_loss(recon, latent, x, alpha=0.0, beta=1e-2, gamma=1e-7, return_components=True)
    no_com = polca_loss(recon, latent, x, alpha=1e-2, beta=0.0, gamma=1e-7, return_components=True)
    no_var = polca_loss(recon, latent, x, alpha=1e-2, beta=1e-2, gamma=0.0, return_components=True)

    # the individual components must be identical (only the weight changes)
    assert abs(float(full.ort) - float(no_ort.ort)) < 1e-9
    assert abs(float(full.com) - float(no_com.com)) < 1e-9
    assert abs(float(full.var) - float(no_var.var)) < 1e-9

    # the composite loss must change accordingly
    assert abs(float(full.total) - float(no_ort.total)) > 1e-12
    assert abs(float(full.total) - float(no_com.total)) > 1e-12

    # gamma = 1e-7 is too small to be observable in float32 for the default
    # weights, so the variance term is checked with an amplified weight.
    rec_only = polca_loss(
        recon, latent, x, alpha=0.0, beta=0.0, gamma=0.0, return_components=True
    )
    big_gamma = polca_loss(
        recon, latent, x, alpha=0.0, beta=0.0, gamma=1.0, return_components=True
    )
    assert abs(float(big_gamma.total) - float(rec_only.total)) > 1e-12
    assert abs(float(big_gamma.var) - float(full.var)) < 1e-9
    assert abs(float(big_gamma.total) - float(rec_only.total) - float(full.var)) < 1e-6

    # the training wrapper must zero the corresponding weight
    out = train_polca(
        np.random.RandomState(1).rand(32, 16).astype(np.float32),
        None,
        input_dim=16,
        latent_dim=4,
        hidden_dims=(8,),
        epochs=1,
        device="cpu",
        use_ort=False,
        use_com=False,
        use_var=False,
        log_every=0,
        verbose=False,
    )
    assert out["config"]["alpha"] == 0.0
    assert out["config"]["beta"] == 0.0
    assert out["config"]["gamma"] == 0.0

    # the plain autoencoder baseline must only carry the reconstruction term
    ae = train_plain_ae(
        np.random.RandomState(2).rand(32, 16).astype(np.float32),
        None,
        input_dim=16,
        latent_dim=4,
        hidden_dims=(8,),
        epochs=2,
        device="cpu",
        log_every=0,
        verbose=False,
    )
    for key in ("ort", "com", "var"):
        assert all(abs(v) < 1e-12 for v in ae["history"][key]), f"{key} must be zero for the plain AE"
    assert all(np.isfinite(v) and v >= 0 for v in ae["history"]["rec"])
    print("  ok: the ablation flags change the composite loss")


# --------------------------------------------------------------------------- #
# 3. Dataset shapes
# --------------------------------------------------------------------------- #
def test_digits_dataset_shapes() -> None:
    """``get_dataset('digits')`` must expose the documented shapes."""
    bundle = get_dataset("digits", train=True)
    X = np.asarray(bundle["X"])
    y = np.asarray(bundle["y"])
    images = np.asarray(bundle["images"])

    assert X.ndim == 2 and X.shape[1] == 64, f"unexpected X shape {X.shape}"
    assert X.shape[0] == 1797, f"unexpected number of samples {X.shape[0]}"
    assert X.dtype == np.float32, f"unexpected dtype {X.dtype}"
    assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0, "X must be in [0, 1]"
    assert y.shape == (1797,) and y.dtype == np.int64
    assert set(np.unique(y).tolist()) == set(range(10))
    assert images.shape == (1797, 8, 8)
    assert images.dtype == np.uint8
    assert int(bundle["input_dim"]) == 64
    assert tuple(bundle["image_shape"]) == (8, 8)
    assert int(bundle["num_classes"]) == 10

    train_idx, test_idx = train_test_split_idx(X.shape[0], 0.2, 5)
    assert len(train_idx) + len(test_idx) == X.shape[0]
    assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
    print("  ok: the digits dataset has the expected shapes")


# --------------------------------------------------------------------------- #
# 4. Orthogonality of an orthonormal set
# --------------------------------------------------------------------------- #
def test_orthogonality_of_orthonormal_set() -> None:
    """A random orthonormal basis has a (numerically) zero orthogonality error."""
    rng = np.random.RandomState(0)
    n_samples, n_dims = 200, 6
    # The paper's convention treats every latent *dimension* as a vector of
    # length N, so the columns of the latent matrix must be orthonormal.
    q, _ = np.linalg.qr(rng.randn(n_samples, n_dims))
    latent = q  # (N, n) with orthonormal columns

    err = orthogonality_error(latent)
    assert err < 1e-6, f"orthogonality error should be ~0, got {err}"
    print(f"  ok: the orthogonality error of an orthonormal set is {err:.2e}")


# --------------------------------------------------------------------------- #
# 5. End-to-end evaluation
# --------------------------------------------------------------------------- #
def test_end_to_end_evaluation() -> None:
    """A tiny evaluation must write every result file."""
    from src.evaluate import run_full_evaluation
    from src.make_figures import make_all_figures

    tmp_dir = Path(tempfile.mkdtemp(prefix="polca_smoke_"))
    try:
        result = run_full_evaluation(
            datasets=("digits",),
            latent_dims=(4,),
            epochs=2,
            seeds=(5,),
            methods=("pca", "lda", "ae", "polca", "polca_sup"),
            out_dir=str(tmp_dir),
            run_ablation=True,
            verbose=False,
        )

        assert result["rows"], "the evaluation produced no rows"

        for name in ("metrics.csv", "metrics.json", "summary.md", "ablation.csv"):
            path = tmp_dir / name
            assert path.exists(), f"{name} was not written"
            assert path.stat().st_size > 0, f"{name} is empty"

        with (tmp_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        assert "aggregated" in payload
        assert "digits" in payload["aggregated"]
        assert "polca" in payload["aggregated"]["digits"]

        import csv

        with (tmp_dir / "metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert csv_rows, "metrics.csv is empty"
        for column in ("dataset", "method", "seed", "test_nrmse", "accuracy", "f1"):
            assert column in csv_rows[0], f"missing column {column!r}"
        methods = {r["method"] for r in csv_rows}
        assert {"pca", "lda", "ae", "polca", "polca_sup"}.issubset(methods)

        with (tmp_dir / "ablation.csv").open("r", encoding="utf-8", newline="") as handle:
            ablation_rows = list(csv.DictReader(handle))
        assert len(ablation_rows) == 4, "the ablation study must contain 4 variants"

        figures = make_all_figures(out_dir=str(tmp_dir), datasets=("digits",), latent_dim=4, seed=5)
        assert figures, "no figure was produced"
        for path in figures:
            assert path.exists() and path.stat().st_size > 0
        print(f"  ok: end-to-end evaluation produced {len(figures)} figures")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run every test of the module."""
    tests = [
        test_digits_dataset_shapes,
        test_orthogonality_of_orthonormal_set,
        test_ablation_flags_change_loss,
        test_polca_training_losses,
        test_end_to_end_evaluation,
    ]
    print("Running the POLCA Net pipeline smoke test")
    for test in tests:
        print(f"- {test.__name__}")
        test()
    print(f"\nAll {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
