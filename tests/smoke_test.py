"""Tiny dependency-free smoke test for the POLCA Net implementation.

Run with::

    python tests/smoke_test.py

It only uses plain ``assert`` statements (no pytest required) and covers:

* shapes of the loss components and of a forward pass,
* ``L_ort`` of an exactly orthogonal latent set is (numerically) zero,
* ``L_com`` prefers a decreasing-variance latent ordering over an
  increasing-variance one,
* ``L_var`` and ``L_rec`` behave as specified,
* gradient flow through the composite loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the repository root importable when the file is run directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.polca.losses import (  # noqa: E402
    center_of_mass_loss,
    latent_variances,
    orthogonality_loss,
    orthogonality_matrix,
    polca_loss,
    reconstruction_loss,
    variance_loss,
)
from src.polca.model import Autoencoder, POLCANet  # noqa: E402
from src.polca.utils import AverageMeter, ensure_dir, get_device, save_csv, save_json, set_seed  # noqa: E402


def test_forward_pass() -> None:
    """A forward pass returns a reconstruction and a latent code."""
    model = POLCANet(input_dim=784, latent_dim=8)
    x = torch.rand(16, 784)
    reconstruction, latent = model(x)
    assert reconstruction.shape == (16, 784), reconstruction.shape
    assert latent.shape == (16, 8), latent.shape

    # The baseline autoencoder has the same interface.
    baseline = Autoencoder(input_dim=784, latent_dim=8)
    rec_b, lat_b = baseline(x)
    assert rec_b.shape == (16, 784)
    assert lat_b.shape == (16, 8)

    # Linear-decoder variant: a single Linear layer, no activation.
    linear = POLCANet(input_dim=784, latent_dim=8, linear_decoder=True)
    linear_layers = [m for m in linear.decoder.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 1
    assert linear_layers[0].in_features == 8 and linear_layers[0].out_features == 784
    rec_l, lat_l = linear(x)
    assert rec_l.shape == (16, 784) and lat_l.shape == (16, 8)


def test_loss_components_shapes() -> None:
    """Every loss component is a scalar and the composite loss is finite."""
    torch.manual_seed(0)
    x = torch.rand(16, 784)
    model = POLCANet(784, 8)
    reconstruction, latent = model(x)

    rec = reconstruction_loss(reconstruction, x)
    assert rec.ndim == 0
    assert torch.isfinite(rec)

    out = polca_loss(
        reconstruction,
        latent,
        x,
        torch.randint(0, 10, (16,)),
        torch.nn.CrossEntropyLoss(),
        return_components=True,
    )
    for value in (out.total, out.rec, out.cls, out.ort, out.com, out.var):
        assert value.ndim == 0
        assert torch.isfinite(value)
    assert abs(out.item() - out.to_dict()["total"]) < 1e-9

    # Unsupervised -> classification term is exactly zero.
    unsup = polca_loss(reconstruction, latent, x, None, return_components=True)
    assert float(unsup.cls) == 0.0


def test_orthogonality_loss_zero_for_orthogonal_latent() -> None:
    """An exactly orthogonal latent set has (numerically) zero L_ort."""
    # 3 samples, 3 latent dimensions, mutually orthogonal directions.
    # With N == n the Gram matrix of the normalised latents is exactly I.
    orthogonal = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    value = float(orthogonality_loss(orthogonal))
    assert value < 1e-10, value

    # The Gram matrix of orthogonal latents is the identity.
    s = orthogonality_matrix(orthogonal)
    assert torch.allclose(s, torch.eye(3), atol=1e-6)

    # A perfectly correlated latent set is heavily penalised.
    correlated = torch.ones(8, 4)
    assert float(orthogonality_loss(correlated)) > 0.5


def test_center_of_mass_prefers_decreasing_variance() -> None:
    """L_com is smaller for a decreasing-variance latent ordering."""
    n_samples = 2000
    generator = torch.Generator().manual_seed(7)

    descending_scales = torch.tensor([4.0, 3.0, 2.0, 1.0])
    ascending_scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    noise = torch.randn(n_samples, 4, generator=generator)

    descending = descending_scales * noise
    ascending = ascending_scales * noise

    l_com_desc = float(center_of_mass_loss(descending))
    l_com_asc = float(center_of_mass_loss(ascending))
    assert l_com_desc < l_com_asc, (l_com_desc, l_com_asc)

    # Variances themselves are reported per dimension, in latent order.
    variances = latent_variances(descending)
    assert variances.shape == (4,)
    assert bool((variances[:-1] > variances[1:]).all())

    # L_var is the sum of the per-dimension variances.
    assert abs(float(variance_loss(descending)) - float(variances.sum())) < 1e-3


def test_gradients_flow() -> None:
    """The composite loss back-propagates into the encoder parameters."""
    model = POLCANet(784, 8)
    x = torch.rand(8, 784)
    reconstruction, latent = model(x)
    loss = polca_loss(reconstruction, latent, x, None)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients were produced"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_utils() -> None:
    """Utility helpers behave as documented."""
    set_seed(5)
    first = np.random.rand(3)
    set_seed(5)
    assert np.allclose(first, np.random.rand(3))

    assert get_device() in {"cpu", "cuda"}

    meter = AverageMeter()
    meter.update(1.0)
    meter.update(3.0)
    assert abs(meter.avg - 2.0) < 1e-12

    out_dir = ROOT / "results" / "_smoke"
    ensure_dir(out_dir)
    save_json({"a": 1}, out_dir / "x.json")
    save_csv([{"a": 1, "b": 2}], out_dir / "x.csv")
    assert (out_dir / "x.json").exists()
    assert (out_dir / "x.csv").exists()
    for path in (out_dir / "x.json", out_dir / "x.csv"):
        path.unlink()
    out_dir.rmdir()


def main() -> None:
    """Run all smoke tests."""
    tests = [
        test_forward_pass,
        test_loss_components_shapes,
        test_orthogonality_loss_zero_for_orthogonal_latent,
        test_center_of_mass_prefers_decreasing_variance,
        test_gradients_flow,
        test_utils,
    ]
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
