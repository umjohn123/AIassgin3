"""POLCA Net model: the autoencoder backbone of arXiv:2410.07289.

The network itself is a plain (multi-layer perceptron) autoencoder; what
makes it "POLCA" are the constraints imposed by the composite loss defined
in :mod:`src.polca.losses` (orthogonality of the latent codes, variance
ordering along the latent dimensions and a variance regulariser).

Two classes are provided:

* :class:`POLCANet` -- the full model with the optional *linear decoder*
  variant described in the paper.
* :class:`Autoencoder` -- a baseline autoencoder without any POLCA
  constraint (equivalent to ``POLCANet(..., use_polca=False)``).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

__all__ = ["POLCANet", "Autoencoder", "build_mlp", "ACTIVATIONS"]


#: Mapping of activation names to PyTorch modules.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _activation(name: str | type[nn.Module] | None) -> type[nn.Module]:
    """Resolve an activation name (or module class) to a module class."""
    if name is None:
        return nn.Identity
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    key = str(name).lower()
    if key in ACTIVATIONS:
        return ACTIVATIONS[key]
    raise ValueError(
        f"Unknown activation {name!r}. Available: {sorted(ACTIVATIONS)}"
    )


def build_mlp(
    dims: Sequence[int],
    activation: str | type[nn.Module] = "relu",
    final_activation: bool = False,
) -> nn.Sequential:
    """Build a fully connected MLP.

    Parameters
    ----------
    dims : sequence of int
        Layer widths, e.g. ``[784, 64, 32, 8]``.
    activation : str or type[nn.Module], default ``"relu"``
        Activation used *between* the layers.
    final_activation : bool, default False
        Whether to append the activation after the last layer.

    Returns
    -------
    torch.nn.Sequential
        The constructed network.
    """
    act_cls = _activation(activation)
    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        is_last = index == len(dims) - 2
        if not is_last or final_activation:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class POLCANet(nn.Module):
    """Autoencoder used by POLCA Net.

    Parameters
    ----------
    input_dim : int
        Dimension of the (flattened) input, e.g. ``784`` for 28x28 images.
    latent_dim : int, default 8
        Dimension ``n`` of the latent space.
    hidden_dims : sequence of int, default ``(64, 32)``
        Widths of the hidden layers of the **encoder**.  The decoder mirrors
        this shape when it is non-linear.
    linear_decoder : bool, default False
        When ``True`` the decoder is a single ``nn.Linear(latent_dim,
        input_dim)`` with no activation.  This is the paper's *optional linear
        decoder* property: the decoder map stays linear, hence additive and
        homogeneous, which makes the latent components interpretable in a
        PCA-like fashion.
    activation : str or type[nn.Module], default ``"relu"``
        Activation used in the hidden layers.

    Notes
    -----
    **No activation is applied to the latent code.**  The POLCA loss needs
    well defined variances and inner products of the latent vectors
    (``E[(z_i - E[z_i])^2]``, ``z/||z||_2``), and a squashing activation
    would distort both the variance ordering and the orthogonality measure.
    The latent code is therefore the raw output of the last encoder layer
    (identity activation).

    Examples
    --------
    >>> model = POLCANet(784, latent_dim=8)
    >>> recon, latent = model(torch.rand(4, 784))
    >>> recon.shape, latent.shape
    (torch.Size([4, 784]), torch.Size([4, 8]))
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        linear_decoder: bool = False,
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims: tuple[int, ...] = tuple(int(h) for h in hidden_dims)
        self.linear_decoder = bool(linear_decoder)
        self.activation = activation if isinstance(activation, str) else getattr(
            activation, "__name__", str(activation)
        )

        # ---- encoder: input_dim -> hidden... -> latent_dim (identity act) -- #
        encoder_dims = [self.input_dim, *self.hidden_dims, self.latent_dim]
        self.encoder = build_mlp(encoder_dims, activation=activation, final_activation=False)

        # ---- decoder ------------------------------------------------------ #
        if self.linear_decoder:
            self.decoder = nn.Sequential(nn.Linear(self.latent_dim, self.input_dim))
        else:
            decoder_dims = [self.latent_dim, *reversed(self.hidden_dims), self.input_dim]
            self.decoder = build_mlp(decoder_dims, activation=activation, final_activation=False)

    # ------------------------------------------------------------------ #
    # Forward passes
    # ------------------------------------------------------------------ #
    @property
    def use_polca_constraints(self) -> bool:
        """``True`` -- this model is meant to be trained with ``L_polca``."""
        return True

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, input_dim)`` (or any shape whose last
            dimension is ``input_dim``).

        Returns
        -------
        torch.Tensor
            Latent codes of shape ``(..., latent_dim)``.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent code back into the input space.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes of shape ``(..., latent_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstruction of shape ``(..., input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(N, input_dim)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(reconstruction, latent)`` with shapes ``(N, input_dim)`` and
            ``(N, latent_dim)``.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class Autoencoder(POLCANet):
    """Plain autoencoder baseline (no POLCA constraints).

    This is exactly :class:`POLCANet`; the distinction is purely in *how it
    is trained*.  The POLCA-specific regularisers live in the loss, not in
    the network, so the baseline simply optimises ``L_rec`` (and optionally
    ``L_class``) with the same architecture.  Keeping the baseline as a
    subclass guarantees that both models share exactly the same capacity,
    which makes the ablation in the report fair.

    Parameters
    ----------
    input_dim : int
        Dimension of the flattened input.
    latent_dim : int, default 8
        Dimension of the latent space.
    hidden_dims : sequence of int, default ``(64, 32)``
        Hidden layer widths of the encoder.
    linear_decoder : bool, default False
        Use a single linear decoder layer.
    activation : str or type[nn.Module], default ``"relu"``
        Hidden layer activation.

    Examples
    --------
    >>> baseline = Autoencoder(784, latent_dim=8)
    >>> recon, latent = baseline(torch.rand(4, 784))
    >>> recon.shape
    torch.Size([4, 784])
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        linear_decoder: bool = False,
        activation: str | type[nn.Module] = "relu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            linear_decoder=linear_decoder,
            activation=activation,
        )
        self.is_polca = False

    @property
    def use_polca_constraints(self) -> bool:
        """``False`` for the baseline autoencoder."""
        return False
