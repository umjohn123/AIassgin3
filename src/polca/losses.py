"""Loss functions of POLCA Net (arXiv:2410.07289).

The POLCA objective is a weighted sum of five terms::

    L_polca = L_rec + L_class + alpha * L_ort + beta * L_com + gamma * L_var

with the paper's hyper-parameters ``alpha = 1e-2``, ``beta = 1e-2`` and
``gamma = 1e-7`` (Table 1 of the paper).

Notation used throughout this module
------------------------------------
``N``  batch size
``n``  latent dimensionality
``Z``  the ``(N, n)`` matrix whose rows are the latent codes ``z_i``
``z~`` the L2-normalised latent codes, ``z~_i = z_i / ||z_i||_2``
``S``  the ``(n, n)`` Gram matrix ``S = Z~^T Z~``

All tensors are expected to be PyTorch tensors; the helpers accept either
``torch.Tensor`` or ``numpy.ndarray`` inputs and always return the same type
they received (NumPy in, NumPy out; Torch in, Torch out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

try:  # torch is a hard requirement of the project, but keep imports graceful
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

__all__ = [
    "ALPHA",
    "BETA",
    "GAMMA",
    "EPS_COM",
    "POLCALossOutput",
    "reconstruction_loss",
    "classification_loss",
    "orthogonality_matrix",
    "gram_matrix",
    "similarity_matrix",
    "orthogonality_loss",
    "latent_variances",
    "variance_loss",
    "center_of_mass_loss",
    "polca_loss",
]

#: Weight of the orthogonality term (paper, Table 1).
ALPHA: float = 1e-2
#: Weight of the center-of-mass / variance-ordering term (paper, Table 1).
BETA: float = 1e-2
#: Weight of the variance regularisation term (paper, Table 1).
GAMMA: float = 1e-7
#: Exponent ``eps`` used by the center-of-mass term (paper, default 0.25).
EPS_COM: float = 0.25

# Numerical floor used to avoid divisions by zero.
_EPS: float = 1e-12


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_torch(x: Any) -> bool:
    """Return ``True`` when ``x`` is a ``torch.Tensor``."""
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _as_float_tensor(x: Any) -> "torch.Tensor":
    """Convert ``x`` to a float32 torch tensor (detached from the graph)."""
    if _is_torch(x):
        return x.detach().to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def _zeros_like_scalar(reference: Any, value: float) -> Any:
    """Return a scalar zero of the same type family as ``reference``."""
    if _is_torch(reference):
        return reference.new_zeros(())
    return np.float32(value)


def _scalar(value: Any) -> float:
    """Convert a tensor/array scalar to a Python float."""
    if _is_torch(value):
        return float(value.detach().item())
    return float(np.asarray(value).reshape(()).item())


def _as_float_array(x: Any) -> np.ndarray:
    """Convert ``x`` to a 2-D float32 numpy array."""
    if _is_torch(x):
        return x.detach().to(dtype=torch.float32).cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    return arr


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class POLCALossOutput:
    """Container with the composite loss and all of its components.

    Attributes
    ----------
    total : Any
        The weighted composite loss ``L_polca``.
    rec : Any
        Reconstruction term ``L_rec``.
    cls : Any
        Classification term ``L_class`` (zero when unsupervised).
    ort : Any
        Orthogonality term ``L_ort``.
    com : Any
        Center-of-mass (variance ordering) term ``L_com``.
    var : Any
        Variance regularisation term ``L_var``.
    extra : dict
        Free-form additional information (e.g. latent variances).
    """

    total: Any
    rec: Any
    cls: Any
    ort: Any
    com: Any
    var: Any
    extra: dict = field(default_factory=dict)

    def __float__(self) -> float:
        return _scalar(self.total)

    def item(self) -> float:
        """Return the composite loss as a Python float."""
        return _scalar(self.total)

    def to_dict(self) -> dict[str, float]:
        """Return all components as a plain ``dict`` of floats."""
        return {
            "total": _scalar(self.total),
            "rec": _scalar(self.rec),
            "cls": _scalar(self.cls),
            "ort": _scalar(self.ort),
            "com": _scalar(self.com),
            "var": _scalar(self.var),
        }


# --------------------------------------------------------------------------- #
# L_rec and L_class
# --------------------------------------------------------------------------- #
def reconstruction_loss(reconstruction: Any, target: Any) -> Any:
    """Mean squared error between the input and its reconstruction.

    ``L_rec = mean((reconstruction - target)^2)`` over all elements.

    Parameters
    ----------
    reconstruction : torch.Tensor or numpy.ndarray
        Output of the decoder, shape ``(N, input_dim)``.
    target : torch.Tensor or numpy.ndarray
        Original input, shape ``(N, input_dim)``.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss (same type family as the inputs).
    """
    if _is_torch(reconstruction) or _is_torch(target):
        recon = reconstruction if _is_torch(reconstruction) else torch.as_tensor(reconstruction)
        tgt = target if _is_torch(target) else torch.as_tensor(target)
        recon = recon.to(dtype=torch.float32)
        tgt = tgt.to(dtype=torch.float32)
        return F.mse_loss(recon, tgt)
    recon = np.asarray(reconstruction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    return np.float32(np.mean((recon - tgt) ** 2))


def classification_loss(
    logits: Any,
    labels: Any,
    criterion: Callable[..., Any] | None = None,
) -> Any:
    """Optional cross-entropy classification term ``L_class``.

    When ``labels`` is ``None`` (unsupervised training) the term is exactly
    zero.  When ``criterion`` is ``None`` a mean cross-entropy is used.

    Parameters
    ----------
    logits : torch.Tensor
        Raw scores of shape ``(N, C)``.
    labels : torch.Tensor or None
        Integer class indices of shape ``(N,)``.
    criterion : callable, optional
        Any ``(logits, labels) -> scalar`` callable, e.g.
        ``torch.nn.CrossEntropyLoss()``.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise ImportError("classification_loss requires PyTorch")
    if labels is None:
        device = logits.device if _is_torch(logits) else torch.device("cpu")
        return torch.zeros((), device=device)
    labels = labels if _is_torch(labels) else torch.as_tensor(labels)
    labels = labels.to(device=logits.device).long()
    if criterion is None:
        return F.cross_entropy(logits, labels)
    return criterion(logits, labels)


# --------------------------------------------------------------------------- #
# L_ort
# --------------------------------------------------------------------------- #
def _normalized_latents(latent: Any) -> Any:
    """Return the L2-normalised latent codes ``z~ = z / ||z||_2``.

    Rows whose norm is (numerically) zero are kept as zeros, which makes the
    corresponding Gram-matrix row/column zero and contributes nothing to the
    orthogonality penalty.
    """
    if _is_torch(latent):
        norms = latent.norm(p=2, dim=1, keepdim=True)
        return latent / norms.clamp_min(_EPS)
    latent = np.asarray(latent, dtype=np.float32)
    norms = np.linalg.norm(latent, axis=1, keepdims=True)
    return latent / np.maximum(norms, _EPS)


def gram_matrix(latent: Any) -> Any:
    """Signed Gram matrix ``S = Z~^T Z~`` of the normalised latent codes.

    The diagonal is ``1`` for every non-degenerate latent vector, so a
    perfectly decorrelated latent space gives ``S = I``.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        The ``(n, n)`` matrix ``S``.
    """
    z = _normalized_latents(latent)
    if _is_torch(z):
        return z.transpose(0, 1) @ z
    return z.T @ z


def orthogonality_matrix(latent: Any) -> Any:
    """Element-wise absolute Gram matrix ``|S|`` of the latent codes.

    This is the matrix whose off-diagonal entries are penalised by
    :func:`orthogonality_loss`.  Because the penalty uses squared entries,
    ``|S_ij|^2 == S_ij^2``; the absolute value is taken only so that the
    returned matrix can be displayed directly as an "off-diagonal energy"
    heat map.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        The ``(n, n)`` matrix ``|S|``.
    """
    s = gram_matrix(latent)
    if _is_torch(s):
        return s.abs()
    return np.abs(s)


#: Alias kept for readability in downstream heat-map code.
similarity_matrix = orthogonality_matrix


def orthogonality_loss(latent: Any) -> Any:
    """Orthogonality (decorrelation) penalty ``L_ort``.

    .. math::

        L_{ort} = \\frac{2}{n(n-1)} \\sum_{1 \\le i < j \\le n} S_{ij}^2

    with ``S = Z~^T Z~`` and ``Z~`` the L2-normalised latent codes.  Only the
    strict upper triangle is summed (the matrix is symmetric, so the lower
    triangle must not be counted twice); the factor ``2/(n(n-1))`` then
    converts the sum over pairs into the mean squared off-diagonal entry.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss (differentiable w.r.t. ``latent`` for torch inputs).
    """
    s = gram_matrix(latent)
    n = s.shape[-1]
    if n < 2:
        # No pairs exist -> the penalty is undefined, use zero.
        if _is_torch(s):
            return s.sum() * 0.0
        return np.float32(0.0)
    if _is_torch(s):
        off_diagonal = torch.triu(s, diagonal=1)
        sum_sq = (off_diagonal ** 2).sum()
        return 2.0 * sum_sq / (n * (n - 1))
    off_diagonal = np.triu(s, k=1)
    sum_sq = np.sum(off_diagonal ** 2)
    return np.float32(2.0 * sum_sq / (n * (n - 1)))


# --------------------------------------------------------------------------- #
# L_var and L_com
# --------------------------------------------------------------------------- #
def latent_variances(latent: Any, unbiased: bool = False) -> Any:
    """Per-dimension variances of the latent codes.

    ``Var_i = E[(z_i - E[z_i])^2]`` estimated over the batch.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.
    unbiased : bool, default False
        Use the unbiased (``N-1``) estimator instead of the population
        (``N``) estimator.  The paper's ``E[.]`` notation corresponds to the
        population estimator, hence the default.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        1-D tensor/array of length ``n`` with the per-dimension variances.
    """
    if _is_torch(latent):
        return latent.var(dim=0, unbiased=unbiased)
    arr = np.asarray(latent, dtype=np.float32)
    ddof = 1 if unbiased and arr.shape[0] > 1 else 0
    return np.var(arr, axis=0, ddof=ddof).astype(np.float32)


def variance_loss(latent: Any, unbiased: bool = False) -> Any:
    """Variance regularisation term ``L_var``.

    .. math::

        L_{var} = \\sum_{i=0}^{n-1} E[(z_i - E[z_i])^2]

    i.e. the sum of the per-dimension variances of the latent codes.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.
    unbiased : bool, default False
        See :func:`latent_variances`.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss.
    """
    variances = latent_variances(latent, unbiased=unbiased)
    if _is_torch(variances):
        return variances.sum()
    return np.float32(np.sum(variances))


def center_of_mass_loss(
    latent: Any,
    eps: float = EPS_COM,
    unbiased: bool = False,
) -> Any:
    """Center-of-mass / variance-ordering term ``L_com``.

    .. math::

        L_{com} = \\frac{\\sum_{i=0}^{n-1} i^{1+\\varepsilon}\\, E[(z_i - E[z_i])^2]}
                        {n \\cdot L_{var}}, \\qquad \\varepsilon = 0.25

    The per-dimension variances are effectively L1-normalised by dividing
    them through the total variance ``L_var``, and the result is additionally
    scaled by ``1/n``.  Because the weights ``i^(1+eps)`` grow with the
    latent index, the term is small when the variance *decreases* along the
    latent dimensions (the PCA-like ordering) and large when it increases.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.
    eps : float, default 0.25
        The exponent ``eps`` of the paper.
    unbiased : bool, default False
        See :func:`latent_variances`.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss.
    """
    variances = latent_variances(latent, unbiased=unbiased)
    n = variances.shape[-1]
    if _is_torch(variances):
        index = torch.arange(n, device=variances.device, dtype=variances.dtype)
        weights = index.pow(1.0 + eps)
        total_var = variances.sum()
        numerator = (weights * variances).sum()
        return numerator / (n * total_var.clamp_min(_EPS))
    variances = np.asarray(variances, dtype=np.float32)
    weights = np.arange(n, dtype=np.float32) ** (1.0 + eps)
    total_var = float(np.sum(variances))
    numerator = float(np.sum(weights * variances))
    return np.float32(numerator / (n * max(total_var, _EPS)))


# --------------------------------------------------------------------------- #
# Composite loss
# --------------------------------------------------------------------------- #
def _build_default_classifier(latent_dim: int, num_classes: int) -> Any:
    """Create a deterministic linear classifier head over the latent code.

    Used only when the caller asks for a classification term without
    supplying logits or a classifier.  The weights are created with a fixed
    generator so that the procedure stays reproducible.
    """
    generator = torch.Generator().manual_seed(0)
    layer = torch.nn.Linear(latent_dim, num_classes)
    bound = 1.0 / max(latent_dim, 1) ** 0.5
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        layer.bias.zero_()
    return layer


def polca_loss(
    reconstruction: Any,
    latent: Any,
    target: Any,
    labels: Any = None,
    class_criterion: Callable[..., Any] | None = None,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    eps: float = EPS_COM,
    logits: Any = None,
    classifier: Any = None,
    return_components: bool = False,
) -> Any:
    """The composite POLCA objective.

    .. math::

        L_{polca} = L_{rec} + L_{class}
                     + \\alpha L_{ort} + \\beta L_{com} + \\gamma L_{var}

    with the paper's defaults ``alpha = 1e-2``, ``beta = 1e-2``,
    ``gamma = 1e-7``.

    Parameters
    ----------
    reconstruction : torch.Tensor or numpy.ndarray
        Decoder output, shape ``(N, input_dim)``.
    latent : torch.Tensor or numpy.ndarray
        Latent codes, shape ``(N, n)``.
    target : torch.Tensor or numpy.ndarray
        Original input, shape ``(N, input_dim)``.
    labels : torch.Tensor or None, optional
        Class labels used by ``L_class``.  ``None`` (the default) means the
        run is unsupervised and ``L_class`` is exactly zero.
    class_criterion : callable, optional
        ``(logits, labels) -> scalar`` callable, e.g.
        ``torch.nn.CrossEntropyLoss()``.
    alpha : float, default 1e-2
        Weight of the orthogonality term.
    beta : float, default 1e-2
        Weight of the center-of-mass term.
    gamma : float, default 1e-7
        Weight of the variance regularisation term.
    eps : float, default 0.25
        Exponent of the center-of-mass term.
    logits : torch.Tensor, optional
        Pre-computed class scores of shape ``(N, C)``.  When given, they are
        used directly for ``L_class``.
    classifier : torch.nn.Module, optional
        A callable applied to the latent code to produce the class scores.
        Takes precedence over ``logits``.
    return_components : bool, default False
        When ``True`` a :class:`POLCALossOutput` with every component is
        returned; otherwise the scalar composite loss is returned.

    Returns
    -------
    torch.Tensor or POLCALossOutput
        The composite loss (and optionally all of its components).

    Notes
    -----
    ``L_class`` requires PyTorch tensors; when it is requested without
    ``logits``/``classifier`` a freshly initialised linear head is used so
    that the term still back-propagates a class-separation signal into the
    latent code.
    """
    rec = reconstruction_loss(reconstruction, target)
    ort = orthogonality_loss(latent)
    com = center_of_mass_loss(latent, eps=eps)
    var = variance_loss(latent)

    # ---- L_class -------------------------------------------------------- #
    if labels is None:
        cls = _zeros_like_scalar(rec, 0.0)
    else:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("classification_loss requires PyTorch")
        latent_t = latent if _is_torch(latent) else torch.as_tensor(latent)
        if classifier is not None:
            class_logits = classifier(latent_t)
        elif logits is not None:
            class_logits = logits if _is_torch(logits) else torch.as_tensor(logits)
        else:
            labels_t = labels if _is_torch(labels) else torch.as_tensor(labels)
            num_classes = int(labels_t.max().item()) + 1
            head = _build_default_classifier(latent_t.shape[-1], num_classes).to(
                device=latent_t.device, dtype=latent_t.dtype
            )
            class_logits = head(latent_t)
        cls = classification_loss(class_logits, labels, class_criterion)

    total = rec + cls + alpha * ort + beta * com + gamma * var

    if return_components:
        return POLCALossOutput(
            total=total,
            rec=rec,
            cls=cls,
            ort=ort,
            com=com,
            var=var,
            extra={"latent_variances": latent_variances(latent)},
        )
    return total
