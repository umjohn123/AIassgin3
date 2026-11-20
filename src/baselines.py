"""Baselines for the POLCA Net assignment: PCA, LDA and a plain autoencoder.

All baselines expose the *same* interface so that the training / evaluation
scripts can treat POLCA Net and the baselines interchangeably:

.. code-block:: python

    model.fit(X_train)          # or fit(X_train, y_train) for LDA
    z     = model.latent(X)     # (N, n) latent codes
    x_hat = model.reconstruct(X)  # (N, input_dim) reconstruction

Classes
-------
:class:`PCABaseline`
    Principal Component Analysis (unsupervised, linear).
:class:`LDABaseline`
    Linear Discriminant Analysis (supervised, linear).
:class:`PlainAEBaseline`
    The plain autoencoder baseline (no POLCA losses) trained with MSE only.

Module-level helpers (:func:`fit_pca`, :func:`pca_latent_and_reconstruction`,
:func:`fit_lda`, :func:`lda_latent`) expose the same functionality in a
functional style.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BaselineModel",
    "PCABaseline",
    "LDABaseline",
    "PlainAEBaseline",
    "fit_pca",
    "pca_latent_and_reconstruction",
    "fit_lda",
    "lda_latent",
    "DEFAULT_SEED",
]

#: Global seed (the paper uses 5).
DEFAULT_SEED = 5


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (local copy of the helper).

    This duplicates ``src.polca.utils.set_seed`` on purpose so that this
    module does not depend on a file that is written concurrently.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _import_autoencoder():
    """Import :class:`src.polca.model.Autoencoder` lazily and robustly.

    The model module may be written concurrently by another process, so the
    import is deferred until a baseline is actually instantiated and the
    project root is added to ``sys.path`` if needed.
    """
    import importlib
    import os
    import sys

    try:
        module = importlib.import_module("src.polca.model")
    except ModuleNotFoundError:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("src.polca.model")
    return module.Autoencoder


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensors / arrays to a float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Common protocol
# --------------------------------------------------------------------------- #
class BaselineModel:
    """Base class defining the uniform baseline interface.

    Sub-classes must implement :meth:`fit`, :meth:`latent` and
    :meth:`reconstruct`.
    """

    #: Name used in result tables / figures.
    name: str = "baseline"

    def fit(self, X_train, y_train=None):  # pragma: no cover - interface
        """Fit the baseline on the training data."""
        raise NotImplementedError

    def latent(self, X):  # pragma: no cover - interface
        """Return the latent codes of ``X`` with shape ``(N, n)``."""
        raise NotImplementedError

    def reconstruct(self, X):  # pragma: no cover - interface
        """Return the reconstruction of ``X`` with shape ``(N, input_dim)``."""
        raise NotImplementedError

    # Convenience alias used by the evaluation scripts.
    def encode(self, X):
        """Alias of :meth:`latent`."""
        return self.latent(X)

    def decode(self, z):  # pragma: no cover - optional
        """Decode latent codes (only meaningful for generative models)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def fit_pca(X_train, n_components: int, random_state: int = DEFAULT_SEED):
    """Fit a scikit-learn :class:`~sklearn.decomposition.PCA` on ``X_train``.

    Parameters
    ----------
    X_train : array-like
        ``(N, D)`` training data (already in ``[0, 1]``).
    n_components : int
        Number of principal components to keep.
    random_state : int, default 5
        Passed to the (full) SVD solver for reproducibility.

    Returns
    -------
    sklearn.decomposition.PCA
        The *fitted* PCA object.
    """
    from sklearn.decomposition import PCA

    x = _to_numpy(X_train)
    pca = PCA(n_components=int(n_components), random_state=random_state)
    pca.fit(x)
    return pca


def pca_latent_and_reconstruction(pca, X) -> Tuple[np.ndarray, np.ndarray]:
    """Project ``X`` on the fitted PCA and reconstruct it.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        A fitted PCA model.
    X : array-like
        ``(N, D)`` data.

    Returns
    -------
    tuple of np.ndarray
        ``(latent, reconstruction)`` with shapes ``(N, n_components)`` and
        ``(N, D)``.
    """
    x = _to_numpy(X)
    latent = pca.transform(x)
    recon = pca.inverse_transform(latent)
    return latent, recon


class PCABaseline(BaselineModel):
    """PCA baseline with the uniform baseline interface.

    Parameters
    ----------
    n_components : int
        Number of principal components (latent dimension).
    random_state : int, default 5
        Reproducibility seed.

    Examples
    --------
    >>> pca = PCABaseline(n_components=8)
    >>> _ = pca.fit(X_train)                      # doctest: +SKIP
    >>> z = pca.latent(X_test)                    # doctest: +SKIP
    >>> x_hat = pca.reconstruct(X_test)           # doctest: +SKIP
    """

    name = "PCA"

    def __init__(self, n_components: int = 8, random_state: int = DEFAULT_SEED) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.pca = None

    def fit(self, X_train, y_train=None):
        """Fit the PCA on ``X_train`` (``y_train`` is ignored)."""
        self.pca = fit_pca(X_train, self.n_components, self.random_state)
        return self

    def latent(self, X) -> np.ndarray:
        """Return the principal-component scores of ``X``."""
        if self.pca is None:
            raise RuntimeError("PCABaseline must be fitted before calling latent()")
        return self.pca.transform(_to_numpy(X))

    def reconstruct(self, X) -> np.ndarray:
        """Return the PCA reconstruction of ``X``."""
        if self.pca is None:
            raise RuntimeError("PCABaseline must be fitted before calling reconstruct()")
        return self.pca.inverse_transform(self.pca.transform(_to_numpy(X)))


# --------------------------------------------------------------------------- #
# LDA
# --------------------------------------------------------------------------- #
def fit_lda(
    X_train,
    y_train,
    n_components: int,
    random_state: int = DEFAULT_SEED,
):
    """Fit a robust :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`.

    The solver is ``'lsqr'`` with ``shrinkage='auto'`` (Ledoit-Wolf), which
    is numerically stable even when the number of components is large or
    the covariance is singular (e.g. the highly correlated pixels of the
    digits dataset).

    Parameters
    ----------
    X_train : array-like
        ``(N, D)`` training data.
    y_train : array-like
        ``(N,)`` class labels.
    n_components : int
        Requested number of discriminant dimensions (automatically clipped
        to ``min(n_classes - 1, n_features)``).
    random_state : int, default 5
        Kept for API consistency.

    Returns
    -------
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        The *fitted* LDA object.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    x = _to_numpy(X_train)
    y = np.asarray(y_train).reshape(-1)
    max_components = int(min(len(np.unique(y)) - 1, x.shape[1]))
    n = int(max(1, min(int(n_components), max_components)))

    lda = LinearDiscriminantAnalysis(
        n_components=n, solver="lsqr", shrinkage="auto"
    )
    lda.fit(x, y)
    # sklearn's 'lsqr' solver is classification-only: it does not expose
    # ``transform``.  We therefore build the discriminant basis ourselves
    # from the fitted within-class covariance and class means so that the
    # baseline still provides a genuine dimensionality-reduction latent
    # space (see :func:`lda_latent`).
    lda._polca_discriminant_basis = _compute_lda_basis(lda, x, y)
    return lda


def _compute_lda_basis(lda, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the LDA discriminant directions for the ``lsqr`` solver.

    The generalised eigenvectors of the between-class scatter ``Sb`` with
    respect to the within-class scatter ``Sw`` define the classic Fisher
    discriminant axes.  Here the *shrinked* within-class covariance used by
    the ``lsqr`` solver is reused, so the axes are consistent with the
    fitted decision function.

    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        The fitted LDA object (its ``covariance_`` is the shrunk within
        class covariance).
    X : np.ndarray
        ``(N, D)`` training data.
    y : np.ndarray
        ``(N,)`` training labels.

    Returns
    -------
    np.ndarray
        ``(D, k)`` matrix whose columns are the discriminant directions,
        ordered by decreasing between/within scatter ratio.
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    max_components = int(min(len(classes) - 1, n_features))

    sw = np.asarray(lda.covariance_, dtype=np.float64)
    # The class means are computed in float64 but the accumulation is done
    # per class on the float32 data, which keeps the peak memory footprint
    # at one class at a time instead of a float64 copy of the whole design
    # matrix (a 60000 x 784 float64 array needs ~360 MB on its own).
    means = np.stack(
        [np.asarray(X[y == c], dtype=np.float64).mean(axis=0) for c in classes]
    )

    # Between-class scatter
    overall_mean = np.asarray(X, dtype=np.float64).mean(axis=0)
    sb = np.zeros((n_features, n_features))
    for idx, c in enumerate(classes):
        diff = (means[idx] - overall_mean).reshape(-1, 1)
        sb += np.asarray(lda.priors_[idx]) * (diff @ diff.T)

    # generalised eigenproblem  Sb w = lambda Sw w
    try:
        from scipy.linalg import eigh as _eigh

        evals, evecs = _eigh(sb, sw)
    except Exception:
        # numpy fallback (eig of the whitened problem)
        inv_sw = np.linalg.pinv(sw)
        evals, evecs = np.linalg.eig(inv_sw @ sb)
        evals, evecs = np.real(evals), np.real(evecs)

    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    k = max(1, min(max_components, int(lda._max_components)))
    basis = evecs[:, :k]
    # fix an arbitrary sign so the result is deterministic
    for j in range(basis.shape[1]):
        col = basis[:, j]
        if col[np.argmax(np.abs(col))] < 0:
            basis[:, j] = -col
    return basis


def lda_latent(lda, X) -> np.ndarray:
    """Project ``X`` on the fitted LDA discriminant space.

    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        A fitted LDA model.
    X : array-like
        ``(N, D)`` data.

    Returns
    -------
    np.ndarray
        ``(N, n_components)`` discriminant coordinates.
    """
    x = _to_numpy(X)
    basis = getattr(lda, "_polca_discriminant_basis", None)
    if basis is None:
        return lda.transform(x)
    return x @ basis


class LDABaseline(BaselineModel):
    """LDA baseline with the uniform baseline interface.

    Parameters
    ----------
    n_components : int
        Number of discriminant dimensions.
    random_state : int, default 5
        Reproducibility seed.
    """

    name = "LDA"

    def __init__(self, n_components: int = 8, random_state: int = DEFAULT_SEED) -> None:
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self.lda = None

    def fit(self, X_train, y_train=None):
        """Fit the LDA on ``(X_train, y_train)``."""
        if y_train is None:
            raise ValueError("LDABaseline.fit requires the training labels y_train")
        self.lda = fit_lda(X_train, y_train, self.n_components, self.random_state)
        return self

    def latent(self, X) -> np.ndarray:
        """Return the discriminant coordinates of ``X``."""
        if self.lda is None:
            raise RuntimeError("LDABaseline must be fitted before calling latent()")
        return lda_latent(self.lda, X)

    def reconstruct(self, X) -> np.ndarray:
        """Reconstruct ``X`` from its discriminant coordinates.

        LDA has no generative model, so the reconstruction is the orthogonal
        projection of ``X`` onto the LDA discriminant subspace (the
        least-squares linear reconstruction from the discriminant
        coordinates).
        """
        if self.lda is None:
            raise RuntimeError("LDABaseline must be fitted before calling reconstruct()")
        x = _to_numpy(X)
        basis = getattr(self.lda, "_polca_discriminant_basis", None)
        if basis is None:
            z = self.lda.transform(x)
            coef = self.lda.coef_.T
            intercept = self.lda.intercept_
            return (z - intercept) @ coef.T
        # least squares reconstruction: X ~= Z @ W with W = pinv(Z) @ X
        z = x @ basis
        w = np.linalg.pinv(z) @ x
        return z @ w


# --------------------------------------------------------------------------- #
# Plain autoencoder
# --------------------------------------------------------------------------- #
class PlainAEBaseline(BaselineModel):
    """Plain autoencoder baseline (no POLCA regularisers).

    The network is :class:`src.polca.model.Autoencoder` -- the very same
    architecture as POLCA Net but trained with the reconstruction loss
    only.  The import of the model is **lazy** (inside ``__init__``) so that
    this module can be imported even while ``src/polca/model.py`` is being
    written by another process.

    Parameters
    ----------
    input_dim : int
        Dimension of the flattened input (64 for digits, 784 for MNIST).
    latent_dim : int, default 8
        Dimension of the latent space.
    hidden_dims : sequence of int, default ``(64, 32)``
        Hidden widths of the encoder (the decoder mirrors them).
    linear_decoder : bool, default False
        Use a single linear decoder layer.
    activation : str, default ``"relu"``
        Hidden activation.
    """

    name = "PlainAE"

    def __init__(
        self,
        input_dim: int = 784,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        linear_decoder: bool = False,
        activation: str = "relu",
    ) -> None:
        # Lazy import: src.polca.model may be written concurrently.
        Autoencoder = _import_autoencoder()

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.model = Autoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=tuple(hidden_dims),
            linear_decoder=linear_decoder,
            activation=activation,
        )
        self.device = None
        self.history: list = []

    # -- training ---------------------------------------------------------- #
    def fit(
        self,
        X_train,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        y_train=None,
        verbose: bool = False,
    ):
        """Train the autoencoder with the plain MSE reconstruction loss.

        Parameters
        ----------
        X_train : array-like
            ``(N, input_dim)`` training data in ``[0, 1]``.
        epochs : int, default 50
            Number of passes over the data.
        lr : float, default 1e-3
            Learning rate of the Adam optimiser.
        batch_size : int, default 64
            Mini-batch size.
        device : str, optional
            ``"cpu"`` / ``"cuda"``; inferred when omitted.
        seed : int, default 5
            Seed applied before the training loop.
        y_train : optional
            Unused (kept for interface compatibility).
        verbose : bool, default False
            Print the epoch loss.

        Returns
        -------
        PlainAEBaseline
            ``self`` (chainable).
        """
        import torch
        from torch import nn

        set_seed(seed)
        x = np.asarray(_to_numpy(X_train), dtype=np.float32)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = self.model.to(device)

        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=int(batch_size), shuffle=True, drop_last=False
        )

        optimiser = torch.optim.Adam(self.model.parameters(), lr=float(lr))
        criterion = nn.MSELoss()

        self.history = []
        self.model.train()
        for epoch in range(int(epochs)):
            total, count = 0.0, 0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                total += float(loss.item()) * batch.shape[0]
                count += int(batch.shape[0])
            epoch_loss = total / max(count, 1)
            self.history.append(epoch_loss)
            if verbose:
                print(f"[PlainAE] epoch {epoch + 1}/{epochs} loss={epoch_loss:.6f}")
        self.model.eval()
        return self

    # -- inference --------------------------------------------------------- #
    def _infer(self, X) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
        x = np.asarray(_to_numpy(X), dtype=np.float32)
        with torch.no_grad():
            recon, latent = self.model(torch.as_tensor(x, device=self.device))
        return _to_numpy(latent), _to_numpy(recon)

    def latent(self, X) -> np.ndarray:
        """Return the latent codes of ``X`` with shape ``(N, latent_dim)``."""
        return self._infer(X)[0]

    def reconstruct(self, X) -> np.ndarray:
        """Return the reconstruction of ``X`` with shape ``(N, input_dim)``."""
        return self._infer(X)[1]
