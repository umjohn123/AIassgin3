"""Evaluation metrics for the POLCA Net assignment (arXiv:2410.07289).

The module implements the three groups of metrics used in the paper:

**Reconstruction quality** (on images in ``[0, 1]``)
    :func:`nrmse`, :func:`psnr`, :func:`ssim`.

**Latent-space PCA properties**
    :func:`orthogonality_error`, :func:`cosine_similarity_matrix`,
    :func:`latent_variance_per_dim`, :func:`variance_ordering_score`.

**Downstream classification**
    :func:`classification_report_from_latent`.

All functions are pure ``numpy``/``torch`` code and do not require a GPU.
``scikit-image`` is used for SSIM when available (it is installed in the
assignment environment) but a self-contained Wang et al. fallback is
provided so that the module never hard-depends on an optional package.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "nrmse",
    "psnr",
    "ssim",
    "orthogonality_error",
    "cosine_similarity_matrix",
    "latent_variance_per_dim",
    "variance_ordering_score",
    "spearmanr",
    "classification_report_from_latent",
    "evaluate_reconstruction",
    "aggregate_metrics",
    "DEFAULT_SEED",
]

#: Global seed (the paper uses 5).
DEFAULT_SEED = 5


# --------------------------------------------------------------------------- #
# Generic array helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x) -> np.ndarray:
    """Convert torch tensors / numpy arrays / lists to a numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _as_image_stack(x, image_shape: Optional[tuple] = None) -> np.ndarray:
    """Reshape ``x`` to a ``(N, H, W)`` float array.

    If ``x`` is 2-D it is reshaped using ``image_shape`` when given, else a
    square root of the feature count is assumed (8x8 for 64, 28x28 for
    784).
    """
    arr = _to_numpy(x)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 2:
        if image_shape is not None:
            return arr.reshape(arr.shape[0], image_shape[0], image_shape[1])
        d = int(arr.shape[1])
        side = int(round(np.sqrt(d)))
        if side * side != d:
            raise ValueError(
                f"Cannot infer a square image shape from {d} features; "
                "pass image_shape=(H, W) explicitly."
            )
        return arr.reshape(arr.shape[0], side, side)
    raise ValueError(f"Unsupported array shape {arr.shape}")


# --------------------------------------------------------------------------- #
# Reconstruction metrics
# --------------------------------------------------------------------------- #
def nrmse(orig, recon) -> float:
    """Normalized Root Mean Square Error.

    ``NRMSE = RMSE / (max(orig) - min(orig))`` -- the "Normalized Mean Square
    Error" variant of the paper: the RMSE is normalised by the dynamic range
    of the *original* signal, so the result is 0 for a perfect
    reconstruction and 1 when the error equals the signal range.

    Parameters
    ----------
    orig : array-like
        Original data, shape ``(N, ...)``.
    recon : array-like
        Reconstruction with the same shape as ``orig``.

    Returns
    -------
    float
        The (scalar) normalised RMSE.
    """
    o = _to_numpy(orig)
    r = _to_numpy(recon)
    if o.shape != r.shape:
        raise ValueError(f"Shape mismatch: {o.shape} vs {r.shape}")
    mse = float(np.mean((o - r) ** 2))
    rmse = float(np.sqrt(mse))
    rng = float(o.max() - o.min())
    if rng <= 0:
        # constant image: fall back to plain RMSE
        return rmse
    return rmse / rng


def psnr(orig, recon, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in decibel.

    ``PSNR = 20 * log10(data_range / RMSE)``.  A perfect reconstruction
    returns ``inf`` (clipped to a large finite value, 200 dB, to keep the
    metrics JSON-serialisable).

    Parameters
    ----------
    orig, recon : array-like
        Arrays with identical shapes, values in ``[0, data_range]``.
    data_range : float, default 1.0
        Dynamic range of the signal.

    Returns
    -------
    float
        PSNR in dB.
    """
    o = _to_numpy(orig)
    r = _to_numpy(recon)
    if o.shape != r.shape:
        raise ValueError(f"Shape mismatch: {o.shape} vs {r.shape}")
    rmse = float(np.sqrt(np.mean((o - r) ** 2)))
    if rmse <= 0:
        return 200.0
    return float(20.0 * np.log10(float(data_range) / rmse))


def _gaussian_kernel_1d(ksize: int = 7, sigma: float = 1.5) -> np.ndarray:
    """Normalised 1-D Gaussian window (Wang et al. 2004)."""
    coords = np.arange(ksize) - (ksize - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _ssim_fallback(
    orig: np.ndarray,
    recon: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Self-contained SSIM (Wang et al. 2004) with a 7x7 Gaussian window.

    Parameters
    ----------
    orig, recon : np.ndarray
        ``(N, H, W)`` arrays.
    data_range : float, default 1.0
        Dynamic range of the images.

    Returns
    -------
    float
        Mean SSIM over the batch.
    """
    ksize = 7
    g = _gaussian_kernel_1d(ksize, 1.5)
    # Reflect padding: the border is mirrored instead of dropped, so the
    # output keeps the full (N, H, W) shape and no border pixels are lost.
    pad = ksize // 2

    def filt(x: np.ndarray) -> np.ndarray:
        """Gaussian filter along both spatial axes.

        The 2-D window ``g x g`` is separable, so the filter is applied as
        two consecutive 1-D convolutions (height, then width).  This costs
        ``O(N * H * W * k)`` instead of ``O(N * H * W * k * k)`` of the naive
        two-dimensional convolution -- on the 9600-image MNIST test split
        the naive version needed ~30 s per call, the separable one ~0.1 s.
        """
        x = np.asarray(x, dtype=np.float64)
        # vertical (H axis)
        xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(x)
        for i in range(ksize):
            out += g[i] * xp[:, i : i + x.shape[1], :]
        # horizontal (W axis)
        xp = np.pad(out, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        out2 = np.zeros_like(out)
        for i in range(ksize):
            out2 += g[i] * xp[:, :, i : i + x.shape[2]]
        return out2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = filt(orig)
    mu_y = filt(recon)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = filt(orig * orig) - mu_x2
    sigma_y2 = filt(recon * recon) - mu_y2
    sigma_xy = filt(orig * recon) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def _ssim_vectorised(
    orig: np.ndarray,
    recon: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> Optional[float]:
    """Vectorised SSIM reproducing ``skimage``'s reference implementation.

    ``skimage.metrics.structural_similarity`` uses a **uniform (box)**
    7x7 window -- not a Gaussian one -- and discards the border pixels that
    the window cannot cover.  This function computes the same quantity for a
    whole batch at once:

    * local means / variances / covariance come from a separable uniform
      filter (:func:`scipy.ndimage.uniform_filter`) applied to the two
      spatial axes only,
    * the covariance normalisation uses ``NP / (NP - 1)`` with
      ``NP = win_size ** 2``,
    * the border of ``win_size // 2`` pixels is cropped before averaging.

    Doing this for the whole batch instead of looping over the images in
    Python turns a ~30 s call on the 9600-image MNIST test split into a
    fraction of a second.

    Parameters
    ----------
    orig : np.ndarray
        ``(N, H, W)`` array.
    recon : np.ndarray
        ``(N, H, W)`` array.
    data_range : float, default 1.0
        Dynamic range of the images.
    win_size : int, default 7
        Size of the uniform filter window.

    Returns
    -------
    float or None
        The mean SSIM over the batch, or ``None`` when SciPy is unavailable.
    """
    try:
        from scipy.ndimage import uniform_filter
    except Exception:  # pragma: no cover - SciPy is an optional dependency
        return None

    x = np.asarray(orig, dtype=np.float64)
    y = np.asarray(recon, dtype=np.float64)
    npix = float(win_size ** 2)
    cov_norm = npix / (npix - 1.0)

    # uniform_filter over the two spatial axes only (axis 0 is the batch)
    ux = uniform_filter(x, size=win_size, axes=(1, 2))
    uy = uniform_filter(y, size=win_size, axes=(1, 2))
    uxx = uniform_filter(x * x, size=win_size, axes=(1, 2))
    uyy = uniform_filter(y * y, size=win_size, axes=(1, 2))
    uxy = uniform_filter(x * y, size=win_size, axes=(1, 2))

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    a1 = 2 * ux * uy + c1
    a2 = 2 * vxy + c2
    b1 = ux * ux + uy * uy + c1
    b2 = vx + vy + c2
    ssim_map = (a1 * a2) / (b1 * b2)

    pad = (win_size - 1) // 2
    if pad > 0 and ssim_map.shape[1] > 2 * pad and ssim_map.shape[2] > 2 * pad:
        ssim_map = ssim_map[:, pad:-pad, pad:-pad]
    return float(np.mean(ssim_map))


def ssim(orig, recon, data_range: float = 1.0, image_shape: Optional[tuple] = None) -> float:
    """Structural Similarity Index (mean SSIM over the batch).

    The reference implementation (``scikit-image``) is used when it is
    installed; a *vectorised* equivalent of it handles the whole batch in
    one call, and a self-contained Gaussian-window implementation is kept as
    a last-resort fallback so that the function never fails because of a
    missing optional dependency.

    Parameters
    ----------
    orig, recon : array-like
        Arrays of identical shape; either ``(N, H, W)`` images or
        ``(N, D)`` feature matrices (in which case ``image_shape`` may be
        used to reshape them).
    data_range : float, default 1.0
        Dynamic range of the images (``1.0`` because everything is
        normalised to ``[0, 1]``).
    image_shape : tuple, optional
        ``(H, W)`` used to reshape 2-D inputs.

    Returns
    -------
    float
        Mean SSIM over the images of the batch.
    """
    o = _as_image_stack(orig, image_shape)
    r = _as_image_stack(recon, image_shape)
    if o.shape != r.shape:
        raise ValueError(f"Shape mismatch: {o.shape} vs {r.shape}")

    # Fast path: vectorised reference implementation for the whole batch.
    value = _ssim_vectorised(o, r, data_range=data_range)
    if value is not None:
        return value

    try:  # pragma: no cover - depends on the environment
        from skimage.metrics import structural_similarity as _sk_ssim

        # NOTE: no ``channel_axis`` is passed on purpose.  The images are
        # single-channel, so passing ``channel_axis=-1`` would make
        # scikit-image treat the *columns* of a 28x28 digit as 28 separate
        # channels and compute the SSIM of 28 one-dimensional signals -- a
        # silently wrong number.  The vectorised path above reproduces the
        # correct (no channel axis) result bit for bit.
        values = [
            float(_sk_ssim(o[i], r[i], data_range=data_range))
            for i in range(o.shape[0])
        ]
        return float(np.mean(values))
    except Exception:
        return _ssim_fallback(o, r, data_range=data_range)

# --------------------------------------------------------------------------- #
# Latent-space PCA properties
# --------------------------------------------------------------------------- #
def cosine_similarity_matrix(latent) -> np.ndarray:
    """Cosine-similarity matrix over the **latent dimensions**.

    Following the paper's convention, the latent matrix ``Z`` has shape
    ``(N, n)`` (``N`` samples, ``n`` latent dimensions) and the paper builds
    ``S = Z^T Z`` over the latent *dimensions*, i.e. each latent dimension
    is treated as a vector of length ``N``.  This function therefore returns
    the ``(n, n)`` matrix whose entry ``(i, j)`` is the cosine similarity
    between latent dimension ``i`` and latent dimension ``j``:

    .. math::
        S_{ij} = \\frac{\\sum_k z_{ki} z_{kj}}
                      {\\sqrt{\\sum_k z_{ki}^2} \\sqrt{\\sum_k z_{kj}^2}}

    A perfectly orthogonal latent space gives ``S == I``.

    Parameters
    ----------
    latent : array-like
        ``(N, n)`` latent codes.  A 1-D array is treated as ``(N, 1)``.

    Returns
    -------
    np.ndarray
        ``(n, n)`` cosine-similarity matrix.
    """
    z = _to_numpy(latent)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2:
        raise ValueError(f"latent must be 2-D, got {z.shape}")
    norms = np.linalg.norm(z, axis=0)
    norms = np.where(norms == 0, np.inf, norms)
    z_norm = z / norms
    return z_norm.T @ z_norm


def orthogonality_error(latent) -> float:
    """Mean absolute off-diagonal cosine similarity of the latent codes.

    This quantifies how far the latent space is from being orthogonal.
    ``0`` means perfectly orthogonal; the value is close to 0 for both PCA
    and POLCA (the latter because of the ``L_ort`` loss term).

    Parameters
    ----------
    latent : array-like
        ``(N, n)`` latent codes.

    Returns
    -------
    float
        Mean of ``|S_ij|`` over the off-diagonal entries of the cosine
        similarity matrix.
    """
    s = cosine_similarity_matrix(latent)
    n = s.shape[0]
    if n < 2:
        return 0.0
    off = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(s[off])))


def latent_variance_per_dim(latent) -> np.ndarray:
    """Variance of every latent dimension across the batch.

    Parameters
    ----------
    latent : array-like
        ``(N, n)`` latent codes.

    Returns
    -------
    np.ndarray
        ``(n,)`` array of per-dimension variances (population variance,
        ``ddof=0``).
    """
    z = _to_numpy(latent)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return np.var(z, axis=0)


def spearmanr(a, b) -> float:
    """Spearman rank correlation coefficient (no SciPy dependency).

    The ranks are computed with *average* ranks for ties and the Pearson
    correlation of the two rank vectors is returned.

    Parameters
    ----------
    a, b : array-like
        Two 1-D sequences of the same length.

    Returns
    -------
    float
        The Spearman correlation (0.0 if either input is constant).
    """
    x = _to_numpy(a).reshape(-1)
    y = _to_numpy(b).reshape(-1)
    if x.size != y.size:
        raise ValueError("a and b must have the same length")
    if x.size < 2:
        return 0.0

    def rankdata(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        # average ranks for ties
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
            i = j + 1
        return ranks

    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if denom == 0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def variance_ordering_score(latent) -> float:
    """Spearman correlation between latent dimension index and its variance.

    The score is defined as ``rho(index, -variance)``, i.e. the rank
    correlation between the dimension index and the *negated* variance.
    Therefore a **positive** value means the variance *decreases* with the
    dimension index, which is the ordering property that the paper's
    ``L_com`` loss term encourages (the most "important" latent component
    comes first, the least important last).

    Parameters
    ----------
    latent : array-like
        ``(N, n)`` latent codes.

    Returns
    -------
    float
        Spearman correlation in ``[-1, 1]``.  ``+1`` means the variances are
        perfectly sorted in decreasing order, ``-1`` perfectly increasing.
    """
    var = latent_variance_per_dim(latent)
    idx = np.arange(var.size, dtype=np.float64)
    return spearmanr(idx, -var)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_report_from_latent(
    latent_train,
    y_train,
    latent_test,
    y_test,
    classifiers: Optional[Mapping[str, object]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Dict[str, float]]:
    """Train linear classifiers on the latent space and score them.

    The paper evaluates four linear classifiers on the learned latent
    representation: Perceptron, Ridge Classifier, Logistic Regression and
    a linear SVM.

    Parameters
    ----------
    latent_train : array-like
        ``(N_train, n)`` training latent codes.
    y_train : array-like
        ``(N_train,)`` training labels.
    latent_test : array-like
        ``(N_test, n)`` test latent codes.
    y_test : array-like
        ``(N_test,)`` test labels.
    classifiers : mapping, optional
        Custom ``{name: estimator}`` mapping.  Defaults to the four
        classifiers used in the paper.
    seed : int, default 5
        ``random_state`` used for every estimator.

    Returns
    -------
    dict
        ``{classifier_name: {"accuracy": float, "f1_macro": float}}``.
    """
    from sklearn.linear_model import (
        LogisticRegression,
        Perceptron,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score

    xtr = _to_numpy(latent_train)
    ytr = np.asarray(y_train).reshape(-1)
    xte = _to_numpy(latent_test)
    yte = np.asarray(y_test).reshape(-1)

    if classifiers is None:
        classifiers = {
            "Perceptron": Perceptron(random_state=seed, max_iter=1000, tol=1e-3),
            "Ridge": RidgeClassifier(random_state=seed),
            "LogisticRegression": LogisticRegression(
                random_state=seed, max_iter=2000
            ),
            "LinearSVM": LinearSVC(random_state=seed, max_iter=5000),
        }

    results: Dict[str, Dict[str, float]] = {}
    for name, clf in classifiers.items():
        est = clf
        if hasattr(clf, "get_params") and "random_state" in est.get_params():
            est.set_params(random_state=seed)
        est.fit(xtr, ytr)
        pred = est.predict(xte)
        results[str(name)] = {
            "accuracy": float(accuracy_score(yte, pred)),
            "f1_macro": float(f1_score(yte, pred, average="macro")),
        }
    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def evaluate_reconstruction(
    latent,
    model,
    X_test,
    image_shape: Optional[tuple] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Reconstruct ``X_test`` with ``model`` and return the metric dict.

    Parameters
    ----------
    latent : array-like
        Latent codes of the test set.  They are used only for the latent
        diagnostics (orthogonality / variance ordering); the reconstruction
        itself is produced by decoding the *encoded* input so that the
        result does not depend on the caller having decoded the codes.
    model : object
        Either a ``torch.nn.Module`` with a ``forward(x) -> (recon, latent)``
        signature (e.g. :class:`src.polca.model.POLCANet`) or any object
        exposing ``.reconstruct(X) -> array``.
    X_test : array-like
        ``(N, input_dim)`` test inputs in ``[0, 1]``.
    image_shape : tuple, optional
        ``(H, W)`` used to reshape the flat data for SSIM.
    device : str, optional
        Torch device string; inferred from the model when omitted.

    Returns
    -------
    dict
        ``{"nrmse": float, "psnr": float, "ssim": float}``.
    """
    x = _to_numpy(X_test)

    recon = None
    if hasattr(model, "reconstruct"):
        try:
            recon = _to_numpy(model.reconstruct(x))
        except Exception:
            recon = None
    if recon is None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"torch is required to run the model: {exc}") from exc

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model_dev = model.to(dev)
        except Exception:
            model_dev = model
        with torch.no_grad():
            out = model_dev(torch.as_tensor(x, dtype=torch.float32, device=dev))
        if isinstance(out, (tuple, list)):
            recon = _to_numpy(out[0])
        else:
            recon = _to_numpy(out)

    recon = recon.reshape(x.shape)
    return {
        "nrmse": nrmse(x, recon),
        "psnr": psnr(x, recon),
        "ssim": ssim(x, recon, data_range=1.0, image_shape=image_shape),
    }


def aggregate_metrics(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Average a list of metric dictionaries.

    Keys that are missing from a row are ignored; the returned dictionary
    contains the mean of every key seen across the rows.

    Parameters
    ----------
    rows : iterable of mapping
        Metric dictionaries, e.g. the output of
        :func:`evaluate_reconstruction`.

    Returns
    -------
    dict
        ``{key: mean_value}``.
    """
    rows = list(rows)
    if not rows:
        return {}
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None]
        if values:
            out[key] = float(np.mean(values))
    return out
