"""Dataset loading for the POLCA Net assignment (CISC3024 Assignment 3).

Three datasets are supported through a single uniform API:

============  ======  ===============  =====================================
name          dim     images           source
============  ======  ===============  =====================================
``digits``    64      8x8  grayscale   ``sklearn.datasets.load_digits`` (100% offline)
``mnist``     784     28x28 grayscale  ``torchvision.datasets.MNIST``
``fashion``   784     28x28 grayscale  ``torchvision.datasets.FashionMNIST``
============  ======  ===============  =====================================

Every loader returns the data **already normalised to float32 in [0, 1]**
and offers two views:

* ``images`` -- ``(N, H, W)`` ``uint8`` array used for visualisation.
* ``X``      -- ``(N, H*W)`` ``float32`` array in ``[0, 1]`` used as the
  network input (the "flattened" view).

The digit dataset has a different input dimension (64) than MNIST/Fashion
(784); the model must therefore always be told the ``input_dim`` of the
dataset in use.  :func:`to_28x28` provides a nearest-neighbour upsample of
the 8x8 digits to 28x28 so that reconstruction figures of the different
datasets can be compared visually.  **The upsample is only used for
plotting -- the model still receives the native 64-dimensional digits.**
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DatasetBundle",
    "get_dataset",
    "available_datasets",
    "DATASETS",
    "load_digits_dataset",
    "load_mnist_dataset",
    "load_fashion_mnist_dataset",
    "train_test_split_idx",
    "flatten_images",
    "to_28x28",
    "DEFAULT_SEED",
    "DEFAULT_DATA_ROOT",
]

#: Global seed used by the whole project (paper: seed = 5).
DEFAULT_SEED = 5

#: Root folder where torchvision caches the downloaded datasets.
DEFAULT_DATA_ROOT = os.environ.get("POLCA_DATA_ROOT", "./data")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten a stack of images into a 2-D feature matrix.

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(N, H, W)`` (or ``(N, H*W)`` which is returned
        unchanged).

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(N, H*W)``.
    """
    arr = np.asarray(images)
    if arr.ndim == 2:
        out = arr.astype(np.float32, copy=False)
    else:
        out = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
    return out


def to_28x28(images: np.ndarray) -> np.ndarray:
    """Nearest-neighbour upsample of 8x8 images to 28x28.

    Uses true nearest-neighbour resampling: the output pixel ``(i, j)``
    takes the value of the input pixel ``(i*8 // 28, j*8 // 28)``.  No
    interpolation is applied, so the pixel values stay exactly in the
    original range (0..16 for the digits dataset).

    Parameters
    ----------
    images : np.ndarray
        Array of shape ``(N, 8, 8)`` or ``(N, 64)``.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(N, 28, 28)``.
    """
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], 8, 8)
    if arr.shape[-2:] != (8, 8):
        raise ValueError(f"to_28x28 expects 8x8 images, got {arr.shape[-2:]}")
    if arr.dtype != np.uint8:
        # assume the values are already normalised to [0, 1]
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    idx = (np.arange(28) * 8) // 28  # nearest-neighbour index map
    return arr[:, idx][:, :, idx].astype(np.uint8)


def train_test_split_idx(
    n: int,
    test_size: float = 0.2,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic train/test index split.

    Parameters
    ----------
    n : int
        Total number of samples.
    test_size : float, default 0.2
        Fraction of the data used for testing.
    seed : int, default 5
        Random seed (the paper uses 5).

    Returns
    -------
    tuple of np.ndarray
        ``(train_idx, test_idx)`` -- two ``int64`` arrays with disjoint,
        sorted indices.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test)) if n > 1 else 0
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    """Uniform container returned by every loader.

    Attributes
    ----------
    X : np.ndarray
        ``(N, input_dim)`` ``float32`` features in ``[0, 1]``.
    y : np.ndarray
        ``(N,)`` ``int64`` labels.
    images : np.ndarray
        ``(N, H, W)`` ``uint8`` images for visualisation.
    num_classes : int
        Number of distinct classes present in the split.
    input_dim : int
        Length of the flattened image (``H*W``).
    image_shape : tuple of int
        ``(H, W)``.
    name : str
        Dataset name.
    """

    X: np.ndarray
    y: np.ndarray
    images: np.ndarray
    num_classes: int
    input_dim: int
    image_shape: Tuple[int, int]
    name: str
    train: bool = True
    extra: Dict[str, object] = field(default_factory=dict)

    # dict-like access so that ``bundle['X']`` also works (the assignment
    # scripts index the returned object with strings).
    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def keys(self) -> Sequence[str]:
        """Return the field names (for ``dict(bundle)``-like usage)."""
        return tuple(self.__dataclass_fields__.keys())

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DatasetBundle(name={self.name!r}, n={len(self)}, "
            f"input_dim={self.input_dim}, image_shape={self.image_shape}, "
            f"num_classes={self.num_classes})"
        )


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_digits_dataset(train: bool = True, flatten: bool = True) -> DatasetBundle:
    """Load the UCI-style handwritten digits bundled with scikit-learn.

    Fully **offline**: the data ships inside scikit-learn.

    Parameters
    ----------
    train : bool, default True
        Kept for API compatibility with the torchvision loaders.  The
        sklearn digits dataset has no official train/test split, so the
        *same* 1797 samples are returned in both cases; use
        :func:`train_test_split_idx` to create a local split.
    flatten : bool, default True
        If ``True`` the ``X`` field holds the flattened ``(N, 64)`` matrix;
        otherwise ``X`` holds the ``(N, 8, 8)`` images.

    Returns
    -------
    DatasetBundle
    """
    from sklearn.datasets import load_digits

    data = load_digits()
    images_u8 = data.images.astype(np.uint8)  # (N, 8, 8) in 0..16
    y = data.target.astype(np.int64)

    X = images_u8.reshape(images_u8.shape[0], -1).astype(np.float32) / 16.0
    if not flatten:
        X = X.reshape(-1, 8, 8)

    return DatasetBundle(
        X=X,
        y=y,
        images=images_u8,
        num_classes=int(y.max()) + 1,
        input_dim=64,
        image_shape=(8, 8),
        name="digits",
        train=bool(train),
        extra={"source": "sklearn.datasets.load_digits", "scale": 16.0},
    )


def _load_torchvision(
    name: str,
    train: bool,
    root: str,
    download: bool,
    flatten: bool = True,
) -> DatasetBundle:
    """Shared helper for the torchvision based datasets."""
    try:
        import torchvision  # noqa: F401
        from torchvision import datasets
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"torchvision is required to load the {name!r} dataset: {exc}"
        ) from exc

    os.makedirs(root, exist_ok=True)
    cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
    try:
        ds = cls(root=root, train=bool(train), download=bool(download))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the {name!r} dataset from {root!r} "
            f"(download error: {exc}).\n"
            "Please check your internet connection, or use the fully offline "
            "dataset instead, e.g. get_dataset('digits')."
        ) from exc

    data = ds.data.numpy() if hasattr(ds.data, "numpy") else np.asarray(ds.data)
    targets = ds.targets.numpy() if hasattr(ds.targets, "numpy") else np.asarray(ds.targets)

    images_u8 = data.astype(np.uint8)  # (N, 28, 28) in 0..255
    y = targets.astype(np.int64)
    X = images_u8.reshape(images_u8.shape[0], -1).astype(np.float32) / 255.0
    if not flatten:
        X = X.reshape(-1, 28, 28)

    return DatasetBundle(
        X=X,
        y=y,
        images=images_u8,
        num_classes=10,
        input_dim=784,
        image_shape=(28, 28),
        name=name,
        train=bool(train),
        extra={"source": f"torchvision.datasets.{cls.__name__}", "root": root},
    )


def load_mnist_dataset(
    train: bool = True,
    root: str = DEFAULT_DATA_ROOT,
    download: bool = True,
    flatten: bool = True,
) -> DatasetBundle:
    """Load MNIST through torchvision (28x28, normalised by 255).

    Parameters
    ----------
    train : bool, default True
        ``True`` -> the 60 000 training images, ``False`` -> the 10 000
        test images.
    root : str, default ``"./data"``
        Cache folder (the assignment environment already ships a populated
        ``data/`` directory).
    download : bool, default True
        Download the dataset if it is not present in ``root``.
    flatten : bool, default True
        See :func:`load_digits_dataset`.

    Returns
    -------
    DatasetBundle
    """
    return _load_torchvision("mnist", train, root, download, flatten)


def load_fashion_mnist_dataset(
    train: bool = True,
    root: str = DEFAULT_DATA_ROOT,
    download: bool = True,
    flatten: bool = True,
) -> DatasetBundle:
    """Load Fashion-MNIST through torchvision (28x28, normalised by 255).

    See :func:`load_mnist_dataset` for the parameters.
    """
    return _load_torchvision("fashion", train, root, download, flatten)


# --------------------------------------------------------------------------- #
# Registry / unified entry point
# --------------------------------------------------------------------------- #
#: Registry mapping the dataset name to its loader function.
DATASETS: Dict[str, Callable[..., DatasetBundle]] = {
    "digits": load_digits_dataset,
    "mnist": load_mnist_dataset,
    "fashion": load_fashion_mnist_dataset,
}


def available_datasets() -> Tuple[str, ...]:
    """Return the names of the available datasets."""
    return tuple(DATASETS.keys())


def get_dataset(
    name: str,
    train: bool = True,
    flatten: bool = True,
    root: str = DEFAULT_DATA_ROOT,
    download: bool = True,
) -> DatasetBundle:
    """Unified dataset accessor.

    Parameters
    ----------
    name : str
        One of ``'digits'``, ``'mnist'``, ``'fashion'`` (see
        :func:`available_datasets`).
    train : bool, default True
        Select the train split (torchvision datasets only).
    flatten : bool, default True
        If ``True`` the returned ``X`` is flattened to ``(N, input_dim)``;
        if ``False`` ``X`` keeps the ``(N, H, W)`` shape.
    root : str, default ``"./data"``
        Download/cache root for the torchvision datasets.
    download : bool, default True
        Allow downloading missing torchvision datasets.

    Returns
    -------
    DatasetBundle
        Bundle exposing ``X``, ``y``, ``images``, ``num_classes``,
        ``input_dim``, ``image_shape`` and ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not a registered dataset.
    RuntimeError
        If a torchvision dataset cannot be downloaded.  The message points
        the user to the fully offline ``'digits'`` dataset.
    """
    key = str(name).lower().strip()
    if key in ("fashion_mnist", "fashion-mnist", "fmnist"):
        key = "fashion"
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available datasets: {available_datasets()}"
        )
    loader = DATASETS[key]
    if key == "digits":
        return loader(train=train, flatten=flatten)
    return loader(train=train, root=root, download=download, flatten=flatten)
