"""Utility helpers: reproducibility, device selection, logging, dumping.

All randomness in this project is controlled by :func:`set_seed`, whose
default value is ``5`` -- the seed reported in the POLCA Net paper
(arXiv:2410.07289, Table 1 / experimental setup).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_SEED",
    "set_seed",
    "get_device",
    "ensure_dir",
    "save_json",
    "save_csv",
    "AverageMeter",
    "get_logger",
]

#: Seed used throughout the assignment (paper's default).
DEFAULT_SEED: int = 5

_LOGGER_NAME = "polca"


# --------------------------------------------------------------------------- #
# Reproducibility & hardware
# --------------------------------------------------------------------------- #
def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed every random source used by the project.

    Seeds Python's :mod:`random`, NumPy's legacy global RNG and PyTorch
    (CPU and all CUDA devices).  When ``deterministic`` is ``True`` cuDNN is
    put into deterministic mode, which makes the training curves reproducible
    at the cost of a small speed penalty.

    Parameters
    ----------
    seed : int, default 5
        The seed value.
    deterministic : bool, default True
        Whether to force deterministic cuDNN kernels.

    Returns
    -------
    int
        The seed that was applied (convenient for logging).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional for pure-numpy usage of this module
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def get_device(prefer_cuda: bool = True) -> str:
    """Return the device string to use for training.

    Parameters
    ----------
    prefer_cuda : bool, default True
        If ``True`` and a CUDA device is available, ``"cuda"`` is returned,
        otherwise ``"cpu"``.

    Returns
    -------
    str
        Either ``"cuda"`` or ``"cpu"``.
    """
    if not prefer_cuda:
        return "cpu"
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory (and its parents) if it does not exist yet.

    Parameters
    ----------
    path : str or os.PathLike
        Directory to create.

    Returns
    -------
    pathlib.Path
        The (now existing) directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(obj: Any, path: str | os.PathLike[str]) -> Path:
    """Serialise ``obj`` to JSON.

    Numpy scalars/arrays are converted to plain Python objects so that the
    result is always JSON-serialisable.

    Parameters
    ----------
    obj : Any
        Object to serialise.
    path : str or os.PathLike
        Destination file. Parent directories are created automatically.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    return destination


def _json_default(value: Any) -> Any:
    """Fallback encoder for NumPy / Path objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def save_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | os.PathLike[str],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write a list of dictionaries to a CSV file.

    Parameters
    ----------
    rows : sequence of mapping
        The records to write.
    path : str or os.PathLike
        Destination file. Parent directories are created automatically.
    fieldnames : sequence of str, optional
        Column order.  When omitted the union of the keys of ``rows`` is
        used (order of first appearance).

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    destination = Path(path)
    ensure_dir(destination.parent)

    rows = list(rows)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in names:
                    names.append(key)
        fieldnames = names or []

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return destination


# --------------------------------------------------------------------------- #
# Metrics / logging
# --------------------------------------------------------------------------- #
class AverageMeter:
    """Running average accumulator.

    Examples
    --------
    >>> meter = AverageMeter()
    >>> meter.update(1.0)
    >>> meter.update(3.0)
    >>> round(meter.avg, 3)
    2.0
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.total: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` (observed ``n`` times) to the running statistics."""
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        """The current average (``0.0`` when nothing was accumulated)."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def sum(self) -> float:
        """The sum of all accumulated values."""
        return self.total

    def __float__(self) -> float:
        return self.avg

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AverageMeter(avg={self.avg:.6f}, count={self.count})"


def get_logger(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a stream handler.

    Repeated calls with the same ``name`` reuse the existing logger instead
    of adding duplicate handlers.

    Parameters
    ----------
    name : str, default ``"polca"``
        Logger name.
    level : int, default ``logging.INFO``
        Logging level.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(getattr(h, "_polca_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        handler._polca_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def iter_batches(data: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    """Split an iterable into consecutive lists of at most ``batch_size``.

    Parameters
    ----------
    data : iterable
        The data to chunk.
    batch_size : int
        Maximum chunk size.

    Yields
    ------
    list
        Each chunk of the input iterable.
    """
    batch: list[Any] = []
    for item in data:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
