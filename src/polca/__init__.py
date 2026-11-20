"""POLCA Net: Principal Orthogonal Latent Components Analysis.

Reference
---------
H. Martin, F. Perozo, M. Lopez, "POLCA Net: Principal Orthogonal Latent
Components Analysis", arXiv:2410.07289, 2024.

The public API re-exports the most frequently used symbols so that other
modules can simply do::

    from src.polca import POLCANet, polca_loss, set_seed, get_device
"""

from .losses import (
    POLCALossOutput,
    center_of_mass_loss,
    classification_loss,
    latent_variances,
    orthogonality_loss,
    orthogonality_matrix,
    polca_loss,
    reconstruction_loss,
    similarity_matrix,
    variance_loss,
)
from .model import Autoencoder, POLCANet
from .utils import (
    AverageMeter,
    ensure_dir,
    get_device,
    get_logger,
    save_csv,
    save_json,
    set_seed,
)

__all__ = [
    # losses
    "POLCALossOutput",
    "polca_loss",
    "reconstruction_loss",
    "classification_loss",
    "orthogonality_loss",
    "orthogonality_matrix",
    "similarity_matrix",
    "center_of_mass_loss",
    "variance_loss",
    "latent_variances",
    # model
    "POLCANet",
    "Autoencoder",
    # utils
    "set_seed",
    "get_device",
    "ensure_dir",
    "save_json",
    "save_csv",
    "AverageMeter",
    "get_logger",
]
