"""CISC3024 Pattern Recognition -- Assignment 3 (POLCA Net).

This package contains the AI-generated implementation of the POLCA Net
algorithm (Martin H., Perozo F., Lopez M., "POLCA Net: Principal Orthogonal
Latent Components Analysis", arXiv:2410.07289, 2024).

Sub-packages / modules
----------------------
``src.polca.losses``
    The five loss terms of the POLCA composite objective.
``src.polca.model``
    The :class:`POLCANet` autoencoder and the plain :class:`Autoencoder`
    baseline.
``src.polca.utils``
    Seeding, device selection, logging and result-dumping helpers.
"""

__all__ = ["polca"]
