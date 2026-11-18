"""Rigidly combine inertias expressed about their own COM in a shared link frame."""

import numpy as np


def combine(mass_a, center_a, inertia_a, mass_b, center_b, inertia_b):
    masses = np.array([mass_a, mass_b], dtype=float)
    centers = np.array([center_a, center_b], dtype=float)
    tensors = np.array([inertia_a, inertia_b], dtype=float)
    if not (
        np.isfinite(masses).all() and np.isfinite(centers).all() and np.isfinite(tensors).all()
    ):
        raise ValueError("Nonfinite inertial parameters")
    if np.any(masses <= 0):
        raise ValueError("Both component masses must be positive")
    for tensor in tensors:
        if not np.allclose(tensor, tensor.T) or np.linalg.eigvalsh(tensor).min() <= 0:
            raise ValueError("Inertia must be symmetric positive definite")
    mass = float(masses.sum())
    center = (masses[:, None] * centers).sum(0) / mass
    inertia = tensors.sum(0)
    for m, c in zip(masses, centers):
        delta = c - center
        inertia += m * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return mass, center, inertia
