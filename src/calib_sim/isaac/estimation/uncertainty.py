"""Uncertainty helpers for the Isaac estimation stack."""

from __future__ import annotations

import math

import numpy as np


def is_positive_semidefinite(matrix: np.ndarray, *, tolerance: float = 1e-9) -> bool:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(np.isfinite(eigenvalues)) and np.min(eigenvalues) >= -float(tolerance))


def sanitize_covariance(matrix: np.ndarray, *, floor: float = 1e-9) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    eigvals = np.maximum(eigvals, float(floor))
    return (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float64)


def radius_95_from_covariance(matrix: np.ndarray) -> float:
    sanitized = sanitize_covariance(np.asarray(matrix, dtype=np.float64))
    trace_value = float(np.trace(sanitized))
    return float(2.44774683068 * math.sqrt(max(trace_value, 0.0) / max(sanitized.shape[0], 1)))
