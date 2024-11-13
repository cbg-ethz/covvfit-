"""Spline functions utilities."""

from typing import Optional

import numpy as np
from scipy.interpolate import BSpline


def create_spline_matrix(
    xs: np.ndarray,
    n_coefs: int = 5,
    degree: int = 3,
    knots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Creates basis spline functions evaluated at points provided.

    Args:
        xs: points to evaluate the spline functions on, shape (N,)
        n_coefs: number of coefficients
        degree: B-spline degree
        knots: knot positions, should be of shape `n_coefs + degree + 1`

    Returns:
        B-splines evaluated at the points provided. Shape (N, n_coefs)

    Note:
        The number of knots is given by `n_coefs + degree + 1`
    """
    n_knots = n_coefs + degree + 1

    if knots is None:
        knots = np.linspace(np.min(xs), np.max(xs), n_knots)
    else:
        knots = np.asarray(knots)

    assert isinstance(knots, np.ndarray)
    assert knots.shape[0] == n_knots

    def coeff(i: int) -> np.ndarray:
        """One-hot vector with 1 at position `i`"""
        return np.eye(n_coefs)[i]

    return np.vstack(
        [BSpline(t=knots, c=coeff(i), k=degree)(xs) for i in range(n_coefs)]
    ).T
