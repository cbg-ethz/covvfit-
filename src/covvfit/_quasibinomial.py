"""The quasibinomial models."""
import jax.numpy as jnp

_EPSILON: float = 1e-9


def _clip(p: float, _epsilon: float) -> float:
    return jnp.clip(p, _epsilon, 1.0 - _epsilon)


def variance1(mu: float) -> float:
    return mu * (1.0 - mu)


def quasiloglikelihood1(p: float, y: float, _epsilon: float = _EPSILON) -> float:
    p = _clip(p, _epsilon)
    logp = jnp.log(p)
    log1p = jnp.log1p(-p)
    return y * logp + (1.0 - y) * log1p


def variance2(mu: float) -> float:
    return jnp.square(mu * (1.0 - mu))


def quasiloglikelihood2(p: float, y: float, _epsilon: float = _EPSILON) -> float:
    p = _clip(p, _epsilon)
    logp = jnp.log(p)
    log1p = jnp.log1p(-p)
    return (2 * y - 1) * (logp - log1p) + (2 * y - 1) / (1 - p) - y / (p * (1 - p))


def varianceK(mu: float, k: float) -> float:
    return jnp.power(mu * (1.0 - mu), k)


def _quasiscore(t: float, y: float, k: float):
    return (y - t) / varianceK(t, k)


def quasiloglikelihoodK(
    p: float,
    y: float,
    k: float,
    _epsilon: float = _EPSILON,
    _n_points: int = 100,
    _start=0.5,
) -> float:
    """Quasi-log-likelihood for the model with variance given by

        V(p) = (p(1-p))^k

    Args:
        p: mean prediction
        y: observed value
        k: exponent controlling mean-variance relation
        _epsilon: clipping applied to `p`, introduced for larger numerical stability.
            Use _epsilon = 0 for no clipping
        _n_points: number of points to calculate the quasi-log-likelihood integral numerically
        _start: reference point (lower limit) for the integration. We recommend `0.5` or `y` (if `y` is far from 0 and 1)

    Returns:
        numerical approximation to the quasi-log-likelihood `q(p; y)`.
    """
    p = _clip(p, _epsilon)

    t_values = jnp.linspace(_start, p, _n_points)

    # Evaluate f(t) for each t in the interval
    f_values = _quasiscore(t_values, y, k)

    # Use JAX's trapezoidal rule to approximate the integral
    integral_approx = jnp.trapezoid(f_values, t_values)
    return integral_approx
