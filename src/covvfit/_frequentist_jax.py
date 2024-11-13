"""Frequentist fitting functions powered by JAX."""

import dataclasses
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy import optimize


def calculate_linear(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    shape = (1,) * ts.ndim + (-1,)
    m = midpoints.reshape(shape)
    g = growths.reshape(shape)

    return (ts[..., None] - m) * g
    # return (ts[..., None] ) * g + m


_Float = float | Float[Array, " "]


def calculate_logps(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    linears = calculate_linear(
        ts=ts,
        midpoints=midpoints,
        growths=growths,
    )
    return jax.nn.log_softmax(linears, axis=-1)


def calculate_proportions(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    linear = calculate_linear(
        ts=ts,
        midpoints=midpoints,
        growths=growths,
    )
    return jax.nn.softmax(linear, axis=-1)


def loss(
    y: Float[Array, "*batch variants"],
    logp: Float[Array, "*batch variants"],
    n: _Float,
) -> Float[Array, " *batch"]:
    # Note: we want loss (lower is better), rather than
    # total loglikelihood (higher is better),
    # so we add the negative sign.
    return -jnp.sum(n * y * logp, axis=-1)


class CityData(NamedTuple):
    ts: Float[Array, " timepoints"]
    ys: Float[Array, "timepoints variants"]
    n: _Float


_ThetaType = Float[Array, "(cities+1)*(variants-1)"]


def add_first_variant(vec: Float[Array, " variants-1"]) -> Float[Array, " variants"]:
    return jnp.concatenate([jnp.zeros_like(vec)[0:1], vec])


def construct_total_loss(
    cities: Sequence[CityData],
    average_loss: bool = False,
) -> Callable[[_ThetaType], _Float]:
    cities = tuple(cities)
    n_variants = cities[0].ys.shape[-1]
    for city in cities:
        assert (
            city.ys.shape[-1] == n_variants
        ), "All cities must have the same number of variants"

    if average_loss:  # trick for numerical stability (average loss doesnt blow up)
        n_points_total = 1.0 * sum(city.ts.shape[0] for city in cities)
    else:
        n_points_total = 1.0

    def total_loss(theta: _ThetaType) -> _Float:
        rel_growths = get_relative_growths(theta, n_variants=n_variants)
        rel_midpoints = get_relative_midpoints(theta, n_variants=n_variants)

        growths = add_first_variant(rel_growths)
        return (
            jnp.sum(
                jnp.asarray(
                    [
                        loss(
                            y=city.ys,
                            n=city.n,
                            logp=calculate_logps(
                                ts=city.ts,
                                midpoints=add_first_variant(midp),
                                growths=growths,
                            ),
                        ).sum()
                        for midp, city in zip(rel_midpoints, cities)
                    ]
                )
            )
            / n_points_total
        )

    return total_loss


def construct_theta(
    relative_growths: Float[Array, " variants-1"],
    relative_midpoints: Float[Array, "cities variants-1"],
) -> _ThetaType:
    flattened_midpoints = relative_midpoints.flatten()
    theta = jnp.concatenate([relative_growths, flattened_midpoints])
    return theta


def get_relative_growths(
    theta: _ThetaType,
    n_variants: int,
) -> Float[Array, " variants-1"]:
    return theta[: n_variants - 1]


def get_relative_midpoints(
    theta: _ThetaType,
    n_variants: int,
) -> Float[Array, "cities variants-1"]:
    n_cities = theta.shape[0] // (n_variants - 1) - 1
    return theta[n_variants - 1 :].reshape(n_cities, n_variants - 1)


class StandardErrorsMultipliers(NamedTuple):
    CI95: float = 1.96

    @staticmethod
    def convert(confidence: float) -> float:
        """Calculates the multiplier for a given confidence level.

        Example:
            StandardErrorsMultipliers.convert(0.95)  # 1.9599
        """
        return float(jax.scipy.stats.norm.ppf((1 + confidence) / 2))


def get_standard_errors(
    jacobian: Float[Array, "*output_shape n_inputs"],
    covariance: Float[Array, "n_inputs n_inputs"],
) -> Float[Array, " *output_shape"]:
    """Delta method to calculate standard errors of a function
    from `n_inputs` to `output_shape`.

    Args:
        jacobian: Jacobian of the function to be fitted, shape (output_shape, n_inputs)
        covariance: Covariance matrix of the inputs, shape (n_inputs, n_inputs)

    Returns:
        Standard errors of the fitted parameters, shape (output_shape,)

    Note:
        `output_shape` can be a vector, in which case the output is a vector
        of standard errors or a tensor of any other shape,
        in which case the output is a tensor of standard errors for each output
        coordinate.
    """
    return jnp.sqrt(jnp.einsum("...L,KL,...K -> ...", jacobian, covariance, jacobian))


def triangular_mask(n_variants, valid_value: float = 0, masked_value: float = jnp.nan):
    """Creates a triangular mask. Helpful for masking out redundant parameters
    in anti-symmetric matrices."""
    a = jnp.arange(n_variants)
    nan_mask = jnp.where(a[:, None] < a[None, :], valid_value, masked_value)
    return nan_mask


def get_relative_advantages(theta, n_variants: int):
    # Shape (n_variants-1,) describing relative advantages
    # over the 0th variant
    rel_growths = get_relative_growths(theta, n_variants=n_variants)

    growths = jnp.concatenate((jnp.zeros(1, dtype=rel_growths.dtype), rel_growths))
    diffs = growths[None, :] - growths[:, None]
    return diffs


def get_softmax_predictions(
    theta: _ThetaType, n_variants: int, city_index: int, ts: Float[Array, " timepoints"]
) -> Float[Array, "timepoints variants"]:
    rel_growths = get_relative_growths(theta, n_variants=n_variants)
    growths = add_first_variant(rel_growths)

    rel_midpoints = get_relative_midpoints(theta, n_variants=n_variants)
    midpoints = add_first_variant(rel_midpoints[city_index])

    y_linear = calculate_linear(
        ts=ts,
        midpoints=midpoints,
        growths=growths,
    )

    y_softmax = jax.nn.softmax(y_linear, axis=-1)
    return y_softmax


def get_logit_predictions(
    theta: _ThetaType,
    n_variants: int,
    city_index: int,
    ts: Float[Array, " timepoints"],
) -> Float[Array, "timepoints variants"]:
    return jax.scipy.special.logit(
        get_softmax_predictions(
            theta=theta,
            n_variants=n_variants,
            city_index=city_index,
            ts=ts,
        )
    )


@dataclasses.dataclass
class OptimizeMultiResult:
    x: np.ndarray
    fun: float
    best: optimize.OptimizeResult
    runs: list[optimize.OptimizeResult]


def construct_theta0(
    n_cities: int,
    n_variants: int,
) -> _ThetaType:
    return np.zeros((n_cities * (n_variants - 1) + n_variants - 1,), dtype=float)


def jax_multistart_minimize(
    loss_fn,
    theta0: np.ndarray,
    n_starts: int = 10,
    random_seed: int = 42,
    maxiter: int = 10_000,
) -> OptimizeMultiResult:
    # Create loss function and its gradient
    _loss_grad_fun = jax.jit(jax.value_and_grad(loss_fn))

    def loss_grad_fun(theta):
        loss, grad = _loss_grad_fun(theta)
        return np.asarray(loss), np.asarray(grad)

    solutions: list[optimize.OptimizeResult] = []
    rng = np.random.default_rng(random_seed)

    for i in range(1, n_starts + 1):
        starting_point = theta0 + (i / n_starts) * rng.normal(size=theta0.shape)
        sol = optimize.minimize(
            loss_grad_fun, jac=True, x0=starting_point, options={"maxiter": maxiter}
        )
        solutions.append(sol)

    # Find the optimal solution
    optimal_index = None
    optimal_value = np.inf
    for i, sol in enumerate(solutions):
        if sol.fun < optimal_value:
            optimal_index = i
            optimal_value = sol.fun

    return OptimizeMultiResult(
        best=solutions[optimal_index],
        x=solutions[optimal_index].x,
        fun=solutions[optimal_index].fun,
        runs=solutions,
    )
