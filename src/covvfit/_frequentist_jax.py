"""Frequentist fitting functions powered by JAX."""

import dataclasses
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

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

    # return (ts[..., None] - m) * g
    return (ts[..., None]) * g + m



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


def get_covariance(
    loss_fn: Callable[[_ThetaType], _Float],
    theta: _ThetaType,
) -> Float[Array, "(n_params n_params)"]:
    """Calculates the covariance matrix of the parameters.

    Args:
        loss_fn: The loss function for which the covariance matrix is calculated.
        theta: The optimized parameters at which to evaluate the Hessian.

    Returns:
        The covariance matrix, which is the inverse of the Hessian matrix.
    """
    hessian_matrix = jax.hessian(loss_fn)(theta)
    covariance_matrix = jnp.linalg.inv(hessian_matrix)

    return covariance_matrix


def get_standard_errors(
    covariance: Float[Array, "n_inputs n_inputs"],
    jacobian: Optional[Float[Array, "*output_shape n_inputs"]] = None,
) -> Float[Array, " *output_shape"]:
    """Delta method to calculate standard errors of a function
    from `n_inputs` to `output_shape`.

    Args:
        jacobian: Jacobian of the function to be fitted, shape (output_shape, n_inputs).
                  If None, uses an identity matrix with shape `(n_inputs, n_inputs)`.
        covariance: Covariance matrix of the inputs, shape (n_inputs, n_inputs).

    Returns:
        Standard errors of the fitted parameters, shape (output_shape,).

    Note:
        `output_shape` can be a vector, in which case the output is a vector
        of standard errors, or a tensor of any other shape, in which case
        the output is a tensor of standard errors for each output coordinate.
    """
    # If jacobian is not provided, default to the identity matrix
    if jacobian is None:
        n_inputs = covariance.shape[0]
        jacobian = jnp.eye(n_inputs)

    return jnp.sqrt(jnp.einsum("...L,KL,...K -> ...", jacobian, covariance, jacobian))


def get_confidence_intervals(
    estimates: Float[Array, " *output_shape"],
    standard_errors: Float[Array, " *output_shape"],
    confidence_level: float = 0.95,
) -> tuple[Float[Array, " *output_shape"], Float[Array, " *output_shape"]]:
    """Calculates confidence intervals for parameter estimates.

    Args:
        estimates: Estimated parameters, shape (output_shape,).
        standard_errors: Standard errors of the estimates, shape (output_shape,).
        confidence_level: Confidence level for the intervals (default is 0.95).

    Returns:
        A tuple of two arrays (lower_bound, upper_bound), each with shape (output_shape,)
        representing the confidence interval for each estimate.

    Note:
        Assumes a normal distribution for the estimates.
    """
    # Calculate the multiplier based on the confidence level
    z_score = jax.scipy.stats.norm.ppf((1 + confidence_level) / 2)

    # Compute the lower and upper bounds of the confidence intervals
    lower_bound = estimates - z_score * standard_errors
    upper_bound = estimates + z_score * standard_errors

    return lower_bound, upper_bound


def fitted_values(
    times: List[Float[Array, " timepoints"]],
    theta: _ThetaType,
    cities: list,
    n_variants: int,
) -> list[Float[Array, "variants-1 timepoints"]]:
    """Generates the fitted values of a model based on softmax predictions.

    Args:
        times: A list of arrays, each containing timepoints for a city.
        theta: Parameter array for the model.
        cities: A list of city data objects (used only for iteration).
        n_variants: The number of variants.


    Returns:
        A list of fitted values for each city, each array having shape (variants, timepoints).
    """
    y_fit_lst = [
        get_softmax_predictions(
            theta=theta, n_variants=n_variants, city_index=i, ts=times[i]
        ).T[1:, :]
        for i, _ in enumerate(cities)
    ]

    return y_fit_lst


def create_logit_predictions_fn(
    n_variants: int, city_index: int, ts: Float[Array, " timepoints"]
) -> Callable[
    [Float[Array, " (cities+1)*(variants-1)"]], Float[Array, "timepoints variants"]
]:
    """Creates a version of get_logit_predictions with fixed arguments.

    Args:
        n_variants: Number of variants.
        city_index: Index of the city to consider.
        ts: Array of timepoints.

    Returns:
        A function that takes only theta as input and returns logit predictions.
    """

    def logit_predictions_with_fixed_args(
        theta: _ThetaType,
    ):
        return get_logit_predictions(
            theta=theta, n_variants=n_variants, city_index=city_index, ts=ts
        )[:, 1:]

    return logit_predictions_with_fixed_args


def get_confidence_bands_logit(
    solution_x: Float[Array, " (cities+1)*(variants-1)"],
    variants_count: int,
    ts_lst_scaled: List[Float[Array, " timepoints"]],
    covariance_scaled: Float[Array, "n_params n_params"],
    confidence_level: float = 0.95,
) -> List[tuple]:
    """Computes confidence intervals for logit predictions using the Delta method,
    back-transforms them to the linear scale

    Args:
        solution_x: Optimized parameters for the model.
        variants_count: Number of variants.
        ts_lst_scaled: List of timepoint arrays for each city.
        covariance_scaled: Covariance matrix for the parameters.
        confidence_level: Desired confidence level for intervals (default is 0.95).

    Returns:
        A list of dictionaries for each city, each with "lower" and "upper" bounds
        for the confidence intervals on the linear scale.
    """

    y_fit_lst_logit = [
        get_logit_predictions(solution_x, variants_count, i, ts).T[1:, :]
        for i, ts in enumerate(ts_lst_scaled)
    ]

    y_fit_lst_logit_se = []
    for i, ts in enumerate(ts_lst_scaled):
        # Compute the Jacobian of the transformation and project standard errors
        jacobian = jax.jacobian(create_logit_predictions_fn(variants_count, i, ts))(
            solution_x
        )
        standard_errors = get_standard_errors(
            jacobian=jacobian, covariance=covariance_scaled
        ).T
        y_fit_lst_logit_se.append(standard_errors)

    # Compute confidence intervals on the logit scale
    y_fit_lst_logit_confint = [
        get_confidence_intervals(fitted, se, confidence_level=confidence_level)
        for fitted, se in zip(y_fit_lst_logit, y_fit_lst_logit_se)
    ]

    # Project confidence intervals to the linear scale
    y_fit_lst_logit_confint_expit = [
        (jax.scipy.special.expit(confint[0]), jax.scipy.special.expit(confint[1]))
        for confint in y_fit_lst_logit_confint
    ]

    return y_fit_lst_logit_confint_expit


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
