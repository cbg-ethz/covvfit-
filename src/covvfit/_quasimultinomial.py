"""Frequentist fitting functions powered by JAX."""

import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as distrib
from jaxtyping import Array, Bool, Float
from scipy import optimize

from covvfit._padding import create_padded_array


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


_ThetaType = Float[Array, "(cities+1)*(variants-1)"]


def _add_first_variant(vec: Float[Array, " variants-1"]) -> Float[Array, " variants"]:
    """Prepends 0 to the beginning of the vector."""
    return jnp.concatenate((jnp.zeros(1, dtype=vec.dtype), vec))


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
) -> Float[Array, "n_params n_params"]:
    """Calculates the covariance matrix of the parameters.

    Args:
        loss_fn: The loss function for which the covariance matrix is calculated.
        theta: The optimized parameters at which to evaluate the Hessian.

    Returns:
        The covariance matrix, which is the inverse of the Hessian matrix.


    Note:
        `loss_fn` should *not* be averaged over the data points: otherwise,
        the covariance matrix won't shrink even when a very big data
        set is used
    """
    hessian_matrix = jax.hessian(loss_fn)(theta)
    covariance_matrix = jnp.linalg.inv(hessian_matrix)

    return covariance_matrix


def get_standard_errors(
    covariance: Float[Array, "n_inputs n_inputs"],
    jacobian: Float[Array, "*output_shape n_inputs"] | None = None,
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
    times: list[Float[Array, " timepoints"]],
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
    ts_lst_scaled: list[Float[Array, " timepoints"]],
    covariance_scaled: Float[Array, "n_params n_params"],
    confidence_level: float = 0.95,
) -> list[tuple]:
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

    growths = _add_first_variant(rel_growths)
    diffs = growths[None, :] - growths[:, None]
    return diffs


def get_softmax_predictions(
    theta: _ThetaType, n_variants: int, city_index: int, ts: Float[Array, " timepoints"]
) -> Float[Array, "timepoints variants"]:
    rel_growths = get_relative_growths(theta, n_variants=n_variants)
    growths = _add_first_variant(rel_growths)

    rel_midpoints = get_relative_midpoints(theta, n_variants=n_variants)
    midpoints = _add_first_variant(rel_midpoints[city_index])

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


class _ProblemData(NamedTuple):
    """Internal representation of the data used
    to efficiently construct the quasilikelihood.

    Attrs:
        ts: array of shape (cities, timepoints)
            which is padded with 0 for days where
            there is no measurement for a particular city
        ys: array of shape (cities, timepoints, variants)
            which is padded with the vector (1/variants, ..., 1/variants)
            for timepoints where there is no measurement for a particular city
        mask: array of shape (cities, timepoints) with 0 when there is
            no measurement for a particular city and 1 otherwise
        n_quasimul: quasimultinomial number of trials for each city and timepoint
        overdispersion: overdispersion factor for each city and timepoint
    """

    n_cities: int
    n_variants: int
    ts: Float[Array, "cities timepoints"]
    ys: Float[Array, "cities timepoints variants"]
    mask: Bool[Array, "cities timepoints"]
    n_quasimul: Float[Array, "cities timepoints"]
    overdispersion: Float[Array, "cities timepoints"]


_OverDispersionType = (
    float | list[float] | list[jax.Array] | list[list[float]] | Float[Array, " cities"]
)


def _validate_and_pad(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns_quasimul: _OverDispersionType,
    overdispersion: _OverDispersionType,
) -> _ProblemData:
    """Validation function, parsing the input provided in
    the format convenient for the user to the internal
    representation compatible with JAX."""
    # Get the number of cities
    n_cities = len(ys)
    if len(ts) != n_cities:
        raise ValueError(f"Number of cities not consistent: {len(ys)} != {len(ts)}.")
    if n_cities < 1:
        raise ValueError("There has to be at least one city.")

    # Get the number of variants
    n_variants = ys[0].shape[-1]
    for i, y in enumerate(ys):
        if y.ndim != 2:
            raise ValueError(f"City {i} has {y.ndim} dimension, rather than 2.")
        if y.shape[-1] != n_variants:
            raise ValueError(
                f"City {i} has {y.shape[-1]} variants rather than {n_variants}."
            )

    # Ensure that the number of timepoints is consistent for t and y
    max_timepoints = 0
    for i, (t, y) in enumerate(zip(ts, ys)):
        if t.ndim != 1:
            raise ValueError(
                f"City {i} has time axis with dimension {t.ndim}, rather than 1."
            )
        if t.shape[0] != y.shape[0]:
            raise ValueError(
                f"City {i} has timepoints mismatch: {t.shape[0]} != {y.shape[0]}."
            )

        max_timepoints = max(max_timepoints, t.shape[0])

    _lengths = [t.shape[0] for t in ts]
    out_n = create_padded_array(
        values=ns_quasimul,
        lengths=_lengths,
        padding_length=max_timepoints,
        padding_value=0.0,
    )
    out_overdispersion = create_padded_array(
        values=overdispersion,
        lengths=_lengths,
        padding_length=max_timepoints,
        padding_value=1.0,  # Use 1.0 as we divide by it and want to avoid NaNs
    )

    # Now create the arrays representing the data
    out_ts = create_padded_array(
        values=ts,
        lengths=_lengths,
        padding_length=max_timepoints,
        padding_value=0.0,
    )
    out_mask = create_padded_array(
        values=1,
        lengths=_lengths,
        padding_length=max_timepoints,
        padding_value=0,
        _out_dtype=bool,
    )

    # Create the array with variant proportions, padded with constant vectors
    out_ys = jnp.full(
        shape=(n_cities, max_timepoints, n_variants), fill_value=1.0 / n_variants
    )

    for i, y in enumerate(ys):
        n_timepoints = y.shape[0]
        out_ys = out_ys.at[i, :n_timepoints, :].set(y)

    return _ProblemData(
        n_cities=n_cities,
        n_variants=n_variants,
        ts=out_ts,
        ys=out_ys,
        mask=out_mask,
        n_quasimul=out_n,
        overdispersion=out_overdispersion,
    )


def _quasiloglikelihood_single_city(
    relative_growths: Float[Array, " variants-1"],
    relative_offsets: Float[Array, " variants-1"],
    ts: Float[Array, " timepoints"],
    ys: Float[Array, "timepoints variants"],
    mask: Float[Array, " timepoints"],
    n_quasimul: Float[Array, " timepoints"],
    overdispersion: Float[Array, " timepoints"],
) -> float:
    logps = calculate_logps(
        ts=ts,
        midpoints=_add_first_variant(relative_offsets),
        growths=_add_first_variant(relative_growths),
    )
    # Ensure compatible shapes:
    mask = jnp.asarray(mask, dtype=float)[:, None]
    weight = (n_quasimul / overdispersion)[:, None]

    return jnp.sum(mask * weight * ys * logps)


_RelativeGrowthsAndOffsetsFunction = Callable[
    [Float[Array, " variants-1"], Float[Array, " variants-1"]], _Float
]


def _generate_quasiloglikelihood_function(
    data: _ProblemData,
) -> _RelativeGrowthsAndOffsetsFunction:
    """Creates the quasilikelihood function with signature:

    def quasiloglikelihood(
        relative_growths: array of shape (variants-1,)
        relative_offsets: array of shape (cities, variants-1)
    ) -> float
    """

    def quasiloglikelihood(
        relative_growths: Float[Array, " variants-1"],
        relative_offsets: Float[Array, "cities variants-1"],
    ) -> _Float:
        # Broadcast the array, to use the same relative growths
        # for each city
        _new_shape = (data.n_cities, relative_growths.shape[-1])
        tiled_growths = jnp.broadcast_to(relative_growths, _new_shape)

        logps = jax.vmap(_quasiloglikelihood_single_city)(
            relative_growths=tiled_growths,
            relative_offsets=relative_offsets,
            ts=data.ts,
            ys=data.ys,
            mask=data.mask,
            n_quasimul=data.n_quasimul,
            overdispersion=data.overdispersion,
        )
        return jnp.sum(logps)

    return quasiloglikelihood


def construct_model(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns: _OverDispersionType = 1.0,
    overdispersion: _OverDispersionType = 1.0,
    sigma_growth: float = 10.0,
    sigma_offset: float = 1000.0,
) -> Callable:
    """Builds a NumPyro model suitable for sampling from the quasiposterior.

    Args:
        ys: list of variant proportions array for each city.
            The ith entry should be an array
            of shape (n_timepoints[i], n_variants)
        ts: list of timepoint arrays. The ith entry should be an array
            of shape (n_timepoints[i],)
            Note: `ts` should be appropriately normalized
        ns: controls the quasimultinomial sample size of each city. It can be:
              - a single float (sample size is constant across all cities and timepoints)
              - a sequence of floats, describing one sample size for each city
              - a list of arrays, with the `i`th entry having length `n_timepoints[i]`
        overdispersion: controls the overdispersion factor as in the
            quasilikelihood approach. The shape restrictions are the same as in `ns`.
        sigma_growth: controls the standard deviation of the prior
            on the relative growths
        sigma_offset: controls the standard deviation of the prior
            on the relative offsets

    Note:
        The "loglikelihood" is effectively rescaled by `ns/overdispersion`
        factor. Hence, using both `ns` and `overdispersion` should generally
        be avoided.
    """
    data = _validate_and_pad(
        ys=ys,
        ts=ts,
        ns_quasimul=ns,
        overdispersion=overdispersion,
    )

    quasi_ll_fn = _generate_quasiloglikelihood_function(data)

    def model():
        # Sample growth differences. Note that we sample from the N(0, 1)
        # distribution and then resample, for numerical stability
        _scaled_rel_growths = numpyro.sample(
            "_scaled_relative_growths",
            distrib.Normal().expand((data.n_variants - 1,)),
        )
        rel_growths = numpyro.deterministic(
            "relative_growths",
            sigma_growth * _scaled_rel_growths,
        )

        # Sample offsets. We use scaling the same scaling trick as above
        _scaled_rel_offsets = numpyro.sample(
            "_scaled_relative_offsets",
            distrib.Normal().expand((data.n_cities, data.n_variants - 1)),
        )
        rel_offsets = numpyro.deterministic(
            "relative_offsets",
            _scaled_rel_offsets * sigma_offset,
        )

        numpyro.factor("quasiloglikelihood", quasi_ll_fn(rel_growths, rel_offsets))

    return model


def construct_total_loss(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns: _OverDispersionType = 1.0,
    overdispersion: _OverDispersionType = 1.0,
    accept_theta: bool = True,
    average_loss: bool = False,
) -> Callable[[_ThetaType], _Float] | _RelativeGrowthsAndOffsetsFunction:
    """Constructs the loss function, suitable e.g., for optimization.

    Args:
        ys: list of variant proportions for each city.
            The ith entry should be an array
            of shape (n_timepoints[i], n_variants)
        ts: list of timepoints. The ith entry should be an array
            of shape (n_timepoints[i],)
            Note: `ts` should be appropriately normalized
        ns: controls the quasimultinomial sample size of each city. It can be:
              - a single float (sample size is constant across all cities and timepoints)
              - a sequence of floats, describing one sample size for each city
              - a list of arrays, with the `i`th entry having length `n_timepoints[i]`
        overdispersion: controls the overdispersion factor as in the
            quasilikelihood approach. The shape restrictions are the same as in `ns`.
        accept_theta: whether the returned loss function should accept the
            `theta` vector (suitable for optimization)
            or should be parameterized by the relative growths
            and relative offsets, as in
            ```
            def loss(
                relative_growths: array of shape (variants-1,)
                relative_offsets: array of shape (cities, variants-1)
            ) -> float
            ```
        average_loss: whether the loss should be divided by the
            total number of points. By default it is false, as the loss
            is used to calculate confidence intervals. Setting it to true
            can improve the convergence of the optimization procedure

    Note:
        The "loglikelihood" is effectively rescaled by `ns/overdispersion`
        factor. Hence, using both `ns` and `overdispersion` should generally
        be avoided.
    """

    data: _ProblemData = _validate_and_pad(
        ys=ys,
        ts=ts,
        ns_quasimul=ns,
        overdispersion=overdispersion,
    )

    if average_loss:
        scaling = jnp.sum(data.mask, dtype=float)
    else:
        scaling = 1.0

    # Get the quasilikelihood function
    quasi_ll_fn = _generate_quasiloglikelihood_function(data)

    # Define the loss function parameterized
    # with relative growths and offsets
    def _loss_fn(relative_growths, relative_offsets):
        return -quasi_ll_fn(relative_growths, relative_offsets) / scaling

    # Define the loss function in terms of the theta variable
    def _loss_fn_theta(theta):
        rel_growths = get_relative_growths(theta, n_variants=data.n_variants)
        rel_offsets = get_relative_midpoints(theta, n_variants=data.n_variants)
        return _loss_fn(relative_growths=rel_growths, relative_offsets=rel_offsets)

    if accept_theta:
        return _loss_fn_theta
    else:
        return _loss_fn


def compute_alleged_pearson_residuals(
    ys_lst: list[Float[Array, "timepoints variants"]],
    y_fit_lst: list[Float[Array, "timepoints variants"]],
    ns_lst: _OverDispersionType,
) -> list[Float[Array, "timepoints variants"]]:
    [
        (ys_lst[i] - y_fit_lst[i]) ** 2 / (y_fit_lst[i] * (1 - y_fit_lst[i]))
        for i, _ in enumerate(ys_lst)
    ]


def compute_overdispersion(
    ys_lst: list[Float[Array, "timepoints variants"]],
    y_fit_lst: list[Float[Array, "timepoints variants"]],
    cities: list,
) -> tuple[
    list[Float[Array, "timepoints variants"]],
    Float[Array, " cities"],
    Float[Array, " "],
]:
    """
    Compute overdispersion from a quasimultinomial model.

    Args:
        ys_lst: A list of observed variant proportions for each city,
                each with shape (timepoints, variants).
        y_fit_lst: A list of fitted variant proportions for each city,
                   each with shape (timepoints, variants).
        cities: A list of city identifiers (used only for iteration).

    Returns:
        A tuple containing:
        - A list of Pearson residuals for each city.
        - An array of overdispersion values for each city.
        - A single value of fixed overdispersion across all cities.
    """
    # Compute Pearson residuals for each city
    pearson_r_lst = [
        (y_fit_lst[k] - ys_lst[k]) ** 2 / (y_fit_lst[k] * (1 - y_fit_lst[k]))
        for k, city in enumerate(cities)
    ]

    # Compute overdispersion for each city
    overdisp_list2 = [r.sum() / (r.shape[0] * (r.shape[1] - 1)) for r in pearson_r_lst]

    # Compute fixed overdispersion across all cities
    total_residuals = jnp.concatenate(pearson_r_lst, axis=1).sum()
    total_degrees_of_freedom = sum(r.shape[0] * (r.shape[1] - 1) for r in pearson_r_lst)
    overdisp_fixed = total_residuals / total_degrees_of_freedom

    return pearson_r_lst, overdisp_list2, overdisp_fixed
