"""Frequentist fitting functions powered by JAX."""

import dataclasses
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as distrib
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
        n: quasimultinomial number of trials for each city
        overdispersion: overdispersion factor for each city
    """

    n_cities: int
    n_variants: int
    ts: Float[Array, "cities timepoints"]
    ys: Float[Array, "cities timepoints variants"]
    mask: Float[Array, "cities timepoints"]
    n: Float[Array, " cities"]
    overdispersion: Float[Array, " cities"]


def _validate_and_pad(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns: Float[Array, " cities"] | list[float] | float = 1.0,
    overdispersion: Float[Array, " cities"] | list[float] | float = 1.0,
) -> _ProblemData:
    """Validation function, parsing the input provided in
    the format convenient for the user to the internal
    representation compatible with JAX."""
    # Get the number of cities
    n_cities = len(ys)
    if len(ts) != n_cities:
        raise ValueError(f"Number of cities not consistent: {len(ys)} != {len(ts)}.")

    # Create arrays representing `n` and `overdispersion`
    if hasattr(ns, "__len__"):
        if len(ns) != n_cities:
            raise ValueError(
                f"Provided `ns` has length {len(ns)} rather than {n_cities}."
            )
    if hasattr(overdispersion, "__len__"):
        if len(overdispersion) != n_cities:
            raise ValueError(
                f"Provided `overdispersion` has length {len(overdispersion)} rather than {n_cities}."
            )

    out_n = jnp.asarray(ns) * jnp.ones(n_cities, dtype=float)
    out_overdispersion = jnp.asarray(overdispersion) * jnp.ones_like(out_n)

    # Get the number of variants
    n_variants = ys[0].shape[-1]
    for i, y in enumerate(ys):
        if y.ndim != 2:
            raise ValueError(f"City {i} has {y.ndim} dimension, rather than 2.")
        if y.shape[-1] != n_variants:
            raise ValueError(
                f"City {i} has {y.shape[-1]} variants rather than {n_variants}."
            )

    # Ensure that the number of timepoints is consistent
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

        max_timepoints = t.shape[0]

    # Now create the arrays representing the data
    out_ts = jnp.zeros((n_cities, max_timepoints))  # Pad with zeros
    out_mask = jnp.zeros((n_cities, max_timepoints))  # Pad with zeros
    out_ys = jnp.full(
        shape=(n_cities, max_timepoints, n_variants), fill_value=1.0 / n_variants
    )  # Pad with constant vectors

    for i, (t, y) in enumerate(zip(ts, ys)):
        n_timepoints = t.shape[0]

        out_ts = out_ts.at[i, :n_timepoints].set(t)
        out_ys = out_ys.at[i, :n_timepoints, :].set(y)
        out_mask = out_mask.at[i, :n_timepoints].set(1)

    return _ProblemData(
        n_cities=n_cities,
        n_variants=n_variants,
        ts=out_ts,
        ys=out_ys,
        mask=out_mask,
        n=out_n,
        overdispersion=out_overdispersion,
    )


def construct_model(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns: Float[Array, " cities"] | list[float] | float = 1.0,
    overdispersion: Float[Array, " cities"] | list[float] | float = 1.0,
    sigma_growth: float = 10.0,
    sigma_offset: float = 1000.0,
) -> Callable:
    """Builds a NumPyro model sampling from the quasiposterior.

    Args:
        ys: list of variant proportions for each city.
            The ith entry should be an array
            of shape (n_timepoints[i], n_variants)
        ts: list of timepoints. The ith entry should be an array
            of shape (n_timepoints[i],)
            Note: `ts` should be appropriately normalized
        ns: controls the overdispersion of each city
    """
    data = _validate_and_pad(
        ys=ys,
        ts=ts,
        ns=ns,
        overdispersion=overdispersion,
    )

    def model():
        # Sample growth differences. Note that we sample from the N(0, 1)
        # distribution and then resample, for numerical stability
        _scaled_rel_growths = numpyro.sample(
            "_scaled_relative_growths",
            distrib.Normal().expand((data.n_variants - 1,)),
        )
        numpyro.deterministic(
            "relative_growths",
            sigma_growth * _scaled_rel_growths,
        )

        # Sample offsets. We use scaling the same scaling trick as above
        _scaled_rel_offsets = numpyro.sample(
            "_scaled_offsets",
            distrib.Normal().expand((data.n_cities, data.n_variants - 1)),
        )
        numpyro.deterministic(
            "offsets",
            _scaled_rel_growths * sigma_offset,
        )

        # Construct the loglikelihood
        # TODO(Pawel): Add the implementation
        raise NotImplementedError()

    return model


def construct_total_loss_new(
    ys: list[jax.Array],
    ts: list[jax.Array],
    ns: list[float] | float = 1.0,
    overdispersion: list[float] | float = 1.0,
    average_loss: bool = True,
) -> Callable[[_ThetaType], _Float]:
    data = _validate_and_pad(
        ys=ys,
        ts=ts,
        ns=ns,
        overdispersion=overdispersion,
    )
    assert data is not None
    # TODO(Pawel): Finish this implementation.
    raise NotImplementedError()
