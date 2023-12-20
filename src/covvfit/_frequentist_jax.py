"""Frequentist fitting functions powered by JAX."""
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp

from jaxtyping import Float, Array


def calculate_linear(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    shape = (1,) * ts.ndim + (-1,)
    m = midpoints.reshape(shape)
    g = growths.reshape(shape)

    return (ts[..., None] - m) * g


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


def _add_first_variant(vec: Float[Array, " variants-1"]) -> Float[Array, " variants"]:
    return jnp.concatenate([jnp.zeros_like(vec)[0:1], vec])


def construct_total_loss(
    cities: Sequence[CityData],
) -> Callable[[_ThetaType], _Float]:
    cities = tuple(cities)
    n_variants = cities[0].ys.shape[-1]
    for city in cities:
        assert (
            city.ys.shape[-1] == n_variants
        ), "All cities must have the same number of variants"

    def total_loss(theta: _ThetaType) -> _Float:
        rel_growths = get_relative_growths(theta, n_variants=n_variants)
        rel_midpoints = get_relative_midpoints(theta, n_variants=n_variants)

        growths = _add_first_variant(rel_growths)
        return jnp.sum(
            jnp.asarray(
                [
                    loss(
                        y=city.ys,
                        n=city.n,
                        logp=calculate_logps(
                            ts=city.ts,
                            midpoints=_add_first_variant(midp),
                            growths=growths,
                        ),
                    ).sum()
                    for midp, city in zip(rel_midpoints, cities)
                ]
            )
        )

    return total_loss


def construct_theta(
    relative_growths: Float[Array, " variants-1"],
    relative_midpoints: Float[ArithmeticError, "cities variants-1"],
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
