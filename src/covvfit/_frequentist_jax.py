"""Frequentist fitting functions powered by JAX."""
from typing import NamedTuple

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

    return ts[..., None] * g + m


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
    n: float | Float,
) -> Float[Array, " *batch"]:
    # TODO(Pawel): How to include n here?
    return jnp.sum(n * y * logp, axis=-1)


class CityData(NamedTuple):
    ts: Float[Array, " timepoints"]
    ys: Float[Array, "timepoints variants"]
    n: float | Float


_ThetaType = Float[Array, "(cities+1)*(variants-1)"]


def total_loss(
    theta: _ThetaType,
    data: tuple[CityData, ...],
) -> Float[Array, " "]:
    



def construct_theta(
    growths: Float[Array, " variants-1"],
    midpoints: Float[ArithmeticError, "cities variants-1"],
) -> _ThetaType:
    pass


def get_growths(
    theta: _ThetaType
) -> Float[Array, " variants-1"]:
    pass


def get_midpoints(
    theta: _ThetaType,
) -> Float[Array, "cities variants-1"]:
    pass

