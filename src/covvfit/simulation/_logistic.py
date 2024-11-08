"""Logistic growth simulations."""

import jax
from jaxtyping import Array, Float


def generate_logistic(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    shape = (1,) * ts.ndim + (-1,)
    m = midpoints.reshape(shape)
    g = growths.reshape(shape)

    linear = (ts[..., None] - m) * g

    return jax.nn.softmax(linear, axis=-1)
