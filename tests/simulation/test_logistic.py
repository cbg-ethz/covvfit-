import covvfit.simulation._logistic as ls
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from covvfit._random import JAXRNG


@pytest.mark.parametrize("t_shape", [(2, 5), (10,), (3, 5, 2)])
@pytest.mark.parametrize("n_variants", [2, 3, 5])
def test_generate_logistic(t_shape, n_variants):
    rng = JAXRNG(seed=42)

    growth_rates = jax.random.normal(rng.key, shape=(n_variants,))
    midpoints = jax.random.normal(rng.key, shape=growth_rates.shape)

    ts = jax.random.normal(rng.key, shape=t_shape)

    ys = ls.generate_logistic(ts=ts, midpoints=midpoints, growths=growth_rates)

    assert ys.shape == tuple(list(t_shape) + [n_variants])

    npt.assert_allclose(
        ys.sum(axis=-1),
        jnp.ones(dtype=float, shape=ts.shape),
        rtol=1e-5,
    )
