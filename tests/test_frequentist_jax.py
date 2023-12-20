import pytest
import jax

import numpy.testing as npt

import covvfit._frequentist_jax as fj


@pytest.mark.parametrize("n_cities", [1, 5, 12])
@pytest.mark.parametrize("n_variants", [2, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parameter_conversions_1(seed: int, n_cities: int, n_variants: int) -> None:
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    growth_rel = jax.random.uniform(key1, shape=(n_variants - 1,))
    midpoint_rel = jax.random.uniform(key2, shape=(n_cities, n_variants - 1))

    theta = fj.construct_theta(
        relative_growths=growth_rel,
        relative_midpoints=midpoint_rel,
    )

    npt.assert_allclose(
        fj.get_relative_growths(theta, n_variants=n_variants),
        growth_rel,
    )
    npt.assert_allclose(
        fj.get_relative_midpoints(theta, n_variants=n_variants),
        midpoint_rel,
    )


@pytest.mark.parametrize("n_cities", [1, 5, 12])
@pytest.mark.parametrize("n_variants", [2, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parameter_conversions_2(seed: int, n_cities: int, n_variants: int) -> None:
    theta = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n_cities * (n_variants - 1) + n_variants - 1,)
    )

    growth_rel = fj.get_relative_growths(theta, n_variants=n_variants)
    midpoint_rel = fj.get_relative_midpoints(theta, n_variants=n_variants)

    npt.assert_allclose(
        fj.construct_theta(
            relative_growths=growth_rel,
            relative_midpoints=midpoint_rel,
        ),
        theta,
    )
