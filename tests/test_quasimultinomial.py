import covvfit._quasimultinomial as qm
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest


@pytest.mark.parametrize("n_cities", [1, 5, 12])
@pytest.mark.parametrize("n_variants", [2, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parameter_conversions_1(seed: int, n_cities: int, n_variants: int) -> None:
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    growth_rel = jax.random.uniform(key1, shape=(n_variants - 1,))
    midpoint_rel = jax.random.uniform(key2, shape=(n_cities, n_variants - 1))

    theta = qm.construct_theta(
        relative_growths=growth_rel,
        relative_midpoints=midpoint_rel,
    )

    npt.assert_allclose(
        qm.get_relative_growths(theta, n_variants=n_variants),
        growth_rel,
    )
    npt.assert_allclose(
        qm.get_relative_midpoints(theta, n_variants=n_variants),
        midpoint_rel,
    )


@pytest.mark.parametrize("n_cities", [1, 5, 12])
@pytest.mark.parametrize("n_variants", [2, 3, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parameter_conversions_2(seed: int, n_cities: int, n_variants: int) -> None:
    theta = jax.random.uniform(
        jax.random.PRNGKey(seed), shape=(n_cities * (n_variants - 1) + n_variants - 1,)
    )

    growth_rel = qm.get_relative_growths(theta, n_variants=n_variants)
    midpoint_rel = qm.get_relative_midpoints(theta, n_variants=n_variants)

    npt.assert_allclose(
        qm.construct_theta(
            relative_growths=growth_rel,
            relative_midpoints=midpoint_rel,
        ),
        theta,
    )


def test_softmax_predictions(
    n_cities: int = 2, n_variants: int = 3, n_timepoints: int = 50
) -> None:
    theta0 = qm.construct_theta0(n_cities=n_cities, n_variants=n_variants)
    theta = jax.random.normal(jax.random.PRNGKey(42), shape=theta0.shape)

    ts = jnp.linspace(0, 1, n_timepoints)

    for city in range(n_cities):
        predictions = qm.get_softmax_predictions(
            theta,
            n_variants=n_variants,
            city_index=city,
            ts=ts,
        )

        assert predictions.shape == (n_timepoints, n_variants)

        npt.assert_allclose(
            predictions.sum(axis=-1),
            jnp.ones(n_timepoints),
            atol=1e-6,
        )


def test_get_relative_advantages(n_cities: int = 1, n_variants: int = 5) -> None:
    theta0 = qm.construct_theta0(n_cities=n_cities, n_variants=n_variants)
    # The variants are ordered by increasing fitness
    relative = jnp.arange(1, n_variants)
    theta = qm.construct_theta(
        relative_growths=relative,
        relative_midpoints=qm.get_relative_midpoints(theta0, n_variants=n_variants),
    )

    A = qm.get_relative_advantages(theta, n_variants=n_variants)
    for v2 in range(n_variants):
        for v1 in range(n_variants):
            assert pytest.approx(A[v1, v2]) == v2 - v1

    for v1 in range(n_variants):
        for v2 in range(n_variants):
            for v3 in range(n_variants):
                assert pytest.approx(A[v1, v3]) == A[v1, v2] + A[v2, v3]
