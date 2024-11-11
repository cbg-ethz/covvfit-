import jax
import jax.numpy as jnp
import numpy.testing as npt
from covvfit.simulation._sde import (
    simplex_complete,
    solve_stochastic_replicator_dynamics,
)


def simplex_complete_single(y: jnp.ndarray) -> jnp.ndarray:
    return jnp.append(y, 1 - y.sum())


def test_simplex_complete() -> None:
    y0 = jnp.array([0.1, 0.2, 0.3])

    npt.assert_allclose(simplex_complete(y0), simplex_complete_single(y0))

    y1 = jnp.linspace(0, 0.3, 10).reshape(5, 2)

    npt.assert_allclose(jax.vmap(simplex_complete_single)(y1), simplex_complete(y1))


def test_solve_replicator(dim: int = 3) -> None:
    y0 = jnp.linspace(0.1, 0.9, dim)
    y0 = y0 / y0.sum()

    t_span = jnp.linspace(0, 0.5, 5)
    fitness = jnp.linspace(0.0, 2.0, dim)
    noise = 0.05

    ys_solved, sol = solve_stochastic_replicator_dynamics(
        y0=y0,
        t_span=t_span,
        fitness=fitness,
        noise=noise,
        brownian_tol=0.05,
        solver_dt=0.05,
        key=42,
    )

    assert ys_solved.shape == (t_span.shape[0], dim)
    assert sol.ys.shape == (t_span.shape[0], dim - 1)  # pyright: ignore

    assert ys_solved.min() >= 0
    assert ys_solved.max() <= 1
