import jax
import jax.random as jrandom
import jax.numpy as jnp
from diffrax import (
    diffeqsolve,
    ControlTerm,
    Euler,
    MultiTerm,
    ODETerm,
    SaveAt,
    Solution,
    VirtualBrownianTree,
)


def simplex_complete(y: jax.Array) -> jax.Array:
    """Completes the parametrization to the point on the simplex.

    Args:
        y: shape (..., dim-1)

    Returns:
        y_ext, shape (..., dim)
    """
    ones_minus_sum = 1 - y.sum(axis=-1)
    ones_minus_sum_expanded = ones_minus_sum[..., jnp.newaxis]
    return jnp.concatenate([y, ones_minus_sum_expanded], axis=-1)


def simplex_truncate(y: jax.Array) -> jax.Array:
    """Truncates the last entry.

    Args:
        y: shape (..., dim)

    Returns:
        y_trunc, shape (..., dim-1)
    """
    return y[..., :-1]


def _get_drift_term(fitness: jax.Array, sigmas: jax.Array):
    """Generates the drift term of the SDE:

    drift(t, y, args) -> jnp.ndarray

    Note that `t` and `args` are ignored. Moreover, `y` is assumed
    to be of shape (variants-1,) as the last entry being implicitly
    defined by the other entries due to summing up to 1 constraint.

    Args:
        fitness: fitness vector, shape (variants,)
        sigmas: noise vector, shape (variants,)
    """

    def drift(t, y, args):
        xs = simplex_complete(y)
        phi = jnp.sum(xs * fitness)

        square_term = jnp.sum(jnp.square(sigmas * xs))
        return y * (
            simplex_truncate(fitness)
            - phi
            - y * jnp.square(simplex_truncate(sigmas))
            + square_term
        )

    return drift


def _get_diffusion_term(sigmas: jnp.ndarray):
    """Generates the diffusion term of the SDE:

    diffusion(t, y, args) -> jnp.ndarray

    Note that `t` and `args` are ignored. Moreover, `y` is assumed
    to be of shape (variants-1,) as the last entry being implicitly
    defined by the other entries due to summing up to 1 constraint.

    The returned array is of shape (variants-1, variants)
    as we have independent noise for each variant, but we still
    write the SDEs only for the first `variants-1` entries of `y`.
    """

    def diffusion(t, y, args):
        k = y.shape[0] + 1

        term1 = (y * simplex_truncate(sigmas))[:, None] * jnp.eye(k)[:-1, :]
        term2 = jnp.outer(y, sigmas * simplex_complete(y))

        return term1 - term2

    return diffusion


def solve_stochastic_replicator_dynamics(
    y0: jax.Array,
    t_span: jax.Array,
    fitness: jax.Array,
    noise: jax.Array | float = 0.05,
    brownian_tol: float = 1e-3,
    solver_dt: float = 1e-2,
    key: jax.Array | int = 42,
    jit_terms: bool = False,
) -> tuple[jax.Array, Solution]:
    """Solves

    Args:
        y0: starting point with
            positive entries summing up to 1,  shape (variants,)
        t_span: time span, shape (steps,)
        fitness: fitness vector, shape (variants,)
        noise: noise level, float or array of shape (variants,)
        brownian_tol: tolerance for the Brownian tree, float
        solver_dt: default time step for the solver, float

    Returns:
        y, shape (steps, variants)
        sol, diffrax's Solution. Note that the `sol.ys` is of shape (steps, variants-1)
          as the last entry is implicitly defined by the summing up to 1 constraint.
    """
    # Infer the number of variants and check the dimensions
    dim = y0.shape[0]
    assert fitness.shape == (dim,), "Fitness vector has wrong shape"
    noise = (
        jnp.ones(dim) * noise
    )  # This should work independently on whether `noise` is float or array
    assert noise.shape == (dim,), "Noise vector has wrong shape"

    # Make sure that `key` is JAX key
    if isinstance(key, int):
        key = jrandom.PRNGKey(key)

    # Check solver hyperparameters
    assert brownian_tol > 0, "Brownian tolerance must be positive"
    assert solver_dt > 0, "Solver time step must be positive"

    t0, t1 = t_span.min(), t_span.max()

    # Generate the drift and diffusion terms
    drift = _get_drift_term(fitness, noise)
    diffusion = _get_diffusion_term(noise)

    if jit_terms:
        drift = jax.jit(drift)
        diffusion = jax.jit(diffusion)

    brownian_motion = VirtualBrownianTree(
        t0, t1, tol=brownian_tol, shape=(dim,), key=key  # pyright: ignore
    )
    terms = MultiTerm(ODETerm(drift), ControlTerm(diffusion, brownian_motion))

    solver = Euler()
    saveat = SaveAt(ts=t_span)  # pyright: ignore

    # Note that we truncate the last entry of the solution as it is implicitly defined
    sol = diffeqsolve(
        terms,  # pyright: ignore
        solver,
        t0,  # pyright: ignore
        t1,  # pyright: ignore
        dt0=solver_dt,  # pyright: ignore
        y0=simplex_truncate(y0),  # pyright: ignore
        saveat=saveat,  # pyright: ignore
    )

    ys = simplex_complete(sol.ys)  # pyright: ignore
    return ys, sol
