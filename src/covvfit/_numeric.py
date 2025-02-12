import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy import optimize

# Numerically stable functions to work with logarithms

LOG_THRESHOLD = 1e-7


def log_matrix(
    a: Float[Array, " *shape"],
    threshold: float = LOG_THRESHOLD,
) -> Float[Array, " *shape"]:
    """Takes the logarithm of the entries, in a numerically stable manner.
    I.e., replaces values smaller than `threshold` with minimum value for the provided data type.

    Args:
        a: matrix which entries should be logarithmied
        threshold: threshold used when to not calculate the logarithm

    Returns:
        log_a: matrix with logarithmied entries
    """
    log_a = jnp.log(a)
    neg_inf = jnp.finfo(a.dtype).min
    return jnp.where(a > threshold, log_a, neg_inf)


def log1mexp(x: Float[Array, " *shape"]) -> Float[Array, " *shape"]:
    """Computes `log(1 - exp(x))` in a numerically stable way.

    Args:
        x: array

    Returns:
        log1mexp(x): array of the same shape as `x`
    """
    x = jnp.minimum(x, -jnp.finfo(x.dtype).eps)
    return jnp.where(x > -0.693, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))


@dataclasses.dataclass
class OptimizeMultiResult:
    """Multi-start optimization result.

    Args:
        x: array of shape `(dim,)` representing minimum found
        fun: value of the optimized function at `x`
        best: optimization result (for the best start, yielding `x`)
        runs: all the optimization results (for all starts)
    """

    x: np.ndarray
    fun: float
    best: optimize.OptimizeResult
    runs: list[optimize.OptimizeResult]


def jax_multistart_minimize(
    loss_fn,
    theta0: np.ndarray,
    n_starts: int = 10,
    random_seed: int = 42,
    maxiter: int = 10_000,
) -> OptimizeMultiResult:
    """Multi-start gradient-based minimization.

    Args:
        loss_fn: loss function to be optimized
        theta0: vector of shape `(dim,)`
            providing an example starting point
        n_starts: number of different starts
        random_seed: seed used to perturb `theta0`
        maxiter: maximum number of iterations per run

    Returns:
        result: OptimizeMultiResult with the optimization information
    """
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
