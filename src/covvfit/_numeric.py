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
    log_a = jnp.log(a)
    neg_inf = jnp.finfo(a.dtype).min
    return jnp.where(a > threshold, log_a, neg_inf)


def log1mexp(x):
    """Compute log(1 - exp(x)) in a numerically stable way."""
    x = jnp.minimum(x, -jnp.finfo(x.dtype).eps)
    return jnp.where(x > -0.693, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))


@dataclasses.dataclass
class OptimizeMultiResult:
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
