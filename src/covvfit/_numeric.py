import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
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


def logsumexp_excluding_column(
    y: Float[Array, "*batch variants"],
    axis: int = -1,
) -> Float[Array, "*batch variants"]:
    """
    Compute logsumexp across the given axis for each element, excluding
    the 'current' element at that axis index.

    Args:
        y: An array of shape [..., variants, ...].
        axis: The axis along which we exclude each index before computing
              logsumexp.

    Returns:
        An array of the same shape as `y`, whose element at index i along
        `axis` is the log-sum-exp of all other entries (j != i).
    """
    # Number of elements along the specified axis
    n = y.shape[axis]
    dtype = y.dtype

    # This function will exclude the i-th entry along `axis` by masking it.
    def exclude_index(i: int) -> jnp.ndarray:
        # Create a 1D mask of length n: True for j != i, False for j == i
        mask_1d = jnp.arange(n) != i

        # Reshape it so it can broadcast along the desired `axis`
        # e.g., if y.ndim=3 and axis=1, shape might be (1, n, 1)
        mask_shape = [1] * y.ndim
        mask_shape[axis] = n
        mask_1d = mask_1d.reshape(mask_shape)

        # Replace the excluded positions with -∞
        masked_y = jnp.where(mask_1d, y, jnp.finfo(dtype).min)

        # Compute logsumexp over the masked array along `axis`
        return logsumexp(masked_y, axis=axis)

    # Vectorize over each index (0...n-1) in the chosen axis
    # This produces a new array with shape=(n, [all the other axes]).
    results = jax.vmap(exclude_index)(jnp.arange(n))

    # Move the new axis (of length n) back to `axis`, matching the original shape of y
    results = jnp.moveaxis(results, 0, axis)
    return results


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
