import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from covvfit import numeric


def logsumexp_excluding_column_simple(y, axis: int = -1):
    # Numerical stability by shifting with max_val
    max_val = jnp.max(y, axis=axis, keepdims=True)
    shifted = y - max_val
    # Compute sum exp shifted,
    # Substract sum exp shifted for each column
    # Take the log and add back the max_val
    sum_exp_shifted = jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True)
    logsumexp_excl = jnp.log(sum_exp_shifted - jnp.exp(shifted)) + max_val
    return logsumexp_excl


@pytest.mark.parametrize("shape", [(2, 3, 5), (3,), (1, 3), (2, 4, 6)])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_logsumexp_excluding_column_standard(shape, axis):
    """Test on arrays of different shapes, but with entries for which
    the simple implementation is stable."""
    rng = np.random.default_rng(42)

    if axis > len(shape):
        return

    y = rng.normal(size=shape)
    out = numeric.logsumexp_excluding_column(y)
    exp = logsumexp_excluding_column_simple(y)

    npt.assert_allclose(out, exp, rtol=0.01, atol=1e-3)


def test_logsumexp_excluding_column_hard():
    """Test on examples where the simple implementation does not suffice."""
    y = jnp.array([[1e8, 1.0, 1.0]])
    out = numeric.logsumexp_excluding_column(y)
    expected = jnp.array([[1 + jnp.log(2), 1e8, 1e8]])

    npt.assert_allclose(out, expected)
