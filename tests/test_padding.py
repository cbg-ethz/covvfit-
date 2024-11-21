import jax.numpy as jnp
import numpy.testing as npt
import pytest
from covvfit._padding import create_padded_array
from jax import random


def test_scalar_value():
    """Test with a single scalar value."""
    values = 3.14
    lengths = [2, 3]
    padding_length = 4
    padding_value = 0.0

    expected = jnp.array([[3.14, 3.14, 0.0, 0.0], [3.14, 3.14, 3.14, 0.0]])

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_list_of_scalars():
    """Test with a list of scalar values."""
    values = [1.0, 2.0, 3.0]
    lengths = [1, 2, 3]
    padding_length = 4
    padding_value = -1.0

    expected = jnp.array(
        [[1.0, -1.0, -1.0, -1.0], [2.0, 2.0, -1.0, -1.0], [3.0, 3.0, 3.0, -1.0]]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_list_of_arrays():
    """Test with a list of JAX arrays."""
    values = [
        jnp.array([1.1, 1.2]),
        jnp.array([2.1, 2.2, 2.3]),
        jnp.array([3.1, 3.2, 3.3, 3.4]),
    ]
    lengths = [2, 3, 4]
    padding_length = 5
    padding_value = 0.0

    expected = jnp.array(
        [
            [1.1, 1.2, 0.0, 0.0, 0.0],
            [2.1, 2.2, 2.3, 0.0, 0.0],
            [3.1, 3.2, 3.3, 3.4, 0.0],
        ]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_nested_list_of_floats():
    """Test with a nested list of floats."""
    values = [[1.0, 1.1], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2, 3.3]]
    lengths = [2, 3, 4]
    padding_length = 5
    padding_value = -5.0

    expected = jnp.array(
        [
            [1.0, 1.1, -5.0, -5.0, -5.0],
            [2.0, 2.1, 2.2, -5.0, -5.0],
            [3.0, 3.1, 3.2, 3.3, -5.0],
        ]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_empty_cities():
    """Test with zero cities, expecting a ValueError."""
    values = []
    lengths = []
    padding_length = 3
    padding_value = 0.0

    with pytest.raises(ValueError, match="There has to be at least one city."):
        create_padded_array(values, lengths, padding_length, padding_value)


def test_max_length_exceeds_padding():
    """Test when the maximum expected length exceeds the padding length."""
    values = [1.0, 2.0, 3.0]
    lengths = [2, 5, 3]
    padding_length = 4
    padding_value = 0.0

    with pytest.raises(
        ValueError, match="Maximum length is 5, which is greater than the padding 4."
    ):
        create_padded_array(values, lengths, padding_length, padding_value)


def test_values_length_mismatch():
    """Test when the length of values does not match the number of cities."""
    values = [1.0, 2.0]  # Only 2 values
    lengths = [1, 2, 3]
    padding_length = 3
    padding_value = 0.0

    with pytest.raises(ValueError, match="Provided list has length 2 rather than 3."):
        create_padded_array(values, lengths, padding_length, padding_value)


def test_value_length_mismatch_per_city():
    """Test when a city's value length does not match its expected length."""
    values = [[1.0, 1.1], [2.0], [3.0, 3.1, 3.2]]  # Expected length is 3
    lengths = [2, 3, 3]
    padding_length = 4
    padding_value = 0.0

    with pytest.raises(
        ValueError,
        match="For 1th component the provided array has length 1 rather than 3.",
    ):
        create_padded_array(values, lengths, padding_length, padding_value)


def test_different_out_dtype():
    """Test with a different output data type."""
    values = [1, 2, 3]
    lengths = [1, 2, 3]
    padding_length = 3
    padding_value = 0
    _out_dtype = jnp.int32

    expected = jnp.array([[1, 0, 0], [2, 2, 0], [3, 3, 3]], dtype=jnp.int32)

    result = create_padded_array(
        values, lengths, padding_length, padding_value, _out_dtype=_out_dtype
    )
    npt.assert_array_equal(result, expected)
    assert result.dtype == _out_dtype


def test_large_input():
    """Test with a large number of cities and large padding."""
    n_cities = 20
    padding_length = 10
    padding_value = -1.0
    lengths = [
        random.randint(random.PRNGKey(i), minval=1, maxval=padding_length + 1, shape=())
        for i in range(n_cities)
    ]
    values = [float(i) for i in range(n_cities)]

    result = create_padded_array(values, lengths, padding_length, padding_value)

    assert result.shape == (n_cities, padding_length)

    for i in range(n_cities):
        length = lengths[i]
        expected_row = jnp.full((padding_length,), padding_value)
        if length > 0:
            expected_row = expected_row.at[:length].set(values[i])
        npt.assert_allclose(result[i], expected_row)


def test_float_array_input():
    values = [[1.0, 1.1], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2, 3.3]]
    values = [jnp.asarray(v) for v in values]
    lengths = [2, 3, 4]
    padding_length = 4
    padding_value = 0.0

    expected = jnp.array(
        [
            [1.0, 1.1, 0.0, 0.0],
            [2.0, 2.1, 2.2, 0.0],
            [3.0, 3.1, 3.2, 3.3],
        ]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_allclose(result, expected)


def test_non_finite_padding_value():
    """Test with a non-finite padding value (e.g., NaN)."""
    values = [1.0, 2.0]
    lengths = [1, 2]
    padding_length = 3
    padding_value = float("nan")

    result = create_padded_array(values, lengths, padding_length, padding_value)

    expected = jnp.array([[1.0, jnp.nan, jnp.nan], [2.0, 2.0, jnp.nan]])

    # Since NaN != NaN, use isnan to check padding positions
    for i in range(len(lengths)):
        length = lengths[i]
        # Check the filled values
        npt.assert_array_equal(result[i, :length], expected[i, :length])
        # Check the padding values are NaN
        if length < padding_length:
            assert jnp.isnan(result[i, length:]).all()


def test_zero_padding_length():
    """Test with zero padding length."""
    values = [1.0, 2.0]
    lengths = [0, 0]
    padding_length = 0
    padding_value = 0.0

    expected = jnp.empty((2, 0))

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_expected_length_zero_with_padding():
    """Test with some expected lengths zero and padding_length greater than zero."""
    values = [1.0, 2.0, 3.0]
    lengths = [0, 2, 0]
    padding_length = 3
    padding_value = -1.0

    expected = jnp.array([[-1.0, -1.0, -1.0], [2.0, 2.0, -1.0], [-1.0, -1.0, -1.0]])

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_non_float_values():
    """Test with integer values and float padding."""
    values = [1, 2, 3]
    lengths = [1, 2, 3]
    padding_length = 4
    padding_value = 0.5

    expected = jnp.array(
        [[1.0, 0.5, 0.5, 0.5], [2.0, 2.0, 0.5, 0.5], [3.0, 3.0, 3.0, 0.5]]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    npt.assert_array_equal(result, expected)


def test_floating_point_precision():
    """Test with floating point precision to ensure correct handling."""
    values = [1.123456789, 2.987654321]
    lengths = [3, 2]
    padding_length = 4
    padding_value = 0.0

    expected = jnp.array(
        [
            [1.123456789, 1.123456789, 1.123456789, 0.0],
            [2.987654321, 2.987654321, 0.0, 0.0],
        ]
    )

    result = create_padded_array(values, lengths, padding_length, padding_value)
    # Use allclose for floating point comparisons
    npt.assert_allclose(result, expected, atol=1e-8)
