from typing import Sequence, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def _is_scalar(value) -> bool:
    try:
        length = len(value)
        if length != 0:
            return False
        return True
    except TypeError:
        return True


def create_padded_array(
    values: T | Sequence[T] | Sequence[jax.Array] | Sequence[Sequence[T]],
    lengths: list[int],
    padding_length: int,
    padding_value: T,
    _out_dtype=float,
) -> jax.Array:
    """Parsing utility, which pads `values`
    into a two-dimensional array describing multiple time series.

    Args:
        values: provided values for each group.
            It can be the following a scalar value
            (constant for all time series) or a sequence of values
            describing the observations of each time series.
            If it is a sequence, then each entry can be either
            a single value (constant for the time series) or
            an array specifying values for all the time points in the
            particular time series
        lengths: lengths of the timeseries, one per time series
        padding_length: padding length, must be larger than all entries
            in `lengths`

    Returns:
        JAX array of shape (n_timeseries, padding_length)
    """
    n_cities = len(lengths)
    if n_cities < 1:
        raise ValueError("There has to be at least one city.")
    if max(lengths) > padding_length:
        raise ValueError(
            f"Maximum length is {max(lengths)}, which is greater than the padding {padding_length}."
        )

    out_array = jnp.full(
        shape=(n_cities, padding_length), fill_value=padding_value, dtype=_out_dtype
    )

    # First case: `values` argument is a single number (not an iterable)
    if _is_scalar(values):
        for i, length in enumerate(lengths):
            out_array = out_array.at[i, :length].set(values)
        return out_array

    # Second case: `values` argument is not a scalar, but rather an iterable:
    if len(values) != n_cities:
        raise ValueError(
            f"Provided list has length {len(values)} rather than {n_cities}."
        )

    for i, (value, exp_len) in enumerate(zip(values, lengths)):
        if _is_scalar(value):  # For this city we have constant value provided
            out_array = out_array.at[i, :exp_len].set(value)
        else:  # We have a vector of values provided
            if len(value) != exp_len:
                raise ValueError(
                    f"For {i}th component the provided array has length {len(value)} rather than {exp_len}."
                )
            vals = jnp.asarray(value, dtype=out_array.dtype)
            out_array = out_array.at[i, :exp_len].set(vals)

    return out_array
