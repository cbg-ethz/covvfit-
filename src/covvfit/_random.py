"""Random number generation."""

import jax


class JAXRNG:
    """JAX stateful random number generator.

    Example:
      key = jax.random.PRNGKey(5)
      rng = JAXRNG(key)
      a = jax.random.bernoulli(rng.key, shape=(10,))
      b = jax.random.bernoulli(rng.key, shape=(10,))
    """

    def __init__(self, key: jax.Array | None = None, seed: int = 42) -> None:
        """
        Args:
            key: initialization key
            seed: initialization seed

        Note:
            if `key` is provided, it takes precedence over the `seed`
        """
        if key is not None:
            self._key = key
        else:
            self._key = jax.random.PRNGKey(seed)

    @property
    def key(self) -> jax.Array:
        """Generates a new key."""
        key, subkey = jax.random.split(self._key)
        self._key = key
        return subkey

    def __repr__(self) -> str:
        """Used by the repr() method."""
        return f"{type(self).__name__}(key={self._key})"

    def __str__(self) -> str:
        """Used by the str() method."""
        return repr(self)
