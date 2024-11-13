"""Logistic growth simulations."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from numpyro.distributions import Multinomial


def generate_logistic(
    ts: Float[Array, " *batch"],
    midpoints: Float[Array, " variants"],
    growths: Float[Array, " variants"],
) -> Float[Array, "*batch variants"]:
    shape = (1,) * ts.ndim + (-1,)
    m = midpoints.reshape(shape)
    g = growths.reshape(shape)

    linear = (ts[..., None] - m) * g

    return jax.nn.softmax(linear, axis=-1)


@dataclass
class SimulationSettings:
    """Settings of the simulation."""

    n_cities: int  # Number of cities
    n_variants: int  # Number of
    growth_rates: jax.Array  # Growth rates of individual variants, shape (n_variants,)
    midpoints: (
        jax.Array
    )  # Midpoints for each variant and each sity, shape (n_cities, n_variants)
    n_multinomial: (
        jax.Array
    )  # Settings for multinomial noise for each city, shape (n_cities,).
    n_observations: jax.Array  # Number of observations per city
    time0: float = 0.0
    time1: float = 1.0

    def __post_init__(self):
        if self.n_cities < 1:
            raise ValueError("At least one city is required.")

        if self.n_variants < 2:
            raise ValueError("At least two variants are required.")

        if self.growth_rates.shape != (self.n_variants,):
            raise ValueError("Growth rates has wrong shape.")

        if self.midpoints.shape != (self.n_cities, self.n_variants):
            raise ValueError("Midpoints have wrong shape.")

        if self.n_multinomial.shape != (self.n_cities,):
            raise ValueError("n_multinomal has wrong shape.")

        if self.n_observations.shape != (self.n_cities,):
            raise ValueError("n_observations has wrong shape.")

    def time(self) -> tuple[float, float]:
        return (self.time0, self.time1)

    def _assert_city_index(self, index: int) -> None:
        if index >= self.n_cities:
            raise ValueError(
                f"Index {index} is larger than the number of cities, {self.n_cities}."
            )

    def _default_time_index(self, city_index: int) -> Float[Array, " n_obs"]:
        self._assert_city_index(city_index)
        return jnp.linspace(self.time0, self.time1, self.n_observations[city_index])

    def calculate_abundances_one_city(
        self, city_index: int, ts: Float[Array, " timepoints"] | None = None
    ) -> tuple[Float[Array, " n_obs"], Float[Array, "n_obs n_variants"]]:
        """Calculates the abundances for the specified city.

        Args:
            city_index: index of the city
            ts: timepoints at which abundances should be evaluated. By default (None), we measure uniformly in the range (time0, time1).

        Returns:
            ts: timepoints, shape (n_obs,)
            ys: abundances, shape (n_obs, n_variants).
                Note that ys.sum(axis=-1) results in (approximately) array of 1s.
        """
        self._assert_city_index(city_index)

        if ts is None:
            ts = self._default_time_index(city_index)
        ys = generate_logistic(
            ts, midpoints=self.midpoints[city_index], growths=self.growth_rates
        )

        return ts, ys

    def generate_sample_one_city(
        self, key: jax.Array, city_index: int
    ) -> tuple[jax.Array, jax.Array]:
        """Generates observed samples from a given city."""
        self._assert_city_index(city_index)

        ts, ys = self.calculate_abundances_one_city(city_index=city_index)

        draws = Multinomial(
            total_count=self.n_multinomial[city_index], probs=ys
        ).sample(key)
        proportions = jnp.asarray(draws, dtype=float) / jnp.sum(
            draws, axis=-1, keepdims=True
        )

        return ts, proportions

    def generate_sample_all_cities(
        self, key: jax.Array
    ) -> dict[int, tuple[jax.Array, jax.Array]]:
        ret = {}
        for city_index in range(self.n_cities):
            subkey = jax.random.fold_in(key, city_index)
            ret[city_index] = self.generate_sample_one_city(
                key=subkey, city_index=city_index
            )

        return ret
