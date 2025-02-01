from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

import covvfit._quasimultinomial as qm


class JointLogisticGrowthParams(NamedTuple):
    """This is a model of logistic growth (selection dynamics)
    in `K` cities for `V` competing variants.

    We assume that the relative growth advantages
    do not change between the cities, however
    we allow different introduction times, resulting
    in different offsets in the logistic growth model.

    This model has `V-1` relative growth rate parameters
    and `K*(V-1)` offsets.
    """

    relative_growths: Float[Array, " variants-1"]
    relative_offsets: Float[Array, "cities variants-1"]

    @property
    def n_cities(self) -> int:
        return self.relative_offsets.shape[0]

    @property
    def n_variants(self) -> int:
        return 1 + self.relative_offsets.shape[1]

    @property
    def n_params(self) -> int:
        return (self.n_variants - 1) * (self.n_cities + 1)

    @staticmethod
    def _predict_log_abundance_single(
        relative_growths: Float[Array, " variants-1"],
        relative_offsets: Float[Array, " variants-1"],
        timepoints: Float[Array, " timepoints"],
    ) -> Float[Array, "timepoints variants"]:
        return qm.calculate_logps(
            ts=timepoints,
            midpoints=qm._add_first_variant(relative_offsets),
            growths=qm._add_first_variant(relative_growths),
        )

    def predict_log_abundance(
        self,
        timepoints: Float[Array, "cities timepoints"],
    ) -> Float[Array, "cities timepoints variants"]:
        _new_shape = (self.n_cities, self.n_variants - 1)
        tiled_growths = jnp.broadcast_to(self.relative_growths[None, :], _new_shape)

        return jax.vmap(self._predict_log_abundance_single, in_axes=0)(
            tiled_growths, self.relative_offsets, timepoints
        )

    @classmethod
    def from_vector(cls, theta, n_variants: int) -> "JointLogisticGrowthParams":
        return JointLogisticGrowthParams(
            relative_growths=qm.get_relative_growths(theta, n_variants=n_variants),
            relative_offsets=qm.get_relative_midpoints(theta, n_variants=n_variants),
        )

    def to_vector(self) -> Float[Array, " *batch"]:
        return qm.construct_theta(
            relative_growths=self.relative_growths,
            relative_midpoints=self.relative_offsets,
        )
