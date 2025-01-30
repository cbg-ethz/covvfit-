"""Simultaneous deconvolution."""
from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

import covvfit._quasimultinomial as qm


# Numerically stable functions to work with logarithms
def _log_matrix(
    a: Float[Array, " *shape"],
    threshold: float = 1e-7,
) -> Float[Array, " *shape"]:
    log_a = jnp.log(a)
    neg_inf = jnp.finfo(a.dtype).min
    return jnp.where(a > threshold, log_a, neg_inf)


def log1mexp(x):
    """Compute log(1 - exp(x)) in a numerically stable way."""
    x = jnp.minimum(x, -jnp.finfo(x.dtype).eps)
    return jnp.where(x > -0.693, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))


class _DeconvolutionProblemData(NamedTuple):
    """

    Attrs:
        Dimensions:
            n_cities: number of cities
            n_variants: number of variants
            n_mutations: number of loci

        Observed data:
            ts: array storing timepoints for each city
            ms: array representing the fraction of loci in which
                the mutation was detected at a particular timepoint
                and location
            mask: binary mask. Use 0 to ignore a particular entry
                (e.g., if the particular loci was not sequenced properly
                we have missing data)
            n_quasibin: scaling. Can be used to attribute some data entries
                some lower credibility than others
            overdispersion: has the same effect as `1 / n_quasibin`

        Variant definitions:
            log_variant_defs: logarithmied variant definitions,
                `log E[mutation occurred | variant]`.
                Note that `_log_matrix` is the preferred way
                of taking logarithm in a stable manner.
    """

    n_cities: int
    n_variants: int
    n_mutations: int
    ts: Float[Array, "cities timepoints"]
    ms: Float[Array, "cities timepoints mutations"]
    mask: Bool[Array, "cities timepoints mutations"]
    n_quasibin: Float[Array, "cities timepoints mutations"]
    overdispersion: Float[Array, "cities timepoints mutations"]

    log_variant_defs: Float[Array, "variants mutations"]


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
    def from_vector(cls, theta) -> "JointLogisticGrowthParams":
        return JointLogisticGrowthParams(
            relative_growths=qm.get_relative_growths(theta),
            relative_offsets=qm.get_relative_midpoints(theta),
        )

    def to_vector(self) -> Float[Array, " *batch"]:
        return qm.construct_theta(
            relative_growths=self.relative_growths,
            relative_midpoints=self.relative_offsets,
        )


PyTree = TypeVar("PyTree")

# The epidemic growth models takes as input model parameters
# and predicts the vector log log-variant prevalence
# at each requested timepoint in each city
# Note that each city has separate timepoints vector
GrowthModel = Callable[
    [
        PyTree,  # Model parameters
        Float[Array, "cities timepoints"],  # Timepoints for each city
    ],
    # Log-prevalence vector for each city and timepoint
    Float[Array, "cities timepoints variants"],
]


def _calculate_log_mutation_probabilities(
    log_variant_definitions: Float[Array, "variants mutations"],
    log_variant_abundances: Float[Array, " variants"],
) -> Float[Array, " mutations"]:
    """Calculates the log-probabilities of observing
    mutations at each loci.
    """
    # TODO(Pawel): Write tests for this
    log_A = log_variant_definitions
    log_y = log_variant_abundances

    log_B = log_A + log_y[..., None]  # Shape (variants, mutations)
    return logsumexp(log_B, axis=0)


def _quasiloglikelihood_single_city(
    log_abundance: Float[Array, "timepoints variants"],
    log_variant_defs: Float[Array, "variants mutations"],
    ms: Float[Array, "timepoints mutations"],
    mask: Bool[Array, "timepoints mutations"] | float,
    n_quasibin: Float[Array, "timepoints mutations"] | float,
    overdispersion: Float[Array, "timepoints mutations"] | float,
) -> float:
    # Obtain a matrix of shape (timepoints, mutations)
    log_p = jax.vmap(
        _calculate_log_mutation_probabilities,
        in_axes=(None, 0),
    )(log_variant_defs, log_abundance)

    log1_minusp = log1mexp(log_p)

    log_quasi = (
        mask * n_quasibin * (ms * log_p + (1.0 - ms) * log1_minusp) / overdispersion
    )
    return jnp.sum(log_quasi)


def _generate_quasiloglikelihood_function(data: _DeconvolutionProblemData):
    def quasiloglikelihood(model: JointLogisticGrowthParams) -> float:
        # cities, timepoints, variants
        log_abundances = model.predict_log_abundance(data.ts)

        # quasiloglikelihood for each city
        quasis = jax.vmap(
            _quasiloglikelihood_single_city, in_axes=(0, None, 0, 0, 0, 0)
        )(
            log_abundances,
            data.log_variant_defs,
            data.ms,
            data.mask,
            data.n_quasibin,
            data.overdispersion,
        )

        return jnp.sum(quasis)

    return quasiloglikelihood
