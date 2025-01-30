"""Simultaneous deconvolution."""
from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float


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
    relative_growths: Float[Array, " variants-1"]
    relative_offsets: Float[Array, "cities variants-1"]

    def log_predict(
        self,
        timepoints: Float[Array, "cities timepoints"],
    ) -> Float[Array, "cities timepoints variants"]:
        # TODO(Pawel)
        raise NotImplementedError

    @classmethod
    def from_vector(cls) -> "JointLogisticGrowthParams":
        # TODO(Pawel)
        raise NotImplementedError

    def to_vector(self) -> Float[Array, " *batch"]:
        # TODO(Pawel)
        raise NotImplementedError


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
    mask: Bool[Array, "timepoints mutations"],
    n_quasibin: Float[Array, "timepoints mutations"],
    overdispersion: Float[Array, "timepoints mutations"],
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
