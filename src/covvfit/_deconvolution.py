"""Simultaneous deconvolution."""
from typing import Callable, NamedTuple, TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

import covvfit._quasimultinomial as qm


class _DeconvolutionProblemData(NamedTuple):
    """

    Attrs:
        log_variant_defs: logarithmied variant definitions,
            log E[mutation occurred | variant]
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


class LogisticGrowthParams(NamedTuple):
    relative_growths: Float[Array, " variants-1"]
    relative_offsets: Float[Array, "cities variants-1"]


PyTree = TypeVar("PyTree")

GrowthModel = Callable[
    [PyTree, Float[Array, " timepoints"]], Float[Array, "cities timepoints variants"]
]


def logistic_growth(
    params: LogisticGrowthParams, ts: Float[Array, " timepoints"]
) -> Float[Array, "cities timepoints variants"]:
    return qm.calculate_logps(
        ts=ts,
        midpoints=qm._add_first_variant(params.relative_offsets),
        growths=qm._add_first_variant(params.relative_growths),
    )


def _log_abundance(
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


def _quasiloglikelihood_single_city(
    log_abundance: Float[Array, "timepoints variants"],
    log_variant_defs: Float[Array, "variants mutations"],
    ms: Float[Array, "timepoints mutations"],
    mask: Bool[Array, "timepoints mutations"],
    n_quasibin: Float[Array, "timepoints mutations"],
    overdispersion: Float[Array, "timepoints mutations"],
) -> float:
    # Now generate the log-probability of
    # finding the mutation at each locus
    # with shape (timepoints, mutations)
    # Note that this is a quasibinomial model for each entry,
    # so that the sums obtained by `a.sum(axis=-1)`
    # can be as large as `mutations`, rather than 1.
    log_p = jnp.einsum("tv,vm->tm", log_abundance, log_variant_defs)
    log1_minusp = log1mexp(log_p)

    log_quasi = (
        mask * n_quasibin * (ms * log_p + (1.0 - ms) * log1_minusp) / overdispersion
    )
    return jnp.sum(log_quasi)
