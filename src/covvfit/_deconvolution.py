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
    relative_offsets: Float[Array, " variants-1"]


PyTree = TypeVar("PyTree")

GrowthModel = Callable[
    [PyTree, Float[Array, " timepoints"]], Float[Array, "timepoints variants"]
]


def logistic_growth(
    params: LogisticGrowthParams, ts: Float[Array, " timepoints"]
) -> Float[Array, "timepoints variants"]:
    return qm.calculate_logps(
        ts=ts,
        midpoints=qm._add_first_variant(params.relative_offsets),
        growths=qm._add_first_variant(params.relative_growths),
    )


def _quasiloglikelihood_single_city(
    growth_model: GrowthModel,
    params: PyTree,
    ts: Float[Array, " timepoints"],
    ms: Float[Array, "timepoints mutations"],
    mask: Bool[Array, "timepoints mutations"],
    n_quasibin: Float[Array, "timepoints mutations"],
    overdispersion: Float[Array, "timepoints mutations"],
    log_variant_defs: Float[Array, "variants mutations"],
) -> float:
    # Generate relative variant abundances,
    # shape (timepoints, variants)
    log_variant_abundances = growth_model(params, ts)

    # Now generate the log-probability of
    # finding the mutation at each locus
    # with shape (timepoints, mutations)
    # Note that this is a quasibinomial model for each entry,
    # so that the sums obtained by `a.sum(axis=-1)`
    # can be as large as `mutations`, rather than 1.
    jnp.einsum("tv,vm->tm", log_variant_abundances, log_variant_defs)

    # Now calculate the quasiloglikelihood
