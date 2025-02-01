"""Simultaneous deconvolution."""
from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

import covvfit._quasimultinomial as qm
from covvfit._quasimultinomial import create_padded_array

# Numerically stable functions to work with logarithms

_LOG_THRESHOLD = 1e-7


def _log_matrix(
    a: Float[Array, " *shape"],
    threshold: float = _LOG_THRESHOLD,
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
    mutations: Float[Array, "cities timepoints mutations"]
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
            data.mutations,
            data.mask,
            data.n_quasibin,
            data.overdispersion,
        )

        return jnp.sum(quasis)

    return quasiloglikelihood


def _validate_and_pad(
    timepoints: list[Float[Array, " timepoints"]],
    mutations: list[Float[Array, "timepoints loci"]],
    variant_def: Bool[Array, "variants loci"],
    mask: list[Bool[Array, "timepoints loci"]] | None = None,
    ns: float = 1.0,
    overdispersion: float = 1.0,
    _threshold: float = _LOG_THRESHOLD,
) -> _DeconvolutionProblemData:
    """Validation function, parsing the input provided in
    the format convenient for the user to the internal
    representation compatible with JAX."""
    # TODO(Pawel): Allow specifying the `ns` and `overdispersion` as lists of arrays

    # Get the number of cities
    n_cities = len(mutations)
    if len(timepoints) != n_cities:
        raise ValueError(
            f"Number of cities not consistent: {n_cities} != {len(timepoints)}."
        )
    if n_cities < 1:
        raise ValueError("There has to be at least one city.")

    # Get the number of variants and loci
    variant_def = jnp.asarray(variant_def, dtype=float)
    if variant_def.ndim != 2:
        raise ValueError(
            f"Variant definitions has to be "
            f"a two-dimensional array, but is {variant_def.ndim}."
        )

    n_variants, n_loci = variant_def.shape[0], variant_def.shape[1]

    for i, y in enumerate(mutations):
        if y.ndim != 2:
            raise ValueError(f"City {i} has {y.ndim} dimension, rather than 2.")
        if y.shape[-1] != n_loci:
            raise ValueError(f"City {i} has {y.shape[-1]} loci rather than {n_loci}.")

    # Ensure that the number of timepoints is consistent for t and mutations
    max_timepoints = 0
    for i, (t, y) in enumerate(zip(timepoints, mutations)):
        if t.ndim != 1:
            raise ValueError(
                f"City {i} has time axis with dimension {t.ndim}, rather than 1."
            )
        if t.shape[0] != y.shape[0]:
            raise ValueError(
                f"City {i} has timepoints mismatch: {t.shape[0]} != {y.shape[0]}."
            )

        max_timepoints = max(max_timepoints, t.shape[0])

    # Now create the arrays representing the data

    #    -- timepoints array, padded with zeros
    _lengths = [t.shape[0] for t in timepoints]
    out_ts = create_padded_array(
        values=timepoints,
        lengths=_lengths,
        padding_length=max_timepoints,
        padding_value=0.0,
    )

    #    -- mask array, padded with zeros and taking into account given mask list
    if mask is None:
        mask = [jnp.ones((n_ts, n_loci), dtype=bool) for n_ts in _lengths]

    out_mask = jnp.full(
        shape=(n_cities, max_timepoints, n_loci), fill_value=0, dtype=bool
    )
    for city in range(n_cities):
        n_ts = _lengths[city]
        city_mask = mask[city]
        out_mask = out_mask.at[city, :n_ts, ...].set(city_mask)

    #     --- mutations data
    out_mutations = jnp.full(
        shape=(n_cities, max_timepoints, n_loci), fill_value=0.0, dtype=float
    )
    for city in enumerate(n_cities):
        n_ts = _lengths[city]
        data = mutations[city]
        out_mutations = out_mutations.at[city, :n_ts, ...].set(data)

    #     --- overdispersion and quasimultinomial
    out_n = jnp.full_like(out_mutations, fill_value=ns)
    out_overdispersion = jnp.full_like(out_n, fill_value=overdispersion)

    return _DeconvolutionProblemData(
        n_cities=n_cities,
        n_variants=n_variants,
        n_mutations=n_loci,
        ts=out_ts,
        mutations=out_mutations,
        mask=out_mask,
        n_quasibin=out_n,
        overdispersion=out_overdispersion,
        log_variant_defs=_log_matrix(
            jnp.asarray(variant_def, dtype=float), threshold=_threshold
        ),
    )


def construct_total_loss(
    timepoints: list[Float[Array, " timepoints"]],
    mutations: list[Float[Array, "timepoints loci"]],
    variant_def: Bool[Array, "variants loci"],
    mask: list[Bool[Array, "timepoints loci"]] | None = None,
    ns: qm._OverDispersionType = 1.0,
    overdispersion: qm._OverDispersionType = 1.0,
    accept_vector: bool = True,
):
    """Constructs the loss function for deconvolution.

    Args:
        timepoints: list of arrays representing the timepoints
            at which the data were collected at each location.
            Shape of the `i`th array is `(n_timepoints[i],)`
        mutations: list of arrays representing observed mutations in
            the samples at each location. Shape of `i`th array is
            (n_timepoints[i], n_loci)
        mask: list of binary masks of the same structure as `mutations`.
            Entry 1 means that we trust sequencing at a particular locus
            and 0 means that we do not have sufficient coverage to determine
            the value and we prefer to treat it as missing data
        variant_def: variant definitions matrix, shape (n_variants, n_loci)
    """

    # Take the logarithm of the variant definition matrix
    # in a numerically stable manner

    data = _validate_and_pad(
        timepoints=timepoints,
        mutations=mutations,
        variant_def=variant_def,
        mask=mask,
    )

    quasiloglikelihood_fn = _generate_quasiloglikelihood_function(data)

    def loss_fn(params: JointLogisticGrowthParams) -> float:
        return -quasiloglikelihood_fn(params)

    def loss_fn_vector(theta: jax.Array) -> float:
        params = JointLogisticGrowthParams.from_vector(theta)
        return loss_fn(params)

    if accept_vector:
        return loss_fn_vector
    else:
        return loss_fn
