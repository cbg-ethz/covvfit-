# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
from subplots_from_axsize import subplots_from_axsize

import jax.numpy as jnp

import covvfit._quasimultinomial as qm
import covvfit._deconvolution as dec

# %%
A = jnp.asarray(
    [
        [1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1],
    ]
)

n_variants = A.shape[0]
n_loci = A.shape[1]

print(
    f"Variant definition matrix has rank {jnp.linalg.matrix_rank(A)}. We require rank {n_variants}."
)


n_cities = 2


relative_offsets = jnp.asarray(
    [
        [0.3, -0.3, -4.0],
        [0.2, -0.45, -5.0],
    ]
)
relative_growth_rates = jnp.asarray([0.2, 1.0, 5.0])

n_timepoints = 30

timepoints = jnp.asarray(
    [
        jnp.linspace(0, 1, n_timepoints),
        jnp.linspace(0.1, 0.9, n_timepoints),
    ]
)

assert relative_offsets.shape == (n_cities, n_variants - 1)
assert relative_growth_rates.shape == (n_variants - 1,)

# %%
model = dec.JointLogisticGrowthParams(
    relative_growths=relative_growth_rates,
    relative_offsets=relative_offsets,
)

log_ys = model.predict_log_abundance(timepoints)
ys = jnp.exp(log_ys)

fig, axs = subplots_from_axsize(
    1, n_cities, axsize=(2, 1.5), sharex=True, sharey=True, dpi=180
)

for city, ax in enumerate(axs.ravel()):
    y = ys[city]

    for variant in range(n_variants):
        ax.plot(timepoints[city], y[:, variant], c=f"C{variant}")

    ax.set_title(f"City {city}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variant abundances")
    ax.spines[["top", "right"]].set_visible(False)

# %%
ms_perfect = jnp.einsum("vm,ctv->ctm", A, ys)

rng = np.random.default_rng(42)

sample_size = 200

ms_sampled = rng.binomial(sample_size, jnp.clip(ms_perfect, 1e-5, 1 - 1e-5)) / float(
    sample_size
)

# %%
fig, axs = subplots_from_axsize(
    1, n_cities, axsize=(2, 1.5), sharex=True, sharey=True, dpi=180
)

markers = list(".osP+xDv^*hX")

for city, ax in enumerate(axs.ravel()):
    for locus in range(n_loci):
        ax.plot(
            timepoints[city],
            ms_perfect[city, :, locus],
            c=f"C{locus}",
            linestyle="-",
            alpha=0.3,
        )
        ax.scatter(
            timepoints[city],
            ms_sampled[city, :, locus],
            c=f"C{locus}",
            s=2,
            marker=markers[locus],
        )

    ax.set_title(f"City {city}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mutation probability")
    ax.spines[["top", "right"]].set_visible(False)

# %%
log_A = dec._log_matrix(jnp.asarray(A, dtype=float))


problem_data = dec._DeconvolutionProblemData(
    n_cities=n_cities,
    n_variants=n_variants,
    n_mutations=n_loci,
    ts=timepoints,
    mutations=ms_sampled,
    mask=jnp.ones_like(ms_sampled),
    n_quasibin=jnp.ones_like(ms_sampled),
    overdispersion=jnp.ones_like(ms_sampled),
    log_variant_defs=log_A,
)

# %%
dec._calculate_log_mutation_probabilities(log_A, log_ys[0, 0])

# %%
jnp.log(A.T @ ys[0, 0])

# %%
qll = dec._generate_quasiloglikelihood_function(problem_data)

# %%
qll(model)

# %%
model2 = dec.JointLogisticGrowthParams(
    relative_growths=relative_growth_rates + 1.3,
    relative_offsets=relative_offsets - 1.2,
)

qll(model2)

# %%
