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

# %% [markdown]
# # Model-based deconvolution
#
# This notebook illustrates how to simultaneously perform deconvolution of wastewater data together with the growth rate estimation.
#
# First, let's import the right modules:

# %%
import numpy as np
import jax.numpy as jnp
import seaborn as sns
from subplots_from_axsize import subplots_from_axsize

import covvfit

dyn = covvfit.dynamics
dec = covvfit.deconvolution
qm = covvfit.quasimultinomial

# %% [markdown]
# Now, we construct a synthetic variant definition matrix, representing which loci are mutated in particular variants:

# %%
A = jnp.asarray(
    [
        [1, 1, 0, 0, 1, 0, 1],
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

# %% [markdown]
# We also need to define the ground-truth selection dynamics parameters. We assume that the growth rates are shared between different cities, although the time offsets (depending on the introduction times), can differ:

# %%
params_true = dyn.JointLogisticGrowthParams(
    relative_offsets=jnp.asarray(
        [
            [0.3, -0.3, -4.0],
            [0.2, -0.45, -5.0],
        ]
    ),
    relative_growths=jnp.asarray([0.2, 1.0, 5.0]),
)

n_cities = params_true.n_cities
print(f"Constructed a model for {n_cities} cities.")

assert params_true.n_variants == n_variants

timepoints_ideal = jnp.asarray([jnp.linspace(0, 1, 50)] * n_cities)

log_ys = params_true.predict_log_abundance(timepoints_ideal)
ys = jnp.exp(log_ys)

fig, axs = subplots_from_axsize(
    1, n_cities, axsize=(2, 1.5), sharex=True, sharey=True, dpi=180
)

for city, ax in enumerate(axs.ravel()):
    y = ys[city]

    for variant in range(n_variants):
        ax.plot(timepoints_ideal[city], y[:, variant], c=f"C{variant}")

    ax.set_title(f"City {city + 1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variant abundances")
    ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The variant abundances are not directly observed, however. Let's simulate the mutation abundances corresponding to different loci and plot them.

# %%
ms_perfect = jnp.einsum("vm,ctv->ctm", A, ys)

fig, axs = subplots_from_axsize(
    1, n_cities, axsize=(2, 1.5), sharex=True, sharey=True, dpi=180
)

for city, ax in enumerate(axs.ravel()):
    for locus in range(n_loci):
        ax.plot(
            timepoints_ideal[city],
            ms_perfect[city, :, locus],
            c=f"C{locus}",
            linestyle="-",
        )
    ax.set_title(f"City {city + 1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mutation frequency")
    ax.spines[["top", "right"]].set_visible(False)


# %% [markdown]
# Great, we have the model for how the mutation frequencies, occurring at different loci, change over time! In practice, however, we cannot obtain such frequencies directly.
#
# We should take into account the following:
#
# 1. We do not know the frequencies, but rather we have to obtain them from sequencing data, which is subject to various sources of noise.
# 2. We do not measure the frequency continously, but rather collect the data at various cities at different times.
# 3. Often, the amount of genetic material corresponding to a particular locus is not large enough in a given sample, resulting in a missing frequency value.
#
#
# Let's simulate a more realistic data set. First, we need to create a function:


# %%
def simulate_from_city(
    rng,
    city: int,
    timepoints,
    sample_size: int,
    missing: float,
):
    assert 0 < missing < 1
    timepoints = jnp.asarray([timepoints] * n_cities)

    log_ys = params_true.predict_log_abundance(timepoints)
    ys = jnp.exp(log_ys)

    ms_perfect = jnp.einsum("vm,ctv->ctm", A, ys)

    # Shape (n_timepoints, n_loci)
    ms_perfect = ms_perfect[city, ...]

    frequencies = rng.binomial(
        sample_size, jnp.clip(ms_perfect, 1e-5, 1 - 1e-5)
    ) / float(sample_size)
    mask = rng.binomial(1, jnp.full_like(a=ms_perfect, fill_value=1.0 - missing))
    mask = np.asarray(mask, dtype=bool)

    frequencies[
        ~mask
    ] = 0.5  # Set the unobserved values to 0.5, for numerical stability

    return frequencies, mask


# %% [markdown]
# Second, let's define the values for the cities. Consider the following scenario, in which one city started sequencing earlier, but uses a less performant protocol, resulting in larger noise and larger missing rate:

# %%
rng = np.random.default_rng(42)

timepoints = [
    jnp.linspace(0, 1, 50),
    jnp.linspace(0.2, 1, 40),
]
sample_size = [20, 30]
missing = [0.8, 0.7]

# TODO(Pawel): Remove after debugging
# timepoints = [
#     jnp.linspace(0,  1, 100),
#     jnp.linspace(0.2, 1, 100),
# ]
# sample_size = [100, 100]
# missing = [0.1, 0.2]


assert len(timepoints) == n_cities
assert len(sample_size) == n_cities
assert len(missing) == n_cities


mutation_frequencies = []
masks = []

for city in range(n_cities):
    freq, msk = simulate_from_city(
        rng,
        city=city,
        timepoints=timepoints[city],
        sample_size=sample_size[city],
        missing=missing[city],
    )
    mutation_frequencies.append(freq)
    masks.append(msk)

# %% [markdown]
# We can plot the frequencies corresponding to the collected samples:

# %%
fig, axs = subplots_from_axsize(
    n_cities, 1, axsize=(4, 1), sharex=False, sharey=False, dpi=180, hspace=0.8
)

for city, ax in enumerate(axs.ravel()):
    ax.set_title(f"City {city+1}")
    sns.heatmap(
        mutation_frequencies[city].T,
        cmap="Blues",
        mask=~masks[city].T,
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
        yticklabels=[],
    )
    ax.set_xlabel("Sample number")
    ax.set_ylabel("Locus")

# %% [markdown]
# Let's also quickly plot the idealised mutation frequencies together with the (synthetic) observed ones:

# %%
fig, axs = subplots_from_axsize(
    n_cities,
    1,
    axsize=(3, 1.5),
    sharex=True,
    sharey=True,
    dpi=180,
    hspace=0.8,
)

markers = list(".osP+xDv^*hX")

for city, ax in enumerate(axs.ravel()):
    freq = mutation_frequencies[city]
    for locus in range(n_loci):
        mask = masks[city][:, locus]

        ax.plot(
            timepoints_ideal[city],
            ms_perfect[city, :, locus],
            c=f"C{locus}",
            linestyle="-",
            alpha=0.3,
        )
        ax.scatter(
            timepoints[city][mask],
            freq[:, locus][mask],
            c=f"C{locus}",
            s=2,
            marker=markers[locus],
        )

    ax.set_title(f"City {city}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mutation frequency")
    ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# Having simulated the data, let's proceed to fitting the joint model for deconvolution and growth rate estimation.

# %%
loss_fn = dec.construct_total_loss(
    timepoints=timepoints,
    mutations=mutation_frequencies,
    variant_def=A,
    mask=masks,
    accept_vector=True,
)


# Generate an example point and use a multistart optimizer based on gradient descent
theta0 = 0.0 * params_true.to_vector()
theta_star = covvfit.numeric.jax_multistart_minimize(loss_fn, theta0).x

params_star = dec.JointLogisticGrowthParams.from_vector(theta_star, n_variants)

print(params_star)

# %% [markdown]
# Having fitted the model, let's compare the coefficients found with the ground-truth values:

# %%
fig, axs = subplots_from_axsize(
    1, 2, axsize=(2, 1.5), sharex=False, sharey=False, dpi=180
)
for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)

n_variants = params_true.n_variants

ax = axs[0]
ax.scatter(
    jnp.arange(n_variants - 1),
    params_true.relative_growths,
    marker=".",
    label="Ground truth",
    c="limegreen",
)
ax.scatter(
    jnp.arange(n_variants - 1),
    params_star.relative_growths,
    marker="x",
    label="Inferred",
    c="darkblue",
)
ax.set_xlabel("Growth rates (relative to 0th variant)")

ax = axs[1]
x_ax = jnp.arange(len(params_star.relative_offsets.ravel()))
ax.scatter(
    x_ax,
    params_true.relative_offsets.ravel(),
    marker=".",
    label="Ground truth",
    c="green",
)
ax.scatter(
    x_ax,
    params_star.relative_offsets.ravel(),
    marker="x",
    label="Inferred",
    c="darkblue",
)

ax.set_xlabel("Offsets (relative to 0th variant)")

# %% [markdown]
# Let's also compare how the abundance estimates change over time, according to the fitted model and to the ground-truth one:

# %%
ys_true = jnp.exp(params_true.predict_log_abundance(timepoints_ideal))
ys_star = jnp.exp(params_star.predict_log_abundance(timepoints_ideal))

fig, axs = subplots_from_axsize(
    1, n_cities, axsize=(2, 1.5), sharex=True, sharey=True, dpi=180
)

for city, ax in enumerate(axs.ravel()):
    for variant in range(n_variants):
        ax.plot(timepoints_ideal[city], ys_true[city, :, variant], c=f"C{variant}")
        ax.plot(
            timepoints_ideal[city],
            ys_star[city, :, variant],
            c=f"C{variant}",
            linestyle=":",
        )

    ax.set_title(f"City {city + 1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Variant abundances")
    ax.spines[["top", "right"]].set_visible(False)

# %%
