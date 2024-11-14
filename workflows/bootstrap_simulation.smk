# ===============================================
# == Simulations using bootstrap to construct  ==
# ==        the confidence intervals           ==
# ===============================================
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from covvfit import freq_jax as fj
from covvfit.simulation import logistic

matplotlib.use("Agg")  # Use the Agg backend for compatibility with Snakemake
workdir: "generated/boostrap_simulation/"   # Set the directory for the generated output

N_REPETITIONS: int = 50  # Number of repetitions to calculate the coverage
N_BOOTSTRAPS: int = 100  # Number of bootstrap samples

SETTINGS = {
    "simple": logistic.SimulationSettings(
        n_cities=2,
        n_variants=3,
        growth_rates=jnp.asarray([0.0, 1.0, 3.0]),
        midpoints=jnp.asarray([
            [-1.0, 0.5, 0.90],
            [-1.2, 0.4, 0.95],
        ]),
        n_observations=jnp.asarray([30, 40]),
        n_multinomial=jnp.asarray([20, 10]), 
    )
}

rule all:
    input:
        example_plots = expand("{simulation}/plot_example_data.pdf", simulation=SETTINGS),
        other = expand("{simulation}/bootstrap/plots/{repetition}/pairwise_differences.pdf", simulation=SETTINGS, repetition=range(N_REPETITIONS)),
        coverage = expand("{simulation}/{method}/coverage_plot.pdf", simulation=SETTINGS, method=["bootstrap"]),


rule assign_desired_coverage:
    output: "{simulation}/aux/desired_coverage.npy"
    run:
        coverages = np.arange(0.1, 0.91, 0.1)
        np.save(str(output), coverages)


rule get_coverage:
    input:
        intervals = expand("{simulation}/{method}/uncertainty_intervals/{repetition}.npy", repetition=range(N_REPETITIONS), allow_missing=True),
        desired_coverage = "{simulation}/aux/desired_coverage.npy",
    output: "{simulation}/{method}/coverage_plot.pdf"
    run:
        settings = SETTINGS[wildcards.simulation]

        true_growths = settings.growth_rates

        CIs = [np.load(fp) for fp in input.intervals]
        desired_coverage = np.load(input.desired_coverage)
        
        assert CIs[0].shape[0] == desired_coverage.shape[0]
        assert len(CIs) == N_REPETITIONS, f"Mismatch. N_REPETITIONS: {N_REPETITIONS}, CIs: {len(CIs)}"

        # Only the relative growth rates matter, so we plot them
        fig, axs = plt.subplots(ncols=settings.n_variants, nrows=settings.n_variants)

        for i in range(settings.n_variants):
            for j in range(settings.n_variants):
                ax = axs[i, j]
                if i >= j:
                    ax.set_axis_off()
                else:
                    # The true difference
                    val_true = true_growths[i] - true_growths[j]

                    # Construct empirical coverages
                    buckets = np.zeros_like(desired_coverage, dtype=float)
                    for ind, _ in enumerate(desired_coverage):
                        for rep in CIs:
                            # rep has shape (intervals, i, j, 2)
                            low = rep[ind, i, j, 0]
                            high = rep[ind, i, j, 1]
                            if low <= val_true <= high:
                                buckets[ind] += 1

                    # Now plot the coverages
                    ax.plot(desired_coverage, desired_coverage, linestyle="--", linewidth=0.2, alpha=0.5, c="k")
                    ax.scatter(desired_coverage, buckets / N_REPETITIONS, c="maroon", s=8)

                    ax.axis("equal")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)                    
                    ax.set_title(f"Growth rates diff. {i}–{j}")
                    ax.set_xlabel("Desired coverage")
                    ax.set_ylabel("Empirical coverage")
                    ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(output))




rule get_bootstrap_uncertainty_intervals:
    input: 
        assembled = "{simulation}/bootstrap/samples/assembled/{repetition}.npy",
        desired = "{simulation}/aux/desired_coverage.npy",
    output: "{simulation}/bootstrap/uncertainty_intervals/{repetition}.npy"
    run:
        settings = SETTINGS[wildcards.simulation]

        desired_coverages = np.load(input.desired)

        n_intervals = desired_coverages.shape[0]
        n_variants = settings.n_variants

        samples = np.load(input.assembled)

        # We need to create an array of shape (n_intervals, n_variants, n_variants, 2)
        # such that CI[coverage, variant1, variant2, 0] is the
        # lower end of the uncertainty interval with required coverage for the growth rate difference
        # between variant1 and variant2.
        # Similarly, CI[coverage, variant1, variant2, 1] is the upper end of the uncertainty interval 
        CI = np.zeros((n_intervals, n_variants, n_variants, 2))

        for index_coverage, coverage in enumerate(desired_coverages):
            for variant1 in range(n_variants):
                for variant2 in range(n_variants):
                    if variant1 == variant2:
                        continue
                    else:
                        xs = samples[:, variant1] - samples[:, variant2]
                        alpha = 1 - coverage
                        CI[index_coverage, variant1, variant2, 0] = np.quantile(xs, q=0.5 * alpha)
                        CI[index_coverage, variant1, variant2, 1] = np.quantile(xs, q=1 - 0.5 * alpha)
        
        np.save(str(output), CI)


rule plot_bootstrap_samples:
    input: 
        assembled = "{simulation}/bootstrap/samples/assembled/{repetition}.npy"
    output: "{simulation}/bootstrap/plots/{repetition}/pairwise_differences.pdf"
    run:
        settings = SETTINGS[wildcards.simulation]

        true_growths = settings.growth_rates
        samples_growths = np.load(input.assembled)

        # Only the relative growth rates matter.
        fig, axs = plt.subplots(ncols=settings.n_variants, nrows=settings.n_variants, sharex=False, sharey=False)

        for i in range(settings.n_variants):
            for j in range(settings.n_variants):
                ax = axs[i, j]
                if i >= j:
                    ax.set_axis_off()
                else:
                    val_true = true_growths[i] - true_growths[j]
                    val_samples = samples_growths[:, i] - samples_growths[:, j]
                    ax.axvline(val_true, color="black")
                    ax.hist(val_samples, bins=10, density=True)
                    ax.set_xlabel(f"Growth rates diff. {i}–{j}")
                    ax.set_yticks([])
                    ax.spines[["top", "right", "left"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(output))

        
rule assemble_bootstrap_samples:
    input:
        bootstraps = expand("{simulation}/bootstrap/samples/raw/{repetition}/{bootstrap}.npy", bootstrap=range(N_BOOTSTRAPS), allow_missing=True)
    output:
        assembled = "{simulation}/bootstrap/samples/assembled/{repetition}.npy"
    run:
        bootstrap_samples = []
        for fp in input.bootstraps:
            inferred_growths = np.load(fp)
            bootstrap_samples.append(inferred_growths)

        # Shape (n_boostrap, n_variants)
        np.save(output.assembled, np.stack(bootstrap_samples))


rule fit_to_bootstrapped_sample:
    input:
        data = "{simulation}/observed_data/{repetition}.npz"
    output: "{simulation}/bootstrap/samples/raw/{repetition}/{bootstrap}.npy"
    run:
        arrays = np.load(input.data)
        settings = SETTINGS[wildcards.simulation]
        boostrap_index = int(wildcards.bootstrap)
        key = jax.random.PRNGKey(boostrap_index)

        ts_all = []
        ys_all = []

        for city_index in range(settings.n_cities):
            subkey = jax.random.fold_in(key, city_index)
            ts_obs = arrays[f"time{city_index}"]
            ys_obs = arrays[f"abun{city_index}"]

            # Get the index
            n_points = len(ts_obs)
            index = jax.random.randint(subkey, minval=0, maxval=n_points, shape=(n_points,))
            index = jnp.sort(index)  # Sort the index. Technically, it's not necessary

            # Obtain the bootstrapped sample
            ts_obs = ts_obs[index]
            ys_obs = ys_obs[index, :]

            ts_all.append(ts_obs)
            ys_all.append(ys_obs)


        # Note that we don't use any priors: we optimize just the (quasi-)likelihood
        loss = fj.construct_total_loss(ys=ys_all, ts=ts_all, accept_theta=True)
        theta0 = fj.construct_theta0(n_cities=settings.n_cities, n_variants=settings.n_variants)

        solution = fj.jax_multistart_minimize(
            loss,
            theta0,
        )

        # Obtain the growths vector of shape (n_variants,)
        # Note that the first entry has zero growth rate and only the differences have actual meaning
        inferred_growths = jnp.concatenate((jnp.zeros(1), fj.get_relative_growths(solution.x, n_variants=settings.n_variants)))

        np.save(str(output), inferred_growths)




rule plot_data:
    input:
        data0 = "{simulation}/observed_data/0.npz"
    output: "{simulation}/plot_example_data.pdf"
    run:
        settings = SETTINGS[wildcards.simulation]

        arrays = np.load(input.data0)

        fig, axs = plt.subplots(1, settings.n_cities, sharex=True, sharey=True)
        
        for city_index, ax in enumerate(axs.ravel()):
            ts_obs = arrays[f"time{city_index}"]
            ys_obs = arrays[f"abun{city_index}"]

            ts_eval = jnp.linspace(settings.time0, settings.time1, 51)
            ts_eval, ys_eval = settings.calculate_abundances_one_city(city_index=city_index, ts=ts_eval)

            for variant_index in range(settings.n_variants):
                color = f"C{variant_index}"
                ax.plot(ts_eval, ys_eval[:, variant_index], c=color)
                ax.scatter(ts_obs, ys_obs[:, variant_index], c=color, s=3)

        # Apply some styling
        for ax in axs.ravel():
            ax.set_xlabel("Time")
            ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(str(output))



rule generate_data:
    output:
        data = "{simulation}/observed_data/{seed}.npz"
    run:
        seed = int(wildcards.seed)
        settings: logistic.SimulationSettings = SETTINGS[wildcards.simulation]
        
        samples = settings.generate_sample_all_cities(jax.random.PRNGKey(seed))
        dct = {}
        for key, (t, y) in samples.items():
            dct[f"time{key}"] = t
            dct[f"abun{key}"] = y

        np.savez(output.data, **dct)
