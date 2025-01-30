# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Quasilikelihood data analysis notebook
#
# This notebook shows how to estimate growth advantages by fiting the model within the quasimultinomial framework.

# +
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import yaml
from pathlib import Path

from covvfit import plot, preprocess
from covvfit import quasimultinomial as qm

plot_ts = plot.timeseries
# -


# ## Load and preprocess data
#
# We start by loading the data:

# +
DATA_DIR = Path("../data/main/")
DATA_PATH = DATA_DIR / "deconvolved.csv"
VAR_DATES_PATH = DATA_DIR / "var_dates.yaml"


data = pd.read_csv(DATA_PATH, sep="\t")
data.head()

# +
# Load the YAML file
with open(VAR_DATES_PATH, "r") as file:
    var_dates_data = yaml.safe_load(file)

# Access the var_dates data
var_dates = var_dates_data["var_dates"]


data_wide = data.pivot_table(
    index=["date", "location"], columns="variant", values="proportion", fill_value=0
).reset_index()
data_wide = data_wide.rename(columns={"date": "time", "location": "city"})

# Define the list with cities:
cities = list(data_wide["city"].unique())

## Set limit times for modeling

max_date = pd.to_datetime(data_wide["time"]).max()
delta_time = pd.Timedelta(days=240)
start_date = max_date - delta_time

# Print the data frame
data_wide.head()
# -

# Now we look at the variants in the data and define the variants of interest:

# +
# Convert the keys to datetime objects for comparison
var_dates_parsed = {
    pd.to_datetime(date): variants for date, variants in var_dates.items()
}


# Function to find the latest matching date in var_dates
def match_date(start_date):
    start_date = pd.to_datetime(start_date)
    closest_date = max(date for date in var_dates_parsed if date <= start_date)
    return closest_date, var_dates_parsed[closest_date]


variants_full = match_date(start_date + delta_time)[1]  # All the variants in this range

variants_investigated = [
    "KP.2",
    "KP.3",
    "XEC",
]  # Variants found in the data, which we focus on in this analysis
variants_other = [
    i for i in variants_full if i not in variants_investigated
]  # Variants not of interest
# -

# Apart from the variants of interest, we define the "other" variant, which artificially merges all the other variants into one. This allows us to model the data as a compositional time series, i.e., the sum of abundances of all "variants" is normalized to one.

# +
variants_effective = ["other"] + variants_investigated
data_full = preprocess.preprocess_df(
    data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
)

data_full["other"] = data_full[variants_other].sum(axis=1)
data_full[variants_effective] = data_full[variants_effective].div(
    data_full[variants_effective].sum(axis=1), axis=0
)

# +
ts_lst, ys_effective = preprocess.make_data_list(
    data_full, cities=cities, variants=variants_effective
)

# Scale the time for numerical stability
time_scaler = preprocess.TimeScaler()
ts_lst_scaled = time_scaler.fit_transform(ts_lst)
# -


# ## Fit the quasimultinomial model
#
# Now we fit the quasimultinomial model, which allows us to find the maximum quasilikelihood estimate of the parameters:

# +
# %%time

# no priors
loss = qm.construct_total_loss(
    ys=ys_effective,
    ts=ts_lst_scaled,
    average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
)

n_variants_effective = len(variants_effective)

# initial parameters
theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=n_variants_effective)

# Run the optimization routine
solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)

theta_star = solution.x  # The maximum quasilikelihood estimate

print(
    f"Relative growth advantages: \n",
    qm.get_relative_growths(theta_star, n_variants=n_variants_effective),
)
# -

# ## Confidence intervals of the growth advantages
#
# To obtain confidence intervals, we will take into account overdispersion. To do this, we need to compare the predictions with the observed values. Then, we can use overdispersion to attempt to correct the covariance matrix and obtain the confidence intervals.

# +
## compute fitted values
ys_fitted = qm.fitted_values(
    ts_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
)

## compute covariance matrix
covariance = qm.get_covariance(loss, theta_star)

overdispersion_tuple = qm.compute_overdispersion(
    observed=ys_effective,
    predicted=ys_fitted,
)

overdisp_fixed = overdispersion_tuple.overall

print(f"Overdispersion factor: {float(overdisp_fixed):.3f}.")
print("Note that values lower than 1 signify underdispersion.")

## scale covariance by overdisp
covariance_scaled = overdisp_fixed * covariance

## compute standard errors and confidence intervals of the estimates
standard_errors_estimates = qm.get_standard_errors(covariance_scaled)
confints_estimates = qm.get_confidence_intervals(
    theta_star, standard_errors_estimates, confidence_level=0.95
)


print("\n\nRelative growth advantages:")
for variant, m, l, u in zip(
    variants_effective[1:],
    qm.get_relative_growths(theta_star, n_variants=n_variants_effective),
    qm.get_relative_growths(confints_estimates[0], n_variants=n_variants_effective),
    qm.get_relative_growths(confints_estimates[1], n_variants=n_variants_effective),
):
    print(f"  {variant}: {float(m):.2f} ({float(l):.2f} – {float(u):.2f})")
# -


# We can propagate this uncertainty to the observed values. Let's generate confidence bands around the fitted lines and predict the future behaviour.

# +
ys_fitted_confint = qm.get_confidence_bands_logit(
    theta_star,
    n_variants=n_variants_effective,
    ts=ts_lst_scaled,
    covariance=covariance_scaled,
)


## compute predicted values and confidence bands
horizon = 60
ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)

ys_pred = qm.fitted_values(
    ts_pred_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
)
ys_pred_confint = qm.get_confidence_bands_logit(
    theta_star,
    n_variants=n_variants_effective,
    ts=ts_pred_lst_scaled,
    covariance=covariance_scaled,
)
# -

# ## Plot
#
# Finally, we plot the abundance data and the model predictions. Note that the 0th element in each array corresponds to the artificial "other" variant and we decided to plot only the explicitly defined variants.

# +
colors = [plot_ts.COLORS_COVSPECTRUM[var] for var in variants_investigated]


figure_spec = plot.arrange_into_grid(len(cities), axsize=(4, 1.5), dpi=350, wspace=1)


def plot_city(ax, i: int) -> None:
    def remove_0th(arr):
        """We don't plot the artificial 0th variant 'other'."""
        return arr[:, 1:]

    # Plot fits in observed and unobserved time intervals.
    plot_ts.plot_fit(ax, ts_lst[i], remove_0th(ys_fitted[i]), colors=colors)
    plot_ts.plot_fit(
        ax, ts_pred_lst[i], remove_0th(ys_pred[i]), colors=colors, linestyle="--"
    )

    plot_ts.plot_confidence_bands(
        ax,
        ts_lst[i],
        jax.tree.map(remove_0th, ys_fitted_confint[i]),
        colors=colors,
    )
    plot_ts.plot_confidence_bands(
        ax,
        ts_pred_lst[i],
        jax.tree.map(remove_0th, ys_pred_confint[i]),
        colors=colors,
    )

    # Plot the data points
    plot_ts.plot_data(ax, ts_lst[i], remove_0th(ys_effective[i]), colors=colors)

    # Plot the complements
    plot_ts.plot_complement(ax, ts_lst[i], remove_0th(ys_fitted[i]), alpha=0.3)
    plot_ts.plot_complement(
        ax, ts_pred_lst[i], remove_0th(ys_pred[i]), linestyle="--", alpha=0.3
    )

    # format axes and title
    def format_date(x, pos):
        return plot_ts.num_to_date(x, date_min=start_date)

    date_formatter = ticker.FuncFormatter(format_date)
    ax.xaxis.set_major_formatter(date_formatter)
    tick_positions = [0, 0.5, 1]
    tick_labels = ["0%", "50%", "100%"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_ylabel("Relative abundances")
    ax.set_title(cities[i])


figure_spec.map(plot_city, range(len(cities)))
# -

# ## Quasiposterior modelling
#
# Above we fitted the model using the maximum quasilikelihood approach, and then constructed approximate confidence intervals basing on the assumed covariance matrix structure and adjusting it by the estimated overdispersion factor.
# There exists also another method of quantifying uncertainty, which is based on generalized Bayesian paradigm, where the likelihood is replaced by the quasilikelihood.
#
# These methods of quantifying uncertainty do not have to be necessarily compatible and may reveal that the quasiposterior on growth advantage estimates is e.g., not symmetric.
#
# In fact, we attempt to use separate overdispersion for each city. Let's compare both approaches.

# +
import arviz as az
from numpyro.infer import MCMC, NUTS
from functools import partial


def sample_from_model(share_overdispersion: bool):
    if share_overdispersion:
        _overdispersion = overdispersion_tuple.overall
    else:
        _overdispersion = overdispersion_tuple.cities

    model = qm.construct_model(
        ys=ys_effective,
        ts=ts_lst_scaled,
        overdispersion=_overdispersion,
        sigma_offset=100.0,
    )

    mcmc = MCMC(NUTS(model), num_chains=4, num_samples=2000, num_warmup=2000)
    mcmc.run(jax.random.PRNGKey(42))
    return mcmc


mcmc_shared = sample_from_model(share_overdispersion=True)
mcmc_indivi = sample_from_model(share_overdispersion=False)
# -

# Before we proceed with the analysis of the quasiposteriors, let's see if we can trust the obtained samples.
#
# **Shared overdispersion**

idata = az.from_numpyro(mcmc_shared)
az.summary(idata, filter_vars="regex", var_names="^r.*")

az.plot_trace(idata, filter_vars="regex", var_names="^r.*")
plt.tight_layout()
plt.show()

# **Individual overdispersion parameters**

idata = az.from_numpyro(mcmc_indivi)
az.summary(idata, filter_vars="regex", var_names="^r.*")

az.plot_trace(idata, filter_vars="regex", var_names="^r.*")
plt.tight_layout()
plt.show()

# If we do not see sampling problems, we can try to understand the quasiposterior distributions.
#
# Let's compare both quasiposteriors additionally with the confidence intervals.

# +
from subplots_from_axsize import subplots_from_axsize


def plot_posterior(ax, i, mcmc):
    max_quasilikelihood = qm.get_relative_growths(
        theta_star, n_variants=n_variants_effective
    )
    lower = qm.get_relative_growths(
        confints_estimates[0], n_variants=n_variants_effective
    )
    upper = qm.get_relative_growths(
        confints_estimates[1], n_variants=n_variants_effective
    )

    # Plot maximum quasilikelihood and confidence interval bands
    ax.axvline(max_quasilikelihood[i], c="k")
    ax.axvspan(lower[i], upper[i], alpha=0.3, facecolor="k", edgecolor=None)

    # Plot quasiposterior samples using a histogram
    samples = mcmc.get_samples()["relative_growths"][:, i]
    ax.hist(samples, bins=40, color="maroon")

    # Plot the credible interval calculated using quantiles
    credibility = 0.95
    _a = (1 - credibility) / 2.0
    ax.axvline(jnp.quantile(samples, q=_a), c="maroon", linestyle=":")
    ax.axvline(jnp.quantile(samples, q=1.0 - _a), c="maroon", linestyle=":")

    # Apply some styling
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_yticks([])


fig, axs = subplots_from_axsize(
    ncols=n_variants_effective - 1,
    axsize=(2, 0.8),
    nrows=2,
    sharex="col",
    hspace=0.25,
    dpi=400,
)

for i in range(n_variants_effective - 1):
    plot_posterior(axs[0, i], i, mcmc_shared)
    plot_posterior(axs[1, i], i, mcmc_indivi)

axs[0, 0].set_ylabel("Shared")
axs[1, 0].set_ylabel("Individual")

for i, variant in enumerate(variants_effective[1:]):
    axs[0, i].set_title(f"Advantage of {variant}")


# -

# We see two things:
#
#   - Quasiposterior employing shared overdispersion gives similar results to the ones obtained with confidence intervals.
#   - When we use individual overdispersion factors (one per city), we see a discrepancy.
#
# Let's compare the predictive plots between two quasiposteriors and the confidence bands obtained earlier.


# +
def plot_predictions(
    ax,
    i: int,
    *,
    fitted_line,
    fitted_lower,
    fitted_upper,
    predicted_line,
    predicted_lower,
    predicted_upper,
) -> None:
    def remove_0th(arr):
        """We don't plot the artificial 0th variant 'other'."""
        return arr[:, 1:]

    # Plot fits in observed and unobserved time intervals.
    plot_ts.plot_fit(ax, ts_lst[i], remove_0th(fitted_line[i]), colors=colors)
    plot_ts.plot_fit(
        ax, ts_pred_lst[i], remove_0th(predicted_line[i]), colors=colors, linestyle="--"
    )
    plot_ts.plot_confidence_bands(
        ax,
        ts_lst[i],
        (remove_0th(fitted_lower[i]), remove_0th(fitted_upper[i])),
        colors=colors,
    )
    plot_ts.plot_confidence_bands(
        ax,
        ts_pred_lst[i],
        (remove_0th(predicted_lower[i]), remove_0th(predicted_upper[i])),
        colors=colors,
    )

    # Plot the data points
    plot_ts.plot_data(ax, ts_lst[i], remove_0th(ys_effective[i]), colors=colors)

    # Plot the complements
    plot_ts.plot_complement(ax, ts_lst[i], remove_0th(fitted_line[i]), alpha=0.3)
    plot_ts.plot_complement(
        ax, ts_pred_lst[i], remove_0th(predicted_line[i]), linestyle="--", alpha=0.3
    )

    # format axes and title
    def format_date(x, pos):
        return plot_ts.num_to_date(x, date_min=start_date)

    date_formatter = ticker.FuncFormatter(format_date)
    ax.xaxis.set_major_formatter(date_formatter)
    tick_positions = [0, 0.5, 1]
    tick_labels = ["0%", "50%", "100%"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)


fig, axs = subplots_from_axsize(
    ncols=3,
    axsize=(2, 0.8),
    nrows=len(cities),
    sharex=True,
    sharey=True,
    hspace=0.4,
    dpi=400,
)

for i, city in enumerate(cities):
    axs[i, 0].set_ylabel(city)

for ax, name in zip(
    axs[0, :], ["Confidence", "Credible (shared)", "Credible (individual)"]
):
    ax.set_title(name)


# Plot the quasilikelihood fits
for i, ax in enumerate(axs[:, 0]):
    plot_predictions(
        ax,
        i,
        fitted_line=ys_fitted,
        fitted_lower=[y.lower for y in ys_fitted_confint],
        fitted_upper=[y.upper for y in ys_fitted_confint],
        predicted_line=ys_pred,
        predicted_lower=[y.lower for y in ys_pred_confint],
        predicted_upper=[y.upper for y in ys_pred_confint],
    )


# Plot the quasiposterior with shared MCMC


def obtain_predictions(mcmc, _a=0.05):
    def get_fit(sample):
        theta = qm.construct_theta(
            relative_growths=sample["relative_growths"],
            relative_midpoints=sample["relative_offsets"],
        )

        y_fit = qm.fitted_values(
            ts_lst_scaled, theta, cities=cities, n_variants=n_variants_effective
        )
        y_pre = qm.fitted_values(
            ts_pred_lst_scaled, theta, cities=cities, n_variants=n_variants_effective
        )
        return y_fit, y_pre

    def get_line(ys):
        return jnp.mean(ys, axis=0)

    def get_lower(ys):
        return jnp.quantile(ys, q=_a / 2, axis=0)

    def get_upper(ys):
        return jnp.quantile(ys, q=1 - _a / 2, axis=0)

    # Apply some thinning for computational speedup
    samples = jax.tree.map(lambda x: x[::10, ...], mcmc.get_samples())

    fits, preds = jax.vmap(get_fit)(samples)
    return dict(
        fitted_line=jax.tree.map(get_line, fits),
        fitted_lower=jax.tree.map(get_lower, fits),
        fitted_upper=jax.tree.map(get_upper, fits),
        predicted_line=jax.tree.map(get_line, preds),
        predicted_lower=jax.tree.map(get_lower, preds),
        predicted_upper=jax.tree.map(get_upper, preds),
    )


for i, ax in enumerate(axs[:, 1]):
    plot_predictions(ax, i, **obtain_predictions(mcmc_shared))

# Plot individual overdispersions
for i, ax in enumerate(axs[:, 2]):
    plot_predictions(ax, i, **obtain_predictions(mcmc_indivi))
