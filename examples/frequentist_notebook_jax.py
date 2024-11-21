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

from covvfit import plot, preprocess
from covvfit import quasimultinomial as qm

plot_ts = plot.timeseries
# -


# ## Load and preprocess data
#
# We start by loading the data:

# +
_dir_switch = False  # Change this to True or False, depending on the laptop you are on
if _dir_switch:
    DATA_PATH = "../../LolliPop/lollipop_covvfit/deconvolved.csv"
    VAR_DATES_PATH = "../../LolliPop/lollipop_covvfit/var_dates.yaml"
else:
    DATA_PATH = "../new_data/deconvolved.csv"
    VAR_DATES_PATH = "../new_data/var_dates.yaml"


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
