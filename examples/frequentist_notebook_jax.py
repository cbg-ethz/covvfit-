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

# +
import jax
import jax.numpy as jnp

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from scipy.stats import norm

import yaml

import covvfit._preprocess_abundances as prec
import covvfit.plotting._timeseries as plot_ts

from covvfit import quasimultinomial as qm

import numpyro

# -


# # Load and preprocess data

# +
DATA_PATH = "../../LolliPop/lollipop_covvfit/deconvolved.csv"
VAR_DATES_PATH = "../../LolliPop/lollipop_covvfit/var_dates.yaml"

DATA_PATH = "../new_data/deconvolved.csv"
VAR_DATES_PATH = "../new_data/var_dates.yaml"


data = pd.read_csv(DATA_PATH, sep="\t")
data.head()


# Load the YAML file
with open(VAR_DATES_PATH, "r") as file:
    var_dates_data = yaml.safe_load(file)

# Access the var_dates data
var_dates = var_dates_data["var_dates"]
# -

data_wide = data.pivot_table(
    index=["date", "location"], columns="variant", values="proportion", fill_value=0
).reset_index()
data_wide = data_wide.rename(columns={"date": "time", "location": "city"})
data_wide.head()

# +
## Set limit times for modeling

max_date = pd.to_datetime(data_wide["time"]).max()
delta_time = pd.Timedelta(days=240)
start_date = max_date - delta_time


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


variants_full = match_date(start_date + delta_time)[1]

variants = ["KP.2", "KP.3", "XEC"]

variants_other = [i for i in variants_full if i not in variants]
# -

cities = list(data_wide["city"].unique())

variants2 = ["other"] + variants
data2 = prec.preprocess_df(
    data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
)

# +
data2["other"] = data2[variants_other].sum(axis=1)
data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

ts_lst, ys_lst = prec.make_data_list(data2, cities, variants2)
ts_lst, ys_lst2 = prec.make_data_list(data2, cities, variants)

t_max = max([x.max() for x in ts_lst])
t_min = min([x.min() for x in ts_lst])

ts_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_lst]
# -


# # fit in jax

# +
# %%time

# Recall that the input should be (n_timepoints, n_variants)
# TODO(Pawel, David): Resolve Issue https://github.com/cbg-ethz/covvfit/issues/24
observed_data = [y.T for y in ys_lst]


# no priors
loss = qm.construct_total_loss(
    ys=observed_data,
    ts=ts_lst_scaled,
    average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
)

# initial parameters
theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants2))

# Run the optimization routine
solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)
# -

# ## Make fitted values and confidence intervals

# +
## compute fitted values
fitted_values = qm.fitted_values(
    ts_lst_scaled, theta=solution.x, cities=cities, n_variants=len(variants2)
)

# ... and because of https://github.com/cbg-ethz/covvfit/issues/24
# we need to transpose again
y_fit_lst = [y.T[1:] for y in fitted_values]

## compute covariance matrix
covariance = qm.get_covariance(loss, solution.x)

overdispersion_tuple = qm.compute_overdispersion(
    observed=observed_data,
    predicted=fitted_values,
)

overdisp_fixed = overdispersion_tuple.overall


# +
## scale covariance by overdisp
covariance_scaled = overdisp_fixed * covariance

## compute standard errors and confidence intervals of the estimates
standard_errors_estimates = qm.get_standard_errors(covariance_scaled)
confints_estimates = qm.get_confidence_intervals(solution.x, standard_errors_estimates)

## compute confidence intervals of the fitted values on the logit scale and back transform
y_fit_lst_confint = qm.get_confidence_bands_logit(
    solution.x, len(variants2), ts_lst_scaled, covariance_scaled
)

## compute predicted values and confidence bands
horizon = 60
ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
ts_pred_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_pred_lst]

y_pred_lst = qm.fitted_values(
    ts_pred_lst_scaled, theta=solution.x, cities=cities, n_variants=len(variants2)
)
# ... and because of https://github.com/cbg-ethz/covvfit/issues/24
# we need to transpose again
y_pred_lst = [y.T[1:] for y in y_pred_lst]

y_pred_lst_confint = qm.get_confidence_bands_logit(
    solution.x, len(variants2), ts_pred_lst_scaled, covariance_scaled
)


# -

# ## Plotting functions

plot_fit = plot_ts.plot_fit
plot_complement = plot_ts.plot_complement
plot_data = plot_ts.plot_data
plot_confidence_bands = plot_ts.plot_confidence_bands

# ## Plot

# +
colors_covsp = plot_ts.colors_covsp
colors = [colors_covsp[var] for var in variants]
fig, axes_tot = plt.subplots(4, 2, figsize=(15, 10))
axes_flat = axes_tot.flatten()

for i, city in enumerate(cities):
    ax = axes_flat[i]
    # plot fitted and predicted values
    plot_fit(ax, ts_lst[i], y_fit_lst[i], variants, colors)
    plot_fit(ax, ts_pred_lst[i], y_pred_lst[i], variants, colors, linetype="--")

    #     # plot 1-fitted and predicted values
    plot_complement(ax, ts_lst[i], y_fit_lst[i], variants)
    #     plot_complement(ax, ts_pred_lst[i], y_pred_lst[i], variants, linetype="--")
    # plot raw deconvolved values
    plot_data(ax, ts_lst[i], ys_lst2[i], variants, colors)
    # make confidence bands and plot them
    conf_bands = y_fit_lst_confint[i]
    plot_confidence_bands(
        ax,
        ts_lst[i],
        {"lower": conf_bands[0], "upper": conf_bands[1]},
        variants,
        colors,
    )

    pred_bands = y_pred_lst_confint[i]
    plot_confidence_bands(
        ax,
        ts_pred_lst[i],
        {"lower": pred_bands[0], "upper": pred_bands[1]},
        variants,
        colors,
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
    ax.set_ylabel("relative abundances")
    ax.set_title(city)

fig.tight_layout()
fig.show()
# -
