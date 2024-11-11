# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python (covvfit)
#     language: python
#     name: covvfit
# ---

# %%
import pandas as pd
import pymc as pm

import numpy as np

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from scipy.stats import norm

import yaml

import covvfit._frequentist as freq
import covvfit._preprocess_abundances as prec
import covvfit.plotting._timeseries as plot_ts


# %% [markdown]
# # Load and preprocess data

# %%
DATA_PATH = "../../LolliPop/lollipop_test_noisy/deconvolved.csv"
data = pd.read_csv(DATA_PATH, sep="\t")
data.head()

# %%
data_wide = data.pivot_table(
    index=["date", "location"], columns="variant", values="proportion", fill_value=0
).reset_index()
data_wide = data_wide.rename(columns={"date": "time", "location": "city"})
data_wide.head()

# %%
## Set limit times for modeling

max_date = pd.to_datetime(data_wide["time"]).max()
delta_time = pd.Timedelta(days=240)
start_date = max_date - delta_time

# %%
# Path to the YAML file
var_dates_yaml = "../../LolliPop/lollipop_test_noisy/var_dates.yaml"

# Load the YAML file
with open(var_dates_yaml, "r") as file:
    var_dates_data = yaml.safe_load(file)

# Access the var_dates data
var_dates = var_dates_data["var_dates"]


# %%
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

# %%
cities = list(data_wide["city"].unique())

# %%
variants2 = ["other"] + variants
data2 = prec.preprocess_df(
    data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
)

# %%
data2["other"] = data2[variants_other].sum(axis=1)
data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

ts_lst, ys_lst = prec.make_data_list(data2, cities, variants2)
ts_lst, ys_lst2 = prec.make_data_list(data2, cities, variants)

t_max = max([x.max() for x in ts_lst])
t_min = min([x.min() for x in ts_lst])

ts_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_lst]


# %% [markdown]
# # Fit model

# %%
## This model takes into account the complement of the variants to be monitored, and sets its fitness to zero
## However, due to the pm.math.concatenate operation, we cannot use it for finding the hessian


def create_model_fixed2(
    ts_lst,
    ys_lst,
    n=1.0,
    coords={
        "cities": [],
        "variants": [],
    },
    prior_params={
        "mu_midpoint": 0.0,
        "sigma_midpoint": 100.0,
        "mu_rate": 10.0,
        "sigma_rate": 100.0,
    },
):
    """function to create a fixed effect model with varying intercepts and one rate vector"""
    with pm.Model(coords=coords) as model:
        # define priors of rate and midpoint parameters
        midpoint_var = pm.Normal(
            "midpoint",
            prior_params["mu_midpoint"],
            sigma=prior_params["sigma_midpoint"],
            dims=["cities", "variants"],
        )
        rate_var = pm.Normal(
            "rate",
            prior_params["mu_rate"],
            sigma=prior_params["sigma_rate"],
            dims="variants",
        )

        # give logits pred values
        ys_logits = [
            (
                pm.math.concatenate([[0], rate_var])[:, None] * ts_lst[i]
                + pm.math.concatenate([[0], midpoint_var[i, :]])[:, None]
            )
            for i, city in enumerate(coords["cities"])
        ]

        # give predicted log relative abundances of variants
        ys_logp = [pm.math.log_softmax(x, axis=0) for x in ys_logits]

        # make Multinom/n likelihood
        def log_likelihood(y, logp, n):
            return n * pm.math.sum(y * logp, axis=0)

        # compute likelihood of data
        [
            pm.DensityDist(
                f"ys_noisy_{city}",
                ys_logp[i],
                n,
                logp=log_likelihood,
                observed=ys_lst[i],
            )
            for i, city in enumerate(coords["cities"])
        ]

    return model


# %%
# %%time
with create_model_fixed2(
    ts_lst_scaled,
    ys_lst,
    coords={
        "cities": cities,
        "variants": variants,
    },
):
    model_map_fixed = pm.find_MAP(maxeval=50000, seed=12313)


# %%
model_map_fixed

# %% [markdown]
# ## Make uncertainty

# %%
model_map_fixed

# %%
# %%time
with create_model_fixed2(
    ts_lst,
    ys_lst,
    coords={
        "cities": cities,
        "variants": variants,
    },
):
    model_hessian_fixed = pm.find_hessian(model_map_fixed)

# %%
model_map_fixed

# %% jupyter={"outputs_hidden": true}
model_map_fixed

# %%
y_fit_lst = freq.fitted_values(ts_lst_scaled, model_map_fixed, cities)


# %%
# make predicted values
horizon = 60
ts_pred_lst = [np.arange(horizon + 1) + tt.max() for tt in ts_lst]
ts_pred_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_pred_lst]
y_pred_lst = freq.fitted_values(ts_pred_lst_scaled, model_map_fixed, cities)

# %%
pearson_r_lst, overdisp_list, overdisp_fixed = freq.compute_overdispersion(
    ys_lst2, y_fit_lst, cities
)
(
    fitness_diff,
    fitness_diff_se,
    fitness_diff_lower,
    fitness_diff_upper,
) = freq.make_fitness_confints(
    model_map_fixed["rate"], model_hessian_fixed, overdisp_fixed, g=7.0
)

p_variants = len(variants)
p_params = model_hessian_fixed.shape[0]
model_hessian_inv = np.linalg.inv(model_hessian_fixed)

# %% [markdown]
# # Prepare fitted values and intervals

# %%
# model_map_fixed["rate"] = model_map_fixed["rate"] / t_max
# model_hessian_fixed[-len(variants):,-len(variants):] = model_hessian_fixed[-len(variants):,-len(variants):]*t_max

# %% [markdown]
# ## Plotting functions

# %%
make_confidence_bands = freq.make_confidence_bands

# %%
plot_fit = plot_ts.plot_fit
plot_complement = plot_ts.plot_complement
plot_data = plot_ts.plot_data
plot_confidence_bands = plot_ts.plot_confidence_bands

# %% [markdown]
# ## Plot

# %%
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
    plot_complement(ax, ts_pred_lst[i], y_pred_lst[i], variants, linetype="--")
    #     # plot raw deconvolved values
    plot_data(ax, ts_lst[i], ys_lst2[i], variants, colors)
    #     # make confidence bands and plot them
    conf_bands = make_confidence_bands(
        ts_lst_scaled[i],
        y_fit_lst[i],
        model_hessian_inv,
        i,
        model_map_fixed["rate"],
        model_map_fixed["midpoint"][i, :],
        overdisp_list[i],
    )
    plot_confidence_bands(ax, ts_lst[i], conf_bands, variants, colors)

    conf_bands_pred = make_confidence_bands(
        ts_pred_lst_scaled[i],
        y_pred_lst[i],
        model_hessian_inv,
        i,
        model_map_fixed["rate"],
        model_map_fixed["midpoint"][i, :],
        overdisp_list[i],
    )
    plot_confidence_bands(
        ax, ts_pred_lst[i], conf_bands_pred, variants, colors, alpha=0.1
    )

    #     # format axes and title
    #     date_formatter = ticker.FuncFormatter(plot_ts.num_to_date)
    #     ax.xaxis.set_major_formatter(date_formatter)
    #     tick_positions = [0, 0.5, 1]
    #     tick_labels = ["0%", "50%", "100%"]
    #     ax.set_yticks(tick_positions)
    #     ax.set_yticklabels(tick_labels)
    #     ax.set_ylabel("relative abundances")
    ax.set_title(city)

fig.tight_layout()
fig.show()
