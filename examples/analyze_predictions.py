# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: jax
#     language: python
#     name: jax
# ---

# # Prediction analysis notebook
#
# This notebook is to analyze accuracy of prediction.

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

import numpy as np

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
# %matplotlib inline
import matplotlib.pyplot as plt

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 300  # Adjust to 150, 200, or more for higher resolution

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
cities = [
    "Zürich (ZH)",
    "Altenrhein (SG)",
    "Laupen (BE)",
    "Lugano (TI)",
    "Chur (GR)",
    "Genève (GE)",
]

variants_full = [
    "BA.5",
    "BA.2.75",
    "BA.2.86",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "JN.1",
]

variants_investigated = [
    "BA.2.75",
    "BA.2.86",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "JN.1",
]

variants_other = [
    i for i in variants_full if i not in variants_investigated
]  # Variants not of interest

variants_evaluated = [
    "EG.5",
    "JN.1",
    "BA.2.86",
]  # variants for which we will evaluate predictions

# date parameters for the range of the simulation

max_date = pd.to_datetime("2024-01-01")
start_date = pd.to_datetime("2022-10-21")
max_dates = pd.date_range(start="2023-08-01", end="2024-01-01", freq="D")
max_horizon = 45

# smoothing parameters:
sigma = 15
smoothing_method = "median"
max_smooth_date = max_date + pd.to_timedelta(3 * sigma + max_horizon, "D")


# -

# Apart from the variants of interest, we define the "other" variant, which artificially merges all the other variants into one. This allows us to model the data as a compositional time series, i.e., the sum of abundances of all "variants" is normalized to one.

# ### functions to be reran at each iteration
#
# We will define some functions to be reran at each iteration of the experiment. They will prepare the data from a date to another and fit the model. Then we will output some predictions.

# +


def prepare_data(data_wide, start_date, max_date):
    """function to prepare data with the right timespan"""

    data_wide2 = data_wide.copy()

    ## Set limit times for modeling
    data_wide2["time"] = pd.to_datetime(data_wide2["time"])

    data_wide2 = data_wide2[data_wide2["time"] >= start_date]
    data_wide2 = data_wide2[data_wide2["time"] <= max_date]

    variants_effective = ["other"] + variants_investigated

    data_full = preprocess.preprocess_df(
        data_wide2, cities, variants_full, date_min=start_date, zero_date=start_date
    )

    data_full["other"] = data_full[variants_other].sum(axis=1)
    data_full[variants_effective] = data_full[variants_effective].div(
        data_full[variants_effective].sum(axis=1), axis=0
    )

    ts_lst, ys_effective = preprocess.make_data_list(
        data_full, cities=cities, variants=variants_effective
    )

    # Scale the time for numerical stability
    time_scaler = preprocess.TimeScaler()
    ts_lst_scaled = time_scaler.fit_transform(ts_lst)

    return {
        "data_wide2": data_wide2,
        "ts_lst": ts_lst,
        "ys_effective": ys_effective,
        "ts_lst_scaled": ts_lst_scaled,
        "time_scaler": time_scaler,
    }


# -

# function to make inference and output preductions


def inference(ys_effective, ts_lst_scaled, ts_lst, time_scaler, horizon=7):
    # no priors
    loss = qm.construct_total_loss(
        ys=ys_effective,
        ts=ts_lst_scaled,
        average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
    )

    variants_effective = ["other"] + variants_investigated
    n_variants_effective = len(variants_effective)
    # initial parameters
    theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=n_variants_effective)

    # Run the optimization routine
    solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)
    theta_star = solution.x  # The maximum quasilikelihood estimate

    ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
    ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)

    ys_pred = qm.fitted_values(
        ts_pred_lst_scaled,
        theta=theta_star,
        cities=cities,
        n_variants=n_variants_effective,
    )

    return {
        "ys_pred": ys_pred,
        "ts_pred_lst": ts_pred_lst,
        "solution": solution,
    }


# ## Prepare smoothed data for comparison
#
# We will smooth wastewater deconvolved data to average out the sampling noise. This will give us a better baseline to compare our predictions to.


# +
def gaussian_kernel(t, ts, sigma):
    """Compute Gaussian weights for a given t"""
    return np.exp(-0.5 * ((ts - t) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def smooth_value(t, ts, ys, sigma):
    """Compute the smoothed value at time t using a Gaussian-weighted average."""
    weights = gaussian_kernel(t, ts, sigma)
    weights /= weights.sum()  # Normalize weights
    return np.dot(weights, ys)


def smooth_value_median(t, ts, ys, sigma):
    """Compute the smoothed value at time t using a Gaussian-weighted median for each dimension in ys."""
    weights = gaussian_kernel(t, ts, sigma)
    sorted_indices = np.argsort(ys, axis=0)
    ys_sorted = np.take_along_axis(ys, sorted_indices, axis=0)
    weights_sorted = np.take_along_axis(weights[:, np.newaxis], sorted_indices, axis=0)
    cumsum_weights = np.cumsum(weights_sorted, axis=0)
    median_indices = np.argmax(cumsum_weights >= 0.5 * cumsum_weights[-1], axis=0)
    return np.take_along_axis(ys_sorted, median_indices[np.newaxis, :], axis=0)[0]


def smooth_array(tt, ts, ys, sigma, method="mean"):
    """Compute smoothed values for an array of t values."""
    if method == "mean":
        return np.array([smooth_value(t, ts, ys, sigma) for t in tt])
    if method == "median":
        return np.array([smooth_value_median(t, ts, ys, sigma) for t in tt])


# -

# smooth the data. make sure to include data away from the time boundaries of the experiment to avoid common biases from kernel smoothing

# +
prep_dat_full = prepare_data(data_wide, start_date, max_smooth_date)

ys_lst_full = prep_dat_full["ys_effective"]
ts_lst_full = prep_dat_full["ts_lst"]

ts_lst_smooth = []
ys_lst_smooth = []
for i in range(len(ys_lst_full)):
    min_t = int(ts_lst_full[i].min())
    max_t = int(ts_lst_full[i].max())
    ts_tmp = np.linspace(min_t, max_t, max_t - min_t + 1)
    ys_smooth = smooth_array(
        ts_tmp, ts_lst_full[i], ys_lst_full[i], sigma=sigma, method=smoothing_method
    )

    ts_lst_smooth.append(ts_tmp)
    ys_lst_smooth.append(ys_smooth)
# -

# Let's plot the smoothing

# +
colors = [plot_ts.COLORS_COVSPECTRUM[var] for var in variants_investigated]


def remove_0th(arr):
    """We don't plot the artificial 0th variant 'other'."""
    return arr[:, 1:]


fig, axes = plt.subplots(3, 2)
axes = axes.flatten()

for i, city in enumerate(cities):
    ax = axes[i]
    plot_ts.plot_data(ax, ts_lst_full[i], remove_0th(ys_lst_full[i]), colors=colors)
    plot_ts.plot_fit(ax, ts_lst_smooth[i], remove_0th(ys_lst_smooth[i]), colors=colors)
    # ax.set_xlim((300,ts_lst_full[i].max()))

    def format_date(x, pos):
        return plot_ts.num_to_date(x, date_min=start_date)

    date_formatter = ticker.FuncFormatter(format_date)
    ax.xaxis.set_major_formatter(date_formatter)

plt.tight_layout()
plt.show()

# -

# # make a comparison

# Let's define functions to make the comparison between predicted and smoothed


# +
def compare(ts_in, ys_in, ts_smooth, ys_smooth):
    """Compare predicted data ys_in at ts_in with smoothed data ys_smooth at ts_smooth."""
    indices = np.where(np.isin(ts_smooth, ts_in))[0]
    if len(indices) == 0:
        raise ValueError("No matching time points found in ts_smooth")

    differences = ys_in - ys_smooth[indices]
    relative_differences = differences / ys_smooth[indices]
    return (differences, relative_differences)


def compare_all(ts_lst_pred, ys_lst_pred, ts_lst_smooth, ys_lst_smooth):
    """Compare all time series in the given lists."""
    all_differences = []
    all_relative_differences = []
    for ts_in, ys_in, ts_smooth, ys_smooth in zip(
        ts_lst_pred, ys_lst_pred, ts_lst_smooth, ys_lst_smooth
    ):
        differences, relative_differences = compare(ts_in, ys_in, ts_smooth, ys_smooth)
        all_differences.append(differences)
        all_relative_differences.append(relative_differences)
    return (all_differences, all_relative_differences)


# -

# define when are the dates we want to fit at and predict, run the experiment

# +
import tqdm
import warnings

# Initialize list to store results
all_differences_results = []
all_relative_differences_results = []

# Loop through max_dates and perform
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for max_date in tqdm.tqdm(max_dates, leave=False):
        prep_dat = prepare_data(data_wide, start_date, max_date)

        inf_dat = inference(
            prep_dat["ys_effective"],
            prep_dat["ts_lst_scaled"],
            prep_dat["ts_lst"],
            prep_dat["time_scaler"],
            horizon=max_horizon,
        )

        diff_res, rel_diff_res = compare_all(
            inf_dat["ts_pred_lst"], inf_dat["ys_pred"], ts_lst_smooth, ys_lst_smooth
        )

        all_differences_results.append(diff_res)
        all_relative_differences_results.append(rel_diff_res)
# -


# ## plot the results

# +
all_differences_results_array = np.array(all_differences_results)
all_relative_differences_results_array = np.array(all_relative_differences_results)
print(all_differences_results_array.shape)

fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex="col", sharey="row")
# axes = axes.flatten()

variants_effective = ["other"] + variants_investigated

for i, variant in enumerate([9, 2, 8]):
    ## Plot time series
    ax = axes[0, i]
    dates_points = start_date + pd.to_timedelta(ts_lst_full[0], "D")
    ax.scatter(dates_points, ys_lst_full[0][:, variant])
    dates_smooth = start_date + pd.to_timedelta(ts_lst_smooth[0], "D")
    ax.plot(dates_smooth, ys_lst_smooth[0][:, variant])
    ax.set_xlim(pd.to_datetime(["2023-08-01", "2024-01-01"]))
    ax.set_title(variants_effective[variant])
    ax.set_ylabel("rel. abundance")

    ## plot mean error

    ax = axes[1, i]
    for horizon in [0, 7, 14, 21, 28, 35, 42]:
        ax.plot(
            max_dates,
            all_differences_results_array.mean(axis=1)[:, horizon, variant],
            label=f"horizon={int(horizon/7)}[week]",
        )
        # ax.legend()
        ax.set_ylabel("mean forecast error")
        ax.set_ylim((-0.5, 0.5))

    ## plot mean absolute relative error

    # ax=axes[2,i]
    # for horizon in [0,7,14,21]:
    #     ax.plot(
    #         max_dates,
    #         np.abs(all_relative_differences_results_array[:,:,horizon,variant]).mean(axis=1),
    #         label=f"horizon={horizon}[d]"
    #     )
    #     ax.set_ylim((0,2))
    #     ax.set_ylabel("mean abs rel error")

# Deduplicate legend entries
handles, labels = axes[1, 0].get_legend_handles_labels()
unique_labels = {}
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels[label] = handle
        unique_handles.append((handle, label))

# Add unique legend
fig.legend(
    [h for h, _ in unique_handles],
    [l for _, l in unique_handles],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
fig.tight_layout()
plt.show()

# +
all_differences_results_array = np.array(all_differences_results)
all_relative_differences_results_array = np.array(all_relative_differences_results)
print(all_differences_results_array.shape)

fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex="col", sharey="row")
# axes = axes.flatten()

variants_effective = ["other"] + variants_investigated

for i, variant in enumerate([9, 2, 8]):
    ## Plot time series
    ax = axes[0, i]
    dates_points = start_date + pd.to_timedelta(ts_lst_full[0], "D")
    ax.scatter(dates_points, ys_lst_full[0][:, variant])
    dates_smooth = start_date + pd.to_timedelta(ts_lst_smooth[0], "D")
    ax.plot(dates_smooth, ys_lst_smooth[0][:, variant])
    ax.set_xlim(pd.to_datetime(["2023-08-01", "2024-01-01"]))
    ax.set_title(variants_effective[variant])
    ax.set_ylabel("rel. abundance")

    ## plot mean error

    ax = axes[1, i]
    for horizon in [0, 7, 14, 21, 28, 35, 42]:
        ax.plot(
            max_dates + pd.to_timedelta(horizon, "D"),
            all_differences_results_array.mean(axis=1)[:, horizon, variant],
            label=f"horizon={int(horizon/7)}[week]",
            alpha=0.7,
        )
        # ax.legend()
        ax.set_ylabel("mean forecast error")
        ax.set_ylim((-0.5, 0.5))


# Deduplicate legend entries
handles, labels = axes[1, 0].get_legend_handles_labels()
unique_labels = {}
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels[label] = handle
        unique_handles.append((handle, label))

# Add unique legend
fig.legend(
    [h for h, _ in unique_handles],
    [l for _, l in unique_handles],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
fig.tight_layout()
plt.show()
