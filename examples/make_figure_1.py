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

# # Analysis of the full dataset notebook
#
# This notebook shows how to estimate growth advantages by fiting the model within the quasimultinomial framework, on the whole dataset

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
# %matplotlib inline
import matplotlib.pyplot as plt

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 150  # Adjust to 150, 200, or more for higher resolution

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

## Set limit times for modeling

max_date = pd.to_datetime(data_wide["time"]).max()
delta_time = pd.Timedelta(days=1250)
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


variants_full = [
    "B.1.1.7",
    "B.1.351",
    #     "P.1",
    "B.1.617.2",
    "BA.1",
    "BA.2",
    "BA.4",
    "BA.5",
    "BA.2.75",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "BA.2.86",
    "JN.1",
    "KP.2",
    "KP.3",
    "XEC",
]

variants_investigated = [
    # "B.1.1.7",
    "B.1.617.2",
    "BA.1",
    "BA.2",
    "BA.4",
    "BA.5",
    "BA.2.75",
    "BQ.1.1",
    "XBB.1.5",
    "XBB.1.9",
    "XBB.1.16",
    "XBB.2.3",
    "EG.5",
    "BA.2.86",
    "JN.1",
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
    (
        qm.get_relative_growths(theta_star, n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
    (
        qm.get_relative_growths(confints_estimates[0], n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
    (
        qm.get_relative_growths(confints_estimates[1], n_variants=n_variants_effective)
        - time_scaler.t_min
    )
    / (time_scaler.t_max - time_scaler.t_min),
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
    # plot_ts.plot_fit(
    #     ax, ts_pred_lst[i], remove_0th(ys_pred[i]), colors=colors, linestyle="--"
    # )

    plot_ts.plot_confidence_bands(
        ax,
        ts_lst[i],
        jax.tree.map(remove_0th, ys_fitted_confint[i]),
        colors=colors,
    )
    # plot_ts.plot_confidence_bands(
    #     ax,
    #     ts_pred_lst[i],
    #     jax.tree.map(remove_0th, ys_pred_confint[i]),
    #     colors=colors,
    # )

    # Plot the data points
    plot_ts.plot_data(ax, ts_lst[i], remove_0th(ys_effective[i]), colors=colors)

    # Plot the complements
    plot_ts.plot_complement(ax, ts_lst[i], remove_0th(ys_fitted[i]), alpha=0.3)
    # plot_ts.plot_complement(
    #     ax, ts_pred_lst[i], remove_0th(ys_pred[i]), linestyle="--", alpha=0.3
    # )

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

# +
import matplotlib.pyplot as plt

# Example data (adjust as needed)
colors = [plot_ts.COLORS_COVSPECTRUM[var] for var in variants_investigated]

# Create the figure and subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 5))
axes = axes.flatten()
for i in range(6):
    plot_city(axes[i], i)  # Assuming plot_city modifies the axes

# Create a custom legend
legend_elements = [
    plt.Line2D([0], [0], color=color, lw=4, label=variant)
    for color, variant in zip(colors, variants_investigated)
]

# Add the legend to the right outside the plot
fig.legend(
    handles=legend_elements,
    loc="center left",  # Align the legend to the left of the bounding box
    bbox_to_anchor=(1, 0.5),  # Position it to the right of the figure
    title="Variants",
)

# Adjust the layout to make space for the legend

plt.tight_layout()
# Show the plot
plt.show()
# -


# ## Plot matrix of growth advantage estimates

# +
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


relative_growths = (
    qm.get_relative_growths(theta_star, n_variants=n_variants_effective)
    - time_scaler.t_min
) / (time_scaler.t_max - time_scaler.t_min)
relative_growths = jnp.concat([jnp.array([0]), relative_growths])
relative_growths = relative_growths * 100 * 7

pairwise_diff = jnp.expand_dims(relative_growths, axis=1) - jnp.expand_dims(
    relative_growths, axis=0
)

fitness_df = pd.DataFrame(pairwise_diff)


# rename columns and rows
fitness_df.columns = variants_effective
fitness_df.index = variants_effective

# format, remove first row and last col, keep only lower triangle and diagonal
fitness_df = fitness_df.iloc[1:, :]
fitness_df = fitness_df.iloc[:, :-1]
mask = np.triu(np.ones(fitness_df.shape), k=1).astype(bool)
fitness_df = fitness_df.mask(mask, np.nan, inplace=False)
fitness_df.iloc[1:, 0] = np.nan
fitness_df.iloc[3:, 1] = np.nan
fitness_df.iloc[3:, 2] = np.nan
fitness_df.iloc[5:, 3] = np.nan
fitness_df.iloc[7:, 4] = np.nan
fitness_df.iloc[9:, 5] = np.nan
fitness_df.iloc[11:, 6] = np.nan
fitness_df.iloc[12:, 7] = np.nan
fitness_df.iloc[14:, 8] = np.nan
fitness_df.iloc[14:, 9] = np.nan
fitness_df.iloc[14:, 10] = np.nan
fitness_df.iloc[14:, 11] = np.nan
fitness_df.iloc[14:, 12] = np.nan
fitness_df.iloc[14:, 13] = np.nan


fitness_df = fitness_df.iloc[1:, 1:]

ax = sns.heatmap(fitness_df, cmap="Reds", annot=True, fmt=".0f", cbar=True)
ax.set_title("Weekly Growth Advantage (%)")

# -

# ## Look at overdispersion for different thresholds of epsilon
#
# Let's check if our estimate of overdispersion is stable in the vicinity of the selected epsilon=0.001 threshold.

# +
epsilons = np.logspace(-4, -2, 25)

# Compute results for each epsilon
overdispersion_results = [
    qm.compute_overdispersion(ys_effective, ys_fitted, epsilon=eps) for eps in epsilons
]

# Extract overall and cities data
overalls = [res.overall for res in overdispersion_results]
cities_res = np.array([res.cities for res in overdispersion_results])

# Plotting
fig, axes = plt.subplots(1, 1, figsize=(10, 4))
axes = [axes]

# Overall vs Epsilon
axes[0].plot(epsilons, overalls, label="Overall", color="black", linestyle="--")
axes[0].set_title("Overdispersion vs Epsilon")
axes[0].set_xlabel("Epsilon")
axes[0].set_ylabel("Overdispersion")
axes[0].grid(True)
axes[0].set_xscale("log")
# axes[0].set_yscale('log')


# Cities vs Epsilon
for city_idx in range(cities_res.shape[1]):
    axes[0].plot(epsilons, cities_res[:, city_idx], label=f"{cities[city_idx]}")
axes[0].legend()

plt.tight_layout()
plt.show()
