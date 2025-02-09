# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: jax
#     language: python
#     name: jax
# ---

# %%
## analyse results

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

# import covvfit._frequentist as freq
import covvfit._preprocess_abundances as prec
import covvfit.plotting._timeseries as plot_ts

from covvfit import quasimultinomial as qm

import numpyro

import requests
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 300


# %%
with open("../workflows/compare_clinical/config_ba1ba2.yaml", "rb") as f:
    config_file = yaml.safe_load(f)

# %%
merged_ww_data = pd.read_csv(
    "../workflows/compare_clinical/data/config_ba1ba2/wastewater_preprocessed.csv"
)
merged_ww_data["total_count"] = 1
merged_ww_data.head()
grouped_ww_data = (
    merged_ww_data.groupby(["time", "city"])["total_count"].sum().unstack(fill_value=0)
)
# grouped_ww_data

# %%
clin_freq = pd.read_csv(
    "../workflows/compare_clinical/data/config_ba1ba2/normalized_clinical_data.csv"
)
# clin_freq

# %%
variants = ["BA.1*", "BA.2*"]

# Load the data
wastewater_df = pd.read_csv(
    "../workflows/compare_clinical/results/config_ba1ba2/consolidated_wastewater_results.csv",
    parse_dates=["end_date"],
)
clinical_df = pd.read_csv(
    "../workflows/compare_clinical/results/config_ba1ba2/consolidated_clinical_results.csv",
    parse_dates=["end_date"],
)
merged_clinical_data = pd.read_csv(
    "../workflows/compare_clinical/data/config_ba1ba2/merged_clinical_data.csv"
)

# Filter merged clinical data for specific divisions
divisions = ["Zürich", "Geneva", "Ticino", "Graubünden", "Bern", "Sankt Gallen"]
filtered_clinical_data = merged_clinical_data[
    merged_clinical_data["division"].isin(divisions)
]


# Convert date to datetime if not already
filtered_clinical_data["date"] = pd.to_datetime(filtered_clinical_data["date"])
filtered_clinical_data = filtered_clinical_data[
    (filtered_clinical_data["date"] >= x_min)
    & (filtered_clinical_data["date"] <= x_max)
]

# Calculate total counts per day, stratified by division
grouped_clinical_data = (
    filtered_clinical_data.groupby(["date", "division"])["total_count"]
    .sum()
    .unstack(fill_value=0)
)

# Normalize wastewater and clinical solutions
for i in range(4):
    wastewater_df[f"solution_{i}_normalized"] = (
        wastewater_df[f"solution_{i}"] / wastewater_df["t_max"]
    )
    clinical_df[f"solution_{i}_normalized"] = (
        clinical_df[f"solution_{i}"] / clinical_df["t_max"]
    )

variants_evaluated = ["BA.2*"]
reference_variant = "BA.1*"

variants_evaluated_index = np.where(
    np.isin(config_file["variants_investigated"], variants_evaluated)
)[0]

variants_reference_index = np.where(
    np.isin(config_file["variants_investigated"], reference_variant)
)[0]
variants_reference_index

# %%
# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=False)

# Define a colormap for consistent colors
colors = cm.tab10.colors  # Use a colormap with at least 4 colors

ax = axes[1]
# Top plot: Wastewater and clinical normalized solutions
i = 1
color = colors[i]  # Assign the same color for both lines
ax.plot(
    wastewater_df["end_date"],
    (
        wastewater_df[f"solution_{1}_normalized"]
        - wastewater_df[f"solution_{0}_normalized"]
    )
    * 7
    * 100,
    label=f"Wastewater rate {variants[1]}",
    color=color,
    linestyle="-",
)
ax.plot(
    clinical_df["end_date"],
    (clinical_df[f"solution_{1}_normalized"] - clinical_df[f"solution_{0}_normalized"])
    * 7
    * 100,
    label=f"Clinical rate {variants[1]}",
    color=color,
    linestyle="--",
)

ax.set_ylabel("rel. fitness / week")
ax.set_title("BA.2 vs BA.1 Fitness Measurable at Different Dates")
ax.set_ylim(0, 0.25 * 7 * 100)
ax.legend()


import matplotlib.dates as mdates

# Ensure the 'date' column is in datetime format
grouped_clinical_data.reset_index(inplace=True)
grouped_clinical_data["date"] = pd.to_datetime(
    grouped_clinical_data["date"], errors="coerce"
)

# Drop rows with invalid dates
grouped_clinical_data = grouped_clinical_data.dropna(subset=["date"])

# Sort by date
grouped_clinical_data.sort_values("date", inplace=True)

## plot samples

ax = axes[2]

# Plot stacked bar chart
div_col = zip(grouped_clinical_data.columns[1:], colors)
for division, color in div_col:  # Use the same divisions and colors as in the bar plot
    # for division in grouped_clinical_data.columns[1:]:  # Skip the 'date' column
    ax.bar(
        grouped_clinical_data["date"],
        grouped_clinical_data[division],
        label=division,
        color=color,
        bottom=grouped_clinical_data.loc[:, grouped_clinical_data.columns[1:]]
        .iloc[:, : grouped_clinical_data.columns[1:].tolist().index(division)]
        .sum(axis=1, skipna=True),
    )

# Set labels and title
ax.set_ylabel("Samples Count")
ax.set_title("Samples Per Day")
ax.set_xlabel("Date")

# Format x-axis with date labels
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)


# Add legend
ax.axhline(
    y=6,
    color="red",
    linestyle="--",
    linewidth=3,  # Set boldness by adjusting the line width
    label="Wastewater samples (6/day)",  # Add a label for legend if needed
)

ax.axhline(
    y=186,
    color="black",
    linestyle="--",
    linewidth=3,  # Set boldness by adjusting the line width
    label="Clinical sequences (186/day)",  # Add a label for legend if needed
)


# ax.legend(title="")

clin_freq = pd.read_csv(
    "../workflows/compare_clinical/data/config_ba1ba2/normalized_clinical_data.csv"
)
clin_freq["date"] = pd.to_datetime(clin_freq["date"])

# Sort clin_freq by date
clin_freq = clin_freq.sort_values(by="date")

ax = axes[0]

# Add the plot of BA.2* frequency
div_col = zip(grouped_clinical_data.columns[1:], colors)
for division, color in div_col:  # Use the same divisions and colors as in the bar plot
    division_data = clin_freq[clin_freq["division"] == division]
    ax.plot(
        division_data["date"],
        division_data["BA.2*"],
        label=division,
        color=color,
        # marker='o',
        linestyle="-",
    )

# Customize the plot
ax.set_title("BA.2* Frequency in Clinical Sequences")
ax.set_xlabel("Date")
ax.set_ylabel("Frequency of BA.2*")
# ax.legend(title="Division", loc="upper left")
ax.grid(True)

# Format x-axis for better readability
plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="right")
axes[2].xaxis.set_major_locator(
    mdates.MonthLocator(bymonthday=1)
)  # First day of each month
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
axes[2].set_xlabel("")
# Remove x-ticks and labels for the first two subplots
axes[0].tick_params(labelbottom=False)  # Hide x-axis labels for axes[0]
axes[0].set_xlabel("")
axes[1].tick_params(labelbottom=False)  # Hide x-axis labels for axes[1]


# Define x-axis limits based on the earliest and latest date in all datasets
x_min = wastewater_df["end_date"].min()
x_max = max(
    wastewater_df["end_date"].max(),
    clinical_df["end_date"].max(),
    grouped_clinical_data["date"].max(),
    clin_freq["date"].max(),
)

# Apply the same x-axis limits to all subplots
for ax in axes:
    ax.set_xlim(x_min, x_max)


# Ensure layout is correct
plt.tight_layout()
plt.show()
