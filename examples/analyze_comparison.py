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
import matplotlib.lines as mlines


# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 300


# %%
# # Plot
# fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=False)

# # Define a colormap for consistent colors
# colors = cm.tab10.colors  # Use a colormap with at least 4 colors

# ax = axes[1]
# # Top plot: Wastewater and clinical normalized solutions
# for i, var_idx in enumerate(variants_evaluated_index):
#     color = colors[i]  # Assign the same color for both lines
#     ax.plot(
#         wastewater_df["end_date"],
#         (wastewater_df[f"solution_{var_idx}_normalized"]
#         - wastewater_df[f"solution_{variants_reference_index}_normalized"])*7*100,
#         label=f"Wastewater rate {variants[i]}",
#         color=color,
#         linestyle="-",
#     )
#     ax.plot(
#         clinical_df["end_date"],
#         (clinical_df[f"solution_{var_idx}_normalized"]
#          - clinical_df[f"solution_{variants_reference_index}_normalized"])*7*100,
#         label=f"Clinical rate {variants[i]}",
#         color=color,
#         linestyle="--",
#     )

# ax.set_ylabel("rel. fitness / week")
# ax.set_title(f"fitness, relative to {reference_variant}, measurable at different dates")
# ax.set_ylim(0, 0.25*7*100)
# ax.legend()


# import matplotlib.dates as mdates

# # Ensure the 'date' column is in datetime format
# grouped_clinical_data.reset_index(inplace=True)
# grouped_clinical_data["date"] = pd.to_datetime(
#     grouped_clinical_data["date"], errors="coerce"
# )

# # Drop rows with invalid dates
# grouped_clinical_data = grouped_clinical_data.dropna(subset=["date"])

# # Sort by date
# grouped_clinical_data.sort_values("date", inplace=True)

# ## plot samples

# ax = axes[2]

# # Plot stacked bar chart
# div_col = zip(grouped_clinical_data.columns[1:], colors)
# for division, color in div_col:  # Use the same divisions and colors as in the bar plot
# # for division in grouped_clinical_data.columns[1:]:  # Skip the 'date' column
#     ax.bar(
#         grouped_clinical_data['date'],
#         grouped_clinical_data[division],
#         label=division,
#         color=color,
#         bottom=grouped_clinical_data.loc[:, grouped_clinical_data.columns[1:]].iloc[:, :grouped_clinical_data.columns[1:].tolist().index(division)].sum(axis=1, skipna=True)
#     )

# # Set labels and title
# ax.set_ylabel("Samples Count")
# ax.set_title("Samples Per Day")
# ax.set_xlabel("Date")

# # Format x-axis with date labels
# locator = mdates.AutoDateLocator()
# formatter = mdates.ConciseDateFormatter(locator)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)


# # # Add legend
# # ax.axhline(
# #     y=6,
# #     color='red',
# #     linestyle='--',
# #     linewidth=3,  # Set boldness by adjusting the line width
# #     label="Wastewater samples (6/day)"  # Add a label for legend if needed
# # )

# # ax.axhline(
# #     y=186,
# #     color='black',
# #     linestyle='--',
# #     linewidth=3,  # Set boldness by adjusting the line width
# #     label="Clinical sequences (186/day)"  # Add a label for legend if needed
# # )


# # ax.legend(title="")

# ax = axes[0]

# # Add the plot of BA.2* frequency
# div_col = zip(grouped_clinical_data.columns[1:], colors)
# for division, color in div_col:  # Use the same divisions and colors as in the bar plot
#     division_data = clin_freq[clin_freq["division"] == division]
#     ax.plot(
#         division_data["date"],
#         division_data[variants_evaluated[-1]],
#         label=division,
#         color=color,
#         # marker='o',
#         linestyle="-",
#     )

# # Customize the plot
# ax.set_title(f"{variants_evaluated[-1]} Frequency in Clinical Sequences")
# ax.set_xlabel("Date")
# ax.set_ylabel("rel.abundance")
# # ax.legend(title="Division", loc="upper left")
# ax.grid(True)

# # Format x-axis for better readability
# plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="right")
# axes[2].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))  # First day of each month
# axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
# axes[2].set_xlabel("")
# # Remove x-ticks and labels for the first two subplots
# axes[0].tick_params(labelbottom=False)  # Hide x-axis labels for axes[0]
# axes[0].set_xlabel("")
# axes[1].tick_params(labelbottom=False)  # Hide x-axis labels for axes[1]


# # Define x-axis limits based on the earliest and latest date in all datasets
# x_min = wastewater_df["end_date"].min()
# x_max = max(wastewater_df["end_date"].max(), clinical_df["end_date"].max(), grouped_clinical_data["date"].max(), clin_freq["date"].max())

# # Apply the same x-axis limits to all subplots
# for ax in axes:
#     ax.set_xlim(x_min, x_max)


# # Ensure layout is correct
# plt.tight_layout()
# plt.show()


# %%
import yaml
import pandas as pd
import numpy as np


def load_data(folder, divisions, variants_evaluated, reference_variant):
    """
    Load and preprocess wastewater and clinical data from the specified folder.

    Parameters:
    folder (str): Folder containing the necessary data files.

    Returns:
    dict: A dictionary containing all loaded and processed data.
    """

    # Load configuration file
    with open(f"../workflows/compare_clinical/{folder}.yaml", "rb") as f:
        config_file = yaml.safe_load(f)

    # Load wastewater data
    merged_ww_data = pd.read_csv(
        f"../workflows/compare_clinical/data/{folder}/wastewater_preprocessed.csv"
    )
    merged_ww_data["total_count"] = 1
    grouped_ww_data = (
        merged_ww_data.groupby(["time", "city"])["total_count"]
        .sum()
        .unstack(fill_value=0)
    )

    # Load clinical frequency data
    clin_freq = pd.read_csv(
        f"../workflows/compare_clinical/data/{folder}/normalized_clinical_data.csv"
    )

    # Load consolidated results
    wastewater_df = pd.read_csv(
        f"../workflows/compare_clinical/results/{folder}/consolidated_wastewater_results.csv",
        parse_dates=["end_date"],
    )
    clinical_df = pd.read_csv(
        f"../workflows/compare_clinical/results/{folder}/consolidated_clinical_results.csv",
        parse_dates=["end_date"],
    )
    merged_clinical_data = pd.read_csv(
        f"../workflows/compare_clinical/data/{folder}/merged_clinical_data.csv"
    )

    # Filter merged clinical data for specific divisions
    filtered_clinical_data = merged_clinical_data[
        merged_clinical_data["division"].isin(divisions)
    ]
    filtered_clinical_data["date"] = pd.to_datetime(filtered_clinical_data["date"])

    # Define x-axis limits based on the wastewater and clinical data
    x_min = min(wastewater_df["end_date"].min(), clinical_df["end_date"].min())
    x_max = max(wastewater_df["end_date"].max(), clinical_df["end_date"].max())

    # Filter clinical data within date range
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
    for i in range(len(config["variants_investigated"])):
        wastewater_df[f"solution_{i}_normalized"] = (
            wastewater_df[f"solution_{i}"] / wastewater_df["t_max"]
        )
        try:
            wastewater_df[f"confint_lower_{i}_normalized"] = (
                wastewater_df[f"confint_lower_{i}"] / wastewater_df["t_max"]
            )
            wastewater_df[f"confint_upper_{i}_normalized"] = (
                wastewater_df[f"confint_upper_{i}"] / wastewater_df["t_max"]
            )
        except:
            pass
        clinical_df[f"solution_{i}_normalized"] = (
            clinical_df[f"solution_{i}"] / clinical_df["t_max"]
        )
        clinical_df[f"confint_lower_{i}_normalized"] = (
            clinical_df[f"confint_lower_{i}"] / clinical_df["t_max"]
        )
        clinical_df[f"confint_upper_{i}_normalized"] = (
            clinical_df[f"confint_upper_{i}"] / clinical_df["t_max"]
        )

    # Get indices for evaluated and reference variants
    variants_evaluated_index = np.where(
        np.isin(config_file["variants_investigated"], variants_evaluated)
    )[0]
    variants_reference_index = np.where(
        np.isin(config_file["variants_investigated"], reference_variant)
    )[0][0]

    # clinical frequency
    clin_freq = pd.read_csv(
        f"../workflows/compare_clinical/data/{folder}/normalized_clinical_data.csv"
    )
    clin_freq["date"] = pd.to_datetime(clin_freq["date"])
    # Sort clin_freq by date
    clin_freq = clin_freq.sort_values(by="date")

    return {
        "config": config_file,
        "grouped_ww_data": grouped_ww_data,
        "clin_freq": clin_freq,
        "wastewater_df": wastewater_df,
        "clinical_df": clinical_df,
        "grouped_clinical_data": grouped_clinical_data,
        "variants_evaluated_index": variants_evaluated_index,
        "variants_reference_index": variants_reference_index,
        "x_min": x_min,
        "x_max": x_max,
        "merged_ww_data": merged_ww_data,
    }


# %%


def make_plot(
    axes,
    config,
    grouped_ww_data,
    clin_freq,
    wastewater_df,
    clinical_df,
    grouped_clinical_data,
    variants_evaluated_index,
    variants_reference_index,
    x_min,
    x_max,
    merged_ww_data,
):
    colors_covsp = plot_ts.COLORS_COVSPECTRUM
    ax = axes[1]
    # Top plot: Wastewater normalized solutions
    for i, var_idx in enumerate(variants_evaluated_index):
        variant = config["variants_investigated"][var_idx]
        ax.plot(
            wastewater_df["end_date"],
            (
                wastewater_df[f"solution_{var_idx}_normalized"]
                - wastewater_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            label=f"{variant}",
            color=colors_covsp[variant.rstrip("*")],
            linestyle="-",
        )
        ax.fill_between(
            wastewater_df["end_date"],
            (
                wastewater_df[f"confint_lower_{var_idx}_normalized"]
                - wastewater_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            (
                wastewater_df[f"confint_upper_{var_idx}_normalized"]
                - wastewater_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            color=colors_covsp[variant.rstrip("*")],
            alpha=0.2,  # Transparency for the shaded area
        )
        # ax.plot(
        #     clinical_df["end_date"],
        #     (clinical_df[f"solution_{var_idx}_normalized"]
        #      - clinical_df[f"solution_{variants_reference_index}_normalized"])*7*100,
        #     label=f"Clinical {variants[i]}",
        #     color = colors_covsp[variant.rstrip("*")],
        #     linestyle="--",
        # )

    ax.set_ylabel("rel. fitness / week")
    ax.set_title("Wastewater-Derived Selection advantage")
    ax.set_ylim(0, 0.2 * 7 * 100)

    ## clinical solutions

    ax = axes[2]
    # Top plot: Wastewater and clinical normalized solutions
    for i, var_idx in enumerate(variants_evaluated_index):
        variant = config["variants_investigated"][var_idx]
        ax.plot(
            clinical_df["end_date"],
            (
                clinical_df[f"solution_{var_idx}_normalized"]
                - clinical_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            label=f"{variant}",
            color=colors_covsp[variant.rstrip("*")],
            linestyle="--",
        )
        ax.fill_between(
            clinical_df["end_date"],
            (
                clinical_df[f"confint_lower_{var_idx}_normalized"]
                - clinical_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            (
                clinical_df[f"confint_upper_{var_idx}_normalized"]
                - clinical_df[f"solution_{variants_reference_index}_normalized"]
            )
            * 7
            * 100,
            color=colors_covsp[variant.rstrip("*")],
            alpha=0.2,  # Transparency for the shaded area
        )

    ax.set_ylabel("rel. fitness / week")
    ax.set_title(f"Clinical-Derived Selection Advantage")
    ax.set_ylim(0, 0.2 * 7 * 100)

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

    ax = axes[3]

    # Plot  bar chart

    # Ensure 'date' and 'time' are properly set as indices
    grouped_clinical_data = grouped_clinical_data.set_index("date")
    grouped_ww_data.index = pd.to_datetime(grouped_ww_data.index, errors="coerce")

    # Resample both datasets to weekly sums
    weekly_clinical = grouped_clinical_data.resample("W").sum()
    weekly_ww = grouped_ww_data.resample("W").sum()

    # Align the two datasets by reindexing with the union of both weekly indices
    all_weeks = weekly_clinical.index.union(weekly_ww.index)
    weekly_clinical = weekly_clinical.reindex(all_weeks, fill_value=0)
    weekly_ww = weekly_ww.reindex(all_weeks, fill_value=0)

    # Sum across all divisions/cities for total weekly counts
    clinical_totals = weekly_clinical.sum(axis=1)
    ww_totals = weekly_ww.sum(axis=1)

    # Define bar width
    width = 3  # Adjust for side-by-side effect

    # Plot side-by-side bars
    ax.bar(
        clinical_totals.index - pd.Timedelta(days=2),  # Shift left for alignment
        clinical_totals,
        width=width,
        label="Clinical Samples",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        ww_totals.index + pd.Timedelta(days=2),  # Shift right for alignment
        ww_totals,
        width=width,
        label="Wastewater Samples",
        color="orange",
        alpha=0.7,
    )

    # Set labels and title
    ax.set_ylabel("Count")
    ax.set_title("Samples Per Week")
    # ax.set_xlabel("Date")

    # Format x-axis with date labels
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # # Add legend
    # ax.axhline(
    #     y=6,
    #     color='red',
    #     linestyle='--',
    #     linewidth=3,  # Set boldness by adjusting the line width
    #     label="Wastewater samples (6/day)"  # Add a label for legend if needed
    # )

    # ax.axhline(
    #     y=186,
    #     color='black',
    #     linestyle='--',
    #     linewidth=3,  # Set boldness by adjusting the line width
    #     label="Clinical sequences (186/day)"  # Add a label for legend if needed
    # )

    # ax.legend(title="")

    ax = axes[0]

    for i, variant in enumerate(variants_evaluated):
        ax.plot(
            clin_freq[clin_freq["division"] == "Zürich"]["date"],
            clin_freq[clin_freq["division"] == "Zürich"][variant],
            color=colors_covsp[variant.rstrip("*")],
            linestyle="--",
        )
        ax.plot(
            pd.to_datetime(
                merged_ww_data[merged_ww_data["city"] == "Zürich (ZH)"]["time"]
            ),
            merged_ww_data[merged_ww_data["city"] == "Zürich (ZH)"][
                variant.rstrip("*")
            ],
            color=colors_covsp[variant.rstrip("*")],
            linestyle="-",
        )
    # ax.legend(title="Division", loc="upper left")
    ax.grid(True)
    ax.set_title("Relative Abundance")

    # Format x-axis for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="right")
    axes[3].xaxis.set_major_locator(
        mdates.MonthLocator(bymonthday=1)
    )  # First day of each month
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    axes[3].set_xlabel("")
    # Remove x-ticks and labels for the first two subplots
    axes[0].tick_params(labelbottom=False)  # Hide x-axis labels for axes[0]
    axes[0].set_xlabel("")
    axes[1].tick_params(labelbottom=False)  # Hide x-axis labels for axes[1]
    axes[2].tick_params(labelbottom=False)  # Hide x-axis labels for axes[1]

    # Define x-axis limits based on the earliest and latest date in all datasets
    x_min = wastewater_df["end_date"].min()
    x_max = wastewater_df["end_date"].max()

    # Apply the same x-axis limits to all subplots
    for ax in axes:
        ax.set_xlim(x_min, x_max)


# %%
# Plot
fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharey="none")


variants = ["BA.2.86*", "JN.1*"]
divisions = ["Zürich", "Geneva", "Ticino", "Graubünden", "Bern", "Sankt Gallen"]
variants_evaluated = ["BA.2.86*", "JN.1*"]
reference_variant = "EG.5*"
folder = "config_jn1"

(
    config,
    grouped_ww_data,
    clin_freq,
    wastewater_df,
    clinical_df,
    grouped_clinical_data,
    variants_evaluated_index,
    variants_reference_index,
    x_min,
    x_max,
    merged_ww_data,
) = load_data(folder, divisions, variants_evaluated, reference_variant).values()

make_plot(
    axes[:, 1],
    config,
    grouped_ww_data,
    clin_freq,
    wastewater_df,
    clinical_df,
    grouped_clinical_data,
    variants_evaluated_index,
    variants_reference_index,
    x_min,
    x_max,
    merged_ww_data,
)


variants = ["BA.1*", "BA.2*"]
divisions = ["Zürich", "Geneva", "Ticino", "Graubünden", "Bern", "Sankt Gallen"]
variants_evaluated = ["BA.2*"]
reference_variant = "BA.1*"
folder = "config_ba1ba2"

(
    config,
    grouped_ww_data,
    clin_freq,
    wastewater_df,
    clinical_df,
    grouped_clinical_data,
    variants_evaluated_index,
    variants_reference_index,
    x_min,
    x_max,
    merged_ww_data,
) = load_data(folder, divisions, variants_evaluated, reference_variant).values()

make_plot(
    axes[:, 0],
    config,
    grouped_ww_data,
    clin_freq,
    wastewater_df,
    clinical_df,
    grouped_clinical_data,
    variants_evaluated_index,
    variants_reference_index,
    x_min,
    x_max,
    merged_ww_data,
)

axes[0, 1].set_xlim([pd.to_datetime("2023-08-01"), pd.to_datetime("2024-01-01")])
axes[1, 1].set_xlim([pd.to_datetime("2023-08-01"), pd.to_datetime("2024-01-01")])
axes[2, 1].set_xlim([pd.to_datetime("2023-08-01"), pd.to_datetime("2024-01-01")])
axes[3, 1].set_xlim([pd.to_datetime("2023-08-01"), pd.to_datetime("2024-01-01")])
axes[0, 0].set_ylim([0, 1])
axes[0, 1].set_ylim([0, 1])

cutoffs = [0.025, 0.05, 0.10]  # Cutoff values
break_dates = pd.to_datetime(
    [
        "2022-01-15 00:00:00",
        "2022-01-20 12:00:00",
        "2022-01-28 12:00:00",
        "2023-10-14 12:00:00",
        "2023-10-19 00:00:00",
        "2023-10-27 00:00:00",
    ]
)

# First three vlines for axes[0,0] and axes[1,0]
for i in range(3):
    for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
        ax.axvline(
            x=break_dates[i],
            color="black",
            linestyle="dashed",
            linewidth=1,
        )
        ax.text(
            break_dates[i],
            ax.get_ylim()[1] * 0.9,  # Position at 90% of y-axis max
            f"{cutoffs[i] * 100}%",
            color="black",
            fontsize=10,
            ha="right",
            va="top",
            rotation=90,
        )

# Next three vlines for axes[0,1] and axes[1,1]
for i in range(3, 6):
    for ax in [axes[0, 1], axes[1, 1], axes[2, 1]]:
        ax.axvline(
            x=break_dates[i],
            color="black",
            linestyle="dashed",
            linewidth=1,
        )
        ax.text(
            break_dates[i],
            ax.get_ylim()[1] * 0.9,  # Position at 90% of y-axis max
            f"{cutoffs[i-3] * 100}%",  # Using the same labels for both sets
            color="black",
            fontsize=10,
            ha="right",
            va="top",
            rotation=90,
        )


# axes[2,0].set_yscale("log")
# axes[2,1].set_yscale("log")

# Collect legend handles and labels from all axes
handles = []
labels = []
for ax in axes.flat:  # Iterate through all axes in the figure
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Deduplicate legend entries
unique_legend = {}
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_legend:
        unique_legend[label] = handle
        unique_handles.append((handle, label))

# Create custom line handles for Wastewater and Clinical
wastewater_line = mlines.Line2D(
    [], [], color="black", linestyle="-", label="Wastewater"
)
clinical_line = mlines.Line2D([], [], color="black", linestyle="--", label="Clinical")

# Add custom handles to the legend
unique_handles.insert(0, (wastewater_line, "Wastewater"))
unique_handles.insert(1, (clinical_line, "Clinical"))

# Apply the deduplicated legend
fig.legend(
    [h for h, _ in unique_handles],
    [l for _, l in unique_handles],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

import string

# Generate panel labels: a, b, c, d, e, f, g, ...
panel_labels = list(string.ascii_lowercase)
# Loop through all subplots and label them
for i, ax in enumerate(axes.flatten()):
    ax.text(
        -0.1,
        1.2,  # Position: slightly above each subplot
        panel_labels[i],  # Get the next letter
        transform=ax.transAxes,  # Use subplot-relative coordinates
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )


# fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# Ensure layout is correct
plt.tight_layout()
plt.show()
