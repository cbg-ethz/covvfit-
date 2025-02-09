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
import os
import pickle
import string

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from covvfit import plot

plot_ts = plot.timeseries

# %%
# %matplotlib inline

# Set default DPI for high-resolution plots
plt.rcParams["figure.dpi"] = 300  # Adjust to 150, 200, or more for higher resolution


# %% [markdown]
# # load data from the runs


# %%
def load_data(directory):
    """Load all output files from a run."""
    data = {}

    # Load numpy arrays
    for filename in ["diff_array.npy", "rel_diff_array.npy"]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            data[filename.split(".")[0]] = np.load(path)

    # Load pickle files
    for filename in [
        "ys_lst_full.pkl",
        "ts_lst_full.pkl",
        "ts_lst_smooth.pkl",
        "ys_lst_smooth.pkl",
    ]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                data[filename.split(".")[0]] = pickle.load(f)

    return data


# %%
ba1ba2 = load_data("results/ba1ba2/")
with open("config_ba1ba2.yaml", "r") as file:
    ba1ba2_config = yaml.safe_load(file)

jn1 = load_data("results/jn1/")
with open("config_jn1.yaml", "r") as file:
    jn1_config = yaml.safe_load(file)


ba1ba2_config["variants_evaluated"] = ["other"] + ba1ba2_config["variants_evaluated"]
all_configs = [ba1ba2_config, jn1_config]
all_data = [ba1ba2, jn1]

# %%
ba1ba2_config["variants_evaluated"] = ["BA.1", "BA.2"]
jn1_config["variants_evaluated"] = ["BA.2.86", "JN.1"]

# %% [markdown]
# # Plot results

# %%
# Define colormap for error plotting
cmap = cm.coolwarm
colors_covsp = plot_ts.COLORS_COVSPECTRUM
colors_covsp["other"] = "#969696"

horizons = [0, 7, 14, 21, 30, 60, 90]

# Create a 5-row subplot layout to include the new row
fig, axes = plt.subplots(
    6,
    len(all_configs),
    figsize=(12, 12),  # Increase figure height to accommodate extra row
    # sharex="col",
    sharey="row",
    gridspec_kw={"hspace": 0.55},  # Adjust vertical spacing
)

norm = mcolors.Normalize(vmin=min(horizons), vmax=max(horizons))
colors = [cmap(norm(h)) for h in horizons]

for k, (config, proc_data) in enumerate(zip(all_configs, all_data)):
    variants_investigated = config["variants_investigated"]
    variants_effective = ["other"] + variants_investigated
    variants_evaluated = config["variants_evaluated"]
    start_date = pd.to_datetime(config["start_date"])
    max_dates = pd.date_range(
        start=config["max_dates"]["start"], end=config["max_dates"]["end"], freq="D"
    )
    (
        diff_array,
        rel_diff_array,
        ys_lst_full,
        ts_lst_full,
        ts_lst_smooth,
        ys_lst_smooth,
    ) = proc_data.values()

    ax = axes[0, k]
    ## Plot the rel abundance of the unstudied variants
    variants_notevaluated = [
        i for i in variants_investigated if i not in variants_evaluated
    ]
    variants_notevaluated_index = np.where(
        np.isin(variants_effective, variants_notevaluated)
    )[0]

    dates_points = start_date + pd.to_timedelta(ts_lst_full[0], "D")
    dates_smooth = start_date + pd.to_timedelta(ts_lst_smooth[0], "D")

    for var_idx, var_name in zip(variants_notevaluated_index, variants_notevaluated):
        ax.plot(
            dates_smooth,
            ys_lst_smooth[0][:, var_idx],
            c=colors_covsp[var_name],
            alpha=0.25,
            label=var_name,
        )

    ## Plot the rel abundance of the studied variants
    variants_evaluated_index = np.where(
        np.isin(variants_effective, variants_evaluated)
    )[0]

    for var_idx, var_name in zip(variants_evaluated_index, variants_evaluated):
        ax.scatter(
            dates_points, ys_lst_full[0][:, var_idx], c=colors_covsp[var_name], s=2
        )
        ax.plot(
            dates_smooth,
            ys_lst_smooth[0][:, var_idx],
            c=colors_covsp[var_name],
            label=var_name,
        )
    ax.set_xlim(max_dates.min(), max_dates.max())
    ax.set_ylabel("rel. abundance")
    ax.set_title("smoothed deconvolved rel. abundances")

    ## Plot error
    for i, (var_idx, var_name) in enumerate(
        zip(variants_evaluated_index, variants_evaluated)
    ):
        ax = axes[1 + i, k]
        for j, horizon in enumerate(horizons):
            ax.plot(
                max_dates,
                diff_array.mean(axis=1)[:, horizon, var_idx],
                label=int(horizon),
                color=colors[j],
            )
        ax.set_ylabel("mean err.")
        ax.set_ylim((-0.5, 0.5))
        ax.set_title(f"{var_name} pred. error")
        ax.tick_params(axis="x", rotation=0)

    ## New row: Boxplot analysis for variants_evaluated
    for cutoff, newrow in zip([0.025, 0.05, 0.10], [3, 4, 5]):
        ax = axes[newrow, k]
        # Compute break index
        break_index = np.median(
            [np.where(smooth[:, -1] >= cutoff)[0][0] for smooth in ys_lst_smooth]
        )
        max_dates_idx = (
            max_dates - pd.to_datetime(config["start_date"])
        ) / pd.to_timedelta(1, "D")

        err_before = diff_array[max_dates_idx < break_index, :, :, :]
        err_after = diff_array[max_dates_idx >= break_index, :, :, :]

        # Get the number of groups from the last index dimension
        num_groups = err_before.shape[-1]

        # Prepare data for boxplot (only for variants_evaluated)
        boxplot_data = []
        boxplot_labels = []
        boxplot_colors = []
        horizon_positions = []

        for i, (condition, h) in enumerate(
            [(c, h) for c in ["Before", "After"] for h in horizons]
        ):
            for var_idx, var_name in zip(variants_evaluated_index, variants_evaluated):
                if condition == "Before":
                    data_subset = np.abs(err_before[:, :, h, var_idx]).flatten()
                else:
                    data_subset = np.abs(err_after[:, :, h, var_idx]).flatten()

                boxplot_data.append(data_subset)
                boxplot_labels.append(f"{condition}-{h}-{var_name}")
                boxplot_colors.append(colors_covsp[var_name])

        horizon_positions = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41]
        horizon_positions = np.arange(1.5, 2 * len(horizons) * 2 + 1, 2).tolist()

        # Create the boxplot
        box = ax.boxplot(
            boxplot_data,
            labels=boxplot_labels,
            patch_artist=True,
        )

        # Apply colors based on the variant
        for patch, flier, median, color in zip(
            box["boxes"], box["fliers"], box["medians"], boxplot_colors
        ):
            patch.set_facecolor(color)  # Set box interior color
            patch.set_edgecolor("black")  # Set thin black outline
            flier.set(
                marker=".",
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=1,
                linestyle="none",
            )
            median.set(color="black")

        # Set x-ticks only at horizon positions
        ax.set_xticks(horizon_positions)
        ax.set_xticklabels(horizons * 2, rotation=45)
        ax.set_ylabel("abs. err.")
        ax.set_xlabel("horizon (days)")
        ax.set_title(f"abs. pred. error before/after {cutoff*100}% cutoff")

        # add vline for cutoff

        ax.axvline(
            x=np.median(horizon_positions),
            color="black",
            linestyle="dashed",
            linewidth=1.5,
            # alpha=0.8,
        )

        # add cutoff to other plots
        break_date = pd.to_timedelta(break_index, "D") + pd.to_datetime(
            config["start_date"]
        )
        for i in range(3):
            axes[i, k].axvline(
                x=break_date,
                color="black",
                linestyle="dashed",
                linewidth=1.5,
                # alpha=0.8,
            )
            axes[i, k].text(
                break_date,
                axes[i, k].get_ylim()[1] * 0.9,  # Position at 90% of y-axis max
                f"{cutoff * 100}%",
                color="black",
                fontsize=10,
                ha="right",
                va="top",
                rotation=90,
            )


## Legend for horizons
# Deduplicate legend entries
handles, labels = axes[2, 1].get_legend_handles_labels()
unique_labels = {}
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels[label] = handle
        unique_handles.append((handle, label))

# Add unique legend
fig.legend(
    [h for h, _ in unique_handles],
    [z for _, z in unique_handles],
    loc="center",
    bbox_to_anchor=(1, 0.5),
    title="horizon (days)",
)

## Legend for variants
# Create a dictionary to store legend handles for variants
variant_handles = {}

# Collect variant legend handles from the first row of axes
for k in range(len(all_configs)):  # Iterate over columns
    handles, labels = axes[0, k].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in variant_handles:
            variant_handles[label] = handle  # Store unique handles

# Extract unique handles and labels
variant_legend_handles = list(variant_handles.values())
variant_legend_labels = list(variant_handles.keys())

# Add the variant legend above the horizon legend
fig.legend(
    variant_legend_handles,
    variant_legend_labels,
    loc="upper center",
    bbox_to_anchor=(1, 0.9),
    title="Variants",
)


# Share x-axis for rows 0,1,2,3 within each column
for k in range(len(all_configs)):  # Iterate over columns
    for row in range(1, 3):  # Rows 1,2,3 share x-axis with row 0 in the same column
        axes[row, k].sharex(axes[0, k])

# remove x label for second to last row
axes[-3, 0].set_xlabel("")
axes[-3, 1].set_xlabel("")
axes[-2, 0].set_xlabel("")
axes[-2, 1].set_xlabel("")


for k in range(len(all_configs)):  # Iterate over columns
    axes[0, k].xaxis.set_major_locator(
        mdates.MonthLocator(bymonthday=1)
    )  # First day of each month
    axes[0, k].xaxis.set_major_formatter(
        mdates.DateFormatter("%b '%y")
    )  # Format: "Jan '21"

# Generate panel labels: a, b, c, d, e, f, g, ...
panel_labels = list(string.ascii_lowercase)

# Loop through all subplots and label them
for row in range(6):  # 5 rows
    for col in range(len(all_configs)):  # Number of columns
        axes[row, col].text(
            -0.1,
            1.2,  # Position: slightly above each subplot
            panel_labels[row * len(all_configs) + col],  # Get the next letter
            transform=axes[row, col].transAxes,  # Use subplot-relative coordinates
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="right",
        )

fig.tight_layout()
plt.show()


# %%
threshs = np.linspace(0.01, 0.25, 1000)
horizons = [7, 14, 21, 30, 45, 60, 90]

norm = mcolors.Normalize(vmin=min(horizons), vmax=max(horizons))
colors = [cmap(norm(h)) for h in horizons]

fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex="col", sharey="row")

for k, (config, proc_data) in enumerate(zip(all_configs, all_data)):
    variants_investigated = config["variants_investigated"]
    variants_effective = ["other"] + variants_investigated
    variants_evaluated = config["variants_evaluated"]
    start_date = pd.to_datetime(config["start_date"])
    max_dates = pd.date_range(
        start=config["max_dates"]["start"], end=config["max_dates"]["end"], freq="D"
    )
    (
        diff_array,
        rel_diff_array,
        ys_lst_full,
        ts_lst_full,
        ts_lst_smooth,
        ys_lst_smooth,
    ) = proc_data.values()

    err_before_s = []
    err_after_s = []
    for thresh in threshs:
        break_index = np.median(
            [np.where(smooth[:, -1] >= thresh)[0][0] for smooth in ys_lst_smooth]
        )
        max_dates_idx = (
            max_dates - pd.to_datetime(config["start_date"])
        ) / pd.to_timedelta(1, "D")

        err_before = diff_array[max_dates_idx < break_index, :, :, :]
        err_after = diff_array[max_dates_idx >= break_index, :, :, :]

        err_before_s.append(np.abs(err_before).mean(axis=(0, 1)))
        err_after_s.append(np.abs(err_after).mean(axis=(0, 1)))

    axes[0, k].plot(threshs, np.array(err_before_s)[:, 0, -1], c="black", label="0")
    axes[1, k].plot(threshs, np.array(err_after_s)[:, 0, -1], c="black")
    for j, horizon in enumerate(horizons):
        axes[0, k].plot(
            threshs, np.array(err_before_s)[:, horizon, -1], c=colors[j], label=horizon
        )
        axes[1, k].plot(threshs, np.array(err_after_s)[:, horizon, -1], c=colors[j])

    axes[0, k].set_title(f"{variants_effective[-1]} before threshold")
    axes[1, k].set_title(f"{variants_effective[-1]} after threshold")

for i in [0, 1]:
    for j in [0, 1]:
        axes[i, j].set_ylabel("mean abs. err.")
        axes[i, j].set_xlabel("threshold")

handles, labels = axes[0, 0].get_legend_handles_labels()

# Add unique legend
fig.legend(
    handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="horizon (days)"
)
fig.tight_layout()
plt.show()
