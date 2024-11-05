"""utilities to plot"""
import pandas as pd
import numpy as np

from scipy.special import expit
from scipy.stats import norm

import matplotlib.lines as mlines
import matplotlib.patches as mpatches


colors_covsp = {
    "B.1.1.7": "#D16666",
    "B.1.351": "#FF6666",
    "P.1": "#FFB3B3",
    "B.1.617.1": "#66C265",
    "B.1.617.2": "#66A366",
    "BA.1": "#A366A3",
    "BA.2": "#CFAFCF",
    "BA.4": "#8a66ff",
    "BA.5": "#585eff",
    "BA.2.75": "#008fe0",
    "BQ.1.1": "#ac00e0",
    "XBB.1.9": "#bb6a33",
    "XBB.1.5": "#ff5656",
    "XBB.1.16": "#e99b30",
    "XBB.2.3": "#f5e424",
    "EG.5": "#b4e80b",
    "BA.2.86": "#FF20E0",
    "JN.1": "#00e9ff",  # improv
        "KP.2": "#D16666",  # improv
        "KP.3": "#66A366",  # improv   
        "XEC": "#A366A3",  # improv   
    "undetermined": "#969696",
}


def make_legend(colors, variants):
    """make a shared legend for the plot"""
    # Create a patch (i.e., a colored box) for each variant
    variant_patches = [
        mpatches.Patch(color=color, label=variants[i]) for i, color in enumerate(colors)
    ]

    # Create lines for "fitted", "predicted", and "observed" labels
    fitted_line = mlines.Line2D([], [], color="black", linestyle="-", label="fitted")
    predicted_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="predicted"
    )
    observed_points = mlines.Line2D(
        [], [], color="black", marker="o", linestyle="None", label="daily estimates"
    )
    blank_line = mlines.Line2D([], [], color="white", linestyle="", label="")

    # Combine all the legend handles
    handles = variant_patches + [
        blank_line,
        fitted_line,
        predicted_line,
        observed_points,
    ]

    return handles


def num_to_date(num, pos=None, date_min="2023-01-01", fmt="%b. '%y"):
    """convert days number into a date format"""
    date = pd.to_datetime(date_min) + pd.to_timedelta(num, "D")
    return date.strftime(fmt)


def plot_fit(ax, ts, y_fit, variants, colors, linetype='-'):
    """
    Function to plot fitted values with customizable line type.
    
    Parameters:
        ax (matplotlib.axes): The axis to plot on.
        ts (array-like): Time series data.
        y_fit (array-like): Fitted values for each variant.
        variants (list): List of variant names.
        colors (list): List of colors for each variant.
        linetype (str): Line style for plotting (e.g., '-', '--', '-.', ':').
    """
    sorted_indices = np.argsort(ts)
    for i, variant in enumerate(variants):
        ax.plot(
            ts[sorted_indices],
            y_fit[i, :][sorted_indices],
            color=colors[i],
            linestyle=linetype,
            label=f"fit {variant}",
        )
        
def plot_complement(ax, ts, y_fit, variants, color="grey", linetype="-"):
    ## function to plot 1-sum(fitted_values) i.e., the other variant(s)
    sorted_indices = np.argsort(ts)
    ax.plot(
        ts[sorted_indices],
        (1-y_fit.sum(axis=0))[sorted_indices],
        color=color,
        linestyle=linetype
    )
        
        
def plot_data(ax, ts, ys, variants, colors):
    ## function to plot raw values
    for i, variant in enumerate(variants):
        ax.scatter(
            ts,
            ys[i, :],
            label="observed",
            alpha=0.5,
            color=colors[i],
            s=4
        )
        

def plot_confidence_bands(ax, ts, conf_bands, variants, colors, label="Confidence band", alpha=0.2):
    """
    Plot confidence intervals for fitted values on a given axis with customizable confidence level.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        ts (array-like): Time series data.
        y_fit_logit (array-like): Logit-transformed fitted values for each variant.
        logit_se (array-like): Standard errors for the logit-transformed fitted values.
        color (str): Color for the confidence interval.
        confidence (float, optional): Confidence level (e.g., 0.95 for 95%). Default is 0.95.
        label (str, optional): Label for the confidence band. Default is "Confidence band".
    """    
    # Sort indices for time series
    sorted_indices = np.argsort(ts)
    
    # Plot the confidence interval
    for i, variant in enumerate(variants):
        ax.fill_between(
            ts[sorted_indices],
            conf_bands['lower'][i][sorted_indices],
            conf_bands['upper'][i][sorted_indices],
            color=colors[i],
            alpha=alpha,
            label=label
        )
    
