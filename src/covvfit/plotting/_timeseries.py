"""utilities to plot"""
import pandas as pd

import matplotlib.lines as mlines
import matplotlib.patches as mpatches


colors_covsp = {
    "B.1.1.7": "#D16666",
    "B.1.351": "#FF6666",
    "P.1": "#FFB3B3",
    "B.1.617.1": "#A3FFD1",
    "B.1.617.2": "#66C266",
    "BA.1": "#A366A3",
    "BA.2": "#CFAFCF",
    "BA.4": "#8467F6",
    "BA.5": "#595EF6",
    "BA.2.75": "#DE9ADB",
    "BQ.1.1": "#8fe000",
    "XBB.1.9": "#dd6bff",
    "XBB.1.5": "#ff5656",
    "XBB.1.16": "#e99b30",
    "XBB.2.3": "#b4b82a",
    "EG.5": "#359f99",
    "BA.2.86": "#FF20E0",
    "JN.1": "#00e9ff",
    "undetermined": "#999696",
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
