import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _get_simplex_axis(ax: Axes | None = None) -> tuple[Figure | None, Axes]:
    """Creates an axis, which represents a probability simplex."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    assert ax is not None

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    spine_width = ax.spines["left"].get_linewidth()
    spine_color = ax.spines["left"].get_edgecolor()

    # Plot x + y = 1 line as "diagonal spine"
    ax.plot(
        np.linspace(0, 1, 3),
        np.linspace(1, 0, 3),
        c=spine_color,
        linewidth=spine_width,
    )
    return fig, ax


def plot_on_simplex(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str = "x",
    y_label: str = "y",
    ax: Axes | None = None,
    line_color: str | None = "gray",
    line_width: float = 0.5,
    line_alpha: float = 0.7,
    line_style: str = "-",
    t: np.ndarray | None = None,
    scatter_size: float = 3**2,
    scatter_cmap: str = "Blues",
) -> tuple[Figure | None, Axes]:
    """Plots a trajectory on the probability simplex.

    Args:
        x: x-coordinates of the trajectory, shape (n,) or (n, 1)
        y: y-coordinates of the trajectory, shape (n,) or (n, 1)
        x_label: label for the x-axis
        y_label: label for the y-axis
        ax: axis to plot on. By default (None) a new axis is created
        line_color: color of the trajectory line. If None, no line is plotted
        line_width: width of the trajectory line
        t: optional parameter to color the trajectory scatter
        scatter_size: size of the scatter points
        scatter_cmap: colormap for the scatter points

    Returns:
        fig: figure of the plot or None if `ax` is specified
        ax: axis of the plot
    """
    assert x.shape == y.shape

    fig, ax = _get_simplex_axis(ax)

    if t is None:
        t = np.linspace(0, 1, x.shape[0])

    ax.plot(
        x.ravel(),
        y.ravel(),
        color=line_color,
        linewidth=line_width,
        alpha=line_alpha,
        linestyle=line_style,
    )
    ax.scatter(
        x.ravel(),
        y.ravel(),
        c=t,  # pyright: ignore
        cmap=scatter_cmap,
        s=scatter_size,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig, ax
