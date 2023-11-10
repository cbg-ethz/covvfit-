from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def set_axis_off(ax: Axes, i: int = 0, j: int = 0) -> None:
    """Hides the axis."""
    ax.set_axis_off()


def plot_grid(
    nrows: int,
    diag_func: Callable[[Axes, int], Any],
    under_diag: Callable[[Axes, int, int], Any],
    over_diag: Callable[[Axes, int, int], Any] = set_axis_off,
    ncols: int | None = None,
    axsize: tuple[float, float] = (2.0, 2.0),
    **subplot_kw,
) -> tuple[Figure, np.ndarray]:
    """Creates a grid of subplots.

    Args:
        nrows: number of rows
        diag_func: function to plot on the diagonal.
            Should have a signature (Axes, row_index)
        under_diag: function to plot under the diagonal.
            Should have a signature (Axes, row_index, col_index)
        over_diag: function to plot over the diagonal.
            Should have a signature (Axes, row_index, col_index)
        ncols: number of columns. By default equal to the number of rows
        axsize: size of the axes
        subplot_kw: keyword arguments passed to `plt.subplots`, e.g. `dpi=300`

    Returns:
        figure
        array of axes, shape `(nrows, ncols)`
    """
    assert nrows > 0
    if ncols is None:
        ncols = nrows

    assert ncols is not None
    figsize = (ncols * axsize[0], nrows * axsize[1])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **subplot_kw)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if i == j:
                diag_func(ax, i)
            elif i > j:
                under_diag(ax, i, j)
            else:
                over_diag(ax, i, j)

    return fig, axes
