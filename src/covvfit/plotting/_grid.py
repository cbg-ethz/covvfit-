from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from subplots_from_axsize import subplots_from_axsize


def set_axis_off(ax: Axes, i: int = 0, j: int = 0) -> None:
    """Hides the axis."""
    ax.set_axis_off()


def plot_on_rectangular_grid(
    nrows: int,
    diag_func: Callable[[Axes, int], Any],
    under_diag: Callable[[Axes, int, int], Any],
    over_diag: Callable[[Axes, int, int], Any] = set_axis_off,
    ncols: int | None = None,
    axsize: tuple[float, float] = (2.0, 2.0),
    **subplot_kw,
) -> tuple[Figure, np.ndarray]:
    """Creates a rectangular grid of subplots.

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


@dataclass(frozen=False)
class ArrangedGrid:
    """A two-dimensional grid of axes.

    Attrs:
        fig: Matplotlib figure.
        axes: one-dimensional array of active axes,
            with length equal to the number of active plots
        axes_grid: two-dimensional array of all axes.


    Note:
        The number of plots in `axes_grid` is typically
        greater than the one in `axes`, as `axes_grid`
        contains also the axes which are not active
    """

    fig: Figure
    axes: np.ndarray
    axes_grid: np.ndarray

    @property
    def n_active(self) -> int:
        return len(self.axes)

    def map(
        self,
        func: Callable[[Axes], None] | Callable[[Axes, Any], None],
        arguments: list | None = None,
    ) -> None:
        """Applies a function to each active plotting axis.

        Args:
            func: function to be applied. It can have
                signature func(ax: plt.Axes)
                if `arguments` is None, which modifies
                the axis in-place.

                If `arguments` is not None, then the function
                should have the signature
                func(ax: plt.Axes, argument)
                where `argument` is taken from the `arguments`
                list
        """
        if arguments is None:
            for ax in self.axes:
                func(ax)
        else:
            if self.n_active != len(arguments):
                raise ValueError(
                    f"Provide one argument for each active axis, in total {self.n_active}"
                )
            for ax, arg in zip(self.axes, arguments):
                func(ax, arg)

    def set_titles(self, titles: list[str]) -> None:
        for title, ax in zip(titles, self.axes):
            ax.set_title(title)

    def set_xlabels(self, labels: list[str]) -> None:
        for label, ax in zip(labels, self.axes):
            ax.set_xlabel(label)

    def set_ylabels(self, labels: list[str]) -> None:
        for label, ax in zip(labels, self.axes):
            ax.set_ylabel(label)


def _calculate_nrows(n: int, ncols: int):
    if ncols < 1:
        raise ValueError(f"ncols has to be at least 1, was {ncols}.")
    return int(np.ceil(n / ncols))


def arrange_into_grid(
    nplots: int,
    ncols: int = 2,
    axsize: tuple[float, float] = (2.0, 1.0),
    **kwargs,
) -> ArrangedGrid:
    """Builds an array of plots to accommodate
    the axes listed.

    Args:
        nplots: number of plots
        ncols: number of columns
        axsize: axis size
        kwargs: keyword arguments to be passed to
            `subplots_from_axsize`. For example,
            ```
            wspace=0.2,  # Changes the horizontal spacing
            hspace=0.3,  # Changes the vertical spacing
            left=0.5,    # Changes the left margin
            ```
    """
    nrows = _calculate_nrows(nplots, ncols=ncols)

    fig, axs = subplots_from_axsize(
        axsize=axsize,
        nrows=nrows,
        ncols=ncols,
        **kwargs,
    )

    # Set not used axes
    for i, ax in enumerate(axs.ravel()):
        if i >= nplots:
            ax.set_axis_off()

    return ArrangedGrid(
        fig=fig,
        axes=axs.ravel()[:nplots],
        axes_grid=axs,
    )
