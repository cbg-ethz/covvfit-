import covvfit._quasimultinomial as quasimultinomial

try:
    import covvfit.simulation as simulation
except Exception as e:
    import warnings

    warnings.warn(
        f"It is not possible to use `simulation` subpackage due to missing dependencies. Exception raised: {e}"
    )
    simulation = None

import covvfit.plotting as plot
from covvfit._preprocess_abundances import load_data, make_data_list, preprocess_df
from covvfit._splines import create_spline_matrix

VERSION = "0.1.0"


__all__ = [
    "create_spline_matrix",
    "make_data_list",
    "preprocess_df",
    "load_data",
    "VERSION",
    "quasimultinomial",
    "plot",
    "simulation",
]
