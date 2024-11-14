import warnings

try:
    import covvfit._frequentist as freq

    warnings.warn("The `freq` submodule is deprecated.")
except Exception as e:
    warnings.warn(
        f"It is not possible to use `freq` subpackage due to missing dependencies. Exception raised: {e}"
    )
    freq = None

try:
    import covvfit._quasimultinomial as quasimultinomial
except Exception as e:
    warnings.warn(
        f"It is not possible to use `quasimultinomial` subpackage due to missing dependencies. Exception raised: {e}"
    )
    freq_jax = None

try:
    import covvfit.simulation as simulation
except Exception as e:
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
    "freq",
    "quasimultinomial",
    "plot",
    "simulation",
]
