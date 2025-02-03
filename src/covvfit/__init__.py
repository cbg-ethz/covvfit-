import covvfit._deconvolution as deconvolution
import covvfit._dynamics as dynamics
import covvfit._numeric as numeric
import covvfit._quasimultinomial as quasimultinomial

try:
    import covvfit.simulation as simulation
except Exception as e:
    import warnings

    warnings.warn(
        f"It is not possible to use `simulation` subpackage due to missing dependencies. Exception raised: {e}"
    )
    simulation = None

import covvfit._preprocess_abundances as preprocess
import covvfit.plotting as plot
from covvfit._splines import create_spline_matrix

VERSION = "0.1.0"


__all__ = [
    "create_spline_matrix",
    "dynamics",
    "deconvolution",
    "load_data",
    "VERSION",
    "numeric",
    "quasimultinomial",
    "plot",
    "preprocess",
    "simulation",
]
