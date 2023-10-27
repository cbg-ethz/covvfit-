[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![build](https://github.com/cbg-ethz/covvfit/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/cbg-ethz/covvfit/actions/workflows/test.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# covvfit
Fitness estimates of SARS-CoV-2 variants.

  - **Documentation:** [https://cbg-ethz.github.io/covvfit](https://cbg-ethz.github.io/covvfit)
  - **Source code:** [https://github.com/cbg-ethz/covvfit](https://github.com/cbg-ethz/covvfit)
  - **Bug reports:** [https://github.com/cbg-ethz/covvfit/issues](https://github.com/cbg-ethz/covvfit/issues)


## Installation

### Developers
Create a new environment, e.g., using [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
```bash
$ micromamba create -n covvfit -c conda-forge python=3.10
```

Then, install the package. 

For a machine where development happens it comes with developer utilities:

```bash
$ pip install poetry
$ poetry install --with dev
$ pre-commit install
```

## See Also

  - [V-pipe](https://cbg-ethz.github.io/V-pipe/): a bioinformatics pipeline for viral sequencing data.
  - [cojac](https://github.com/cbg-ethz/cojac): command-line tools for the analysis of co-occurrence of mutations on amplicons.

