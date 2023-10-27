# covvfit
Fitness estimates of SARS-CoV-2 variants.

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

