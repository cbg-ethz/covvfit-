# Workflows

This directory contains Snakemake workflows.
Snakemake can be added to an existing Micromamba environment by using

```bash
$ micromamba install snakemake -c bioconda -c conda-forge
```



## Assessing bootstrap confidence intervals

This workflow constructs bootstrap confidence and calculates their
coverage in simulated data setting.

Run Snakemake workflow by using: 

```bash
$ snakemake -s workflows/boostrap_simulation.smk -c6
```
