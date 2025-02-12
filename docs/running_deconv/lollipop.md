# Prepare deconvoluted data with *LolliPop* for *Covvfit* 

Here we explain how to prepare input for *Covvfit* using the [*LolliPop*](https://github.com/cbg-ethz/LolliPop) tool.

We need to deconvolute wastewater data with *LolliPop*, but we need to do it without smoothing (or with minimal smoothing) to avoid introducing bias by kernel smoothing procedures.

## Install *LolliPop*

Follow installation instructions described [here](https://github.com/cbg-ethz/LolliPop). They consist of the following: 

Create environment:
```bash
conda create -n lollipop 
conda activate lollipop
```

Clone repository:

```bash
git clone https://github.com/cbg-ethz/LolliPop.git
cd LolliPop
```

Install dependencies:
```bash
pip install '.[cli]'
```

## Get and prepare data

Make a directory for deconvolution results:

```bash
mkdir lollipop_covvfit
cd lollipop_covvfit
```

Obtain mutation data, for example from Euler:

```bash
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/variants/tallymut.tsv.zst .
zstd -d tallymut.tsv.zst 
```

Obtain the latest configuration files from euler:

```bash
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/var_dates.yaml .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/variant_config.yaml .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/ww_locations.tsv .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/filters_badmut.yaml .
```

Prepare parameters for the deconvolution:
```bash
cat << EOF > deconv_config.yaml
bootstrap: 0

kernel_params:
  bandwidth: 0.1

regressor: robust
regressor_params:
  f_scale: 0.01

deconv_params:
  min_tol: 1e-3
EOF
```

## Run *LolliPop*

To deconvolve the data, run:

```bash
cd ..
ldata="./lollipop_covvfit"
lollipop deconvolute $ldata/tallymut.tsv \
    -o $ldata/deconvolved.csv \
    --variants-config $ldata/variant_config.yaml \
    --variants-dates $ldata/var_dates.yaml \
    --deconv-config $ldata/deconv_config.yaml \
    --filters $ldata/filters_badmut.yaml  \
    --seed=42 \
    --n-cores=2
```

## Run *CovvFit* 

You are ready to use *Covvfit* and can proceed to the tutorial [here](../cli.md).  
