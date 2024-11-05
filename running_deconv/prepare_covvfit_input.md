# Prepare deconvolved data for covvfit

Here we explain how to prepare input for covvfit. We need to deconvolve wastewater data with _LolliPop_, but we need to do it without smoothing (or with minimal smoothing) to avoid biases due to biases in kernel smoothers. 

## Install LolliPop

Follow installation detailed here https://github.com/cbg-ethz/LolliPop 

Create environment:
```console
conda create -n lollipop 
conda activate lollipop
```

Clone repo:
```console
git clone https://github.com/cbg-ethz/LolliPop.git
cd LolliPop
```

Install dependencies:
```console
pip install '.[cli]'
```

## Get and prepare data

Make a dir for our results:

```console
mkdir lollipop_covvfit
cd lollipop_covvfit
```

Get mutation data from euler:

```console
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/variants/tallymut.tsv.zst .
zstd -d tallymut.tsv.zst 
```

Get latest config files from euler:

```console
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/var_dates.yaml .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/variant_config.yaml .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/ww_locations.tsv .
rsync -avz --progress euler:/cluster/project/pangolin/work-vp-test/filters_badmut.yaml .
```

Prepare parameters for the deconv
```console
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

## Run LolliPop

Run:

```console
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

## Run CovvFit 

Run the notebook. 



