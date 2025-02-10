import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import yaml
from pathlib import Path

from covvfit import plot, preprocess
from covvfit import quasimultinomial as qm

import numpy as np
import pickle

plot_ts = plot.timeseries

run_name = config["run_name"]

DATA_DIR = Path("../../data/main/")
DATA_PATH = DATA_DIR / "deconvolved.csv"
VAR_DATES_PATH = DATA_DIR / "var_dates.yaml"

# Define the list with cities:
cities = config["cities"]
variants_full = config["variants_full"]
variants_investigated = config["variants_investigated"]
variants_other = [
    i for i in variants_full if i not in variants_investigated
]  # Variants not of interest
variants_evaluated = config["variants_evaluated"]

# date parameters for the range of the simulation 

max_date = pd.to_datetime(config["max_date"])
start_date = pd.to_datetime(config["start_date"])
max_dates = pd.date_range(
    start=config["max_dates"]["start"],
    end=config["max_dates"]["end"],
    freq='D'
      )
max_horizon = config["max_horizon"]

# smoothing parameters:
sigma = config["sigma"]
smoothing_method = config["smoothing_method"]
max_smooth_date = max_date + pd.to_timedelta(3*sigma + max_horizon, "D")

## function definitions

def prepare_data(data_wide, start_date, max_date):
    """function to prepare data with the right timespan"""

    data_wide2 = data_wide.copy()

    ## Set limit times for modeling
    data_wide2["time"] = pd.to_datetime(data_wide2["time"])
    
    data_wide2 = data_wide2[data_wide2["time"] >= start_date]
    data_wide2 = data_wide2[data_wide2["time"] <= max_date]

    variants_effective = ["other"] + variants_investigated

    
    data_full = preprocess.preprocess_df(
        data_wide2, cities, variants_full, date_min=start_date, zero_date=start_date
    )
    
    data_full["other"] = data_full[variants_other].sum(axis=1)
    data_full[variants_effective] = data_full[variants_effective].div(
        data_full[variants_effective].sum(axis=1), axis=0
    )

    ts_lst, ys_effective = preprocess.make_data_list(
        data_full, cities=cities, variants=variants_effective
    )
    
    # Scale the time for numerical stability
    time_scaler = preprocess.TimeScaler()
    ts_lst_scaled = time_scaler.fit_transform(ts_lst)

    return {
        "data_wide2":data_wide2,
        "ts_lst":ts_lst,
        "ys_effective":ys_effective,
        "ts_lst_scaled":ts_lst_scaled,
        "time_scaler":time_scaler,
    }

def inference(ys_effective, ts_lst_scaled, ts_lst, time_scaler, horizon=7):
    # no priors
    loss = qm.construct_total_loss(
        ys=ys_effective,
        ts=ts_lst_scaled,
        average_loss=False,  # Do not average the loss over the data points, so that the covariance matrix shrinks with more and more data added
    )
    
    variants_effective = ["other"] + variants_investigated
    n_variants_effective = len(variants_effective)
    # initial parameters
    theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=n_variants_effective)

    # Run the optimization routine
    solution = qm.jax_multistart_minimize(loss, theta0, n_starts=10)
    theta_star = solution.x  # The maximum quasilikelihood estimate

    ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
    ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)
    
    ys_pred = qm.fitted_values(
        ts_pred_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
    )

    return {
        "ys_pred":ys_pred,
        "ts_pred_lst":ts_pred_lst,
        "solution":solution,
    }

def gaussian_kernel(t, ts, sigma):
    """Compute Gaussian weights for a given t"""
    return np.exp(-0.5 * ((ts - t) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def smooth_value(t, ts, ys, sigma):
    """Compute the smoothed value at time t using a Gaussian-weighted average."""
    weights = gaussian_kernel(t, ts, sigma)
    weights /= weights.sum()  # Normalize weights
    return np.dot(weights, ys)

def smooth_value_median(t, ts, ys, sigma):
    """Compute the smoothed value at time t using a Gaussian-weighted median for each dimension in ys."""
    weights = gaussian_kernel(t, ts, sigma)
    sorted_indices = np.argsort(ys, axis=0)
    ys_sorted = np.take_along_axis(ys, sorted_indices, axis=0)
    weights_sorted = np.take_along_axis(weights[:, np.newaxis], sorted_indices, axis=0)
    cumsum_weights = np.cumsum(weights_sorted, axis=0)
    median_indices = np.argmax(cumsum_weights >= 0.5 * cumsum_weights[-1], axis=0)
    return np.take_along_axis(ys_sorted, median_indices[np.newaxis, :], axis=0)[0]

def smooth_array(tt, ts, ys, sigma, method="mean"):
    """Compute smoothed values for an array of t values."""
    if method=="mean":
        return np.array([smooth_value(t, ts, ys, sigma) for t in tt])
    if method=="median":
        return np.array([smooth_value_median(t, ts, ys, sigma) for t in tt])

def compare(ts_in, ys_in, ts_smooth, ys_smooth):
    """Compare predicted data ys_in at ts_in with smoothed data ys_smooth at ts_smooth."""
    indices = np.where(np.isin(ts_smooth, ts_in))[0]
    if len(indices) == 0:
        raise ValueError("No matching time points found in ts_smooth")
    
    differences = ys_in - ys_smooth[indices]
    relative_differences = differences / ys_smooth[indices]
    return (differences , relative_differences)

def compare_all(ts_lst_pred, ys_lst_pred, ts_lst_smooth, ys_lst_smooth):
    """Compare all time series in the given lists."""
    all_differences = []
    all_relative_differences = []
    for ts_in, ys_in, ts_smooth, ys_smooth in zip(ts_lst_pred, ys_lst_pred, ts_lst_smooth, ys_lst_smooth):
        differences, relative_differences = compare(ts_in, ys_in, ts_smooth, ys_smooth)
        all_differences.append(differences)
        all_relative_differences.append(relative_differences)
    return (all_differences, all_relative_differences)

# Rule all
rule all:
    input:
        f"results/{run_name}/diff_array.npy",
        f"results/{run_name}/rel_diff_array.npy",
        f"results/{run_name}/ys_lst_full.pkl",
        f"results/{run_name}/ts_lst_full.pkl",
        f"results/{run_name}/ts_lst_smooth.pkl",
        f"results/{run_name}/ys_lst_smooth.pkl"

rule run_experiment:
    output:
        f"results/{run_name}/diff_array.npy",
        f"results/{run_name}/rel_diff_array.npy",
        f"results/{run_name}/ys_lst_full.pkl",
        f"results/{run_name}/ts_lst_full.pkl",
        f"results/{run_name}/ts_lst_smooth.pkl",
        f"results/{run_name}/ys_lst_smooth.pkl"
    run:

        ## load the data
        data = pd.read_csv(DATA_PATH, sep="\t")
        data_wide = data.pivot_table(
            index=["date", "location"], columns="variant", values="proportion", fill_value=0
        ).reset_index()
        data_wide = data_wide.rename(columns={"date": "time", "location": "city"})

        # Load the YAML file
        with open(VAR_DATES_PATH, "r") as file:
            var_dates_data = yaml.safe_load(file)

        # Access the var_dates data
        var_dates = var_dates_data["var_dates"]

        ## make the smoothing:

        prep_dat_full = prepare_data(data_wide, start_date, max_smooth_date)

        ys_lst_full = prep_dat_full["ys_effective"]
        ts_lst_full = prep_dat_full["ts_lst"]

        ts_lst_smooth = []
        ys_lst_smooth = []
        for i in range(len(ys_lst_full)):
            min_t = int(ts_lst_full[i].min())
            max_t = int(ts_lst_full[i].max())
            ts_tmp = np.linspace(min_t, max_t, max_t-min_t+1)
            ys_smooth = smooth_array(ts_tmp, ts_lst_full[i], ys_lst_full[i], sigma=sigma, method=smoothing_method)
            
            ts_lst_smooth.append(ts_tmp)
            ys_lst_smooth.append(ys_smooth)

        # Save smoothed etc
        with open(f"results/{run_name}/ys_lst_full.pkl", "wb") as f:
            pickle.dump(ys_lst_full, f)
        with open(f"results/{run_name}/ts_lst_full.pkl", "wb") as f:
            pickle.dump(ts_lst_full, f)
        with open(f"results/{run_name}/ts_lst_smooth.pkl", "wb") as f:
            pickle.dump(ts_lst_smooth, f)
        with open(f"results/{run_name}/ys_lst_smooth.pkl", "wb") as f:
            pickle.dump(ys_lst_smooth, f)

        import tqdm
        import warnings

        # Initialize list to store results
        all_differences_results = []
        all_relative_differences_results = []

        # Loop through max_dates and perform 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for max_date in tqdm.tqdm(max_dates, leave=False):
                prep_dat = prepare_data(data_wide, start_date, max_date)
                
                inf_dat = inference(
                    prep_dat["ys_effective"],
                    prep_dat["ts_lst_scaled"],
                    prep_dat["ts_lst"],
                    prep_dat["time_scaler"],
                    horizon=max_horizon,
                )
                
                diff_res, rel_diff_res = compare_all(inf_dat["ts_pred_lst"], inf_dat["ys_pred"], ts_lst_smooth, ys_lst_smooth)
                
                all_differences_results.append(diff_res)
                all_relative_differences_results.append(rel_diff_res)

        all_differences_results_array = np.array(all_differences_results)
        all_relative_differences_results_array = np.array(all_relative_differences_results)

        np.save(f"results/{run_name}/diff_array.npy", all_differences_results_array)
        np.save(f"results/{run_name}/rel_diff_array.npy", all_relative_differences_results_array)






