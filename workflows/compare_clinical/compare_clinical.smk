import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Generate list of dates from date_range
start_date = datetime.strptime(config["filter"]["date_range"]["start"], "%Y-%m-%d")
end_date = datetime.strptime(config["filter"]["date_range"]["end"], "%Y-%m-%d")
date_range = [
    (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range((end_date - start_date).days + 1)
]

# where to store data and results
run_name = config["run_name"]

# Rule all
rule all:
    input:
        f"results/{run_name}/consolidated_clinical_results.csv",
        f"results/{run_name}/consolidated_wastewater_results.csv",
        f"results/{run_name}/comparative_estimates_plot.png"


# Rule to fetch total counts
rule fetch_total_counts:
    output:
        "data/total_counts.csv"
    params:
        url=config["api_url"],
        params={"fields": "date,division", "country": config["country"]}
    run:
        import requests
        import pandas as pd

        response = requests.get(params.url, params=params.params)
        if response.status_code == 200:
            data = response.json().get("data", [])
            df = pd.DataFrame(data).rename(columns={"count": "total_count"})
            df.to_csv(output[0], index=False)
        else:
            print(f"Error: {response.status_code}")
            pd.DataFrame().to_csv(output[0])


rule fetch_variant_counts:
    output:
        expand("data/{variant}_counts.csv", variant=config["variant_list"])
    params:
        url=config["api_url"],
        country=config["country"],
        variants=config["variant_list"]
    run:
        import requests
        import pandas as pd

        for variant in params.variants:
            # Prepare API request parameters
            params_variant = {
                "fields": "date,division",
                "country": params.country,
                "nextcladePangoLineage": variant
            }

            # Fetch data
            response = requests.get(params.url, params=params_variant)
            if response.status_code == 200:
                data = response.json().get("data", [])
                df = pd.DataFrame(data).rename(columns={"count": variant})
            else:
                print(f"Error fetching {variant}: {response.status_code}")
                df = pd.DataFrame()

            # Save the data
            output_file = f"data/{variant}_counts.csv"
            df.to_csv(output_file, index=False)


# Rule to merge data
rule merge_data:
    input:
        total="data/total_counts.csv",
        variants=expand("data/{variant}_counts.csv", variant=config["variant_list"])
    output:
        "data/merged_clinical_data.csv"
    run:
        import pandas as pd

        df_total = pd.read_csv(input.total)
        merged_df = df_total.copy()

        for variant_file in input.variants:
            variant = variant_file.split("/")[-1].replace("_counts.csv", "")
            df_variant = pd.read_csv(variant_file).rename(columns={"count": variant})
            merged_df = pd.merge(merged_df, df_variant, on=["date", "division"], how="left")

        # Clean data
        merged_df = merged_df.dropna(subset=["date"])
        merged_df["date"] = pd.to_datetime(merged_df["date"])
        merged_df = merged_df.fillna(0)

        merged_df.to_csv(output[0], index=False)

# Rule to filter and normalize data
rule filter_and_normalize_data:
    input:
        "data/merged_clinical_data.csv"
    output:
        "results/{run_name}/normalized_clinical_data.csv"
    params:
        start_date=config["filter"]["start_date"],
        end_date=config["filter"]["end_date"],
        divisions=config["filter"]["divisions"],
        variants=config["variant_list"]
    run:
        import pandas as pd

        def filter_data(df, start_date, end_date, divisions):
            mask = (
                (df["date"] >= start_date)
                & (df["date"] <= end_date)
                & (df["division"].isin(divisions))
            )
            return df[mask].copy()

        def normalize_data(df, variant_list):
            variant_columns = variant_list
            df = df.copy()
            total_n = df[variant_columns].sum(axis=1)
            for variant in variant_columns:
                df[variant] = df[variant] / total_n
            df["count_sum"] = total_n
            return df

        # Load and process
        merged_df = pd.read_csv(input[0])
        filtered_df = filter_data(
            merged_df, params.start_date, params.end_date, params.divisions
        )
        filtered_df = normalize_data(filtered_df, params.variants)

        filtered_df.to_csv(output[0], index=False)

# Rule to fit clinical data
rule fit_clinical_data:
    input:
        "results/{run_name}/normalized_clinical_data.csv"
    output:
        "results/{run_name}/clinical_models/model_fitting_solution_{date}.json"
    params:
        variants_full=config["variants_full"],
        variants=config["variants"],
        start_date=config["filter"]["start_date"],
        n_starts=10,
        end_date="{date}"
    run:
        import pandas as pd
        import numpy as np
        import jax.numpy as jnp
        import json
        from covvfit import _preprocess_abundances as prec
        from covvfit import quasimultinomial as qm

        # Load input data
        filtered_df = pd.read_csv(input[0])
        filtered_df = filtered_df.rename(columns={"date": "time", "division": "city"})

        # Filter data up to the given end date
        filtered_df["time"] = pd.to_datetime(filtered_df["time"])
        end_date = pd.Timestamp(params.end_date)
        filtered_df = filtered_df[filtered_df["time"] <= end_date]

        cities = filtered_df["city"].unique().tolist()
        filtered_df["undetermined"] = 0.0

        variants_full = params.variants_full
        variants = params.variants
        variants_other = [i for i in variants_full if i not in variants]
        variants2 = ["other"] + variants

        # Preprocess the data
        data2 = prec.preprocess_df(filtered_df, cities, variants_full, zero_date=params.start_date)
        data2["other"] = data2[variants_other].sum(axis=1)
        data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

        # Merge and clean data
        data2["time"] = pd.to_datetime(data2["time"])
        data2 = data2.merge(filtered_df[["time", "city", "count_sum"]], on=["time", "city"], how="left")
        data2 = data2[~data2.isna().any(axis=1)]

        # Prepare time series and observations
        ts_lst, ys_lst, ns_lst = prec.make_data_list(data2, cities, variants2)
        t_min = min([ts.min() for ts in ts_lst])
        t_max = max([ts.max() for ts in ts_lst])
        ts_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_lst]
        observed_data = [y.T for y in ys_lst]

        # Fit the model
        loss = qm.construct_total_loss(
            ys=observed_data,
            ts=ts_lst_scaled,
            ns=ns_lst,
            average_loss=False
        )
        theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants2))
        solution = qm.jax_multistart_minimize(loss, theta0, n_starts=params.n_starts)

        # Save the solution
        solution_data = {
            "solution": solution.x.tolist(),
            "variants": variants,
            "t_min": float(t_min),
            "t_max": float(t_max),
            "end_date": params.end_date
        }
        with open(output[0], "w") as f:
            json.dump(solution_data, f)



rule gather_clinical_results:
    input:
        expand(
            "results/{run_name}/clinical_models/model_fitting_solution_{date}.json",
         date=date_range, run_name=run_name,
         )
    output:
        "results/{run_name}/consolidated_clinical_results.csv"
    run:
        import json
        import pandas as pd

        all_results = []

        # Loop through all input files and collect data
        for file in input:
            with open(file, "r") as f:
                data = json.load(f)

                # Flatten the data into a single dictionary
                flattened_result = {
                    "end_date": data["end_date"],
                    "t_max": data["t_max"]
                }
                # Add each element of "solution" as its own column
                for i, value in enumerate(data["solution"]):
                    flattened_result[f"solution_{i}"] = value

                all_results.append(flattened_result)

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(all_results)

        # Save the DataFrame as a CSV file
        df.to_csv(output[0], index=False)


rule import_wastewater_data:
    input:
        config["wastewater_data_path"]
    output:
        "data/wastewater_preprocessed.csv"
    params:
        start_date=config["filter"]["start_date"],
        end_date=config["filter"]["end_date"],
        cities=config["wastewater_cities"],
        variants_full=config["variants_full"],
        variants=config["variants"]
    run:
        import pandas as pd
        from covvfit import _preprocess_abundances as prec

        # Load and pivot the data
        data = pd.read_csv(input[0], sep="\t")
        data_wide = data.pivot_table(
            index=["date", "location"], columns="variant", values="proportion", fill_value=0
        ).reset_index()
        data_wide = data_wide.rename(columns={"date": "time", "location": "city"})

        # Extract parameters
        start_date = pd.Timestamp(params.start_date)
        end_date = pd.Timestamp(params.end_date)
        cities = params.cities
        variants_full = params.variants_full
        variants = params.variants

        # Preprocess variants
        variants_full = [v.rstrip("*") for v in params.variants_full] + ["undetermined"]
        variants = [v.rstrip("*") for v in params.variants]

        # Define "other" variants
        variants_other = [i for i in variants_full if i not in variants]
        variants2 = ["other"] + variants

        # Preprocess the data
        data2 = prec.preprocess_df(
            data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
        )

        # Add "other" column and normalize
        data2["other"] = data2[variants_other].sum(axis=1)
        data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

        # Filter by date range
        data2 = data2[(data2["time"] >= start_date) & (data2["time"] <= end_date)]

        # Save the preprocessed data
        data2.to_csv(output[0], index=False)

rule fit_wastewater_data:
    input:
        "data/wastewater_preprocessed.csv"
    output:
        "results/{run_name}/wastewater_models/wastewater_model_fitting_solution_{date}.json"
    params:
        variants_full=config["variants_full"],
        variants=config["variants"],
        start_date=config["filter"]["start_date"],
        cities=config["wastewater_cities"],
        n_starts=10,
        end_date="{date}"
    run:
        import pandas as pd
        import numpy as np
        import jax.numpy as jnp
        import json
        from covvfit import _preprocess_abundances as prec
        from covvfit import quasimultinomial as qm

        # Load input data
        data_wide = pd.read_csv(input[0])

        # Filter data up to the given end date
        data_wide["time"] = pd.to_datetime(data_wide["time"])
        end_date = pd.Timestamp(params.end_date)
        data_wide = data_wide[data_wide["time"] <= end_date]

        cities = params.cities  # Extract cities from config
        variants_full = [v.rstrip("*") for v in params.variants_full] + ["undetermined"]
        variants = [v.rstrip("*") for v in params.variants]
        variants_other = [i for i in variants_full if i not in variants]
        variants2 = ["other"] + variants

        # Preprocess the data
        data2 = prec.preprocess_df(data_wide, cities, variants_full, zero_date=params.start_date)
        data2["other"] = data2[variants_other].sum(axis=1)
        data2[variants2] = data2[variants2].div(data2[variants2].sum(axis=1), axis=0)

        # Prepare time series and observations
        ts_lst, ys_lst = prec.make_data_list(data2, cities, variants2)
        ts_lst, ys_lst2 = prec.make_data_list(data2, cities, variants)
        t_max = max([x.max() for x in ts_lst])
        t_min = min([x.min() for x in ts_lst])
        ts_lst_scaled = [(x - t_min) / (t_max - t_min) for x in ts_lst]
        observed_data = [y.T for y in ys_lst]

        # Fit the model
        loss = qm.construct_total_loss(
            ys=observed_data,
            ts=ts_lst_scaled,
            average_loss=False  # No averaging for covariance matrix shrinkage
        )
        theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants2))
        solution = qm.jax_multistart_minimize(loss, theta0, n_starts=params.n_starts)

        # Save the solution
        solution_data = {
            "solution": solution.x.tolist(),
            "variants": variants2,
            "t_min": float(t_min),
            "t_max": float(t_max),
            "end_date": params.end_date
        }
        with open(output[0], "w") as f:
            json.dump(solution_data, f)

rule gather_wastewater_results:
    input:
        expand("results/{run_name}/wastewater_models/wastewater_model_fitting_solution_{date}.json",
         date=date_range,
         run_name=run_name,
         )
    output:
        "results/{run_name}/consolidated_wastewater_results.csv"
    run:
        import json
        import pandas as pd

        all_results = []

        # Loop through all input files and collect data
        for file in input:
            with open(file, "r") as f:
                data = json.load(f)

                # Flatten the data into a single dictionary
                flattened_result = {
                    "end_date": data["end_date"],
                    "t_max": data["t_max"]
                }
                # Add each element of "solution" as its own column
                for i, value in enumerate(data["solution"]):
                    flattened_result[f"solution_{i}"] = value

                all_results.append(flattened_result)

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(all_results)

        # Save the DataFrame as a CSV file
        df.to_csv(output[0], index=False)

rule plot_comparative_estimates:
    input:
        wastewater="results/{run_name}/consolidated_wastewater_results.csv",
        clinical="results/{run_name}/consolidated_clinical_results.csv"
    output:
        "results/{run_name}/comparative_estimates_plot.png"
    params:
        variants=config["variants"] 
    run:
        import matplotlib
        matplotlib.use("Agg")  # Use a non-GUI backend for rendering plots
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, AutoDateLocator

        # Load data
        wastewater_df = pd.read_csv(input.wastewater, parse_dates=["end_date"])
        clinical_df = pd.read_csv(input.clinical, parse_dates=["end_date"])

        # Add normalized solution columns
        for i in range(len(params.variants)):
            wastewater_df[f"solution_{i}_normalized"] = wastewater_df[f"solution_{i}"] / wastewater_df["t_max"]
            clinical_df[f"solution_{i}_normalized"] = clinical_df[f"solution_{i}"] / clinical_df["t_max"]

        # Plotting
        plt.figure(figsize=(12, 6))

        # Plot for wastewater data
        for i, variant in enumerate(params.variants):
            plt.plot(
                wastewater_df["end_date"], 
                wastewater_df[f"solution_{i}_normalized"], 
                label=f"Wastewater rate {variant}", 
            )

        # Plot for clinical data
        for i, variant in enumerate(params.variants):
            plt.plot(
                clinical_df["end_date"], 
                clinical_df[f"solution_{i}_normalized"], 
                linestyle="--", 
                label=f"Clinical rate {variant}",
            )

        # Formatting
        plt.xlabel("End Date")
        plt.ylabel("Relative Growth Rate")
        plt.title("Fitness Advantages Measurable at Different Dates")
        plt.ylim((0, 0.6))
        plt.legend()

        # Format x-axis for dates
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output[0])
        plt.close()

