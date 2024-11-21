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
        "data/{run_name}/total_counts.csv"
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
        expand("data/{run_name}/{variant}_counts.csv", variant=config["variant_list"], run_name=run_name)
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
            output_file = f"data/{run_name}/{variant}_counts.csv"
            df.to_csv(output_file, index=False)


# Rule to merge data
rule merge_data:
    input:
        total="data/{run_name}/total_counts.csv",
        variants=expand("data/{run_name}/{variant}_counts.csv", variant=config["variant_list"], run_name=run_name)
    output:
        "data/{run_name}/merged_clinical_data.csv"
    run:
        import pandas as pd
        from pango_aliasor.aliasor import Aliasor

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

        def is_parent_lineage(parent: str, child: str) -> bool:
            """
            Check if a lineage is a parent of another.

            Parameters:
            - parent (str): The potential parent lineage (e.g., "B.1.1.529").
            - child (str): The potential child lineage (e.g., "BA.1").

            Returns:
            - bool: True if the parent is indeed a parent of the child, False otherwise.
            """
            # Initialize Aliasor
            aliasor = Aliasor()
            
            # Convert both lineages to uncompressed forms
            full_parent = aliasor.uncompress(parent)
            full_child = aliasor.uncompress(child)
            
            # Check if the child starts with the parent
            return full_child.startswith(full_parent + ".")


        # Adjust for children using the is_parent_lineage function, handling nested hierarchies
        def adjust_for_children(data, variants):
            columns = variants
            adjusted_data = data.copy()
            aliasor = Aliasor()  # Initialize Aliasor

            # Sort columns by lineage hierarchy to handle deeper levels first
            sorted_columns = sorted(columns, key=lambda x: aliasor.uncompress(x).count("."), reverse=True)

            # Process columns from deepest to shallowest
            for parent in sorted_columns:
                # Check for children of the parent lineage
                children = [child for child in columns if is_parent_lineage(parent, child)]
                if children:
                    # Subtract the sum of the children counts from the parent
                    adjusted_data[parent] -= adjusted_data[children].sum(axis=1)
            
            return adjusted_data

        merged_df = adjust_for_children(merged_df, input.variants)
        merged_df.to_csv(output[0], index=False)

# Rule to filter and normalize data
rule filter_and_normalize_data:
    input:
        "data/{run_name}/merged_clinical_data.csv"
    output:
        "data/{run_name}/normalized_clinical_data.csv"
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
        "data/{run_name}/normalized_clinical_data.csv"
    output:
        "results/{run_name}/clinical_models/model_fitting_solution_{date}.json"
    params:
        variants_full=config["variants_full"],
        variants_investigated=config["variants_investigated"],
        start_date=config["filter"]["start_date"],
        n_starts=10,
        end_date="{date}"
    run:
        import pandas as pd
        import numpy as np
        import jax.numpy as jnp
        import json
        from covvfit import preprocess as preprocess
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
        variants_investigated = params.variants_investigated
        variants_other = [i for i in variants_full if i not in variants_investigated]
        variants_effective = ["other"] + variants_investigated

        # Preprocess the data
        data_full = preprocess.preprocess_df(
            filtered_df, cities, variants_full, zero_date=start_date
        )

        data_full["other"] = data_full[variants_other].sum(axis=1)
        data_full[variants_effective] = data_full[variants_effective].div(
            data_full[variants_effective].sum(axis=1), axis=0
        )

        # Merge and clean data
        data_full["time"] = pd.to_datetime(data_full["time"])
        data_full = data_full.merge(filtered_df[["time", "city", "count_sum"]], on=["time", "city"], how="left")
        data_full = data_full[~data_full.isna().any(axis=1)]

        # Prepare time series and observations
        ts_lst, ys_effective, ns_lst = preprocess.make_data_list(data_full, cities, variants_effective)
        # Scale the time for numerical stability
        time_scaler = preprocess.TimeScaler()
        ts_lst_scaled = time_scaler.fit_transform(ts_lst)
        t_max = time_scaler.t_max
        t_min = time_scaler.t_min


        # Fit the model
        loss = qm.construct_total_loss(
            ys=ys_effective,
            ts=ts_lst_scaled,
            ns=ns_lst,
            average_loss=False
        )
        theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants_effective))
        solution = qm.jax_multistart_minimize(loss, theta0, n_starts=params.n_starts)

        # Save the solution
        solution_data = {
            "solution": solution.x.tolist(),
            "variants": variants_investigated,
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
        "data/{run_name}/wastewater_preprocessed.csv"
    params:
        start_date=config["filter"]["start_date"],
        end_date=config["filter"]["end_date"],
        cities=config["wastewater_cities"],
        variants_full=config["variants_full"],
        variants=config["variants_investigated"]
    run:
        import pandas as pd
        from covvfit import preprocess as preprocess

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
        variants_effective = ["other"] + variants

        # Preprocess the data
        data2 = preprocess.preprocess_df(
            data_wide, cities, variants_full, date_min=start_date, zero_date=start_date
        )

        # Add "other" column and normalize
        data2["other"] = data2[variants_other].sum(axis=1)
        data2[variants_effective] = data2[variants_effective].div(data2[variants_effective].sum(axis=1), axis=0)

        # Filter by date range
        data2 = data2[(data2["time"] >= start_date) & (data2["time"] <= end_date)]

        # Save the preprocessed data
        data2.to_csv(output[0], index=False)

rule fit_wastewater_data:
    input:
        "data/{run_name}/wastewater_preprocessed.csv"
    output:
        "results/{run_name}/wastewater_models/wastewater_model_fitting_solution_{date}.json"
    params:
        variants_full=config["variants_full"],
        variants_investigated=config["variants_investigated"],
        start_date=config["filter"]["start_date"],
        cities=config["wastewater_cities"],
        n_starts=10,
        end_date="{date}"
    run:
        import pandas as pd
        import numpy as np
        import jax.numpy as jnp
        import json
        from covvfit import preprocess as preprocess
        from covvfit import quasimultinomial as qm

        # Load input data
        data_wide = pd.read_csv(input[0])

        # Filter data up to the given end date
        data_wide["time"] = pd.to_datetime(data_wide["time"])
        end_date = pd.Timestamp(params.end_date)
        data_wide = data_wide[data_wide["time"] <= end_date]

        cities = params.cities  # Extract cities from config
        variants_full = [v.rstrip("*") for v in params.variants_full] + ["undetermined"]
        variants_investigated = [v.rstrip("*") for v in params.variants_investigated]
        variants_other = [i for i in variants_full if i not in variants_investigated]
        variants_effective = ["other"] + variants_investigated

        # Preprocess the data
        data_full = preprocess.preprocess_df(data_wide, cities, variants_full, zero_date=params.start_date)
        data_full["other"] = data_full[variants_other].sum(axis=1)
        data_full[variants_effective] = data_full[variants_effective].div(
            data_full[variants_effective].sum(axis=1), axis=0
        )
        # Prepare time series and observations
        ts_lst, ys_effective = preprocess.make_data_list(
            data_full, cities=cities, variants=variants_effective
        )
        # Scale the time for numerical stability
        time_scaler = preprocess.TimeScaler()
        ts_lst_scaled = time_scaler.fit_transform(ts_lst)
        t_max = time_scaler.t_max
        t_min = time_scaler.t_min

        # Fit the model
        loss = qm.construct_total_loss(
            ys=ys_effective,
            ts=ts_lst_scaled,
            average_loss=False  # No averaging for covariance matrix shrinkage
        )
        theta0 = qm.construct_theta0(n_cities=len(cities), n_variants=len(variants_effective))
        solution = qm.jax_multistart_minimize(loss, theta0, n_starts=params.n_starts)

        # Save the solution
        solution_data = {
            "solution": solution.x.tolist(),
            "variants": variants_effective,
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
        variants=config["variants_investigated"] 
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

