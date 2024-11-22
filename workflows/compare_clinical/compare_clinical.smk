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
        f"results/{run_name}/comparative_estimates_plot.png",
        f"results/{run_name}/fitted_values.gif",


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
            filtered_df, cities, variants_full, zero_date=params.start_date
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

        # print(f"ts_lst = {ts_lst}")
        # print(f"t_max = {t_max}")

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
        data_full = preprocess.preprocess_df(
            data_wide,
            cities,
            variants_full,
            # date_min=start_date,
            zero_date=params.start_date
            )
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



rule plot_fitted_values:
    # TODO (David): 
    #   – plotting capabilities for wastewater data, or generalize
    #   – plotting capabilities for mixed ?
    #   – confidence bands
    input:
        data="data/{run_name}/normalized_clinical_data.csv",
        model="results/{run_name}/clinical_models/model_fitting_solution_{date}.json"
    output:
        "results/{run_name}/fitted_plots/fitted_values_plot_{date}.png"
    params:
        variants_investigated=config["variants_investigated"],
        variants_full=config["variants_full"],
        cities=config["filter"]["divisions"],
        start_date=config["filter"]["start_date"],
        end_date=config["filter"]["end_date"]
    run:
        import matplotlib
        matplotlib.use("Agg")  # Use a non-GUI backend for rendering plots
        import jax.numpy as jnp
        import json
        import matplotlib.pyplot as plt
        from matplotlib import ticker
        from covvfit import quasimultinomial as qm
        from covvfit import preprocess, plot
        plot_ts = plot.timeseries

        # Load input data and model solution
        with open(input.model, "r") as f:
            model_data = json.load(f)

        theta_star = jnp.array(model_data["solution"])
        t_min = model_data["t_min"]
        t_max = model_data["t_max"]
        n_variants_effective = len(params.variants_investigated) + 1  # Add "other"
        start_date = pd.Timestamp(params.start_date)
        end_date = pd.Timestamp(params.end_date)
        cities = params.cities

        # Load normalized data
        normalized_data = pd.read_csv(input.data)
        def make_raw_data(
                input_data=input.data,
                end_date=end_date,
                cities=cities,
                variants_full=params.variants_full,
                variants_investigated=params.variants_investigated,
            ):
            # Load input data
            filtered_df = pd.read_csv(input_data)
            filtered_df = filtered_df.rename(columns={"date": "time", "division": "city"})

            # Filter data up to the given end date
            filtered_df["time"] = pd.to_datetime(filtered_df["time"])
            filtered_df = filtered_df[filtered_df["time"] <= end_date]
            filtered_df = filtered_df[filtered_df["time"] >= start_date]
            filtered_df["undetermined"] = 0.0
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
            return (ts_lst, ys_effective, ns_lst)
        
        ts_lst_raw, ys_effective, ns_lst = make_raw_data()
        
        # Make ts_lst and ts_lst_scaled
        ts_lst = [np.arange(t_max) for i, city in enumerate(cities)]
        time_scaler = preprocess.TimeScaler()
        ts_lst_scaled = time_scaler.fit_transform(ts_lst)

        # Compute fitted values, covariance, and confidence intervals
        ys_fitted = qm.fitted_values(
            ts_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
        )

        # covariance = qm.get_covariance(loss, theta_star)
        # overdispersion_tuple = qm.compute_overdispersion(observed=ys_effective, predicted=ys_fitted)
        # overdisp_fixed = overdispersion_tuple.overall
        # covariance_scaled = overdisp_fixed * covariance
        # ys_fitted_confint = qm.get_confidence_bands_logit(
            # theta_star, n_variants=n_variants_effective, ts=ts_lst_scaled, covariance=covariance_scaled
        # )

        # Prepare predictions
        horizon = 60
        ts_pred_lst = [jnp.arange(horizon + 1) + tt.max() for tt in ts_lst]
        ts_pred_lst_scaled = time_scaler.transform(ts_pred_lst)
        ys_pred = qm.fitted_values(
            ts_pred_lst_scaled, theta=theta_star, cities=cities, n_variants=n_variants_effective
        )
        # ys_pred_confint = qm.get_confidence_bands_logit(
        #     theta_star, n_variants=n_variants_effective, ts=ts_pred_lst_scaled, covariance=covariance_scaled
        # )

        # Colors for variants
        colors = [plot_ts.COLORS_COVSPECTRUM[var.rstrip("*")] for var in params.variants_investigated]

        # Create plot grid
        figure_spec = plot.arrange_into_grid(len(cities), axsize=(4, 1.5), dpi=350, wspace=1)

        def plot_city(ax, i: int) -> None:
            def remove_0th(arr):
                return arr[:, 1:]

            plot_ts.plot_fit(ax, ts_lst[i], remove_0th(ys_fitted[i]), colors=colors)
            plot_ts.plot_fit(
                ax, ts_pred_lst[i], remove_0th(ys_pred[i]), colors=colors, linestyle="--"
            )
            # plot_ts.plot_confidence_bands(
            #     ax, ts_lst[i], jax.tree_map(remove_0th, ys_fitted_confint[i]), colors=colors
            # )
            # plot_ts.plot_confidence_bands(
            #     ax, ts_pred_lst[i], jax.tree_map(remove_0th, ys_pred_confint[i]), colors=colors
            # )
            plot_ts.plot_data(ax, ts_lst_raw[i], remove_0th(ys_effective[i]), colors=colors)
            # plot_ts.plot_complement(ax, ts_lst[i], remove_0th(ys_fitted[i]), alpha=0.3)
            # plot_ts.plot_complement(ax, ts_pred_lst[i], remove_0th(ys_pred[i]), linestyle="--", alpha=0.3)

            def format_date(x, pos):
                return plot_ts.num_to_date(x, date_min=start_date)

            date_formatter = ticker.FuncFormatter(format_date)
            ax.xaxis.set_major_formatter(date_formatter)
            ax.set_yticks([0, 0.5, 1])
            time_scaler_data = preprocess.TimeScaler()
            time_scaler_data.fit(ts_lst_raw)
            ax.set_xlim((time_scaler_data.t_min, time_scaler_data.t_max))
            ax.set_yticklabels(["0%", "50%", "100%"])
            ax.set_ylabel("Relative abundances")
            ax.set_title(cities[i])

        figure_spec.map(plot_city, range(len(cities)))
        plt.savefig(output[0])
        plt.close()


rule generate_gif:
    input:
        plots=expand("results/{run_name}/fitted_plots/fitted_values_plot_{date}.png", 
        date=date_range, run_name=run_name)
    output:
        gif="results/{run_name}/fitted_values.gif"
    params:
        duration=200  # Duration of each frame in milliseconds
    run:
        from PIL import Image

        # Sort input files to ensure chronological order
        plot_files = sorted(input.plots)
        images = [Image.open(file) for file in plot_files]

        # Save as GIF
        images[0].save(
            output.gif,
            save_all=True,
            append_images=images[1:],
            duration=params.duration,
            loop=0  # Infinite loop
        )
        print(f"GIF saved as {output.gif}")
