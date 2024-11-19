"""utilities to preprocess relative abundances"""

import pandas as pd
from jaxtyping import Array, Float


def load_data(file) -> pd.DataFrame:
    wwdat = pd.read_csv(file)
    wwdat = wwdat.rename(columns={wwdat.columns[0]: "time"})  # pyright: ignore
    return wwdat


def preprocess_df(
    df: pd.DataFrame,
    cities: list[str],
    variants: list[str],
    *,
    undetermined_thresh: float = 0.01,
    zero_date: str = "2023-01-01",
    date_min: str | None = None,
    date_max: str | None = None,
    time_col: str = "time",
    city_col: str = "city",
    undetermined_col: str | None = "undetermined",
) -> pd.DataFrame:
    """Preprocessing function.

    Args:
        df: data frame with data
        cities: cities for which the data will be processed
        variants: variants which will be processed, they should
            be represented by different columns in `df`
        undetermined_thresh: threshold of the `undetermined` variant
            used to remove days with too many missing values.
            Use `None` to not remove any data
        zero_date: reference time point
        date_min: the lower bound of the data to be selected.
            Set to `None` to not set a bound
        date_max: see `date_min`
        time_col: column with dates representing days
        city_col: column with cities
        undetermined_col: column with the undetermined variant
    """
    df = df.copy()

    # Convert the 'time' column to datetime
    df[time_col] = pd.to_datetime(df["time"])

    # Remove days with too high undetermined
    if undetermined_col is not None:
        df = df[df[undetermined_col] < undetermined_thresh]  # pyright: ignore

    # Subset the columns corresponding to variants
    df = df[[time_col, city_col] + variants]  # pyright: ignore

    # Subset only the specified cities
    df = df[df[city_col].isin(cities)]  # pyright: ignore

    # Create a new column which is the difference in days between zero_date and the date
    df["days_from"] = (df[time_col] - pd.to_datetime(zero_date)).dt.days

    # Subset dates
    if date_min is not None:
        df = df[df[time_col] >= pd.to_datetime(date_min)]  # pyright: ignore
    if date_max is not None:
        df = df[df[time_col] < pd.to_datetime(date_max)]  # pyright: ignore

    return df


def make_data_list(
    df: pd.DataFrame,
    cities: list[str],
    variants: list[str],
) -> tuple[
    list[Float[Array, " timepoints"]], list[Float[Array, "timepoints variants"]]
]:
    ts_lst = [df[(df.city == city)].days_from.values for city in cities]
    ys_lst = [
        df[(df.city == city)][variants].values for city in cities
    ]  # pyright: ignore
    # TODO(David, Pawel): How should we handle this case?
    #   It *implicitly* changes the output data type, basing on the input value.
    #   Do we even use this feature?
    if "count_sum" in df.columns:
        ns_lst = [df[(df.city == city)].count_sum.values for city in cities]
        return (ts_lst, ys_lst, ns_lst)
    else:
        return (ts_lst, ys_lst)


_ListTimeSeries = list[Float[Array, " timeseries"]]


class TimeScaler:
    """Scales a list of time series, so that the values are normalized."""

    def __init__(self):
        self.t_min = None
        self.t_max = None
        self._fitted = False

    def fit(self, ts: _ListTimeSeries) -> None:
        """Fit the scaler parameters to the provided time series.

        Args:
            ts: list of timeseries, i.e., `ts[i]` is an array
                of some length `n_timepoints[i]`.
        """
        self.t_min = min([x.min() for x in ts])
        self.t_max = max([x.max() for x in ts])
        self._fitted = True

    def transform(self, ts: _ListTimeSeries) -> _ListTimeSeries:
        """Returns scaled values.

        Args:
            ts: list of timeseries, i.e., `ts[i]` is an array
                of some length `n_timepoints[i]`.

        Returns:
            list of exactly the same format as `ts`

        Note:
            The model has to be fitted first.
        """
        if not self._fitted:
            raise RuntimeError("You need to fit the model first.")

        denominator = self.t_max - self.t_min
        return [(x - self.t_min) / denominator for x in ts]

    def fit_transform(self, ts: _ListTimeSeries) -> _ListTimeSeries:
        """Fits the model and returns scaled values.

        Args:
            ts: list of timeseries, i.e., `ts[i]` is an array
                of some length `n_timepoints[i]`.

        Returns:
            list of exactly the same format as `ts`

        Note:
            This function is equivalent to calling
            first `fit` method and then `transform`.
        """
        self.fit(ts)
        return self.transform(ts)
