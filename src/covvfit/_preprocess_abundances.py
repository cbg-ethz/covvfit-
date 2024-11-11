"""utilities to preprocess relative abundances"""
import pandas as pd


def load_data(file) -> pd.DataFrame:
    wwdat = pd.read_csv(file)
    wwdat = wwdat.rename(columns={wwdat.columns[0]: "time"})  # pyright: ignore
    return wwdat


def preprocess_df(
    df: pd.DataFrame,
    cities: list[str],
    variants: list[str],
    undertermined_thresh: float = 0.01,
    zero_date: str = "2023-01-01",
    date_min: str | None = None,
    date_max: str | None = None,
) -> pd.DataFrame:
    """Preprocessing."""
    df = df.copy()

    # Convert the 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Remove days with too high undetermined
    df = df[df["undetermined"] < undertermined_thresh]  # pyright: ignore

    # Subset the 'BQ.1.1' column
    df = df[["time", "city"] + variants]  # pyright: ignore

    # Subset only the specified cities
    df = df[df["city"].isin(cities)]  # pyright: ignore

    # Create a new column which is the difference in days between zero_date and the date
    df["days_from"] = (df["time"] - pd.to_datetime(zero_date)).dt.days

    # Subset dates
    if date_min is not None:
        df = df[df["time"] >= pd.to_datetime(date_min)]  # pyright: ignore
    if date_max is not None:
        df = df[df["time"] < pd.to_datetime(date_max)]  # pyright: ignore

    return df


def make_data_list(
    df: pd.DataFrame,
    cities: list[str],
    variants: list[str],
) -> tuple[list, list]:
    ts_lst = [df[(df.city == city)].days_from.values for city in cities]
    ys_lst = [
        df[(df.city == city)][variants].values.T for city in cities
    ]  # pyright: ignore

    return (ts_lst, ys_lst)
