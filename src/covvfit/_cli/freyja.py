"""Script gathering Freyja files into one CSV."""
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer


@dataclass
class _FreyjaData:
    lineages: list[str]
    abundances: list[float]
    resid: float
    coverage: float

    location: str | None
    timepoint: str | None


def _read_freyja(file_path: str, /) -> _FreyjaData:
    with open(file_path) as fp:
        content = fp.read()
    lines = content.split("\n")
    data = {}
    for line in lines:
        name, _, line_content = line.partition("\t")
        data[name.strip()] = line_content.strip()

    return _FreyjaData(
        lineages=data["lineages"].split(),
        abundances=[float(x) for x in data["abundances"].strip().split()],
        resid=float(data["resid"]),
        coverage=float(data["coverage"]),
        location=data.get("location", None),
        timepoint=data.get("timepoint", None),
    )


def _get_abundances(content: _FreyjaData, /) -> pd.DataFrame:
    abund, names = content.abundances, content.lineages

    # Validate that the abundances are within bounds
    for ab, name in zip(abund, names):
        if not (0 <= ab <= 1):
            raise ValueError(f"Abundance {ab} of variant {name} is out of bounds.")

    return pd.DataFrame(
        {
            "name": names,
            "abundance": abund,
        }
    )


_SAMPLE_COL: str = "sample"


def _parse_freyja(
    path: Path,
    *,
    sample: str,
    min_coverage: float,
) -> pd.DataFrame:
    """Parses Freyja file into a data frame.
    If the coverage is not matched, an empty data frame is returned.
    Adds the column for sample names.
    """
    content = _read_freyja(path)
    coverage = content.coverage

    if coverage < min_coverage:
        return pd.DataFrame()

    abundances = _get_abundances(content)
    abundances[_SAMPLE_COL] = sample

    return abundances


def freyja_gather(
    directory: Annotated[str, typer.Option(help="Directory with Freyja output")],
    output: Annotated[str, typer.Option(help="Desired output location for the CSV")],
    metadata: Optional[
        Annotated[
            Optional[str],
            typer.Option("--metadata", help="File with sample-specific metadata"),
        ]
    ] = None,
    metadata_sep: Annotated[
        str, typer.Option("--metadata-sep", help="Separator used in the metadata file.")
    ] = "\t",
    metadata_sample_col: Annotated[
        str,
        typer.Option(
            "--metadata-sample-col",
            help="Name of the column storing sample names in the metadata file",
        ),
    ] = "sample",
    min_coverage: Annotated[
        Optional[float],
        typer.Option(
            "--min-coverage",
            help="Minimum coverage threshold needed to use the sample. By default no threshold is applied.",
        ),
    ] = None,
    output_sep: Annotated[
        str, typer.Option("--output-sep", help="Separator used in the output file.")
    ] = "\t",
) -> None:
    """Gathers Freyja-demixed files from the given directory into the output CSV, adding the metadata from the file."""
    # TODO(Pawel): Add reading timepoint and location directly from the Freyja files, rather than metadata CSV

    if min_coverage is None:
        min_coverage = -1e9

    path = Path(directory)

    # Obtain the filenames
    sample_paths = list(path.glob("*.tsv"))

    # Get the large data frame with abundances
    abundance = pd.concat(
        [
            _parse_freyja(path=pth, sample=pth.stem, min_coverage=min_coverage)
            for pth in sample_paths
        ],
        ignore_index=True,
    )

    if metadata is not None:
        metadata = pd.read_csv(metadata, sep=metadata_sep)

        if metadata_sample_col not in metadata.columns:
            raise ValueError(
                f"Column {metadata_sample_col} not available in the metadata file with columns {metadata.columns}."
            )

        abundance = pd.merge(
            abundance,
            metadata,
            left_on=_SAMPLE_COL,
            right_on=metadata_sample_col,
            how="left",
        )

    # Generate the parent directories (if not exist)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save file
    abundance.to_csv(output_path, index=False, sep=output_sep)
