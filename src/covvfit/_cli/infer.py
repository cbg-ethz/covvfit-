"""Script running Covvfit inference on the data."""
from typing import Annotated

import typer


def infer(
    data: Annotated[str, typer.Argument(help="Data path")],
    output: Annotated[str, typer.Argument(help="Output directory")],
):
    """Runs inference."""
    typer.echo(f"Processing data file: {data} {output}")
    # Add your processing logic here
