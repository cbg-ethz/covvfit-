"""Script gathering Freyja files into one CSV."""
from typing import Annotated

import typer


def gather(
    directory: Annotated[str, typer.Argument(help="Directory with Freyja output")],
    metadata: Annotated[str, typer.Argument(help="Metadata")],
    output: Annotated[str, typer.Argument(help="Desired output location")],
):
    """Gathers files from the directory into the output CSV."""
    typer.echo(f"Processing data file: {directory}")
    # Add your processing logic here
