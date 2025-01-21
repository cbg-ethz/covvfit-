"""Script gathering Freyja files into one CSV."""
from typing import Annotated

import typer


def freyja_gather(
    directory: Annotated[str, typer.Argument(help="Directory with Freyja output")],
    metadata: Annotated[str, typer.Argument(help="Metadata")],
    output: Annotated[str, typer.Argument(help="Desired output location")],
):
    """Gathers Freyja-demixed files from the given directory into the output CSV,
    adding the metadata."""
    typer.echo(
        "This function is not implemented yet. It will be added at a later time."
    )
    typer.Exit(code=1)
    # Add your processing logic here
