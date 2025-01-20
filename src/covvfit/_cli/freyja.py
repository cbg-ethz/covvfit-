import typer

app = typer.Typer()


@app.command()
def gather(directory: str, output: str):
    """Gathers files from the directory into the output CSV."""
    typer.echo(f"Processing data file: {directory}")
    # Add your processing logic here
