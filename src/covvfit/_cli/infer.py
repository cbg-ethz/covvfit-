import typer

app = typer.Typer()


@app.command()
def infer(data: str, output: str):
    """Runs inference."""
    typer.echo(f"Processing data file: {data} {output}")
    # Add your processing logic here
