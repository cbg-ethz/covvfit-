"""Main script to which subcommands from `covvfit._cli` can be added."""
import typer

from covvfit._cli.freyja import gather
from covvfit._cli.infer import infer

app = typer.Typer()

# Add scripts here
app.command()(gather)
app.command()(infer)


def main():
    app()


if __name__ == "__main__":
    main()
