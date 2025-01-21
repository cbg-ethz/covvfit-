"""Main script to which subcommands from `covvfit._cli` can be added."""
import typer

from covvfit._cli.freyja import freyja_gather
from covvfit._cli.infer import infer

app = typer.Typer()


def _add_script(fn):
    app.command()(fn)


# Add functions processinf the data:
_add_script(freyja_gather)
_add_script(infer)


def main():
    app()


if __name__ == "__main__":
    main()
