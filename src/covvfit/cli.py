import typer

import covvfit._cli.freyja as freyja
import covvfit._cli.infer as infer


def main():
    app = typer.Typer()

    # Add scripts here
    app.add_typer(freyja.app, name="gather_freyja")
    app.add_typer(infer.app, name="infer")

    app()


if __name__ == "__main__":
    main()
