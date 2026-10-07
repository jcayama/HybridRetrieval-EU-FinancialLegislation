"""Punto de entrada único del pipeline. Ver `python cli.py --help`."""

import sys

import click

from scripts import download_corpus as corpus


@click.group()
def cli():
    """Pipeline de evaluación de recuperación híbrida sobre legislación financiera de la UE."""


@cli.command("download-corpus")
@click.option("--freeze", is_flag=True, help="Escribe data/corpus.sha256 con los hashes descargados (primera vez).")
def download_corpus(freeze: bool):
    """Descarga el corpus de EUR-Lex y verifica hashes y recuento de artículos."""
    sys.exit(corpus.run(freeze=freeze))


if __name__ == "__main__":
    cli()
