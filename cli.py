"""Punto de entrada único del pipeline. Ver `python cli.py --help`."""

import sys

import click

from scripts import browse_corpus, chunk_corpus
from scripts import download_corpus as corpus


@click.group()
def cli():
    """Pipeline de evaluación de recuperación híbrida sobre legislación financiera de la UE."""


@cli.command("download-corpus")
@click.option("--freeze", is_flag=True, help="Escribe data/corpus.sha256 con los hashes descargados (primera vez).")
def download_corpus(freeze: bool):
    """Descarga el corpus de EUR-Lex y verifica hashes y recuento de artículos."""
    sys.exit(corpus.run(freeze=freeze))


@cli.command("chunk")
def chunk():
    """Genera los chunks (por artículo y por ventanas) a partir del corpus congelado."""
    sys.exit(chunk_corpus.run())


@cli.command("list-articles")
@click.option("--act", help="aifmd, dora o aiact (o un CELEX). Sin esta opción, todos.")
@click.option("--range", "article_range", help="Rango de artículos, p. ej. 17-23.")
def list_articles(act: str | None, article_range: str | None):
    """Lista artículos (acto, número, título y tamaño)."""
    sys.exit(browse_corpus.run_list(act, article_range))


@cli.command("show-article")
@click.argument("act")
@click.argument("number", type=int)
def show_article(act: str, number: int):
    """Muestra el texto completo de un artículo: ACT (aifmd|dora|aiact) y NUMBER."""
    sys.exit(browse_corpus.run_show(act, number))


if __name__ == "__main__":
    cli()
