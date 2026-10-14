"""Punto de entrada único del pipeline. Ver `python cli.py --help`."""

import sys

import click

from scripts import browse_corpus, chunk_corpus, compare, db, evaluate, golden
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


@cli.command("validate-golden")
def validate_golden():
    """Valida data/golden (formato, artículos, grados) y muestra avisos de revisión."""
    sys.exit(golden.run_validate())


@cli.command("freeze-golden")
def freeze_golden():
    """Congela el golden set (SHA-256). Hazlo solo cuando esté completo."""
    sys.exit(golden.run_freeze())


@cli.command("evaluate")
@click.option("--system", "system_name", type=click.Choice(sorted(evaluate.SYSTEMS)), default="bm25", show_default=True)
@click.option("--chunking", type=click.Choice(["article", "window", "both"]), default="both", show_default=True)
def evaluate_command(system_name: str, chunking: str):
    """Evalúa un sistema sobre el golden set congelado y guarda results/<sistema>_<chunking>.json."""
    sys.exit(evaluate.run(system_name, chunking))


@cli.command("init-db")
def init_db_command():
    """Crea (o actualiza) el esquema en Postgres. Requiere `docker compose up -d` y el .env."""
    sys.exit(db.run_init())


@cli.command("load-db")
def load_db_command():
    """Carga data/chunks/*.jsonl en Postgres (idempotente: reemplaza lo anterior)."""
    sys.exit(db.run_load())


@cli.command("compare")
@click.argument("result_a")
@click.argument("result_b")
@click.option("--metric", default="ndcg@10", show_default=True, help="recall@1|3|5|10, ndcg@10 o mrr@10")
def compare_command(result_a: str, result_b: str, metric: str):
    """Diferencia pareada A − B (con IC bootstrap) entre dos resultados de results/, p. ej. bm25_window fts_window."""
    sys.exit(compare.run(result_a, result_b, metric))


if __name__ == "__main__":
    cli()
