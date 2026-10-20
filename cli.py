"""Punto de entrada único del pipeline. Ver `python cli.py --help`."""

import sys

import click

from scripts import abstention, bench_index, browse_corpus, chunk_corpus, compare, db, embeddings, error_analysis, evaluate, gate, golden, report
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
@click.option("--bonferroni", "family_size", type=int, default=None, help="Añade el IC global con corrección de Bonferroni para N comparaciones.")
def compare_command(result_a: str, result_b: str, metric: str, family_size: int | None):
    """Diferencia pareada A − B (con IC bootstrap) entre dos resultados de results/, p. ej. bm25_window fts_window."""
    sys.exit(compare.run(result_a, result_b, metric, family_size))


@cli.command("embed")
@click.option("--verify", "check", is_flag=True, help="No recalcula: comprueba que una muestra se reproduce con los vectores guardados.")
@click.option("--only", multiple=True, help="Calcula solo esta fuente (repetible) y conserva el resto del manifest, p. ej. --only abstention_queries.")
def embed_command(check: bool, only: tuple[str, ...]):
    """Calcula los embeddings (modelo fijado) y los guarda en data/embeddings/. Requiere requirements-embeddings.txt."""
    sys.exit(embeddings.run(check, list(only) or None))


@cli.command("bench-index")
@click.option("--size", type=int, default=bench_index.SYNTHETIC_SIZE, show_default=True, help="Vectores sintéticos.")
@click.option("--queries", "n_queries", type=int, default=bench_index.SYNTHETIC_QUERIES, show_default=True)
@click.option("--skip-synthetic", is_flag=True, help="Solo el corpus real.")
def bench_index_command(size: int, n_queries: int, skip_synthetic: bool):
    """Compara HNSW e IVFFlat (recall y latencia) en un esquema temporal; guarda results/index_benchmark.json."""
    sys.exit(bench_index.run(size, n_queries, skip_synthetic))


@cli.command("gate")
def gate_command():
    """Regression gate: falla si el híbrido (búsqueda exacta) cae bajo los umbrales de data/gate/thresholds.json."""
    sys.exit(gate.run_gate())


@cli.command("calibrate-gate")
@click.option("--write-thresholds", is_flag=True, help="Escribe data/gate/thresholds.json con los umbrales calculados.")
def calibrate_gate_command(write_thresholds: bool):
    """Recalcula el ruido, la sensibilidad a regresiones y los umbrales (local; requiere torch)."""
    sys.exit(gate.run_calibrate(write_thresholds))


@cli.command("gate-margin")
@click.argument("margin", type=float)
def gate_margin_command(margin: float):
    """Fija a mano el margen del gate (>= 0,01) y recalcula los umbrales desde la calibración guardada."""
    sys.exit(gate.run_set_margin(margin))


@cli.command("error-analysis")
def error_analysis_command():
    """Analiza por qué falla cada consulta mala del híbrido y guarda results/error_analysis.json (local; requiere torch)."""
    sys.exit(error_analysis.run())


@cli.command("freeze-abstention")
def freeze_abstention_command():
    """Valida y congela (SHA-256) las consultas sin respuesta de data/abstention."""
    sys.exit(abstention.run_freeze())


@cli.command("abstention")
@click.option("--write-threshold", is_flag=True, help="Escribe data/abstention/threshold.json (el umbral que usa la API) con la regla fijada de antemano.")
def abstention_command(write_threshold: bool):
    """Coseno del mejor chunk: golden set frente a consultas sin respuesta; guarda results/abstention.json (necesita Postgres cargado)."""
    sys.exit(abstention.run(write_threshold))


@cli.command("report")
@click.option("--check", is_flag=True, help="No escribe: falla si la sección de resultados del README está desactualizada.")
def report_command(check: bool):
    """Genera la sección de resultados del README a partir de results/ (o la comprueba con --check)."""
    sys.exit(report.run(check))


if __name__ == "__main__":
    cli()
