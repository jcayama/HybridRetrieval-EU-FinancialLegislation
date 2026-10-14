"""Regression gate: comprueba que el sistema principal (híbrido, búsqueda exacta) no cae bajo unos umbrales.

Un gate NO compara sistemas con estadística (el IC de muestreo del golden set es de ±0,1): detecta que la tubería
haya cambiado de comportamiento sobre los mismos datos congelados. Por eso el margen depende del RUIDO DE
IMPLEMENTACIÓN y de plataforma, no del de muestreo.

Regla del umbral (fijada antes de calcular):
  ruido   = max(diferencia entre entornos observada para FTS, cambio máximo del híbrido al perturbar los vectores)
  margen  = max(MIN_MARGIN, 3 × ruido) redondeado hacia arriba a múltiplos de ROUND
  umbral  = valor actual − margen, redondeado hacia abajo a múltiplos de ROUND

Uso:
    python cli.py gate                              # comprueba (lo ejecuta la CI)
    python cli.py calibrate-gate [--write-thresholds]   # recalcula ruido, sensibilidad y umbrales (local, requiere torch)
"""

import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from scripts import db, embeddings
from scripts import evaluate as ev
from scripts import fts as fts_module
from scripts import hybrid as hybrid_module
from scripts.dense import DenseIndex, precomputed_query_embedder
from scripts.download_corpus import ROOT, sha256_bytes
from scripts.fts import FtsIndex
from scripts.golden import QUERIES_FILE, load_jsonl
from scripts.hybrid import HybridIndex
from scripts.metrics import evaluate_run, load_qrels, mean_metrics

THRESHOLDS_FILE = ROOT / "data" / "gate" / "thresholds.json"
CALIBRATION_FILE = ROOT / "results" / "gate_calibration.json"
CHUNKINGS = ("article", "window")
METRICS = ("recall@10", "ndcg@10")
MIN_MARGIN = 0.01
SAFETY = 3
ROUND = 0.005
# Diferencia máxima observada en las métricas globales de FTS entre PostgreSQL 17.11 de Homebrew (macOS) y el
# contenedor pgvector/pgvector:pg17 (Debian): Recall@3 con ventanas, 0,377 frente a 0,369. Medida UNA vez a mano;
# el resto de métricas difirió menos. No es reproducible desde este script (requiere un segundo servidor).
CROSS_ENVIRONMENT_NOISE = 0.008
FLOAT_LEVEL_SIZES = (1e-6, 1e-5)  # emulan diferencias de coma flotante entre arquitecturas: entran en la regla del margen
LADDER_SIZES = (1e-4, 1e-3, 1e-2)  # añadida tras ver que lo anterior daba 0: solo informativa, NO entra en la regla
PERTURBATION_SEEDS = 20


# ---------- regla del umbral (funciones puras) ----------


def round_up(value: float, step: float = ROUND) -> float:
    return round(math.ceil(value / step - 1e-9) * step, 6)


def round_down(value: float, step: float = ROUND) -> float:
    return round(math.floor(value / step + 1e-9) * step, 6)


def margin_for(noise: float) -> float:
    return max(MIN_MARGIN, round_up(SAFETY * noise))


def floor_for(current: float, margin: float) -> float:
    return round_down(current - margin)


def thresholds_for(current: dict[str, dict[str, float]], margin: float) -> dict[str, dict[str, float]]:
    return {c: {m: floor_for(current[c][m], margin) for m in METRICS} for c in CHUNKINGS}


def check(current: dict[str, dict[str, float]], thresholds: dict[str, dict[str, float]]) -> list[str]:
    """Fallos: métricas por debajo de su umbral, o sin valor. `current` y `thresholds`: {chunking: {métrica: valor}}."""
    failures = []
    for chunking, floors in thresholds.items():
        for metric, floor in floors.items():
            value = current.get(chunking, {}).get(metric)
            if value is None:
                failures.append(f"{chunking} {metric}: sin valor")
            elif value < floor:
                failures.append(f"{chunking} {metric}: {value:.3f} < umbral {floor:.3f}")
    return failures


# ---------- evaluación ----------


def _evaluate_index(index, chunking: str, queries: list[dict], qrels) -> dict[str, float]:
    means = mean_metrics(evaluate_run(ev.run_system(index, queries), qrels))
    return {m: means[m] for m in METRICS}


def hybrid_index(chunking: str, embed_query=None, schema: str | None = None):
    sha = sha256_bytes(ev.CHUNK_FILES[chunking].read_bytes())
    return HybridIndex(chunking, sha, embed_query or precomputed_query_embedder(), schema)


def current_values() -> dict[str, dict[str, float]]:
    """Métricas del híbrido con búsqueda EXACTA sobre el corpus público, para cada chunking."""
    queries, qrels = load_jsonl(QUERIES_FILE), load_qrels()
    return {c: _evaluate_index(hybrid_index(c), c, queries, qrels) for c in CHUNKINGS}


# ---------- calibración: ruido ----------


def perturbed_embedder(sigma: float, seed: int):
    base = precomputed_query_embedder()
    rng = np.random.default_rng(seed)
    cache: dict[str, np.ndarray] = {}

    def embed(query: str) -> np.ndarray:
        if query not in cache:
            v = base(query).astype(np.float64)
            v = v * (1.0 + sigma * rng.standard_normal(v.shape))
            cache[query] = (v / np.linalg.norm(v)).astype(np.float32)
        return cache[query]

    return embed


def perturbation_noise(queries: list[dict], qrels, baseline: dict, sizes: tuple[float, ...]) -> dict:
    out = {}
    for sigma in sizes:
        worst = {c: dict.fromkeys(METRICS, 0.0) for c in CHUNKINGS}
        for seed in range(PERTURBATION_SEEDS):
            for c in CHUNKINGS:
                values = _evaluate_index(hybrid_index(c, perturbed_embedder(sigma, seed)), c, queries, qrels)
                for m in METRICS:
                    worst[c][m] = max(worst[c][m], abs(values[m] - baseline[c][m]))
        out[f"{sigma:g}"] = worst
    return out


# ---------- calibración: sensibilidad a regresiones plausibles ----------


@contextmanager
def patched(module, name: str, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def simple_config_schema():
    """Esquema temporal con FTS en la configuración 'simple' (sin stemming ni palabras vacías)."""
    import secrets

    import psycopg

    name = f"gatecal_{secrets.token_hex(3)}"
    with psycopg.connect(**db.conninfo(), autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {name}")
    try:
        with db.connect(name) as conn:
            conn.execute(db.SCHEMA_FILE.read_text(encoding="utf-8").replace("to_tsvector('english'", "to_tsvector('simple'"))
            for strategy, path in ev.CHUNK_FILES.items():
                ids, matrix = embeddings.load_embeddings(f"chunks_{strategy}")
                manifest = embeddings.read_manifest()["files"][f"chunks_{strategy}"]["sha256"]
                db.load_chunks(conn, strategy, ev.load_chunks(strategy), sha256_bytes(Path(path).read_bytes()), dict(zip(ids, matrix, strict=True)), manifest)
        with patched(fts_module, "SEARCH_SQL", fts_module.SEARCH_SQL.replace("'english'", "'simple'")):
            yield name
    finally:
        with psycopg.connect(**db.conninfo(), autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA {name} CASCADE")


def scenarios(queries: list[dict], qrels) -> dict[str, dict]:
    base = precomputed_query_embedder()
    texts = [q["query"] for q in queries]
    shifted = {texts[i]: base(texts[(i + 1) % len(texts)]) for i in range(len(texts))}
    out: dict[str, dict] = {}

    def run(name: str, factory):
        out[name] = {c: _evaluate_index(factory(c), c, queries, qrels) for c in CHUNKINGS}

    def sha(c):
        return sha256_bytes(ev.CHUNK_FILES[c].read_bytes())

    run("hibrido_actual", lambda c: hybrid_index(c))
    run("solo_denso (el lado léxico no aporta nada)", lambda c: DenseIndex(c, sha(c), base))
    run("solo_fts (el lado denso no aporta nada)", lambda c: FtsIndex(c, sha(c)))
    run("consultas_desalineadas (cada consulta recibe el vector de otra)", lambda c: hybrid_index(c, lambda q: shifted[q]))
    try:
        model = embeddings.load_model()
        raw = dict(zip(texts, embeddings.encode(model, texts), strict=True))  # SIN la instrucción del modelo
        run("consultas_sin_instruccion_del_modelo", lambda c: hybrid_index(c, lambda q: raw[q]))
    except (ImportError, RuntimeError, OSError) as error:
        out["consultas_sin_instruccion_del_modelo"] = {"omitido": str(error)}
    for k in (1, 1000):
        with patched(hybrid_module, "K", k):
            run(f"rrf_k_{k}", lambda c: hybrid_index(c))
    with simple_config_schema() as schema:
        run("fts_config_simple_en_vez_de_english", lambda c: hybrid_index(c, schema=schema))
    return out


# ---------- calibración completa y comprobación ----------


def calibrate(write_thresholds: bool = False) -> dict:
    queries, qrels = load_jsonl(QUERIES_FILE), load_qrels()
    current = current_values()
    noise_runs = perturbation_noise(queries, qrels, current, FLOAT_LEVEL_SIZES)
    ladder = perturbation_noise(queries, qrels, current, LADDER_SIZES)
    perturbation_max = max(v for sizes in noise_runs.values() for per_chunking in sizes.values() for v in per_chunking.values())
    noise = max(CROSS_ENVIRONMENT_NOISE, perturbation_max)
    margin = margin_for(noise)
    thresholds = thresholds_for(current, margin)
    sens = scenarios(queries, qrels)
    for name, per_chunking in sens.items():
        for c, values in per_chunking.items():
            if c in CHUNKINGS:
                values["falla_el_gate"] = bool(check({c: values}, {c: thresholds[c]}))
    report = {
        "rule": {"safety": SAFETY, "min_margin": MIN_MARGIN, "round": ROUND, "cross_environment_noise": CROSS_ENVIRONMENT_NOISE},
        "perturbation_noise_max_abs_change": noise_runs,
        "perturbation_ladder_max_abs_change_informative": ladder,
        "perturbation_max": perturbation_max,
        "noise": noise,
        "margin": margin,
        "current": current,
        "thresholds": thresholds,
        "scenarios": sens,
    }
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_thresholds:
        write_thresholds_file(thresholds, margin, "regla declarada (3 × ruido)")
    return report


def write_thresholds_file(thresholds: dict, margin: float, source: str) -> None:
    THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    spec = {"system": "hybrid (FTS + denso, RRF k=60), búsqueda exacta", "margin": margin, "margin_source": source, "thresholds": thresholds, "calibration": CALIBRATION_FILE.name}
    THRESHOLDS_FILE.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_set_margin(margin: float) -> int:
    """Recalcula los umbrales desde la calibración guardada con un margen decidido a mano (sin recalibrar)."""
    if margin < MIN_MARGIN:
        print(f"ERROR: el margen mínimo es {MIN_MARGIN}", file=sys.stderr)
        return 1
    try:
        current = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))["current"]
    except FileNotFoundError:
        print("ERROR: falta results/gate_calibration.json; ejecuta `python cli.py calibrate-gate`", file=sys.stderr)
        return 1
    thresholds = thresholds_for(current, margin)
    write_thresholds_file(thresholds, margin, "decisión manual")
    print(f"Umbrales con margen {margin} (decisión manual) escritos en {THRESHOLDS_FILE.relative_to(ROOT)}")
    for c in CHUNKINGS:
        print("  " + "  ".join(f"{c} {m} {thresholds[c][m]:.3f}" for m in METRICS))
    return 0


def format_calibration(report: dict) -> str:
    lines = [f"Ruido de plataforma usado: {report['noise']:.4f} (perturbación máx. {report['perturbation_max']:.4f}; entre entornos {report['rule']['cross_environment_noise']})",
             f"Margen = max({MIN_MARGIN}, {SAFETY} × ruido, redondeado) = {report['margin']:.3f}", "", "Umbrales (valor actual → umbral):"]
    for c in CHUNKINGS:
        for m in METRICS:
            lines.append(f"  híbrido {c:<8} {m:<10} {report['current'][c][m]:.3f} → {report['thresholds'][c][m]:.3f}")
    lines.append("\nEscalera de ruido (informativa; cambio máximo absoluto sobre las semillas, recall@10 / nDCG@10):")
    for sigma, per in report["perturbation_ladder_max_abs_change_informative"].items():
        lines.append(f"  ruido relativo {sigma:>6}: " + " | ".join(f"{c[:3]} R {per[c]['recall@10']:.3f} nDCG {per[c]['ndcg@10']:.3f}" for c in CHUNKINGS))
    lines.append("\nSensibilidad: ¿detectaría el gate estas regresiones?")
    for name, per_chunking in report["scenarios"].items():
        if "omitido" in per_chunking:
            lines.append(f"  {name}: OMITIDO ({per_chunking['omitido']})")
            continue
        cells = []
        for c in CHUNKINGS:
            v = per_chunking[c]
            cells.append(f"{c[:3]}: R@10 {v['recall@10']:.3f} nDCG {v['ndcg@10']:.3f} {'FALLA' if v.get('falla_el_gate') else 'pasa '}")
        lines.append(f"  {name:<68} " + " | ".join(cells))
    return "\n".join(lines)


def run_calibrate(write_thresholds: bool) -> int:
    try:
        report = calibrate(write_thresholds)
    except (RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(format_calibration(report))
    print(f"\nGuardado en {CALIBRATION_FILE.relative_to(ROOT)}" + (f" y {THRESHOLDS_FILE.relative_to(ROOT)}" if write_thresholds else ""))
    return 0


def run_gate() -> int:
    try:
        spec = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
        current = current_values()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Regression gate · {spec['system']} · margen {spec['margin']}")
    for c in CHUNKINGS:
        for m in METRICS:
            print(f"  {c:<8} {m:<10} {current[c][m]:.3f}  (umbral {spec['thresholds'][c][m]:.3f})")
    failures = check(current, spec["thresholds"])
    if failures:
        print("\nGATE FALLIDO:", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    print("\nGate superado.")
    return 0
