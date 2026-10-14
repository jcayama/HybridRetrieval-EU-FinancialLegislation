"""Embeddings densos de los chunks y de las consultas, precalculados y versionados.

Decisiones fijadas antes de medir (no se ajustan con el golden set):
- Modelo BAAI/bge-small-en-v1.5 fijado por hash de commit (REVISION); 384 dimensiones; ventana de 512 tokens.
- Consultas con la instrucción de recuperación del modelo; pasajes (chunks) sin prefijo.
- Vectores normalizados (L2), de modo que el producto escalar es la similitud coseno.

Los vectores viven en data/embeddings/ junto a un manifest que registra modelo, revisión, instrucción, versiones
de librerías y los hashes de cada fichero y de su fuente. `load_embeddings` se niega a devolver vectores
desfasados. La CI y la API usan estos ficheros y NO necesitan torch ni sentence-transformers.
Requisitos para calcularlos: requirements-embeddings.txt.
"""

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from scripts.chunk_corpus import ARTICLES_FILE, WINDOWS_FILE
from scripts.download_corpus import ROOT, sha256_bytes
from scripts.golden import QUERIES_FILE, load_jsonl

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DIM = 384
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32

EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
MANIFEST_NAME = "manifest.json"
SOURCES = {"chunks_article": ARTICLES_FILE, "chunks_window": WINDOWS_FILE, "queries": QUERIES_FILE}
ID_FIELD = {"chunks_article": "chunk_id", "chunks_window": "chunk_id", "queries": "query_id"}
STALE_HINT = "ejecuta `python cli.py embed` (requiere requirements-embeddings.txt)"


def load_model():
    """Carga el modelo en CPU con la revisión fijada y comprueba que coincide con lo documentado."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu")
    if model.max_seq_length != MAX_SEQ_LENGTH or model.get_embedding_dimension() != DIM:
        raise RuntimeError(f"el modelo cargado no coincide con lo esperado ({MAX_SEQ_LENGTH} tokens, {DIM} dimensiones)")
    return model


def texts_for(name: str, rows: list[dict]) -> list[str]:
    """Texto que se codifica: las consultas llevan la instrucción; los chunks, no."""
    if name == "queries":
        return [QUERY_INSTRUCTION + row["query"] for row in rows]
    return [row["text"] for row in rows]


def encode(model, texts: list[str]) -> np.ndarray:
    vectors = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def count_truncated(model, texts: list[str]) -> int | None:
    """Cuántos textos superan la ventana del modelo (se truncan). None si el modelo no tiene tokenizador."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    return sum(len(ids) > MAX_SEQ_LENGTH for ids in tokenizer(texts, add_special_tokens=True)["input_ids"])


def _versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for package in ("sentence-transformers", "torch", "transformers", "numpy"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def build(model, out_dir: Path = EMBEDDINGS_DIR, sources: dict[str, Path] = SOURCES) -> dict:
    """Calcula y guarda los embeddings de todas las fuentes, y escribe el manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "query_instruction": QUERY_INSTRUCTION,
        "dim": DIM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "normalized": True,
        "libraries": _versions(),
        "files": {},
    }
    for name, source in sources.items():
        rows = load_jsonl(source)
        texts = texts_for(name, rows)
        vectors = encode(model, texts)
        path = out_dir / f"{name}.npy"
        np.save(path, vectors, allow_pickle=False)
        manifest["files"][name] = {
            "file": path.name,
            "n": len(rows),
            "ids": [row[ID_FIELD[name]] for row in rows],
            "sha256": sha256_bytes(path.read_bytes()),
            "source": source.name,
            "source_sha256": sha256_bytes(source.read_bytes()),
            "truncated_to_max_seq_length": count_truncated(model, texts),
        }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_manifest(out_dir: Path = EMBEDDINGS_DIR) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        raise RuntimeError(f"no hay embeddings en {out_dir}; {STALE_HINT}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_embeddings(name: str, out_dir: Path = EMBEDDINGS_DIR, sources: dict[str, Path] = SOURCES) -> tuple[list[str], np.ndarray]:
    """Devuelve (ids, matriz n×384 float32 normalizada). Falla si los vectores están desfasados."""
    manifest = read_manifest(out_dir)
    if (manifest["model_id"], manifest["revision"], manifest["query_instruction"]) != (MODEL_ID, REVISION, QUERY_INSTRUCTION):
        raise RuntimeError(f"los embeddings se calcularon con otro modelo, revisión o instrucción; {STALE_HINT}")
    entry = manifest["files"][name]
    path = out_dir / entry["file"]
    if sha256_bytes(path.read_bytes()) != entry["sha256"]:
        raise RuntimeError(f"{path.name} no coincide con el hash del manifest (fichero modificado); {STALE_HINT}")
    if sha256_bytes(sources[name].read_bytes()) != entry["source_sha256"]:
        raise RuntimeError(f"{path.name} está desactualizado respecto a {sources[name].name}; {STALE_HINT}")
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (entry["n"], DIM) or matrix.dtype != np.float32:
        raise RuntimeError(f"{path.name} tiene forma o tipo inesperados: {matrix.shape} {matrix.dtype}")
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{path.name} no está normalizado (L2)")
    return entry["ids"], matrix


def verify(model, out_dir: Path = EMBEDDINGS_DIR, sources: dict[str, Path] = SOURCES, sample: int = 20, min_cosine: float = 0.9999) -> float:
    """Recalcula una muestra de cada fichero y comprueba que coincide con lo guardado (coseno >= min_cosine).
    No se exige igualdad bit a bit: entre plataformas los vectores pueden diferir en el último decimal."""
    worst = 1.0
    for name, source in sources.items():
        _, stored = load_embeddings(name, out_dir, sources)
        rows = load_jsonl(source)
        picks = np.linspace(0, len(rows) - 1, num=min(sample, len(rows)), dtype=int)
        fresh = encode(model, [texts_for(name, rows)[i] for i in picks])
        worst = min(worst, float(np.min(np.sum(fresh * stored[picks], axis=1))))
    if worst < min_cosine:
        raise RuntimeError(f"los embeddings guardados no se reproducen: coseno mínimo {worst:.6f} < {min_cosine}")
    return worst


def run(check: bool = False) -> int:
    try:
        model = load_model()
        if check:
            print(f"Verificación correcta: coseno mínimo entre muestra recalculada y guardada = {verify(model):.6f}")
            return 0
        manifest = build(model)
    except (RuntimeError, FileNotFoundError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for name, entry in manifest["files"].items():
        print(f"{name}: {entry['n']} vectores, truncados a {MAX_SEQ_LENGTH} tokens: {entry['truncated_to_max_seq_length']}")
    print(f"Guardado en {EMBEDDINGS_DIR.relative_to(ROOT)}/ (modelo {MODEL_ID} @ {REVISION[:8]})")
    return 0
