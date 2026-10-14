import hashlib
import json

import numpy as np
import pytest

from scripts import embeddings as emb
from scripts.embeddings import DIM, QUERY_INSTRUCTION


class FakeModel:
    """Modelo falso: vector determinista a partir del hash del texto (normalizado) y registro de las llamadas."""

    def __init__(self, salt=""):
        self.salt = salt
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        out = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((self.salt + text).encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).normal(size=DIM)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def workspace(tmp_path):
    chunks = tmp_path / "articles.jsonl"
    chunks.write_text("".join(json.dumps({"chunk_id": f"C:art_{i}", "text": f"Article {i}. T\nbody number {i}"}) + "\n" for i in range(5)), encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text("".join(json.dumps({"query_id": f"q{i}", "type": "paraphrase", "theme": "t", "query": f"question {i}"}) + "\n" for i in range(3)), encoding="utf-8")
    sources = {"chunks_article": chunks, "queries": queries}
    return tmp_path / "emb", sources


def test_instruction_only_on_queries():
    assert emb.texts_for("queries", [{"query": "what?"}]) == [QUERY_INSTRUCTION + "what?"]
    assert emb.texts_for("chunks_article", [{"text": "Article 1. T\nbody"}]) == ["Article 1. T\nbody"]
    assert emb.texts_for("chunks_window", [{"text": "w"}]) == ["w"]


def test_encode_requests_normalized_float32_vectors():
    model = FakeModel()
    vectors = emb.encode(model, ["a", "b"])
    assert vectors.dtype == np.float32 and vectors.shape == (2, DIM)
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_build_then_load_roundtrip(workspace):
    out_dir, sources = workspace
    model = FakeModel()
    manifest = emb.build(model, out_dir, sources)
    assert manifest["model_id"] == emb.MODEL_ID and manifest["revision"] == emb.REVISION
    assert manifest["files"]["chunks_article"]["truncated_to_max_seq_length"] is None  # el modelo falso no tiene tokenizador
    ids, matrix = emb.load_embeddings("chunks_article", out_dir, sources)
    assert ids == [f"C:art_{i}" for i in range(5)] and matrix.shape == (5, DIM) and matrix.dtype == np.float32
    qids, qmatrix = emb.load_embeddings("queries", out_dir, sources)
    assert qids == ["q0", "q1", "q2"] and qmatrix.shape == (3, DIM)
    # las consultas se codificaron CON instrucción y los chunks SIN ella
    sent = {text for texts, _ in model.calls for text in texts}
    assert QUERY_INSTRUCTION + "question 0" in sent and "question 0" not in sent
    assert "Article 0. T\nbody number 0" in sent


def test_load_detects_changed_source(workspace):
    out_dir, sources = workspace
    emb.build(FakeModel(), out_dir, sources)
    sources["chunks_article"].write_text(sources["chunks_article"].read_text() + "\n")
    with pytest.raises(RuntimeError, match="desactualizado"):
        emb.load_embeddings("chunks_article", out_dir, sources)
    emb.load_embeddings("queries", out_dir, sources)  # la otra fuente no cambió


def test_load_detects_tampered_vectors(workspace):
    out_dir, sources = workspace
    emb.build(FakeModel(), out_dir, sources)
    path = out_dir / "chunks_article.npy"
    np.save(path, np.zeros((5, DIM), dtype=np.float32))
    with pytest.raises(RuntimeError, match="hash"):
        emb.load_embeddings("chunks_article", out_dir, sources)


def test_load_detects_other_model_revision(workspace, monkeypatch):
    out_dir, sources = workspace
    emb.build(FakeModel(), out_dir, sources)
    monkeypatch.setattr(emb, "REVISION", "0" * 40)
    with pytest.raises(RuntimeError, match="otro modelo"):
        emb.load_embeddings("chunks_article", out_dir, sources)


def test_load_without_embeddings_explains_what_to_do(tmp_path):
    with pytest.raises(RuntimeError, match="cli.py embed"):
        emb.load_embeddings("queries", tmp_path / "nada")


def test_verify_passes_for_same_model_and_fails_for_a_different_one(workspace):
    out_dir, sources = workspace
    emb.build(FakeModel(), out_dir, sources)
    assert emb.verify(FakeModel(), out_dir, sources) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(RuntimeError, match="no se reproducen"):
        emb.verify(FakeModel(salt="otro"), out_dir, sources)


# ---------- con los embeddings versionados (no necesitan el modelo) ----------

committed = pytest.mark.skipif(not (emb.EMBEDDINGS_DIR / emb.MANIFEST_NAME).exists(), reason="embeddings no calculados (python cli.py embed)")


@committed
def test_committed_embeddings_are_consistent_with_current_sources():
    expected = {"chunks_article": 248, "chunks_window": 719, "queries": 40}
    for name, n in expected.items():
        ids, matrix = emb.load_embeddings(name)
        assert len(ids) == n and matrix.shape == (n, DIM)
        assert len(set(ids)) == n


# ---------- con el modelo real (se saltan si no está instalado o descargado) ----------


@pytest.fixture(scope="module")
def real_model():
    try:
        return emb.load_model()
    except Exception as error:  # sin sentence-transformers, sin red o sin el modelo en caché
        pytest.skip(f"modelo no disponible: {error}")


def test_real_model_properties(real_model):
    vectors = emb.encode(real_model, ["The depositary shall safe-keep the assets.", "Fines for providers of AI models."])
    assert vectors.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    query = emb.encode(real_model, emb.texts_for("queries", [{"query": "who keeps the assets of a fund safe"}]))[0]
    assert float(query @ vectors[0]) > float(query @ vectors[1])  # el pasaje relacionado puntúa más


@committed
def test_real_model_reproduces_committed_embeddings(real_model):
    assert emb.verify(real_model) >= 0.9999
