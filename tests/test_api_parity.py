"""El endpoint debe devolver EXACTAMENTE lo que `evaluate` midió: sin eso, publicar las cifras no tendría sentido.

Se ejecuta contra el corpus público cargado en la base de datos (`python cli.py load-db`). Los modos que necesitan el
modelo se saltan si no está instalado.
"""

import pytest
from fastapi.testclient import TestClient

from functions.retriever import ModelUnavailable, RetrievalService
from main import create_app
from scripts import evaluate as ev
from scripts.golden import QUERIES_FILE, load_jsonl
from tests.helpers import skip_or_fail

QUERIES = load_jsonl(QUERIES_FILE)


@pytest.fixture(scope="module")
def real_service():
    svc = RetrievalService()
    svc.start()  # intenta cargar el modelo; si no puede, queda sin él
    try:
        svc.search("Lead Overseer", mode="fts", chunking="article")
    except Exception as error:  # sin Postgres o sin corpus cargado (en la CI es un fallo, no un salto)
        skip_or_fail(f"base de datos no disponible o sin cargar: {error}")
    return svc


def evaluation_rankings(system, chunking):
    sha = ev.sha256_bytes(ev.CHUNK_FILES[chunking].read_bytes())
    index = ev.SYSTEMS[system](chunking, ev.load_chunks(chunking), sha)
    return ev.run_system(index, QUERIES)


def api_rankings(svc, mode, chunking):
    return {q["query_id"]: [(h.celex, h.article) for h in svc.search(q["query"], top_k=10, mode=mode, chunking=chunking)] for q in QUERIES}


@pytest.mark.parametrize("chunking", ["article", "window"])
def test_fts_endpoint_returns_exactly_the_rankings_that_evaluate_measured(real_service, chunking):
    assert api_rankings(real_service, "fts", chunking) == evaluation_rankings("fts", chunking)


@pytest.mark.parametrize(("mode", "system"), [("dense", "dense"), ("hybrid", "hybrid")])
@pytest.mark.parametrize("chunking", ["article", "window"])
def test_model_modes_return_exactly_the_rankings_that_evaluate_measured(real_service, mode, system, chunking):
    """Aquí la consulta se embebe con el modelo en el momento; `evaluate` usa los vectores precalculados."""
    try:
        got = api_rankings(real_service, mode, chunking)
    except ModelUnavailable:
        pytest.skip("modelo de embeddings no instalado (requirements-embeddings.txt)")
    assert got == evaluation_rankings(system, chunking)


def test_the_http_layer_serves_a_real_query_end_to_end(real_service):
    response = TestClient(create_app(service=real_service)).post("/hybridretrievals", json={"query": "Lead Overseer", "mode": "fts", "top_k": 5})
    assert response.status_code == 200
    results = response.json()["results"]
    assert 0 < len(results) <= 5 and all(r["title"] and r["snippet"] and r["act"] for r in results)
    assert any(r["celex"] == "32022R2554" for r in results)  # el «Lead Overseer» es una figura de DORA
