"""El endpoint debe devolver EXACTAMENTE lo que `evaluate` midió: sin eso, publicar las cifras no tendría sentido.

Se ejecuta contra el corpus público cargado en la base de datos (`python cli.py load-db`). Los modos que necesitan el
modelo se saltan si no está instalado.
"""

import json

import pytest
from fastapi.testclient import TestClient

from functions.retriever import ModelUnavailable, RetrievalService
from main import create_app
from scripts import abstention
from scripts import evaluate as ev
from scripts.golden import QUERIES_FILE, load_jsonl
from tests.helpers import skip_or_fail

QUERIES = load_jsonl(QUERIES_FILE)


@pytest.fixture(scope="module")
def real_service():
    svc = RetrievalService(abstain=False)  # la paridad es de rankings; la abstención se prueba aparte
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


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
@pytest.mark.parametrize("chunking", ["article", "window"])
def test_with_abstention_on_it_only_abstains_below_the_threshold_and_otherwise_matches_evaluate(real_service, mode, chunking):
    """Con la abstención activa, una consulta del golden set o se abstiene (coseno guardado < umbral) o da el ranking medido."""
    saved = json.loads(abstention.RESULTS_FILE.read_text(encoding="utf-8"))["chunkings"][chunking]["golden"]
    cosine = {row["query_id"]: row["cosine"] for row in saved}
    threshold = abstention.load_thresholds().get(chunking)  # None con artículos: la API no se abstiene
    svc = RetrievalService()
    svc.start()
    measured = evaluation_rankings(mode, chunking)
    try:
        outcomes = {q["query_id"]: svc.search_outcome(q["query"], top_k=10, mode=mode, chunking=chunking) for q in QUERIES}
    except ModelUnavailable:
        pytest.skip("modelo de embeddings no instalado (requirements-embeddings.txt)")
    for query_id, outcome in outcomes.items():
        assert outcome.abstained == (threshold is not None and cosine[query_id] < threshold), query_id
        if outcome.abstained:
            assert outcome.hits == []
        else:
            assert [(h.celex, h.article) for h in outcome.hits] == measured[query_id], query_id


def test_the_http_layer_serves_a_real_query_end_to_end(real_service):
    response = TestClient(create_app(service=real_service)).post("/hybridretrievals", json={"query": "Lead Overseer", "mode": "fts", "top_k": 5})
    assert response.status_code == 200
    results = response.json()["results"]
    assert 0 < len(results) <= 5 and all(r["title"] and r["snippet"] and r["act"] for r in results)
    assert any(r["celex"] == "32022R2554" for r in results)  # el «Lead Overseer» es una figura de DORA
