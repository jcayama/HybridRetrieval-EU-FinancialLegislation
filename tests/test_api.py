"""Contrato HTTP del endpoint, con un servicio falso: sin base de datos ni modelo."""

import pytest
from fastapi.testclient import TestClient

from functions.retriever import DatabaseNotReady, ModelUnavailable, SearchHit, SearchOutcome
from main import create_app


def hit(rank, article=None):
    article = article or 100 + rank
    return SearchHit(rank=rank, chunk_id=f"32022R2554:art_{article}:w200:0", act="DORA 2022/2554", celex="32022R2554", article=article, title=f"Título {article}", snippet=f"fragmento {article}", chunk_words=200)


class FakeService:
    def __init__(self, tenants_enabled=False, error=None, hits=None, database=True, model=True, best_cosine=None, abstained=False, abstention=True):
        self.tenants_enabled = tenants_enabled
        self.error = error
        self.hits = hits if hits is not None else [hit(i) for i in range(1, 6)]
        self.database, self.model = database, model
        self.best_cosine, self.abstained, self.abstention = best_cosine, abstained, abstention
        self.calls = []

    def status(self):
        return {"database": self.database, "model": self.model, "tenants_enabled": self.tenants_enabled, "abstention": self.abstention}

    def search_outcome(self, query, top_k=10, mode="hybrid", chunking="window", tenant=None):
        self.calls.append({"query": query, "top_k": top_k, "mode": mode, "chunking": chunking, "tenant": tenant})
        if self.error:
            raise self.error
        if self.abstained:
            return SearchOutcome([], self.best_cosine, True)
        return SearchOutcome(self.hits[:top_k], self.best_cosine)


def post(service, body, headers=None):
    return TestClient(create_app(service=service)).post("/hybridretrievals", json=body, headers=headers or {})


# ---------- salud ----------


def test_root_answers_with_the_project_title():
    response = TestClient(create_app(service=FakeService())).get("/")
    assert response.status_code == 200 and "Hybrid Retrieval" in response.text


def test_health_reports_database_model_and_tenants():
    body = TestClient(create_app(service=FakeService(tenants_enabled=True, model=False))).get("/health")
    assert body.status_code == 200
    assert body.json() == {"status": "ok", "database": True, "model": False, "tenants_enabled": True, "abstention": True}


def test_health_is_503_when_the_database_is_down():
    response = TestClient(create_app(service=FakeService(database=False))).get("/health")
    assert response.status_code == 503 and response.json()["status"] == "degraded"


# ---------- contrato de la consulta ----------


def test_defaults_are_hybrid_window_top_10_without_tenant():
    service = FakeService()
    response = post(service, {"query": "Lead Overseer"})
    assert response.status_code == 200
    assert service.calls == [{"query": "Lead Overseer", "top_k": 10, "mode": "hybrid", "chunking": "window", "tenant": None}]
    body = response.json()
    assert (body["query"], body["mode"], body["chunking"], body["top_k"]) == ("Lead Overseer", "hybrid", "window", 10)


def test_results_have_rank_article_title_and_snippet():
    results = post(FakeService(), {"query": "Lead Overseer"}).json()["results"]
    assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]
    assert set(results[0]) == {"rank", "chunk_id", "act", "celex", "article", "title", "snippet", "chunk_words"}
    assert len({(r["celex"], r["article"]) for r in results}) == len(results)  # artículos distintos
    assert results[0]["title"] == "Título 101" and results[0]["snippet"] == "fragmento 101"


def test_options_are_passed_to_the_service():
    service = FakeService()
    assert post(service, {"query": "x y", "top_k": 3, "mode": "fts", "chunking": "article"}).status_code == 200
    assert service.calls[0] | {"query": None} == {"query": None, "top_k": 3, "mode": "fts", "chunking": "article", "tenant": None}
    assert len(post(FakeService(), {"query": "x", "top_k": 2}).json()["results"]) == 2


def test_short_exact_term_queries_are_accepted():
    """Regresión: la regla antigua de «mínimo 10 caracteres» rechazaba consultas del golden set como «TLPT»."""
    assert post(FakeService(), {"query": "TLPT"}).status_code == 200


def test_the_query_is_stripped_before_searching():
    service = FakeService()
    assert post(service, {"query": "  Lead Overseer \n"}).status_code == 200
    assert service.calls[0]["query"] == "Lead Overseer"


def test_no_matches_is_a_200_with_an_empty_list():
    assert post(FakeService(hits=[]), {"query": "zzzz"}).json()["results"] == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t", "x" * 501])
def test_empty_blank_or_too_long_queries_are_400_with_the_project_error_format(query):
    service = FakeService()
    response = post(service, {"query": query})
    assert response.status_code == 400 and response.json()["errors"][0]["code"] == "E0001"
    assert service.calls == []  # ni siquiera se llega al servicio


@pytest.mark.parametrize(
    "body",
    [{}, {"query": 7}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 51}, {"query": "x", "top_k": "diez"}, {"query": "x", "mode": "magic"}, {"query": "x", "chunking": "chapters"}],
)
def test_structurally_invalid_bodies_are_422(body):
    assert post(FakeService(), body).status_code == 422


def test_a_500_character_query_is_the_longest_accepted():
    assert post(FakeService(), {"query": "x" * 500}).status_code == 200


# ---------- servicio no disponible ----------


def test_model_unavailable_is_503_and_explains_how_to_continue():
    response = post(FakeService(error=ModelUnavailable()), {"query": "Lead Overseer"})
    error = response.json()["errors"][0]
    assert response.status_code == 503 and error["code"] == "E0002"
    assert "requirements-embeddings.txt" in error["title"] and "fts" in error["title"]


def test_database_not_ready_is_503():
    response = post(FakeService(error=DatabaseNotReady("sin chunks")), {"query": "x"})
    assert response.status_code == 503 and response.json()["errors"][0]["code"] == "E0004"
    assert "load-db" in response.json()["errors"][0]["title"]


def test_unexpected_errors_are_500_without_leaking_details():
    response = post(FakeService(error=RuntimeError("fallo en db.interna.example password=hunter2")), {"query": "x"})
    assert response.status_code == 500 and "error_id" in response.json()
    assert "hunter2" not in response.text and "db.interna" not in response.text


# ---------- inquilinos ----------


def test_the_tenant_header_is_ignored_when_tenants_are_disabled():
    service = FakeService(tenants_enabled=False)
    assert post(service, {"query": "x"}, {"X-Tenant-ID": "A"}).status_code == 200
    assert service.calls[0]["tenant"] is None


def test_the_tenant_header_is_used_when_enabled_and_absent_means_public_only():
    service = FakeService(tenants_enabled=True)
    post(service, {"query": "x"}, {"X-Tenant-ID": "tenant_A-1.eu"})
    post(service, {"query": "x"})
    assert [c["tenant"] for c in service.calls] == ["tenant_A-1.eu", None]


@pytest.mark.parametrize("bad", ["", "   ", "has space", "a;drop table", "x" * 65, "a/b"])
def test_an_invalid_tenant_header_is_400_when_tenants_are_enabled(bad):
    service = FakeService(tenants_enabled=True)
    response = post(service, {"query": "x"}, {"X-Tenant-ID": bad})
    assert response.status_code == 400 and response.json()["errors"][0]["code"] == "E0003"
    assert service.calls == []


# ---------- documentación ----------


def test_openapi_documents_the_endpoint_and_the_interactive_docs_exist():
    client = TestClient(create_app(service=FakeService()))
    spec = client.get("/openapi.json").json()
    assert "post" in spec["paths"]["/hybridretrievals"] and "/health" in spec["paths"]
    assert {"RetrievalRequest", "RetrievalResponse"} <= set(spec["components"]["schemas"])
    assert client.get("/docs").status_code == 200


# ---------- abstención ----------


def test_an_abstention_is_a_200_with_no_results_and_the_cosine_that_explains_it():
    response = post(FakeService(abstained=True, best_cosine=0.41), {"query": "pizza"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == [] and body["abstained"] is True and body["best_cosine"] == 0.41


def test_a_normal_answer_is_not_an_abstention_and_carries_the_cosine():
    body = post(FakeService(best_cosine=0.73), {"query": "Lead Overseer"}).json()
    assert body["abstained"] is False and body["best_cosine"] == 0.73 and len(body["results"]) == 5


def test_fts_answers_carry_no_cosine():
    body = post(FakeService(), {"query": "Lead Overseer", "mode": "fts"}).json()
    assert body["abstained"] is False and body["best_cosine"] is None


def test_health_reports_whether_abstention_is_active():
    assert TestClient(create_app(service=FakeService(abstention=False))).get("/health").json()["abstention"] is False
