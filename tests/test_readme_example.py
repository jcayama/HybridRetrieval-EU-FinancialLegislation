"""Los ejemplos de respuesta de los README deben ser lo que el servicio devuelve de verdad (con las abreviaturas declaradas)."""

import json
import re

import pytest

from functions.retriever import ModelUnavailable, RetrievalService
from scripts.report import README_FILES
from tests.helpers import skip_or_fail


def example(readme_text: str, name: str) -> dict:
    match = re.search(rf"<!-- {name}:start -->\s*```json\n(.*?)\n```\s*<!-- {name}:end -->", readme_text, re.S)
    assert match, f"falta el bloque {name} en el README"
    return json.loads(match.group(1))


@pytest.fixture(scope="module")
def service():
    svc = RetrievalService()
    svc.start()
    try:
        svc.search("Lead Overseer", mode="fts", chunking="window")
    except Exception as error:  # sin Postgres o sin corpus cargado (en la CI es un fallo, no un salto)
        skip_or_fail(f"base de datos no disponible o sin cargar: {error}")
    return svc


@pytest.mark.parametrize("lang", sorted(README_FILES))
def test_the_readme_examples_match_a_real_run(service, lang):
    text = README_FILES[lang].read_text(encoding="utf-8")
    for name in ("example", "example-abstention"):
        shown = example(text, name)
        try:
            outcome = service.search_outcome(shown["query"], top_k=shown["top_k"], mode=shown["mode"], chunking=shown["chunking"])
        except ModelUnavailable:
            pytest.skip("modelo de embeddings no instalado (requirements-embeddings.txt)")
        assert shown["abstained"] == outcome.abstained, name
        assert shown["best_cosine"] == pytest.approx(outcome.best_cosine, abs=5e-4), name  # el README lo redondea a 3 decimales
        assert [(r["rank"], r["article"], r["chunk_id"], r["title"], r["chunk_words"]) for r in shown["results"]] == [(h.rank, h.article, h.chunk_id, h.title, h.chunk_words) for h in outcome.hits], name
        assert all(r["snippet"].rstrip("…").strip() in h.snippet for r, h in zip(shown["results"], outcome.hits, strict=True)), name  # el fragmento es un recorte del real


def test_both_languages_show_the_same_example_json():
    en, es = (README_FILES[lang].read_text(encoding="utf-8") for lang in ("en", "es"))
    for name in ("example", "example-abstention"):
        assert example(en, name) == example(es, name)
