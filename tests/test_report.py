import json

import pytest

from scripts import report as rp
from scripts.golden import QUERIES_FILE, load_jsonl


def test_number_formatting_per_language():
    assert rp.fmt(0.6415, lang="es") == "0,641" and rp.fmt(0.5, 2, "es") == "0,50"
    assert rp.fmt(0.6415, lang="en") == "0.641" and rp.fmt(0.5, 2, "en") == "0.50"
    assert rp.fmt_signed(0.062, lang="es") == "+0,062" and rp.fmt_signed(-0.026, lang="es") == "−0,026" and rp.fmt_signed(0.0, lang="en") == "+0.000"
    assert rp.fmt_ci(-0.039, 0.162, signed=True, lang="es") == "[−0,039, +0,162]"
    assert rp.fmt_ci(-0.039, 0.162, signed=True, lang="en") == "[−0.039, +0.162]"


def test_excludes_zero_only_when_the_interval_does_not_contain_it():
    assert rp.excludes_zero(0.011, 0.233) and rp.excludes_zero(-0.3, -0.01)
    assert not rp.excludes_zero(-0.005, 0.08) and not rp.excludes_zero(0.0, 0.1)


def test_table_has_header_separator_and_rows():
    out = rp.table(["A", "B"], [["1", "2"], ["3", "4"]]).splitlines()
    assert out == ["| A | B |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_name_of_uses_labels_of_each_language():
    assert rp.name_of("hybrid_window", "es") == "Híbrido FTS + denso (ventanas)"
    assert rp.name_of("hybrid-bm25_article", "es") == "Híbrido BM25 + denso (exploratorio) (artículo)"
    assert rp.name_of("hybrid_window", "en") == "Hybrid FTS + dense (window)"
    assert rp.name_of("fts_article", "en") == "Postgres FTS (article)"


def test_both_language_packs_define_exactly_the_same_keys():
    es, en = rp.PACKS["es"], rp.PACKS["en"]
    assert set(es) == set(en)
    for key in es:
        if isinstance(es[key], dict):
            assert set(es[key]) == set(en[key]), key
        if isinstance(es[key], list):
            assert len(es[key]) == len(en[key]), key


def row(name, excl95, exclbonf):
    return {"name": name, "excl95": excl95, "exclbonf": exclbonf}


BEST = {"article": ("hybrid", 0.641), "window": ("dense", 0.644)}


def test_findings_are_derived_from_the_rows_in_spanish():
    rows = [row("A − B", True, True), row("C − D", True, False), row("E − F", False, False)]
    text = "\n".join(rp.findings(rows, 15, BEST, "es"))
    assert "**1** lo siguen excluyendo" in text and "**2** excluyen el cero" in text
    assert "**A − B**" in text  # el que sobrevive se nombra
    assert "C − D" in text and "indicios, no hallazgos" in text  # el que solo excluye al 95 % se nombra aparte
    assert "E − F" not in text
    assert "Híbrido FTS + denso" in text and "Denso (bge-small)" in text and "0,641" in text
    assert "no implica" in text  # la advertencia sobre leer el orden como victoria


def test_findings_are_derived_from_the_rows_in_english():
    rows = [row("A − B", True, True), row("C − D", True, False), row("E − F", False, False)]
    text = "\n".join(rp.findings(rows, 15, BEST, "en"))
    assert "**1** still exclude it" in text and "**2** exclude zero" in text
    assert "**A − B**" in text and "C − D" in text and "hints, not findings" in text and "E − F" not in text
    assert "Hybrid FTS + dense" in text and "Dense (bge-small)" in text and "0.641" in text
    assert "does **not imply**" in text


@pytest.mark.parametrize(("lang", "expected"), [("es", "Ninguna comparación sobrevive"), ("en", "No comparison survives")])
def test_findings_when_nothing_survives(lang, expected):
    text = "\n".join(rp.findings([row("A − B", True, False)], 15, BEST, lang))
    assert expected in text and "**0**" in text


def test_replace_block_swaps_only_what_is_between_the_markers():
    readme = f"antes\n{rp.START}\nviejo\n{rp.END}\ndespués\n"
    assert rp.replace_block(readme, "nuevo") == f"antes\n{rp.START}\nnuevo\n{rp.END}\ndespués\n"
    assert rp.replace_block(rp.replace_block(readme, "x"), "x") == rp.replace_block(readme, "x")  # idempotente


def test_replace_block_works_with_the_tldr_markers_independently_of_the_results_ones():
    readme = f"a\n{rp.TLDR_START}\nold\n{rp.TLDR_END}\nb\n{rp.START}\nold2\n{rp.END}\nc\n"
    out = rp.replace_block(readme, "new", rp.TLDR_START, rp.TLDR_END)
    assert "new" in out and "old2" in out and "\nold\n" not in out


@pytest.mark.parametrize("bad", ["sin marcas", f"{rp.START} solo inicio", f"{rp.END} fin {rp.START} inicio", f"{rp.START}{rp.END}{rp.START}{rp.END}"])
def test_replace_block_rejects_missing_duplicated_or_misordered_markers(bad):
    with pytest.raises(ValueError, match="exactamente una vez"):
        rp.replace_block(bad, "x")


def test_the_declared_comparison_families_have_the_declared_sizes():
    assert len(rp.PRIMARY) == 15 and len(rp.EXPLORATORY) == 6 and rp.FAMILY_EXPLORATORY == 21
    assert len(set(rp.PRIMARY)) == 15 and not set(rp.PRIMARY) & set(rp.EXPLORATORY)
    assert all(k.endswith(("_article", "_window")) for pair in rp.PRIMARY + rp.EXPLORATORY for k in pair)


def test_every_primary_comparison_contrasts_the_same_chunking_or_the_same_system():
    for a, b in rp.PRIMARY + rp.EXPLORATORY:
        (sys_a, ch_a), (sys_b, ch_b) = a.rsplit("_", 1), b.rsplit("_", 1)
        assert sys_a == sys_b or ch_a == ch_b, (a, b)  # cambia una sola cosa a la vez


# ---------- con los resultados y los README reales ----------


def test_the_generated_blocks_of_both_readmes_are_up_to_date():
    """Si falla: ejecuta `python cli.py report`. Evita que los README publiquen cifras desfasadas."""
    assert rp.run(check=True) == 0


@pytest.fixture(scope="module")
def real():
    return rp.load_results(), {q["query_id"]: q["type"] for q in load_jsonl(QUERIES_FILE)}


def test_the_spanish_block_reads_correctly(real):
    results, types = real
    block = rp.render(results, types, "es")
    assert "Corpus real, ventanas (719 filas)" in block  # la coma del título sobrevive al formato de miles
    assert "Sintético (100.000 filas)" in block and "\n\n#### Comparaciones exploratorias" in block
    best = results["hybrid_window"]["summary"]["overall"]["mean"]["ndcg@10"]
    assert rp.fmt(best, lang="es") in block  # la cifra del fichero de resultados aparece en el README generado


def test_the_english_block_reads_correctly(real):
    results, types = real
    block = rp.render(results, types, "en")
    assert "Real corpus, window (719 rows)" in block and "Synthetic (100,000 rows)" in block
    assert "\n\n#### Exploratory comparisons" in block and "Exact (reference)" in block
    assert "sí" not in block and "Comparación" not in block  # nada del otro idioma
    assert "0,6" not in block  # separador decimal inglés
    best = results["hybrid_window"]["summary"]["overall"]["mean"]["ndcg@10"]
    assert rp.fmt(best, lang="en") in block


@pytest.mark.parametrize("lang", rp.LANGS)
def test_both_languages_have_the_same_number_of_comparison_rows_and_numbers(real, lang):
    results, types = real
    block = rp.render(results, types, lang)
    rows = [line for line in block.splitlines() if " − " in line and line.startswith("| ") and "Excluye" not in line and "Excludes" not in line]
    assert len(rows) == 21  # 15 principales + 6 exploratorias


def test_both_languages_show_the_same_figures_apart_from_the_decimal_separator(real):
    import re

    results, types = real
    es = [n.replace(",", ".") for n in re.findall(r"[+−]?\d+,\d+", rp.render(results, types, "es"))]  # solo cifras con coma decimal
    en = re.findall(r"[+−]?\d+\.\d+", rp.render(results, types, "en"))
    assert es == en and len(en) > 200  # mismas cifras, en el mismo orden (y no una comparación vacía)


@pytest.mark.parametrize("lang", rp.LANGS)
def test_the_headline_is_derived_from_the_data(real, lang):
    results, types = real
    text = rp.render_headline(results, types, lang)
    best = json.loads((rp.RESULTS_DIR / "hybrid_window.json").read_text())["summary"]["overall"]["mean"]
    assert rp.fmt(best["ndcg@10"], lang=lang) in text and rp.fmt(best["recall@10"], lang=lang) in text
    assert text.count("\n- ") + 1 == 3 and "| " in text  # tres viñetas y una tabla


# ---------- análisis de errores ----------


@pytest.mark.parametrize("lang", rp.LANGS)
def test_the_error_section_has_one_row_per_case_and_text_for_every_probe_and_effect(lang):
    data = json.loads(rp.ERRORS_FILE.read_text(encoding="utf-8"))
    t = rp.PACKS[lang]
    assert {c["query_id"] for c in data["cases"]} <= set(t["errors_probes"])  # ninguna sonda sin texto
    assert {c["probe"]["effect"] for c in data["cases"]} <= set(t["errors_effect"])
    section = rp.section_errors(lang)
    rows = [line for line in section.splitlines() if line.startswith("| q0")]
    assert len(rows) == data["n_cases"] and f"{data['n_cases']} " in section.split("\n\n")[1]
    assert all(line.count("|") == len(t["errors_header"]) + 1 for line in rows)  # ningún «|» suelto rompe la tabla


def test_the_error_section_is_in_the_right_language_and_inside_the_results_block(real):
    results, types = real
    en, es = rp.render(results, types, "en"), rp.render(results, types, "es")
    assert "#### Error analysis" in en and "explains the failure" in en and "Análisis de errores" not in en
    assert "#### Análisis de errores" in es and "explica el fallo" in es and "Error analysis" not in es
