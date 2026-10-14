import pytest

from scripts import report as rp


def test_number_formatting_uses_spanish_decimal_comma_and_a_real_minus_sign():
    assert rp.fmt(0.6415) == "0,641" and rp.fmt(0.5, 2) == "0,50"
    assert rp.fmt_signed(0.062) == "+0,062" and rp.fmt_signed(-0.026) == "−0,026" and rp.fmt_signed(0.0) == "+0,000"
    assert rp.fmt_ci(-0.039, 0.162, signed=True) == "[−0,039, +0,162]"


def test_excludes_zero_only_when_the_interval_does_not_contain_it():
    assert rp.excludes_zero(0.011, 0.233) and rp.excludes_zero(-0.3, -0.01)
    assert not rp.excludes_zero(-0.005, 0.08) and not rp.excludes_zero(0.0, 0.1)


def test_table_has_header_separator_and_rows():
    out = rp.table(["A", "B"], [["1", "2"], ["3", "4"]]).splitlines()
    assert out == ["| A | B |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_name_of_uses_labels():
    assert rp.name_of("hybrid_window") == "Híbrido FTS + denso (ventanas)"
    assert rp.name_of("hybrid-bm25_article") == "Híbrido BM25 + denso (exploratorio) (artículo)"


def row(name, excl95, exclbonf):
    return {"name": name, "excl95": excl95, "exclbonf": exclbonf}


BEST = {"article": ("hybrid", 0.641), "window": ("dense", 0.644)}


def test_findings_are_derived_from_the_rows():
    rows = [row("A − B", True, True), row("C − D", True, False), row("E − F", False, False)]
    text = "\n".join(rp.findings(rows, 15, BEST))
    assert "**1** lo siguen excluyendo" in text and "**2** excluyen el cero" in text
    assert "**A − B**" in text  # el que sobrevive se nombra
    assert "C − D" in text and "indicios, no hallazgos" in text  # el que solo excluye al 95 % se nombra aparte
    assert "E − F" not in text
    assert "Híbrido FTS + denso" in text and "Denso (bge-small)" in text and "0,641" in text
    assert "no implica" in text  # la advertencia sobre leer el orden como victoria


def test_findings_when_nothing_survives():
    text = "\n".join(rp.findings([row("A − B", True, False)], 15, BEST))
    assert "Ninguna comparación sobrevive" in text and "**0** lo siguen" in text


def test_replace_block_swaps_only_what_is_between_the_markers():
    readme = f"antes\n{rp.START}\nviejo\n{rp.END}\ndespués\n"
    assert rp.replace_block(readme, "nuevo") == f"antes\n{rp.START}\nnuevo\n{rp.END}\ndespués\n"
    assert rp.replace_block(rp.replace_block(readme, "x"), "x") == rp.replace_block(readme, "x")  # idempotente


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


# ---------- con los resultados y el README reales ----------


def test_the_readme_results_block_is_up_to_date():
    """Si falla: ejecuta `python cli.py report`. Evita que el README publique cifras desfasadas."""
    assert rp.run(check=True) == 0


def test_the_generated_block_reads_correctly():
    import json

    from scripts.golden import QUERIES_FILE, load_jsonl

    types = {q["query_id"]: q["type"] for q in load_jsonl(QUERIES_FILE)}
    block = rp.render(rp.load_results(), types)
    assert "Corpus real, ventanas (719 filas)" in block  # la coma del título sobrevive al formato de miles
    assert "Sintético (100.000 filas)" in block
    assert "\n\n#### Comparaciones exploratorias" in block
    rows = [line for line in block.splitlines() if " − " in line and line.startswith("| ") and "Excluye" not in line]
    assert len(rows) == 21  # 15 principales + 6 exploratorias
    best = json.loads((rp.RESULTS_DIR / "hybrid_window.json").read_text())["summary"]["overall"]["mean"]["ndcg@10"]
    assert rp.fmt(best) in block  # la cifra del fichero de resultados aparece en el README generado
