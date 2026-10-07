"""Genera los chunks del corpus congelado: por artículo y por ventanas fijas de palabras.

Uso:
    python cli.py chunk

Salidas (JSONL, una línea por chunk, orden determinista):
    data/chunks/articles.jsonl
    data/chunks/windows_w200_o40.jsonl

Reglas:
- Alcance: solo los artículos (`div.eli-subdivision#art_N`); considerandos y anexos quedan fuera.
- Fidelidad: el texto se copia literal; solo se normalizan espacios y saltos de línea. Las filas de las
  tablas de listas ("(a)" + texto) se unen en una sola línea.
- Ventanas: WINDOW palabras del cuerpo con OVERLAP de solape, sin cruzar el límite de un artículo. Si la
  cola restante tiene menos de MIN_TAIL palabras nuevas, se fusiona con la ventana anterior.
- Todo chunk (artículo o ventana) empieza por la cabecera "Article N. Título". Las palabras de la
  cabecera no cuentan para el tamaño de ventana.
"""

import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from scripts.download_corpus import ACTS, HASH_FILE, RAW_DIR, ROOT, Act, read_hashes, sha256_bytes

CHUNKS_DIR = ROOT / "data" / "chunks"
WINDOW = 200
OVERLAP = 40
MIN_TAIL = 50
ARTICLES_FILE = CHUNKS_DIR / "articles.jsonl"
WINDOWS_FILE = CHUNKS_DIR / f"windows_w{WINDOW}_o{OVERLAP}.jsonl"

ART_ID = re.compile(r"^art_(\d+)$")


def norm(text: str) -> str:
    """Normaliza espacios (incluidos los no separables) sin tocar nada más."""
    return " ".join(text.split())


def chars(text: str) -> str:
    """Texto sin ningún espacio: sirve para comprobar fidelidad independientemente del espaciado."""
    return "".join(text.split())


def css_classes(tag: Tag) -> list[str]:
    """Clases de un elemento; el parser XML las devuelve como una cadena y no como una lista."""
    value = tag.get("class") or []
    return value.split() if isinstance(value, str) else list(value)


def _table_rows(table: Tag) -> list[str]:
    lines = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:  # filas de tablas anidadas: las trata la propia celda
            continue
        cells = [" ".join(_blocks(td)) for td in tr.find_all(["td", "th"], recursive=False)]
        line = norm(" ".join(c for c in cells if c))
        if line:
            lines.append(line)
    return lines


def _blocks(element: Tag, skip_classes: tuple[str, ...] = ()) -> list[str]:
    """Líneas de texto de un elemento, en orden de documento (omitiendo las clases indicadas)."""
    lines: list[str] = []
    for child in element.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            text = norm(str(child))
            if text:
                lines.append(text)
        elif isinstance(child, Tag):
            if any(c in skip_classes for c in css_classes(child)):
                continue
            if child.name == "table":
                lines.extend(_table_rows(child))
            elif child.name == "p":
                text = norm(child.get_text(""))
                if text:
                    lines.append(text)
            else:
                lines.extend(_blocks(child, skip_classes))
    return lines


def article_divs(soup: BeautifulSoup) -> list[Tag]:
    divs = [d for d in soup.find_all("div", class_="eli-subdivision") if ART_ID.match(d.get("id", ""))]
    return sorted(divs, key=lambda d: int(ART_ID.match(d["id"]).group(1)))


def extract_article(div: Tag) -> dict:
    """Devuelve número, título, líneas del cuerpo e id del contenedor (capítulo/sección) de un artículo."""
    number = int(ART_ID.match(div["id"]).group(1))
    title_p = div.find("p", class_="oj-sti-art")
    title = norm(title_p.get_text("")) if title_p else ""
    body_lines = _blocks(div, skip_classes=("oj-ti-art", "eli-title"))
    return {
        "number": number,
        "title": title,
        "lines": body_lines,
        "chapter_id": div.parent.get("id") if div.parent is not None else None,
    }


def header(number: int, title: str) -> str:
    return f"Article {number}. {title}" if title else f"Article {number}"


def window_bounds(n_words: int, size: int = WINDOW, overlap: int = OVERLAP, min_tail: int = MIN_TAIL) -> list[tuple[int, int]]:
    """Intervalos [inicio, fin) en palabras. Ver reglas en la cabecera del módulo."""
    if n_words <= size:
        return [(0, n_words)]
    stride = size - overlap
    bounds = [(0, size)]
    while bounds[-1][1] < n_words:
        prev_start, prev_end = bounds[-1]
        if n_words - prev_end < min_tail:
            bounds[-1] = (prev_start, n_words)  # cola corta: se fusiona con la ventana anterior
            break
        start = prev_start + stride
        bounds.append((start, min(start + size, n_words)))
    return bounds


def build_chunks(act: Act, xml: bytes) -> tuple[list[dict], list[dict]]:
    soup = BeautifulSoup(xml, "xml")
    articles, windows = [], []
    for div in article_divs(soup):
        art = extract_article(div)
        head = header(art["number"], art["title"])
        words = " ".join(art["lines"]).split()
        base = {
            "celex": act.celex,
            "act": act.name,
            "article": art["number"],
            "title": art["title"],
            "chapter_id": art["chapter_id"],
        }
        articles.append(
            {
                "chunk_id": f"{act.celex}:art_{art['number']}",
                "strategy": "article",
                **base,
                "text": head + "\n" + "\n".join(art["lines"]),
                "n_words": len(words),
            }
        )
        for k, (start, end) in enumerate(window_bounds(len(words))):
            windows.append(
                {
                    "chunk_id": f"{act.celex}:art_{art['number']}:w{WINDOW}:{k}",
                    "strategy": "window",
                    **base,
                    "window": k,
                    "word_start": start,
                    "word_end": end,
                    "text": head + "\n" + " ".join(words[start:end]),
                    "n_words": end - start,
                }
            )
    return articles, windows


def dump_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def build_all(raw_dir: Path = RAW_DIR, hash_file: Path = HASH_FILE) -> tuple[list[dict], list[dict]]:
    """Comprueba el HTML contra los hashes congelados y construye todos los chunks."""
    frozen = read_hashes(hash_file)
    all_articles, all_windows = [], []
    for act in ACTS:
        path = raw_dir / act.filename
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}; ejecuta `python cli.py download-corpus`.")
        data = path.read_bytes()
        if frozen.get(act.filename) != sha256_bytes(data):
            raise RuntimeError(f"{act.filename}: el SHA-256 no coincide con el congelado en {hash_file.name}.")
        articles, windows = build_chunks(act, data)
        if len(articles) != act.expected_articles:
            raise RuntimeError(f"{act.celex}: {len(articles)} artículos, se esperaban {act.expected_articles}.")
        all_articles += articles
        all_windows += windows
    return all_articles, all_windows


def run() -> int:
    try:
        articles, windows = build_all()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    ARTICLES_FILE.write_text(dump_jsonl(articles), encoding="utf-8")
    WINDOWS_FILE.write_text(dump_jsonl(windows), encoding="utf-8")
    print(f"{len(articles)} chunks por artículo -> {ARTICLES_FILE.relative_to(ROOT)}")
    print(f"{len(windows)} ventanas (w{WINDOW}, o{OVERLAP}) -> {WINDOWS_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
