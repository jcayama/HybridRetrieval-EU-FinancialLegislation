"""Descarga y congela el corpus de EUR-Lex (texto publicado, vía API Cellar).

Uso:
    python cli.py download-corpus [--freeze]     # recomendado
    python -m scripts.download_corpus            # descarga y VERIFICA contra data/corpus.sha256
    python -m scripts.download_corpus --freeze   # descarga y ESCRIBE data/corpus.sha256

La verificación falla si el SHA-256 de una descarga no coincide con el congelado
o si el número de artículos no es el esperado.
"""

import argparse
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HASH_FILE = ROOT / "data" / "corpus.sha256"

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
HEADERS = {
    "Accept": "application/xhtml+xml, text/html",
    "Accept-Language": "eng",
}
ART_ID = re.compile(r"^art_(\d+)$")


@dataclass(frozen=True)
class Act:
    name: str
    celex: str
    expected_articles: int

    @property
    def filename(self) -> str:
        return f"{self.celex}.html"


ACTS = [
    Act("AIFMD 2011/61/UE", "32011L0061", 71),
    Act("DORA 2022/2554", "32022R2554", 64),
    Act("AI Act 2024/1689", "32024R1689", 113),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_articles(html: bytes | str) -> int:
    """Cuenta los `div.eli-subdivision` con id `art_N` (N numérico), sin repetidos."""
    soup = BeautifulSoup(html, "xml")
    ids = {div["id"] for div in soup.select("div.eli-subdivision[id]") if ART_ID.match(div["id"])}
    return len(ids)


def read_hashes(path: Path = HASH_FILE) -> dict[str, str]:
    """Lee un fichero en formato `sha256sum` (`<hash>  <fichero>`)."""
    if not path.exists():
        return {}
    hashes = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, filename = line.split(maxsplit=1)
            hashes[filename.strip()] = digest
    return hashes


def write_hashes(hashes: dict[str, str], path: Path = HASH_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))


def fetch(client: httpx.Client, url: str, retries: int = 3) -> bytes:
    for attempt in range(1, retries + 1):
        response = client.get(url)
        # La página web de EUR-Lex devuelve un 202 vacío a los scripts; Cellar devuelve el texto.
        if response.status_code == 200 and response.content:
            return response.content
        print(f"  intento {attempt}/{retries}: HTTP {response.status_code}, {len(response.content)} bytes", file=sys.stderr)
        time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo descargar {url}")


def run(freeze: bool = False) -> int:
    """Descarga el corpus. Devuelve 0 si todo es correcto y 1 si hay algún fallo."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frozen = read_hashes()
    new_hashes: dict[str, str] = {}
    failures: list[str] = []

    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        for act in ACTS:
            print(f"{act.name} ({act.celex})")
            data = fetch(client, CELLAR_URL.format(celex=act.celex))
            (RAW_DIR / act.filename).write_bytes(data)
            digest = sha256_bytes(data)
            articles = count_articles(data)
            new_hashes[act.filename] = digest
            print(f"  {len(data)} bytes, sha256={digest}, artículos={articles} (esperados {act.expected_articles})")

            if articles != act.expected_articles:
                failures.append(f"{act.celex}: {articles} artículos, se esperaban {act.expected_articles}")
            if not freeze:
                if act.filename not in frozen:
                    failures.append(f"{act.filename}: sin hash congelado (usa --freeze la primera vez)")
                elif frozen[act.filename] != digest:
                    failures.append(f"{act.filename}: el SHA-256 no coincide con el congelado")

    if failures:
        print("\nFALLOS:", *failures, sep="\n  - ", file=sys.stderr)
        return 1
    if freeze:
        write_hashes(new_hashes)
        print(f"\nHashes congelados en {HASH_FILE.relative_to(ROOT)}")
    else:
        print("\nVerificación correcta: hashes y recuento de artículos coinciden.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", action="store_true", help="escribe data/corpus.sha256 con los hashes descargados")
    return run(parser.parse_args().freeze)


if __name__ == "__main__":
    sys.exit(main())
