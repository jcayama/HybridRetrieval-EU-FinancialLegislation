import os

import pytest


def skip_or_fail(reason: str):
    """En la CI (GitHub define CI=true) un servicio que DEBERÍA estar disponible no se salta: falla.

    Si no, una base de datos inaccesible dejaría los tests de RLS y de paridad «saltados» y la CI saldría verde sin
    haber comprobado nada. En local, sin Postgres, se saltan como siempre.
    """
    if os.getenv("CI"):
        pytest.fail(f"[CI] {reason}")
    pytest.skip(reason)
