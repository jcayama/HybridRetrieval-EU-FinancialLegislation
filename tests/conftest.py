import uuid

import psycopg
import pytest

from scripts import db


@pytest.fixture
def schema():
    """Esquema temporal aislado: no toca las tablas de desarrollo."""
    name = f"test_{uuid.uuid4().hex[:10]}"
    try:
        with db.connect() as conn:
            conn.execute(f"CREATE SCHEMA {name}")
    except (RuntimeError, psycopg.OperationalError) as error:
        pytest.skip(f"Postgres no disponible: {error}")
    try:
        with db.connect(name) as conn:
            db.init_db(conn)
        yield name
    finally:
        with db.connect() as conn:
            conn.execute(f"DROP SCHEMA {name} CASCADE")
