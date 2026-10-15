from pydantic import BaseModel

from models.error import Error

ERRORS = {
    "INVALID_QUERY_PARAMETER": ("E0001", "La consulta está vacía o supera los 500 caracteres"),
    "MODEL_UNAVAILABLE": ("E0002", "El modelo de embeddings no está disponible: instala requirements-embeddings.txt o usa mode=fts"),
    "INVALID_TENANT": ("E0003", "Cabecera X-Tenant-ID no válida: de 1 a 64 caracteres entre letras, cifras, «_», «.» y «-»"),
    "DATABASE_NOT_READY": ("E0004", "La base de datos no está disponible o no está cargada: ejecuta `docker compose up -d`, `python cli.py init-db` y `python cli.py load-db`"),
}


class ErrorResponse(BaseModel):
    errors: list[Error]

    @classmethod
    def from_type(cls, error_type: str) -> "ErrorResponse":
        if error_type not in ERRORS:
            raise ValueError(f"Unsupported error type: {error_type}")
        code, title = ERRORS[error_type]
        return cls(errors=[Error(code=code, title=title)])
