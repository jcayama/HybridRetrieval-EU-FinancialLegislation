from typing import List

from pydantic import BaseModel, ConfigDict

from models.error import Error

ERRORS = {
    "INVALID_QUERY_PARAMETER": ("E0001", "Cadena vacía o menor a 10 caracteres"),
    "TEXT_NOT_EXTRACTED": ("E0002", "Texto no procesado"),
}


class Message(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"code": "string", "title": "string"}})

    code: str
    title: str


class ErrorResponse(BaseModel):
    errors: List[Error]

    @classmethod
    def from_type(cls, error_type: str) -> "ErrorResponse":
        if error_type not in ERRORS:
            raise ValueError(f"Unsupported error type: {error_type}")
        code, title = ERRORS[error_type]
        return cls(errors=[Error(code=code, title=title)])
