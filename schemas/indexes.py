from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    id_document: int
    title: str


class Indexes(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"indexes": [{"id_document": 0, "title": "title"}]}})

    indexes: List[Document]


class InputData(BaseModel):
    query: str = Field(..., description="Consulta de búsqueda")
