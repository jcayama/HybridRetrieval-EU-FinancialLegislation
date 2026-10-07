from pydantic import BaseModel


class Error(BaseModel):
    code: str
    title: str
