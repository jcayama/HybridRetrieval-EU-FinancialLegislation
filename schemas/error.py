from pydantic import BaseModel


class Message(BaseModel):
    code: str
    title: str
