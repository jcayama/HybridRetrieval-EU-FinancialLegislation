import fastapi.openapi.utils as fu
from fastapi import APIRouter, Body, Depends, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from functions.generalFunctions import ModelRetrieval
from models.response import ErrorResponse
from schemas.indexes import Indexes, InputData

responses = {
    400: {"model": ErrorResponse, "description": "Error: Bad Request"},
}


hybrid_retrieval_router = APIRouter()


def get_retriever(request: Request) -> ModelRetrieval:
    return request.app.state.retriever


# and overwrite the validation error schema
fu.validation_error_response_definition = {
    "title": "HTTPValidationError",
    "type": "object",
    "properties": {
        "error": {"title": "Message", "type": "string"},
    },
}


# Hybrid Retrieval endpoint
@hybrid_retrieval_router.post(
    "/hybridretrievals", tags=["hybrid retrieval"], response_model=Indexes, responses=responses
)
def hybrid_retriever(
    input_data: InputData = Body(...),
    retriever: ModelRetrieval = Depends(get_retriever),
) -> Indexes:

    if len(input_data.query.strip()) < 10:
        response = ErrorResponse.from_type("INVALID_QUERY_PARAMETER")
        return JSONResponse(status_code=400, content=jsonable_encoder(response))

    # TODO: la recuperación (FTS / denso / híbrido con RRF) aún no está implementada.
    return Indexes(indexes=[])
