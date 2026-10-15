import asyncio
import re

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse

from functions.retriever import DatabaseNotReady, ModelUnavailable, RetrievalService
from models.response import ErrorResponse
from schemas.retrieval import MAX_QUERY_CHARS, HealthResponse, Hit, RetrievalRequest, RetrievalResponse

TENANT_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")

hybrid_retrieval_router = APIRouter()


def get_service(request: Request) -> RetrievalService:
    return request.app.state.service


def error(status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorResponse.from_type(error_type).model_dump())


@hybrid_retrieval_router.post(
    "/hybridretrievals",
    tags=["hybrid retrieval"],
    response_model=RetrievalResponse,
    summary="Recuperar artículos de legislación financiera de la UE",
    description=(
        "Devuelve artículos distintos de AIFMD, DORA y el AI Act ordenados por relevancia. "
        "`mode=fts` funciona con `requirements.txt`; `hybrid` y `dense` necesitan además el modelo de embeddings. "
        "`X-Tenant-ID` solo tiene efecto si el servidor se arranca con `API_ENABLE_TENANTS=true`; no hay autenticación, "
        "así que no es una medida de seguridad sino una demostración del aislamiento por RLS con datos sintéticos."
    ),
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def hybrid_retriever(
    body: RetrievalRequest,
    service: RetrievalService = Depends(get_service),
    x_tenant_id: str | None = Header(default=None, description="Inquilino (solo con API_ENABLE_TENANTS=true)."),
):
    query = body.query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        return error(400, "INVALID_QUERY_PARAMETER")
    tenant = None
    if service.tenants_enabled and x_tenant_id is not None:  # sin inquilinos habilitados, la cabecera se ignora
        if not TENANT_PATTERN.fullmatch(x_tenant_id):
            return error(400, "INVALID_TENANT")
        tenant = x_tenant_id
    try:
        hits = await asyncio.to_thread(service.search, query, body.top_k, body.mode, body.chunking, tenant)
    except ModelUnavailable:
        return error(503, "MODEL_UNAVAILABLE")
    except DatabaseNotReady:
        return error(503, "DATABASE_NOT_READY")
    return RetrievalResponse(query=query, mode=body.mode, chunking=body.chunking, top_k=body.top_k, results=[Hit(**vars(h)) for h in hits])


@hybrid_retrieval_router.get("/health", tags=["home"], response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def health(service: RetrievalService = Depends(get_service)):
    state = await asyncio.to_thread(service.status)
    body = HealthResponse(status="ok" if state["database"] else "degraded", **state)
    return JSONResponse(status_code=200 if state["database"] else 503, content=body.model_dump())
