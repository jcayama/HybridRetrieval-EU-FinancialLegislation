import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from functions.retriever import RetrievalService
from middlewares.error_handler import ErrorHandler
from routers.hybridretrieval import hybrid_retrieval_router

ORIGINS = ["https://localhost:5013"]
DESCRIPTION = """
Recuperación sobre AIFMD, DORA y el AI Act: FTS de Postgres, búsqueda densa con pgvector o híbrida con RRF.
Use **POST /hybridretrievals** desde esta documentación interactiva para probar consultas.
"""


def tenants_enabled_from_env() -> bool:
    load_dotenv(Path(__file__).parent / ".env")  # el .env no se había cargado todavía en este punto del arranque
    return os.getenv("API_ENABLE_TENANTS", "").strip().lower() in ("1", "true", "yes")


def create_app(service: RetrievalService | None = None) -> FastAPI:
    """Si no se inyecta un servicio (los tests lo hacen), se crea uno al arrancar y se intenta cargar el modelo."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.service = RetrievalService(enable_tenants=tenants_enabled_from_env())
            await asyncio.to_thread(app.state.service.start)  # el modelo se carga fuera del bucle de eventos
        yield

    app = FastAPI(title="Hybrid Retrieval on EU Financial Legislation", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
    if service is not None:
        app.state.service = service
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(ErrorHandler)
    app.include_router(hybrid_retrieval_router)

    @app.get("/", tags=["home"], include_in_schema=False)
    def home():
        return HTMLResponse('<h1>Hybrid Retrieval on EU Financial Legislation</h1><p><a href="/docs">Documentación interactiva</a> · <a href="/health">Estado</a></p>')

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("main:app", port=5013, reload=True)
