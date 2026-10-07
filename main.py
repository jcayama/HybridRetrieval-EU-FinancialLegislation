from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from routers.hybridretrieval import hybrid_retrieval_router
from functions.generalFunctions import ModelRetrieval
from middlewares.error_handler import ErrorHandler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Se carga una sola vez por worker: modelo de embeddings (por ejemplo)
    app.state.retriever = ModelRetrieval()
    yield


app = FastAPI(lifespan=lifespan)
app.title = "Hybrid Retrieval on EU Financial Legislation"
app.version = "0.0.1"

origins = [
    "https://localhost:5013",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ErrorHandler)
app.include_router(hybrid_retrieval_router)


@app.get('/', tags=['home'])
def health_checker():
    """Health checker endpoint that returns a simple HTML response."""
    return HTMLResponse('<h1>Hybrid Retrieval on EU Financial Legislation</h1>')


if __name__ == "__main__":
    uvicorn.run("main:app", port=5013, reload=True)