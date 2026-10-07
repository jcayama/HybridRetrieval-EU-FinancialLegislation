import sys
import traceback
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorHandler(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response | JSONResponse:
        try:
            return await call_next(request)
        except Exception:
            # El detalle técnico va SOLO al log. Devolver str(e) al cliente
            # filtra lo que traiga la excepción: hostnames internos (p. ej. el
            # gateway del LLM), rutas del sistema de ficheros o detalles de
            # configuración.
            #
            # El 500 se mantiene a propósito: una avería no es un resultado de
            # negocio, y quien llama a la API debe poder distinguirla para
            # reintentar. El error_id permite atar lo que reporta el cliente
            # con la traza del log (gunicorn recoge stderr en errorlog).
            # A stderr, igual que traceback.print_exc(), para que el
            # identificador y su traza no acaben en streams distintos.
            error_id = uuid.uuid4().hex[:12]
            print(
                f"[ERROR] {error_id} - {request.method} {request.url.path}",
                file=sys.stderr,
            )
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "Se ha producido un error interno al procesar la "
                        "petición. Vuelva a intentarlo; si el problema "
                        "persiste, facilite el identificador del error."
                    ),
                    "error_id": error_id,
                },
            )