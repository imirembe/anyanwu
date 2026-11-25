"""Cloud ASGI entry point retaining the app's existing /v1/turns contract."""
from pathlib import Path
import logging

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from backend.http_boundary import HTTPBoundary, model_call
from backend.server import APIError, Config, AnyanwuService, MAX_REQUEST_BYTES, load_environment


def create_app(config: Config | None = None, *, providers=None):
    if config is None:
        load_environment(Path(__file__).with_name(".env"))
        config = Config.from_environment()
    if len(config.access_token) < 32 or not config.access_token.isascii():
        raise ValueError("Set an ASCII ANYANWU_ACCESS_TOKEN of at least 32 characters for the cloud gateway.")
    service = AnyanwuService(config, providers)
    api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @api.exception_handler(APIError)
    async def known_error(request, error):
        return JSONResponse({"error": str(error)}, status_code=error.status)

    @api.exception_handler(Exception)
    async def unexpected_error(request, error):
        logging.getLogger("anyanwu.gateway").error("Turn failed: %s", type(error).__name__)
        return JSONResponse({"error": "Anyanwụ couldn’t complete this request. Please try again."}, status_code=500)

    @api.get("/health")
    async def health():
        return service.health()

    @api.post("/v1/turns")
    async def turn(request: Request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise APIError(415, "Use application/json.")
        try:
            payload = await request.json()
        except ValueError:
            raise APIError(400, "Invalid JSON.") from None
        return await model_call(service.turn, payload)

    return HTTPBoundary(api, token=config.access_token, max_body=MAX_REQUEST_BYTES, concurrency=4)
