from contextlib import asynccontextmanager
import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

from app.core.config import settings
from app.core.http_client import close_http_client
from app.routers import ai, chat

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set INFO level for our app modules when CHAT_TRACE_ENABLED is on,
    otherwise keep WARNING to stay quiet alongside uvicorn."""
    level = logging.INFO if settings.CHAT_TRACE_ENABLED else logging.WARNING
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger("app")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    if settings.CHAT_TRACE_ENABLED:
        logger.info("[TRACE] Chat trace enabled — logs to stdout + %s", settings.CHAT_TRACE_DIR)
    yield
    await close_http_client()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservice for Data Crawling and AI Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["POST", "GET"],
    allow_headers=["X-Internal-Api-Key", "Content-Type"],
)

app.include_router(ai.router)
app.include_router(chat.router)


@app.exception_handler(httpx.TimeoutException)
async def _handle_llm_timeout(request: Request, exc: httpx.TimeoutException) -> JSONResponse:
    logger.warning("Upstream timeout: %s", exc)
    return JSONResponse(status_code=504, content={"detail": "LLM timeout"})


@app.exception_handler(httpx.HTTPStatusError)
async def _handle_llm_upstream(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    logger.warning("Upstream HTTP error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": "LLM upstream error"})


@app.get("/")
def read_root():
    return {"message": "Welcome to FinFlow Data & AI Service"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
