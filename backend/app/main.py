from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import logger, settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Ensure runtime directories exist on startup
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.model_cache_path.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("AstraML consolidated backend starting up...")
    logger.info("Upload path configured at: %s", settings.upload_path)
    logger.info("Model cache path configured at: %s", settings.model_cache_path)
    logger.info("Database path configured at: %s", settings.sqlite_db_path)
    yield
    logger.info("AstraML consolidated backend shutting down...")


app = FastAPI(
    title="AstraML API",
    description="Unified production-grade backend API for AstraML",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
origins = []
if settings.frontend_origin:
    if "," in settings.frontend_origin:
        origins = [o.strip() for o in settings.frontend_origin.split(",")]
    else:
        origins = [settings.frontend_origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "astraml-backend"}
