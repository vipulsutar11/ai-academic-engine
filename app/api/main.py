import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from config import config

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {config.APP_NAME} in '{config.APP_ENV}' environment...")
    # Pre-warm embedding model to prevent first-request cold-start latency
    try:
        from app.rag.embeddings import EmbeddingService
        svc = EmbeddingService()
        svc.encode("warmup query")
        logger.info("Pre-warmed EmbeddingService model successfully.")
    except Exception as e:
        logger.warning(f"Could not pre-warm EmbeddingService: {e}")
    yield
    logger.info(f"Shutting down {config.APP_NAME}...")

app = FastAPI(
    title=config.APP_NAME,
    description="Production AI/ML Content Engine for CBSE, ICSE, and UP Board Curricula.",
    version="1.0.0",
    lifespan=lifespan,
    servers=[{"url": "/", "description": "Local Server"}]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router)
