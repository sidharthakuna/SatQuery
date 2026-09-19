"""
SatQuery AI — FastAPI Application Entry Point
Main application with lifespan handler, CORS, static files, and health check.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── Fix import paths ─────────────────────────────────────────
# Ensure the backend directory is in sys.path for clean imports
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from app.api.v1.router import api_router
from app.inference.factory import InferenceFactory
from config.settings import settings

# ── Configure Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("satquery")


# ── Lifespan Handler ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    Initializes inference engine, tool registry, and data directories.
    """
    logger.info("=" * 60)
    logger.info("  🛰️  SatQuery AI — Backend Starting")
    logger.info("=" * 60)
    logger.info(f"  Inference Mode : {settings.inference_mode.value}")
    logger.info(f"  Host           : {settings.host}:{settings.port}")
    logger.info(f"  Upload Dir     : {settings.upload_dir}")
    logger.info(f"  Data Dir       : {settings.data_dir}")
    logger.info(f"  Max VRAM       : {settings.max_vram_gb} GB")
    logger.info("=" * 60)

    # Ensure data directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.samples_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference provider
    try:
        provider = InferenceFactory.get_provider()
        logger.info(f"Inference provider ready: {type(provider).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize inference provider: {e}")

    yield  # Application is running

    # Shutdown
    logger.info("🛰️  SatQuery AI — Backend Shutting Down")


# ── Create FastAPI Application ───────────────────────────────
app = FastAPI(
    title="SatQuery AI",
    description=(
        "Interactive vision-language platform for multimodal remote sensing "
        "image analysis through natural-language queries. "
        "ISRO / Space Applications Centre — SIH Problem Statement ID: 26167"
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ensure Static Directories Exist Before Mounting ──────────
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.report_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.samples_dir.mkdir(parents=True, exist_ok=True)

# ── Dynamic Raster Image & Preview Handlers ──────────────────
@app.get("/static/uploads/{file_name:path}")
async def serve_static_upload(file_name: str):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, Response
    from app.api.v1.endpoints_tiles import _find_raster_file, _render_preview_bytes

    clean = Path(file_name).name
    p = _find_raster_file(clean) or (settings.upload_dir / clean)
    if p and p.exists() and p.is_file():
        if p.suffix.lower() in [".tif", ".tiff", ".geotiff"]:
            try:
                b = _render_preview_bytes(str(p))
                return Response(content=b, media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})
            except Exception as e:
                logger.warning(f"Static preview render error for {p}: {e}")
        return FileResponse(p)
    raise HTTPException(status_code=404, detail="File not found")

# ── Static Files (thumbnails, masks, reports) ────────────────
app.mount(
    "/static/uploads",
    StaticFiles(directory=str(settings.upload_dir)),
    name="uploads",
)
app.mount(
    "/static/reports",
    StaticFiles(directory=str(settings.report_dir)),
    name="reports",
)
app.mount(
    "/static/samples",
    StaticFiles(directory=str(settings.samples_dir)),
    name="samples",
)
app.mount(
    "/samples",
    StaticFiles(directory=str(settings.samples_dir)),
    name="samples_root",
)

# ── Mount API Router ─────────────────────────────────────────
app.include_router(api_router)


# ── Health Check ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "version": "0.2.0",
        "inference_mode": settings.inference_mode.value,
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "SatQuery AI",
        "description": "Vision-Language Assistant for Remote Sensing",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": "/api/v1",
        "organization": "ISRO / Space Applications Centre (SAC)",
        "problem_statement": "SIH 26167",
    }
