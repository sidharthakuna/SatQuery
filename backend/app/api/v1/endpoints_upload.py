"""
SatQuery AI — GeoTIFF Upload Endpoint
Handles multi-part file upload, metadata extraction, and thumbnail generation.
"""

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.geospatial.reader import inspect_geotiff
from app.schemas.query import ImageUploadResponse
from config.constants import SUPPORTED_EXTENSIONS
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    summary="Upload a GeoTIFF satellite image",
    description=(
        "Upload a single GeoTIFF file (Optical or SAR). "
        "The server validates the format, extracts geospatial metadata, "
        "auto-detects the modality (OPTICAL/SAR), and generates a preview thumbnail."
    ),
)
async def upload_geotiff(file: UploadFile = File(...)):
    """
    Multi-part GeoTIFF ingestion endpoint.

    Accepts: .tif, .tiff, .geotiff
    Returns: File ID, metadata, and thumbnail URL
    """
    # ── Validate file extension ──────────────────────────────
    filename = file.filename or "upload.tif"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{ext}'. Expected: {SUPPORTED_EXTENSIONS}",
        )

    # ── Pre-check Content-Length header if provided ──────────────
    if hasattr(file, "headers") and file.headers:
        try:
            content_length = int(file.headers.get("content-length", 0) or 0)
            if content_length > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large (> {settings.max_upload_size_mb} MB). "
                        f"Maximum allowed: {settings.max_upload_size_mb} MB."
                    ),
                )
        except (ValueError, TypeError):
            pass

    # ── Stream directly to disk in chunks (zero memory duplication) ─
    CHUNK_SIZE = 1024 * 1024  # 1 MB
    file_id = uuid4().hex[:12]
    safe_filename = f"{file_id}_{filename}"
    save_path = settings.upload_dir / safe_filename
    total_bytes = 0

    try:
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large (> {settings.max_upload_size_mb} MB). "
                            f"Maximum allowed: {settings.max_upload_size_mb} MB."
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {e}")
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to write uploaded file to server storage.")

    logger.info(f"Saved upload: {save_path} ({total_bytes} bytes)")

    # ── Extract metadata ─────────────────────────────────────
    try:
        metadata = inspect_geotiff(str(save_path), file_id=file_id)
    except Exception as e:
        # Clean up on failure
        logger.warning(f"Metadata extraction failed for {save_path}: {e}")
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Invalid GeoTIFF raster: unable to read image headers or spatial metadata.",
        )

    # ── Set dynamic GeoTIFF preview ──────────────────────────
    thumbnail_url = f"/api/v1/preview/{file_id}"

    # ── Build response ───────────────────────────────────────
    bounds_dict = None
    if metadata.bounds_latlon:
        bounds_dict = metadata.bounds_latlon.model_dump()

    return ImageUploadResponse(
        file_id=file_id,
        filename=filename,
        file_size_bytes=total_bytes,
        modality=metadata.modality,
        crs=metadata.crs,
        width=metadata.width,
        height=metadata.height,
        band_count=metadata.band_count,
        bounds_latlon=bounds_dict,
        thumbnail_url=thumbnail_url,
    )
