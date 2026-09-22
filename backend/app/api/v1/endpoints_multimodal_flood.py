"""
SatQuery AI — Multimodal Flood & Cloud-Free API Endpoints
Dedicated endpoints for all-weather flood inundation assessment,
cloud-penetrating ground reconstruction, and safe evacuation zone routing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.v1._file_utils import find_uploaded_file
from app.core.geospatial.reader import inspect_geotiff
from app.core.orchestrator.multimodal_flood_agentic_router import MultimodalFloodAgenticRouter
from app.schemas.query import SatQueryResult
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/multimodal-flood", tags=["Multimodal Flood & Safe Zones"])

_router_instance: Optional[MultimodalFloodAgenticRouter] = None


def get_multimodal_router() -> MultimodalFloodAgenticRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = MultimodalFloodAgenticRouter()
    return _router_instance


class MultimodalFloodRequest(BaseModel):
    query: Optional[str] = Field(
        "Fuse Sentinel-1 SAR and cloudy optical imagery to penetrate storm clouds, reconstruct a cloud-free ground view, calculate total flooded area, and pinpoint elevated safe evacuation zones.",
        description="Disaster intelligence query",
    )
    optical_image_id: Optional[str] = Field(None, description="Uploaded Sentinel-2 optical image ID")
    sar_image_id: Optional[str] = Field(None, description="Uploaded Sentinel-1 SAR image ID")


@router.post(
    "/analyze",
    response_model=SatQueryResult,
    summary="Execute Multimodal Cloud-Penetrating Flood & Safe Zone Assessment",
    description="Fuses cloud-obscured optical imagery with penetrating SAR radar to clear clouds, quantify flood extent, and delineate elevated safe evacuation zones.",
)
async def analyze_multimodal_flood(request: MultimodalFloodRequest):
    """
    Executes the multimodal agentic routing pipeline for cloud-penetrating flood disaster assessment.
    """
    agentic_router = get_multimodal_router()
    image_metas = []

    # 1. Resolve Optical image
    opt_file = None
    if request.optical_image_id:
        opt_file = find_uploaded_file(request.optical_image_id)
    if not opt_file or not Path(opt_file).exists():
        opt_file = settings.samples_dir / "fusion_optical.tif"
        if not opt_file.exists():
            opt_file = settings.samples_dir / "public_flood_cloudy_optical.tif"
    
    if opt_file and Path(opt_file).exists():
        try:
            image_metas.append(inspect_geotiff(str(opt_file), file_id=request.optical_image_id or opt_file.stem))
        except Exception as e:
            logger.warning(f"Error inspecting optical raster {opt_file}: {e}")

    # 2. Resolve SAR image
    sar_file = None
    if request.sar_image_id:
        sar_file = find_uploaded_file(request.sar_image_id)
    if not sar_file or not Path(sar_file).exists():
        sar_file = settings.samples_dir / "fusion_sar.tif"
        if not sar_file.exists():
            sar_file = settings.samples_dir / "public_flood_sentinel1_sar.tif"

    if sar_file and Path(sar_file).exists():
        try:
            image_metas.append(inspect_geotiff(str(sar_file), file_id=request.sar_image_id or sar_file.stem))
        except Exception as e:
            logger.warning(f"Error inspecting SAR raster {sar_file}: {e}")

    try:
        result = await agentic_router.route_and_execute(
            query=request.query,
            image_metas=image_metas,
        )
        return result
    except Exception as e:
        logger.exception(f"Multimodal flood routing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal flood assessment failed: {str(e)}",
        )


@router.get(
    "/sample",
    summary="Get Public Satellite Flood Sample Metadata",
    description="Returns metadata and sample paths for public Copernicus Sentinel-1 SAR and Sentinel-2 cloudy optical flood imagery.",
)
async def get_flood_sample_metadata():
    """Returns information on the pre-bundled public satellite flood sample."""
    return {
        "title": "Copernicus Sentinel-1 SAR & Sentinel-2 Cloud-Penetrating Flood Assessment",
        "description": "Active monsoonal flood surge over low-lying river basin with 52% storm cloud obstruction.",
        "optical_sample": {
            "satellite": "Sentinel-2 MSI",
            "band_combination": "RGB (B04, B03, B02)",
            "cloud_cover": "52.4% dense cumulus",
            "url": "/static/samples/public_flood_cloudy_optical.tif",
        },
        "sar_sample": {
            "satellite": "Sentinel-1 C-Band SAR",
            "polarization": "Dual-Pol (VV + VH)",
            "penetration": "All-weather microwave (0% cloud loss)",
            "url": "/static/samples/public_flood_sentinel1_sar.tif",
        },
        "default_query": "Fuse Sentinel-1 SAR and cloudy optical imagery to penetrate storm clouds, reconstruct a cloud-free ground view, calculate total flooded area, and pinpoint elevated safe evacuation zones.",
    }
