"""
SatQuery AI — API v1 Aggregated Router
Combines all endpoint modules under /api/v1/.
"""

from fastapi import APIRouter

from app.api.v1.endpoints_upload import router as upload_router
from app.api.v1.endpoints_query import router as query_router
from app.api.v1.endpoints_stream import router as stream_router
from app.api.v1.endpoints_report import router as report_router
from app.api.v1.endpoints_tiles import router as tiles_router
from app.api.v1.endpoints_multimodal_flood import router as flood_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(upload_router, tags=["Upload"])
api_router.include_router(query_router, tags=["Query"])
api_router.include_router(stream_router, tags=["Streaming"])
api_router.include_router(report_router, tags=["Reports"])
api_router.include_router(tiles_router, tags=["Tiles"])
api_router.include_router(flood_router, tags=["Multimodal Flood & Safe Zones"])
