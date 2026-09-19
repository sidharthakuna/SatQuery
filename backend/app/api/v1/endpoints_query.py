"""
SatQuery AI — Query Execution Endpoint
Main endpoint that routes natural-language queries through the orchestrator.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1._file_utils import find_uploaded_file
from app.core.geospatial.reader import inspect_geotiff
from app.core.orchestrator.agent import SatQueryAgent, get_orchestrator_agent
from app.schemas.query import QueryRequest, SatQueryResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


def get_agent() -> SatQueryAgent:
    """Dependency that returns the shared singleton orchestrator agent."""
    return get_orchestrator_agent()


@router.post(
    "/query",
    response_model=SatQueryResult,
    summary="Execute a natural-language query on satellite images",
    description=(
        "Submit a natural-language question along with 1 or 2 uploaded image IDs. "
        "The orchestrator validates inputs, classifies intent, routes to specialist tools, "
        "and returns an answer with a complete execution audit trace."
    ),
)
async def execute_query(
    request: QueryRequest,
    agent: SatQueryAgent = Depends(get_agent),
):
    """
    Main query processing endpoint.

    Flow: Validate → Classify → Execute Tools → Fuse Evidence → Return Trace
    """
    # ── Resolve image file paths from IDs ────────────────────
    image_metas = []
    for image_id in request.image_ids:
        file_path = find_uploaded_file(image_id)
        if file_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {image_id}. Upload it first via POST /api/v1/upload.",
            )
        try:
            meta = inspect_geotiff(file_path, file_id=image_id)
            image_metas.append(meta)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to read image {image_id}: {str(e)}",
            )

    # ── Run orchestrator ─────────────────────────────────────
    try:
        result = await agent.process_query(
            query=request.query,
            image_metas=image_metas,
            history=request.history,
        )
        return result
    except Exception as e:
        logger.exception(f"Query execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {str(e)}",
        )

@router.get(
    "/models",
    summary="Get Specialist Model Registry & Benchmark Catalog",
    description="Returns detailed metadata, architectures, input/output contracts, and empirical benchmark results (VRSBench, RSVQA, CDVQA, BigEarthNet) for all specialist models.",
)
async def get_models_catalog():
    """Return catalog of specialized AI models."""
    from app.core.orchestrator.model_registry import SpecialistModelRegistry
    return SpecialistModelRegistry.get_catalog()

