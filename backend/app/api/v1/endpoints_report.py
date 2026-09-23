"""
SatQuery AI — Report Generation & Download Endpoints
Generates and serves mission briefing PDF and Word (.docx) documents.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.utils.pdf_generator import MissionBriefingGenerator
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


def _cleanup_old_reports(max_age_seconds: int = 86400):
    """Deletes generated report PDFs and DOCX files older than max_age_seconds (default: 24 hours)."""
    try:
        now = time.time()
        for pattern in ["briefing_*.pdf", "briefing_*.docx"]:
            for p in settings.report_dir.glob(pattern):
                if p.is_file() and (now - p.stat().st_mtime > max_age_seconds):
                    p.unlink(missing_ok=True)
    except Exception as e:
        logger.debug(f"Old report cleanup error: {e}")


class ReportRequest(BaseModel):
    """Request to generate a mission briefing PDF / Word document."""
    query: str = Field(..., description="Original user query")
    text_response: str = Field(..., description="Model's text answer")
    audit_trace: dict = Field(..., description="Full execution trace")
    spatial_evidence: Optional[dict] = Field(None, description="Spatial output data")
    mask_image_path: Optional[str] = Field(None, description="Path to mask image for embedding")
    image_metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Raster metadata list")
    thumbnail_paths: Optional[List[str]] = Field(None, description="Paths to thumbnails for embedding")
    classification: Optional[str] = Field("RESTRICTED", description="Classification banner")
    layout_mode: Optional[str] = Field("comprehensive", description="executive_summary or comprehensive")
    include_sensor_telemetry: Optional[bool] = Field(True, description="Include sensor physics table")
    include_audit_trail: Optional[bool] = Field(True, description="Include execution audit trail")


class ReportResponse(BaseModel):
    """Response with the generated report details."""
    report_id: str
    download_url: str
    docx_download_url: Optional[str] = None
    file_size_bytes: int


@router.post(
    "/report/generate",
    response_model=ReportResponse,
    summary="Generate a mission briefing PDF and Word (.docx) document",
    description=(
        "Generates an Autonomous Multimodal Remote Sensing mission intelligence dossier containing the query, "
        "analysis results, geospatial evidence proof, execution audit trace, and multi-model AI audit."
    ),
)
async def generate_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """Create a downloadable mission briefing PDF and Word (.docx) dossier."""
    try:
        # Fallback extraction of thumbnails and mask if caller omitted explicit fields
        thumbs = list(request.thumbnail_paths or [])
        if not thumbs and request.image_metadata:
            for m in request.image_metadata:
                t = m.get("thumbnail_url") or m.get("file_id")
                if t:
                    thumbs.append(str(t))

        mask_p = request.mask_image_path
        if not mask_p and request.spatial_evidence:
            mask_p = request.spatial_evidence.get("mask_url")

        import asyncio
        generator = MissionBriefingGenerator()
        pdf_path = await asyncio.to_thread(
            generator.generate,
            query=request.query,
            text_response=request.text_response,
            audit_trace=request.audit_trace,
            spatial_evidence=request.spatial_evidence,
            mask_image_path=mask_p,
            image_metadata=request.image_metadata,
            thumbnail_paths=thumbs if thumbs else None,
            classification=request.classification or "RESTRICTED",
            layout_mode=request.layout_mode or "comprehensive",
            include_sensor_telemetry=True if request.include_sensor_telemetry is None else request.include_sensor_telemetry,
            include_audit_trail=True if request.include_audit_trail is None else request.include_audit_trail,
        )

        pdf_file = Path(pdf_path)
        report_id = pdf_file.stem.replace("briefing_", "")
        docx_file = settings.report_dir / f"briefing_{report_id}.docx"

        # Clean up orphaned reports older than 24 hours in background task
        background_tasks.add_task(_cleanup_old_reports)

        return ReportResponse(
            report_id=report_id,
            download_url=f"/api/v1/report/{report_id}",
            docx_download_url=f"/api/v1/report/{report_id}/docx" if docx_file.exists() else None,
            file_size_bytes=pdf_file.stat().st_size,
        )

    except Exception as e:
        logger.exception(f"Report generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}",
        )


@router.get(
    "/report/{report_id}",
    summary="Download or view a generated mission briefing PDF",
    description="Retrieve and download or view a previously generated PDF report by its ID.",
)
async def download_report(report_id: str, view: bool = False):
    """Serve a generated PDF report for download or browser viewing."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID format.")

    pdf_path = (settings.report_dir / f"briefing_{report_id}.pdf").resolve()
    if not str(pdf_path).startswith(str(settings.report_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report not found: {report_id}",
        )

    disposition = "inline" if view else "attachment"
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        content_disposition_type=disposition,
        filename=f"SatQuery_Briefing_{report_id}.pdf",
    )


@router.get(
    "/report/{report_id}/docx",
    summary="Download a generated mission briefing Word (.docx) document",
    description="Retrieve and download an editable Microsoft Word (.docx) report by its ID.",
)
async def download_report_docx(report_id: str):
    """Serve a generated Word (.docx) report for download."""
    import re
    if not re.match(r"^[a-zA-Z0-9_\-]+$", report_id):
        raise HTTPException(status_code=400, detail="Invalid report ID format.")

    docx_path = (settings.report_dir / f"briefing_{report_id}.docx").resolve()
    if not str(docx_path).startswith(str(settings.report_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not docx_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Word document report not found for ID: {report_id}",
        )

    return FileResponse(
        path=str(docx_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        content_disposition_type="attachment",
        filename=f"SatQuery_Briefing_{report_id}.docx",
    )

