"""
SatQuery AI — WebSocket Streaming Endpoint
Real-time streaming of orchestrator trace steps as they execute.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.v1._file_utils import find_uploaded_file
from app.core.geospatial.reader import inspect_geotiff
from app.core.orchestrator.agent import SatQueryAgent, get_orchestrator_agent

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_agent() -> SatQueryAgent:
    return get_orchestrator_agent()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time query execution with streaming trace.

    Protocol:
    1. Client connects
    2. Client sends JSON: {"query": "...", "image_ids": ["id1", "id2"]}
    3. Server streams trace steps as JSON messages:
       {"type": "step", "data": {"step_name": "VALIDATION", ...}}
    4. Server sends final result:
       {"type": "result", "data": {full SatQueryResult}}
    5. Connection closes
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        # ── Receive query ────────────────────────────────────
        raw = await websocket.receive_text()
        message = json.loads(raw)

        query = message.get("query", "")
        image_ids = message.get("image_ids", [])
        history = message.get("history", [])

        if not query:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Missing 'query' field"},
            })
            await websocket.close()
            return

        # ── Send acknowledgment ──────────────────────────────
        await websocket.send_json({
            "type": "ack",
            "data": {
                "query": query,
                "image_count": len(image_ids),
                "status": "PROCESSING",
            },
        })

        # ── Resolve images ───────────────────────────────────
        image_metas = []
        for img_id in image_ids:
            file_path = find_uploaded_file(img_id)
            if file_path:
                try:
                    meta = inspect_geotiff(file_path, file_id=img_id)
                    image_metas.append(meta)
                except Exception as e:
                    await websocket.send_json({
                        "type": "warning",
                        "data": {"message": f"Failed to read image {img_id}: {e}"},
                    })
            else:
                await websocket.send_json({
                    "type": "warning",
                    "data": {"message": f"Image not found: {img_id}"},
                })

        # ── Stream: Validation & Classification steps (for imagery tasks) ──
        if image_metas:
            await websocket.send_json({
                "type": "step",
                "data": {
                    "step_index": 0,
                    "step_name": "VALIDATING",
                    "status": "IN_PROGRESS",
                    "duration_ms": 0,
                    "details": {"image_count": len(image_metas)},
                    "message": f"Validating {len(image_metas)} image(s)...",
                },
            })
            await websocket.send_json({
                "type": "step",
                "data": {
                    "step_index": 1,
                    "step_name": "CLASSIFYING",
                    "status": "IN_PROGRESS",
                    "duration_ms": 0,
                    "details": {},
                    "message": "Classifying query intent...",
                },
            })

        # ── Execute full pipeline with real-time streaming ──
        agent = _get_agent()
        is_client_connected = True

        async def stream_step(step):
            nonlocal is_client_connected
            if not is_client_connected:
                raise WebSocketDisconnect()
            try:
                await websocket.send_json({
                    "type": "step",
                    "data": step.model_dump(),
                })
            except (WebSocketDisconnect, RuntimeError):
                is_client_connected = False
                logger.info("WebSocket client disconnected during step streaming")
                raise WebSocketDisconnect()
            except Exception as stream_err:
                logger.debug(f"Failed to stream step: {stream_err}")

        result = await agent.process_query(
            query=query,
            image_metas=image_metas,
            history=history,
            on_step=stream_step,
        )

        # ── Send final result ────────────────────────────────
        if is_client_connected:
            await websocket.send_json({
                "type": "result",
                "data": result.model_dump(),
            })

        logger.info(
            f"WebSocket query complete: task={result.audit_trace.task_identified}, "
            f"confidence={result.audit_trace.confidence_score:.2f}"
        )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "data": {"message": "Invalid JSON format"},
        })
        await websocket.close()
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)},
            })
        except Exception:
            pass
        # Guard against RuntimeError if the connection is already closed
        try:
            await websocket.close()
        except Exception:
            pass



