"""
SatQuery AI — Audit Trace Builder
Accumulates step-by-step execution logs for full observability.
"""

import time
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.audit import ExecutionTrace, TaskType, TraceStep, ValidationReport

logger = logging.getLogger(__name__)


class AuditTraceBuilder:
    """
    Immutable audit trace compiler.
    Records every decision, tool call, timing, and metric from a single
    orchestrator run. The resulting trace is fully inspectable and
    tamper-evident (unique trace_id + timestamps).
    """

    def __init__(self):
        self._trace_id: str = uuid4().hex[:16]
        self._start_time: float = time.time()
        self._last_step_time: float = self._start_time
        self._steps: List[TraceStep] = []
        self._step_counter: int = 0
        self._task_type: TaskType = TaskType.UNKNOWN
        self._validation: Optional[ValidationReport] = None
        self._selected_tools: List[str] = []
        self._parameters: Dict[str, Any] = {}
        self._confidence: float = 0.0
        self._status: str = "IN_PROGRESS"
        self._error: Optional[str] = None
        self._confidence_decomposition = None
        self._evidence_verification = None
        self._evidence_graph = None
        self._interpreted_query = None

    def record_step(
        self,
        step_name: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> TraceStep:
        """Record a single execution step with individual step timing."""
        now = time.time()
        step_duration = round((now - self._last_step_time) * 1000, 2)
        self._last_step_time = now

        step = TraceStep(
            step_index=self._step_counter,
            step_name=step_name,
            status=status,
            duration_ms=step_duration,
            details=details or {},
            message=message,
        )
        self._steps.append(step)
        self._step_counter += 1
        logger.debug(f"Trace step [{step.step_index}] {step_name}: {status} ({step_duration}ms)")
        return step

    def set_task_type(self, task_type: TaskType):
        self._task_type = task_type

    def set_validation(self, report: ValidationReport):
        self._validation = report

    def set_selected_tools(self, tools: List[str]):
        self._selected_tools = tools

    def set_parameters(self, params: Dict[str, Any]):
        self._parameters = params

    def set_confidence(self, score: float):
        self._confidence = max(0.0, min(1.0, score))

    def set_error(self, error: str):
        self._error = error
        self._status = "ERROR"

    def set_success(self):
        self._status = "SUCCESS"

    def set_confidence_decomposition(self, decomp):
        self._confidence_decomposition = decomp

    def set_evidence_verification(self, verif):
        self._evidence_verification = verif

    def set_evidence_graph(self, graph):
        self._evidence_graph = graph

    def set_interpreted_query(self, query):
        self._interpreted_query = query

    def build(self) -> ExecutionTrace:
        """Compile the final immutable execution trace."""
        total_ms = round((time.time() - self._start_time) * 1000, 2)

        return ExecutionTrace(
            trace_id=self._trace_id,
            task_identified=self._task_type,
            input_validation=self._validation or ValidationReport(
                is_valid=False, image_count=0, errors=["No validation performed"]
            ),
            selected_tools=self._selected_tools,
            parameters_applied=self._parameters,
            execution_steps=self._steps,
            total_execution_time_ms=total_ms,
            confidence_score=self._confidence,
            status=self._status,
            error_message=self._error,
            confidence_decomposition=self._confidence_decomposition,
            evidence_verification=self._evidence_verification,
            evidence_graph=self._evidence_graph,
            interpreted_query=self._interpreted_query,
        )

    def get_steps_for_streaming(self) -> List[Dict[str, Any]]:
        """Returns steps as plain dicts for WebSocket streaming."""
        return [step.model_dump() for step in self._steps]
