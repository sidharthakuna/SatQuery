"""
SatQuery AI — Grounded VQA Domain Reasoner
Performs deep conversational scene interpretation and Visual Question Answering,
fusing fine-tuned neural VLM outputs, real raster spectral telemetry, and multi-turn context.
"""

from typing import Any, Dict, List, Optional, Tuple

from app.core.orchestrator.synthesis.conversational_vqa_engine import ConversationalVQAEngine


class VQADomainReasoner:
    """
    Synthesizes articulate, dynamic, context-aware VQA answers
    grounded 100% in image pixels, neural predictions, and multi-turn memory.
    Delegates to the modular ConversationalVQAEngine.
    """

    @staticmethod
    def reason_vqa(
        query: str,
        image_metas: List[Any],
        extra: Dict[str, Any],
        images: Optional[List[Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[str]]:
        return ConversationalVQAEngine.synthesize_answer(
            query=query,
            image_metas=image_metas,
            extra=extra,
            images=images,
            history=history,
        )

