"""
SatQuery AI — Agentic Cognitive Synthesizer Facade
Modularized architecture: Delegates domain reasoning to specialized sub-engines:
- app.core.orchestrator.synthesis.vqa_reasoner
- app.core.orchestrator.synthesis.conversation_memory
- app.core.orchestrator.synthesis.grounding_reasoner
- app.core.orchestrator.synthesis.change_reasoner
- app.core.orchestrator.synthesis.fusion_reasoner
"""

from app.core.orchestrator.synthesis import (
    AgenticCognitiveSynthesizer,
    ConversationMemoryTracker,
    VQADomainReasoner,
    GroundingDomainReasoner,
    ChangeDomainReasoner,
    FusionDomainReasoner,
)

__all__ = [
    "AgenticCognitiveSynthesizer",
    "ConversationMemoryTracker",
    "VQADomainReasoner",
    "GroundingDomainReasoner",
    "ChangeDomainReasoner",
    "FusionDomainReasoner",
]
