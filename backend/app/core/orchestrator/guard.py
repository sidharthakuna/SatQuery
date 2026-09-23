"""
SatQuery AI — Input Compatibility Guard (Legacy Alias)
Forwarding shim pointing to InputIntelligenceGate.
"""

from app.core.orchestrator.input_gate import InputIntelligenceGate

# Backward compatibility alias
InputCompatibilityGuard = InputIntelligenceGate
