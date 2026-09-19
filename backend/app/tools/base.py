"""
SatQuery AI — Base Tool Interface & Registration Decorator
All specialist remote-sensing tools inherit from BaseTool.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Global registry dict populated by @register_tool
_TOOL_REGISTRY: Dict[str, type] = {}


class ToolInput(BaseModel):
    """Standardized input for any specialist tool."""
    images: List[Any] = Field(default_factory=list, description="Preprocessed image tensors or arrays")
    image_metas: List[Any] = Field(default_factory=list, description="GeoTIFF metadata dicts or models")
    query: str = Field("", description="Natural-language query text")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific parameters")
    prior_outputs: Dict[str, Any] = Field(default_factory=dict, description="Outputs from prior tools in execution graph")


class ToolOutput(BaseModel):
    """Standardized output from any specialist tool."""
    tool_id: str
    text_response: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    bounding_boxes: Optional[List[List[float]]] = None
    mask: Optional[Any] = Field(None, description="Binary mask array (not serialized)")
    mask_url: Optional[str] = None
    geojson: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list, description="IDs of tools this execution depended on")


class BaseTool(ABC):
    """
    Abstract base class for all SatQuery AI specialist tools.
    Each tool must implement execute() and validate_parameters().
    """

    tool_id: str = "base_tool"
    tool_name: str = "Base Tool"
    description: str = "Abstract base tool"
    supported_tasks: List[str] = []
    accepts_prior_output: bool = False

    @abstractmethod
    def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Run inference and return structured output."""
        ...

    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate tool-specific parameters before execution."""
        ...

    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameter values for this tool."""
        return {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} tool_id={self.tool_id}>"


def register_tool(cls: type) -> type:
    """
    Decorator that registers a tool class in the global registry.
    Usage:
        @register_tool
        class MyTool(BaseTool):
            tool_id = "my_tool"
            ...
    """
    if hasattr(cls, "tool_id"):
        tool_id = cls.tool_id
        _TOOL_REGISTRY[tool_id] = cls
        logger.info(f"Registered tool: {tool_id} ({cls.__name__})")
    else:
        logger.warning(f"Tool class {cls.__name__} has no tool_id, skipping registration")
    return cls


def get_registered_tools() -> Dict[str, type]:
    """Returns the global tool registry."""
    return dict(_TOOL_REGISTRY)
