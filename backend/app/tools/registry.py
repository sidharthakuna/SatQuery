"""
SatQuery AI — Dynamic Tool Registry
Auto-discovers and manages all registered specialist tools.
"""

import importlib
import logging
import pkgutil
from typing import Dict, List, Optional

from app.tools.base import BaseTool, ToolInput, ToolOutput, get_registered_tools

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry that discovers, instantiates, and manages specialist tools.
    Tools register themselves via the @register_tool decorator on import.
    """

    def __init__(self):
        self._instances: Dict[str, BaseTool] = {}
        self._auto_discover()

    def _auto_discover(self):
        """
        Import all tool_*.py modules in the tools package to trigger
        @register_tool decorators, then instantiate each tool.
        """
        import app.tools as tools_pkg

        for importer, modname, ispkg in pkgutil.iter_modules(tools_pkg.__path__):
            if modname.startswith("tool_"):
                try:
                    importlib.import_module(f"app.tools.{modname}")
                    logger.debug(f"Discovered tool module: {modname}")
                except Exception as e:
                    logger.error(f"Failed to import tool module {modname}: {e}")

        # Instantiate all registered tools
        for tool_id, tool_cls in get_registered_tools().items():
            try:
                self._instances[tool_id] = tool_cls()
                logger.info(f"Instantiated tool: {tool_id}")
            except Exception as e:
                logger.error(f"Failed to instantiate tool {tool_id}: {e}")

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get an instantiated tool by its ID."""
        return self._instances.get(tool_id)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools with their metadata."""
        return [
            {
                "tool_id": tool.tool_id,
                "tool_name": tool.tool_name,
                "description": tool.description,
                "supported_tasks": tool.supported_tasks,
            }
            for tool in self._instances.values()
        ]

    def execute_tool(self, tool_id: str, tool_input: ToolInput) -> ToolOutput:
        """
        Validate parameters and execute a tool by its ID.
        Raises ValueError if the tool is not found or validation fails.
        """
        tool = self.get_tool(tool_id)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_id}")

        # Validate parameters
        if not tool.validate_parameters(tool_input.parameters):
            raise ValueError(
                f"Parameter validation failed for tool {tool_id}: {tool_input.parameters}"
            )

        return tool.execute(tool_input)

    def get_tool_ids(self) -> List[str]:
        """Return all registered tool IDs."""
        return list(self._instances.keys())
