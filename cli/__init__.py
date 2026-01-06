"""
CLI module for IntelliRefactor.

Provides rich terminal output and integration utilities for the command-line interface.
"""

from .rich_output import RichOutputManager, get_rich_output, set_rich_enabled
from .integration import CLIIntegrationManager

__all__ = [
    'RichOutputManager',
    'get_rich_output', 
    'set_rich_enabled',
    'CLIIntegrationManager'
]