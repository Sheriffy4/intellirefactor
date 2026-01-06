"""
Plugin system for IntelliRefactor

Provides plugin discovery, loading, and management capabilities.
Supports extensible analysis and refactoring functionality.
"""

from .plugin_manager import PluginManager
from .plugin_interface import (
    PluginInterface,
    AnalysisPlugin,
    RefactoringPlugin,
    KnowledgePlugin,
    PluginType,
    PluginMetadata,
    PluginError,
    PluginLoadError,
    PluginExecutionError
)
from .hook_system import HookSystem, Hook, HookRegistry

__all__ = [
    'PluginManager',
    'PluginInterface',
    'AnalysisPlugin', 
    'RefactoringPlugin',
    'KnowledgePlugin',
    'PluginType',
    'PluginMetadata',
    'PluginError',
    'PluginLoadError',
    'PluginExecutionError',
    'HookSystem',
    'Hook',
    'HookRegistry'
]