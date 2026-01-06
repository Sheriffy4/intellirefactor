"""
Plugin manager for IntelliRefactor

Handles plugin discovery, loading, registration, and lifecycle management.
"""

import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Set
import logging
import traceback
import json

from .plugin_interface import (
    PluginInterface, PluginType, PluginMetadata, 
    PluginError, PluginLoadError, PluginExecutionError,
    AnalysisPlugin, RefactoringPlugin, KnowledgePlugin,
    ValidationPlugin, ReportingPlugin, OrchestrationPlugin
)
from ..config import PluginConfig

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing loaded plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugins_by_type: Dict[PluginType, List[PluginInterface]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
    
    def register_plugin(self, plugin: PluginInterface) -> None:
        """Register a plugin instance."""
        metadata = plugin.metadata
        plugin_name = metadata.name
        
        if plugin_name in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' is already registered, replacing")
        
        self._plugins[plugin_name] = plugin
        self._plugin_metadata[plugin_name] = metadata
        
        # Add to type-specific registry
        plugin_type = metadata.plugin_type
        if plugin in self._plugins_by_type[plugin_type]:
            self._plugins_by_type[plugin_type].remove(plugin)
        self._plugins_by_type[plugin_type].append(plugin)
        
        logger.info(f"Registered plugin: {plugin_name} (type: {plugin_type.value})")
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin by name."""
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        metadata = self._plugin_metadata[plugin_name]
        
        # Remove from registries
        del self._plugins[plugin_name]
        del self._plugin_metadata[plugin_name]
        self._plugins_by_type[metadata.plugin_type].remove(plugin)
        
        # Cleanup plugin
        try:
            plugin.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up plugin '{plugin_name}': {e}")
        
        logger.info(f"Unregistered plugin: {plugin_name}")
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type."""
        return self._plugins_by_type[plugin_type].copy()
    
    def get_all_plugins(self) -> Dict[str, PluginInterface]:
        """Get all registered plugins."""
        return self._plugins.copy()
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        return self._plugin_metadata.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    def clear(self) -> None:
        """Clear all plugins from registry."""
        for plugin_name in list(self._plugins.keys()):
            self.unregister_plugin(plugin_name)


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.registry = PluginRegistry()
        self._discovered_plugins: Dict[str, str] = {}  # name -> path
        self._failed_plugins: Dict[str, str] = {}  # name -> error message
    
    def discover_plugins(self, directories: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Discover plugins in specified directories.
        
        Args:
            directories: List of directories to search. If None, uses config directories.
            
        Returns:
            Dictionary mapping plugin names to their file paths
        """
        if directories is None:
            directories = self.config.plugin_directories
        
        discovered = {}
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.debug(f"Plugin directory does not exist: {directory}")
                continue
            
            logger.info(f"Discovering plugins in: {directory}")
            
            # Search for Python files
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and not file.startswith('_'):
                        file_path = os.path.join(root, file)
                        plugin_name = os.path.splitext(file)[0]
                        
                        # Check if this looks like a plugin file
                        if self._is_plugin_file(file_path):
                            discovered[plugin_name] = file_path
                            logger.debug(f"Discovered potential plugin: {plugin_name} at {file_path}")
        
        self._discovered_plugins.update(discovered)
        logger.info(f"Discovered {len(discovered)} potential plugins")
        return discovered
    
    def _is_plugin_file(self, file_path: str) -> bool:
        """Check if a file appears to be a plugin."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple heuristic: look for plugin-related imports or classes
                return (
                    'PluginInterface' in content or
                    'AnalysisPlugin' in content or
                    'RefactoringPlugin' in content or
                    'KnowledgePlugin' in content or
                    'ValidationPlugin' in content or
                    'ReportingPlugin' in content or
                    'OrchestrationPlugin' in content
                )
        except Exception:
            return False
    
    def load_plugin(self, plugin_name: str, plugin_path: str) -> Optional[PluginInterface]:
        """
        Load a single plugin from file.
        
        Args:
            plugin_name: Name of the plugin
            plugin_path: Path to the plugin file
            
        Returns:
            Loaded plugin instance or None if loading failed
        """
        try:
            logger.info(f"Loading plugin: {plugin_name} from {plugin_path}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Could not create module spec for {plugin_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and
                    not inspect.isabstract(obj)):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                raise PluginLoadError(f"No plugin classes found in {plugin_path}")
            
            if len(plugin_classes) > 1:
                logger.warning(f"Multiple plugin classes found in {plugin_path}, using first one")
            
            # Instantiate the plugin
            plugin_class = plugin_classes[0]
            plugin_config = self._get_plugin_config(plugin_name)
            plugin_instance = plugin_class(plugin_config)
            
            # Validate plugin
            if not self._validate_plugin(plugin_instance):
                raise PluginLoadError(f"Plugin validation failed for {plugin_name}")
            
            # Initialize plugin
            if not plugin_instance.initialize():
                raise PluginLoadError(f"Plugin initialization failed for {plugin_name}")
            
            plugin_instance._initialized = True
            logger.info(f"Successfully loaded plugin: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            error_msg = f"Failed to load plugin '{plugin_name}': {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            self._failed_plugins[plugin_name] = str(e)
            return None
    
    def _get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get configuration for a specific plugin."""
        # This could be extended to support per-plugin configuration
        return {}
    
    def _validate_plugin(self, plugin: PluginInterface) -> bool:
        """Validate a plugin instance."""
        try:
            # Check that plugin has required metadata
            metadata = plugin.metadata
            if not isinstance(metadata, PluginMetadata):
                logger.error(f"Plugin {plugin.__class__.__name__} has invalid metadata")
                return False
            
            # Check required fields
            if not metadata.name or not metadata.version:
                logger.error(f"Plugin {plugin.__class__.__name__} missing required metadata fields")
                return False
            
            # Validate configuration if provided
            if not plugin.validate_config(plugin.config):
                logger.error(f"Plugin {plugin.__class__.__name__} has invalid configuration")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating plugin {plugin.__class__.__name__}: {e}")
            return False
    
    def load_all_plugins(self, directories: Optional[List[str]] = None) -> Dict[str, PluginInterface]:
        """
        Discover and load all plugins.
        
        Args:
            directories: Directories to search for plugins
            
        Returns:
            Dictionary of successfully loaded plugins
        """
        # Discover plugins if auto-discovery is enabled
        if self.config.auto_discover:
            self.discover_plugins(directories)
        
        loaded_plugins = {}
        
        # Load discovered plugins
        for plugin_name, plugin_path in self._discovered_plugins.items():
            # Check if plugin is explicitly enabled (if enabled_plugins is specified)
            if (self.config.enabled_plugins and 
                plugin_name not in self.config.enabled_plugins):
                logger.debug(f"Skipping plugin {plugin_name} (not in enabled list)")
                continue
            
            plugin = self.load_plugin(plugin_name, plugin_path)
            if plugin:
                self.registry.register_plugin(plugin)
                loaded_plugins[plugin_name] = plugin
        
        logger.info(f"Loaded {len(loaded_plugins)} plugins successfully")
        if self._failed_plugins:
            logger.warning(f"Failed to load {len(self._failed_plugins)} plugins")
        
        return loaded_plugins
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reload was successful, False otherwise
        """
        if plugin_name not in self._discovered_plugins:
            logger.error(f"Cannot reload unknown plugin: {plugin_name}")
            return False
        
        # Unregister existing plugin
        self.registry.unregister_plugin(plugin_name)
        
        # Remove from sys.modules to force reload
        if plugin_name in sys.modules:
            del sys.modules[plugin_name]
        
        # Load plugin again
        plugin_path = self._discovered_plugins[plugin_name]
        plugin = self.load_plugin(plugin_name, plugin_path)
        
        if plugin:
            self.registry.register_plugin(plugin)
            logger.info(f"Successfully reloaded plugin: {plugin_name}")
            return True
        else:
            logger.error(f"Failed to reload plugin: {plugin_name}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if unload was successful, False otherwise
        """
        return self.registry.unregister_plugin(plugin_name)
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names."""
        return list(self.registry.get_all_plugins().keys())
    
    def get_plugin_status(self) -> Dict[str, Any]:
        """Get status of all plugins."""
        return {
            "loaded_plugins": len(self.registry.list_plugins()),
            "failed_plugins": len(self._failed_plugins),
            "discovered_plugins": len(self._discovered_plugins),
            "plugins": {
                name: plugin.get_status() 
                for name, plugin in self.registry.get_all_plugins().items()
            },
            "failed": self._failed_plugins.copy()
        }
    
    def execute_analysis_plugins(self, project_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all analysis plugins on a project."""
        results = {}
        analysis_plugins = self.registry.get_plugins_by_type(PluginType.ANALYSIS)
        
        for plugin in analysis_plugins:
            try:
                plugin_name = plugin.metadata.name
                logger.debug(f"Executing analysis plugin: {plugin_name}")
                
                result = plugin.analyze_project(project_path, context)
                results[plugin_name] = result
                
            except Exception as e:
                logger.error(f"Error executing analysis plugin {plugin.metadata.name}: {e}")
                results[plugin.metadata.name] = {"error": str(e)}
        
        return results
    
    def execute_refactoring_plugins(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all refactoring plugins."""
        results = {}
        refactoring_plugins = self.registry.get_plugins_by_type(PluginType.REFACTORING)
        
        for plugin in refactoring_plugins:
            try:
                plugin_name = plugin.metadata.name
                logger.debug(f"Executing refactoring plugin: {plugin_name}")
                
                opportunities = plugin.identify_opportunities(analysis_results)
                results[plugin_name] = {"opportunities": opportunities}
                
            except Exception as e:
                logger.error(f"Error executing refactoring plugin {plugin.metadata.name}: {e}")
                results[plugin.metadata.name] = {"error": str(e)}
        
        return results
    
    def cleanup(self) -> None:
        """Clean up all plugins and resources."""
        logger.info("Cleaning up plugin manager")
        self.registry.clear()
        self._discovered_plugins.clear()
        self._failed_plugins.clear()