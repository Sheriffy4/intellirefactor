"""
Hook system for IntelliRefactor

Provides extensible hooks for custom analysis and refactoring rules.
Allows plugins to register callbacks for various events and operations.
"""

from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
from functools import wraps

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks supported by the system."""
    # Analysis hooks
    PRE_PROJECT_ANALYSIS = "pre_project_analysis"
    POST_PROJECT_ANALYSIS = "post_project_analysis"
    PRE_FILE_ANALYSIS = "pre_file_analysis"
    POST_FILE_ANALYSIS = "post_file_analysis"
    
    # Refactoring hooks
    PRE_REFACTORING = "pre_refactoring"
    POST_REFACTORING = "post_refactoring"
    PRE_OPPORTUNITY_DETECTION = "pre_opportunity_detection"
    POST_OPPORTUNITY_DETECTION = "post_opportunity_detection"
    
    # Knowledge hooks
    PRE_KNOWLEDGE_EXTRACTION = "pre_knowledge_extraction"
    POST_KNOWLEDGE_EXTRACTION = "post_knowledge_extraction"
    PRE_KNOWLEDGE_QUERY = "pre_knowledge_query"
    POST_KNOWLEDGE_QUERY = "post_knowledge_query"
    
    # Validation hooks
    PRE_VALIDATION = "pre_validation"
    POST_VALIDATION = "post_validation"
    
    # Orchestration hooks
    PRE_WORKFLOW = "pre_workflow"
    POST_WORKFLOW = "post_workflow"
    
    # Custom hooks
    CUSTOM = "custom"


class HookPriority(Enum):
    """Priority levels for hook execution."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


@dataclass
class Hook:
    """Represents a single hook registration."""
    name: str
    hook_type: HookType
    callback: Callable
    priority: HookPriority = HookPriority.NORMAL
    plugin_name: Optional[str] = None
    description: str = ""
    enabled: bool = True
    
    def __post_init__(self):
        # Validate callback signature
        if not callable(self.callback):
            raise ValueError(f"Hook callback must be callable: {self.name}")
        
        # Store callback signature for validation
        self.signature = inspect.signature(self.callback)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the hook callback."""
        if not self.enabled:
            logger.debug(f"Hook {self.name} is disabled, skipping")
            return None
        
        try:
            logger.debug(f"Executing hook: {self.name}")
            return self.callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing hook {self.name}: {e}")
            raise
    
    def validate_args(self, *args, **kwargs) -> bool:
        """Validate arguments against callback signature."""
        try:
            self.signature.bind(*args, **kwargs)
            return True
        except TypeError:
            return False


class HookRegistry:
    """Registry for managing hooks."""
    
    def __init__(self):
        self._hooks: Dict[HookType, List[Hook]] = {
            hook_type: [] for hook_type in HookType
        }
        self._custom_hooks: Dict[str, List[Hook]] = {}
        self._hook_names: Set[str] = set()
    
    def register_hook(self, hook: Hook) -> bool:
        """
        Register a hook.
        
        Args:
            hook: Hook to register
            
        Returns:
            True if registration was successful, False otherwise
        """
        if hook.name in self._hook_names:
            logger.warning(f"Hook with name '{hook.name}' already exists, replacing")
            self.unregister_hook(hook.name)
        
        if hook.hook_type == HookType.CUSTOM:
            # For custom hooks, use the hook name as the key
            custom_key = hook.name
            if custom_key not in self._custom_hooks:
                self._custom_hooks[custom_key] = []
            self._custom_hooks[custom_key].append(hook)
        else:
            self._hooks[hook.hook_type].append(hook)
        
        # Sort hooks by priority
        hook_list = (self._custom_hooks.get(hook.name, []) 
                    if hook.hook_type == HookType.CUSTOM 
                    else self._hooks[hook.hook_type])
        hook_list.sort(key=lambda h: h.priority.value)
        
        self._hook_names.add(hook.name)
        logger.info(f"Registered hook: {hook.name} (type: {hook.hook_type.value}, priority: {hook.priority.value})")
        return True
    
    def unregister_hook(self, hook_name: str) -> bool:
        """
        Unregister a hook by name.
        
        Args:
            hook_name: Name of hook to unregister
            
        Returns:
            True if unregistration was successful, False otherwise
        """
        if hook_name not in self._hook_names:
            return False
        
        # Find and remove the hook
        removed = False
        
        # Check standard hooks
        for hook_type, hooks in self._hooks.items():
            for hook in hooks[:]:  # Create copy to avoid modification during iteration
                if hook.name == hook_name:
                    hooks.remove(hook)
                    removed = True
                    break
        
        # Check custom hooks
        for custom_key, hooks in self._custom_hooks.items():
            for hook in hooks[:]:
                if hook.name == hook_name:
                    hooks.remove(hook)
                    removed = True
                    break
        
        if removed:
            self._hook_names.remove(hook_name)
            logger.info(f"Unregistered hook: {hook_name}")
        
        return removed
    
    def get_hooks(self, hook_type: HookType, custom_key: str = None) -> List[Hook]:
        """
        Get hooks of a specific type.
        
        Args:
            hook_type: Type of hooks to retrieve
            custom_key: For custom hooks, the specific key to retrieve
            
        Returns:
            List of hooks sorted by priority
        """
        if hook_type == HookType.CUSTOM and custom_key:
            return self._custom_hooks.get(custom_key, []).copy()
        else:
            return self._hooks[hook_type].copy()
    
    def list_hooks(self) -> Dict[str, Dict[str, Any]]:
        """List all registered hooks with their metadata."""
        result = {}
        
        # Standard hooks
        for hook_type, hooks in self._hooks.items():
            for hook in hooks:
                result[hook.name] = {
                    "type": hook_type.value,
                    "priority": hook.priority.value,
                    "plugin": hook.plugin_name,
                    "description": hook.description,
                    "enabled": hook.enabled
                }
        
        # Custom hooks
        for custom_key, hooks in self._custom_hooks.items():
            for hook in hooks:
                result[hook.name] = {
                    "type": "custom",
                    "custom_key": custom_key,
                    "priority": hook.priority.value,
                    "plugin": hook.plugin_name,
                    "description": hook.description,
                    "enabled": hook.enabled
                }
        
        return result
    
    def enable_hook(self, hook_name: str) -> bool:
        """Enable a hook by name."""
        return self._set_hook_enabled(hook_name, True)
    
    def disable_hook(self, hook_name: str) -> bool:
        """Disable a hook by name."""
        return self._set_hook_enabled(hook_name, False)
    
    def _set_hook_enabled(self, hook_name: str, enabled: bool) -> bool:
        """Set hook enabled state."""
        if hook_name not in self._hook_names:
            return False
        
        # Find and update the hook
        for hook_type, hooks in self._hooks.items():
            for hook in hooks:
                if hook.name == hook_name:
                    hook.enabled = enabled
                    return True
        
        for custom_key, hooks in self._custom_hooks.items():
            for hook in hooks:
                if hook.name == hook_name:
                    hook.enabled = enabled
                    return True
        
        return False
    
    def clear(self) -> None:
        """Clear all hooks from registry."""
        self._hooks = {hook_type: [] for hook_type in HookType}
        self._custom_hooks.clear()
        self._hook_names.clear()


class HookSystem:
    """Main hook system for managing and executing hooks."""
    
    def __init__(self):
        self.registry = HookRegistry()
        self._execution_context: Dict[str, Any] = {}
    
    def register_hook(self, hook_type: HookType, callback: Callable, 
                     name: str = None, priority: HookPriority = HookPriority.NORMAL,
                     plugin_name: str = None, description: str = "",
                     custom_key: str = None) -> str:
        """
        Register a hook callback.
        
        Args:
            hook_type: Type of hook
            callback: Callback function
            name: Hook name (auto-generated if not provided)
            priority: Hook priority
            plugin_name: Name of plugin registering the hook
            description: Hook description
            custom_key: For custom hooks, the specific key
            
        Returns:
            Name of the registered hook
        """
        if name is None:
            name = f"{plugin_name or 'unknown'}_{callback.__name__}_{len(self.registry._hook_names)}"
        
        # For custom hooks, use custom_key as the hook name if provided
        if hook_type == HookType.CUSTOM and custom_key:
            name = custom_key
        
        hook = Hook(
            name=name,
            hook_type=hook_type,
            callback=callback,
            priority=priority,
            plugin_name=plugin_name,
            description=description
        )
        
        self.registry.register_hook(hook)
        return name
    
    def unregister_hook(self, hook_name: str) -> bool:
        """Unregister a hook by name."""
        return self.registry.unregister_hook(hook_name)
    
    def execute_hooks(self, hook_type: HookType, *args, 
                     custom_key: str = None, 
                     stop_on_error: bool = False,
                     **kwargs) -> List[Any]:
        """
        Execute all hooks of a specific type.
        
        Args:
            hook_type: Type of hooks to execute
            *args: Arguments to pass to hook callbacks
            custom_key: For custom hooks, the specific key
            stop_on_error: Whether to stop execution on first error
            **kwargs: Keyword arguments to pass to hook callbacks
            
        Returns:
            List of results from hook executions
        """
        hooks = self.registry.get_hooks(hook_type, custom_key)
        results = []
        
        logger.debug(f"Executing {len(hooks)} hooks for {hook_type.value}")
        
        for hook in hooks:
            try:
                # Validate arguments
                if not hook.validate_args(*args, **kwargs):
                    logger.warning(f"Invalid arguments for hook {hook.name}, skipping")
                    continue
                
                result = hook.execute(*args, **kwargs)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error executing hook {hook.name}: {e}")
                if stop_on_error:
                    raise
                results.append(None)
        
        return results
    
    def execute_custom_hooks(self, custom_key: str, *args, **kwargs) -> List[Any]:
        """Execute custom hooks by key."""
        return self.execute_hooks(HookType.CUSTOM, *args, custom_key=custom_key, **kwargs)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set execution context value."""
        self._execution_context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get execution context value."""
        return self._execution_context.get(key, default)
    
    def clear_context(self) -> None:
        """Clear execution context."""
        self._execution_context.clear()
    
    def create_hook_decorator(self, hook_type: HookType, 
                            priority: HookPriority = HookPriority.NORMAL,
                            plugin_name: str = None,
                            custom_key: str = None):
        """
        Create a decorator for registering hooks.
        
        Args:
            hook_type: Type of hook
            priority: Hook priority
            plugin_name: Name of plugin
            custom_key: For custom hooks, the specific key
            
        Returns:
            Decorator function
        """
        def decorator(func):
            hook_name = self.register_hook(
                hook_type=hook_type,
                callback=func,
                name=func.__name__,
                priority=priority,
                plugin_name=plugin_name,
                description=func.__doc__ or "",
                custom_key=custom_key
            )
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            wrapper._hook_name = hook_name
            return wrapper
        
        return decorator
    
    def get_hook_status(self) -> Dict[str, Any]:
        """Get status of the hook system."""
        hooks_by_type = {}
        for hook_type in HookType:
            hooks = self.registry.get_hooks(hook_type)
            hooks_by_type[hook_type.value] = len(hooks)
        
        return {
            "total_hooks": len(self.registry._hook_names),
            "hooks_by_type": hooks_by_type,
            "custom_hooks": len(self.registry._custom_hooks),
            "context_keys": list(self._execution_context.keys())
        }
    
    def cleanup(self) -> None:
        """Clean up the hook system."""
        logger.info("Cleaning up hook system")
        self.registry.clear()
        self.clear_context()


# Convenience decorators for common hook types
def analysis_hook(priority: HookPriority = HookPriority.NORMAL, plugin_name: str = None):
    """Decorator for analysis hooks."""
    def decorator(func):
        # This would be set up by the plugin system when loaded
        return func
    return decorator


def refactoring_hook(priority: HookPriority = HookPriority.NORMAL, plugin_name: str = None):
    """Decorator for refactoring hooks."""
    def decorator(func):
        # This would be set up by the plugin system when loaded
        return func
    return decorator


def knowledge_hook(priority: HookPriority = HookPriority.NORMAL, plugin_name: str = None):
    """Decorator for knowledge hooks."""
    def decorator(func):
        # This would be set up by the plugin system when loaded
        return func
    return decorator


def custom_hook(hook_key: str, priority: HookPriority = HookPriority.NORMAL, plugin_name: str = None):
    """Decorator for custom hooks."""
    def decorator(func):
        # This would be set up by the plugin system when loaded
        return func
    return decorator