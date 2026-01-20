"""
Wrapper module to prevent slow imports of responsibility_clusterer.

This module provides lazy loading of the responsibility_clusterer to avoid
slow import times during general package usage.
"""

from typing import TYPE_CHECKING

# Only import the actual module when type checking or when explicitly requested
if TYPE_CHECKING:
    from .responsibility_clusterer_impl import (
        ResponsibilityClusterer,
        ClusteringConfig,
        ClusteringAlgorithm,
        ClusterQuality,
        MethodInfo,
        ResponsibilityCluster,
        ClusteringResult,
        ExtractionComplexity,
        ComponentInterface,
        ExtractionPlan,
    )
else:
    # Lazy loading mechanism
    _responsibility_clusterer_module = None
    
    def _get_responsibility_clusterer_module():
        global _responsibility_clusterer_module
        if _responsibility_clusterer_module is None:
            # Import the actual module only when needed
            from . import responsibility_clusterer_impl
            _responsibility_clusterer_module = responsibility_clusterer_impl
        return _responsibility_clusterer_module
    
    def __getattr__(name):
        """Lazy load attributes from the actual module."""
        module = _get_responsibility_clusterer_module()
        return getattr(module, name)
    
    def __dir__():
        """Return available attributes."""
        module = _get_responsibility_clusterer_module()
        return dir(module)

# Export the same names for type checking compatibility
__all__ = [
    "ResponsibilityClusterer",
    "ClusteringConfig", 
    "ClusteringAlgorithm",
    "ClusterQuality",
    "MethodInfo",
    "ResponsibilityCluster",
    "ClusteringResult",
    "ExtractionComplexity",
    "ComponentInterface",
    "ExtractionPlan",
]