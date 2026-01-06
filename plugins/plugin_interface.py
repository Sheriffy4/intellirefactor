"""
Plugin interface definitions for IntelliRefactor

Defines the base interfaces and types that plugins must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by IntelliRefactor."""
    ANALYSIS = "analysis"
    REFACTORING = "refactoring"
    KNOWLEDGE = "knowledge"
    ORCHESTRATION = "orchestration"
    VALIDATION = "validation"
    REPORTING = "reporting"


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""
    pass


class PluginExecutionError(PluginError):
    """Raised when a plugin fails during execution."""
    pass


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = None
    config_schema: Dict[str, Any] = None
    min_intellirefactor_version: str = "0.1.0"
    max_intellirefactor_version: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config_schema is None:
            self.config_schema = {}


class PluginInterface(ABC):
    """Base interface that all plugins must implement."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
        self._initialized = False
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Default implementation - override in subclasses for custom validation
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get plugin status information.
        
        Returns:
            Dictionary containing plugin status
        """
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "type": self.metadata.plugin_type.value,
            "initialized": self._initialized,
            "config": self.config
        }


class AnalysisPlugin(PluginInterface):
    """Base class for analysis plugins."""
    
    @abstractmethod
    def analyze_project(self, project_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a project.
        
        Args:
            project_path: Path to the project to analyze
            context: Analysis context and existing results
            
        Returns:
            Analysis results dictionary
        """
        pass
    
    @abstractmethod
    def analyze_file(self, file_path: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single file.
        
        Args:
            file_path: Path to the file being analyzed
            content: File content
            context: Analysis context
            
        Returns:
            File analysis results dictionary
        """
        pass
    
    def get_supported_file_types(self) -> List[str]:
        """
        Get list of file extensions this plugin can analyze.
        
        Returns:
            List of file extensions (e.g., ['.py', '.pyx'])
        """
        return ['.py']  # Default to Python files
    
    def get_analysis_metrics(self) -> List[str]:
        """
        Get list of metrics this plugin can calculate.
        
        Returns:
            List of metric names
        """
        return []


class RefactoringPlugin(PluginInterface):
    """Base class for refactoring plugins."""
    
    @abstractmethod
    def identify_opportunities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify refactoring opportunities.
        
        Args:
            analysis_results: Results from project/file analysis
            
        Returns:
            List of refactoring opportunities
        """
        pass
    
    @abstractmethod
    def apply_refactoring(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a refactoring transformation.
        
        Args:
            opportunity: Refactoring opportunity to apply
            context: Refactoring context
            
        Returns:
            Refactoring result dictionary
        """
        pass
    
    def validate_refactoring(self, result: Dict[str, Any]) -> bool:
        """
        Validate that a refactoring was applied correctly.
        
        Args:
            result: Refactoring result to validate
            
        Returns:
            True if refactoring is valid, False otherwise
        """
        return True  # Default implementation
    
    def get_supported_refactoring_types(self) -> List[str]:
        """
        Get list of refactoring types this plugin supports.
        
        Returns:
            List of refactoring type names
        """
        return []


class KnowledgePlugin(PluginInterface):
    """Base class for knowledge management plugins."""
    
    @abstractmethod
    def extract_knowledge(self, analysis_results: Dict[str, Any], 
                         refactoring_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract knowledge from analysis and refactoring results.
        
        Args:
            analysis_results: Project/file analysis results
            refactoring_results: Applied refactoring results
            
        Returns:
            List of knowledge items
        """
        pass
    
    @abstractmethod
    def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.
        
        Args:
            query: Knowledge query parameters
            
        Returns:
            List of matching knowledge items
        """
        pass
    
    def update_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> bool:
        """
        Update knowledge base with new items.
        
        Args:
            knowledge_items: Knowledge items to add/update
            
        Returns:
            True if update was successful, False otherwise
        """
        return True  # Default implementation
    
    def get_knowledge_types(self) -> List[str]:
        """
        Get list of knowledge types this plugin handles.
        
        Returns:
            List of knowledge type names
        """
        return []


class ValidationPlugin(PluginInterface):
    """Base class for validation plugins."""
    
    @abstractmethod
    def validate_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results.
        
        Args:
            analysis_results: Analysis results to validate
            
        Returns:
            Validation results dictionary
        """
        pass
    
    @abstractmethod
    def validate_refactoring(self, refactoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate refactoring results.
        
        Args:
            refactoring_result: Refactoring result to validate
            
        Returns:
            Validation results dictionary
        """
        pass


class ReportingPlugin(PluginInterface):
    """Base class for reporting plugins."""
    
    @abstractmethod
    def generate_report(self, data: Dict[str, Any], report_type: str) -> str:
        """
        Generate a report from data.
        
        Args:
            data: Data to include in the report
            report_type: Type of report to generate
            
        Returns:
            Generated report as string
        """
        pass
    
    def get_supported_report_types(self) -> List[str]:
        """
        Get list of report types this plugin can generate.
        
        Returns:
            List of report type names
        """
        return []
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of output formats this plugin supports.
        
        Returns:
            List of format names (e.g., ['html', 'json', 'markdown'])
        """
        return ['text']


class OrchestrationPlugin(PluginInterface):
    """Base class for orchestration plugins."""
    
    @abstractmethod
    def orchestrate_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate a refactoring workflow.
        
        Args:
            workflow_config: Workflow configuration
            
        Returns:
            Workflow execution results
        """
        pass
    
    def get_supported_workflows(self) -> List[str]:
        """
        Get list of workflow types this plugin supports.
        
        Returns:
            List of workflow type names
        """
        return []