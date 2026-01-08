"""
Generic interfaces for IntelliRefactor components.

This module defines Protocol interfaces that ensure compatibility with different
Python project structures and provide a consistent API across all components.
"""

import ast
from typing import Protocol, Dict, List, Optional, Any, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GenericProjectStructure:
    """Generic representation of a project structure."""

    root_path: str
    source_directories: List[str]
    test_directories: List[str]
    config_files: List[str]
    documentation_files: List[str]
    build_files: List[str]
    total_files: int
    total_lines: int


@dataclass
class GenericAnalysisResult:
    """Generic analysis result that works with any project type."""

    success: bool
    project_path: str
    analysis_type: str
    data: Dict[str, Any]
    metrics: Dict[str, float]
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class GenericRefactoringOpportunity:
    """Generic refactoring opportunity representation."""

    id: str
    type: str
    priority: int
    description: str
    target_files: List[str]
    estimated_impact: Dict[str, float]
    prerequisites: List[str]
    automation_confidence: float
    risk_level: str


class ProjectAnalyzerProtocol(Protocol):
    """Protocol for project analyzers that work with any Python project."""

    def analyze_project(self, project_path: Union[str, Path]) -> GenericAnalysisResult:
        """Analyze a project and return generic analysis results."""
        ...

    def get_project_structure(self, project_path: Union[str, Path]) -> GenericProjectStructure:
        """Get the structure of a project in a generic format."""
        ...

    def identify_source_files(self, project_path: Union[str, Path]) -> List[str]:
        """Identify source files in the project."""
        ...

    def calculate_project_metrics(self, project_path: Union[str, Path]) -> Dict[str, float]:
        """Calculate project-level metrics."""
        ...


class FileAnalyzerProtocol(Protocol):
    """Protocol for file analyzers that work with any Python file."""

    def analyze_file(
        self, file_path: Union[str, Path], external_context: Optional[Any] = None
    ) -> GenericAnalysisResult:
        """Analyze a single file and return generic results."""
        ...

    def calculate_file_metrics(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Calculate file-level metrics."""
        ...

    def identify_issues(self, file_path: Union[str, Path]) -> List[str]:
        """Identify issues in a file."""
        ...

    def suggest_improvements(self, file_path: Union[str, Path]) -> List[str]:
        """Suggest improvements for a file."""


...


class RefactoringSystemProtocol(Protocol):
    """Protocol for refactoring systems that work with any project."""

    def identify_opportunities(
        self, analysis_data: Dict[str, Any]
    ) -> List[GenericRefactoringOpportunity]:
        """Identify refactoring opportunities from analysis data."""
        ...

    def assess_opportunity_quality(self, opportunity: GenericRefactoringOpportunity) -> float:
        """Assess the quality/value of a refactoring opportunity."""
        ...

    def generate_refactoring_plan(
        self, opportunities: List[GenericRefactoringOpportunity]
    ) -> Dict[str, Any]:
        """Generate a refactoring plan from opportunities."""
        ...


class KnowledgeManagerProtocol(Protocol):
    """Protocol for knowledge managers that work with any domain."""

    def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        ...

    def add_knowledge(self, knowledge_item: Dict[str, Any]) -> bool:
        """Add a knowledge item to the knowledge base."""
        ...

    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing knowledge item."""
        ...

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete a knowledge item."""
        ...

    def learn_from_result(self, result: Dict[str, Any]) -> None:
        """Learn from a refactoring result."""
        ...


class ConfigurationProtocol(Protocol):
    """Protocol for configuration objects that work with any project type."""

    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from a file."""
        ...

    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        ...

    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        ...

    def merge(self, other_config: Dict[str, Any]) -> None:
        """Merge with another configuration."""
        ...


class PluginProtocol(Protocol):
    """Protocol for plugins that extend IntelliRefactor functionality."""

    def get_name(self) -> str:
        """Get the plugin name."""
        ...

    def get_version(self) -> str:
        """Get the plugin version."""
        ...

    def get_description(self) -> str:
        """Get the plugin description."""
        ...

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        ...

    def get_supported_operations(self) -> List[str]:
        """Get list of operations this plugin supports."""
        ...


class AnalysisPluginProtocol(PluginProtocol, Protocol):
    """Protocol for analysis plugins."""

    def analyze(self, target: Union[str, Path], context: Dict[str, Any]) -> GenericAnalysisResult:
        """Perform analysis on the target."""
        ...

    def get_analysis_types(self) -> List[str]:
        """Get types of analysis this plugin can perform."""
        ...


class RefactoringPluginProtocol(PluginProtocol, Protocol):
    """Protocol for refactoring plugins."""

    def identify_opportunities(
        self, analysis_data: Dict[str, Any]
    ) -> List[GenericRefactoringOpportunity]:
        """Identify refactoring opportunities."""
        ...

    def apply_refactoring(
        self, opportunity: GenericRefactoringOpportunity, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Apply a refactoring opportunity."""
        ...

    def get_refactoring_types(self) -> List[str]:
        """Get types of refactoring this plugin can perform."""
        ...


class KnowledgePluginProtocol(PluginProtocol, Protocol):
    """Protocol for knowledge plugins."""

    def contribute_knowledge(self) -> List[Dict[str, Any]]:
        """Contribute knowledge items to the knowledge base."""
        ...

    def process_learning_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process learning data and return knowledge insights."""
        ...

    def get_knowledge_domains(self) -> List[str]:
        """Get knowledge domains this plugin covers."""
        ...


# Abstract base classes for concrete implementations


class BaseProjectAnalyzer(ABC):
    """Base class for project analyzers with generic interface implementation."""

    @abstractmethod
    def analyze_project(self, project_path: Union[str, Path]) -> GenericAnalysisResult:
        """Analyze a project and return generic analysis results."""
        pass

    def get_project_structure(self, project_path: Union[str, Path]) -> GenericProjectStructure:
        """Default implementation for getting project structure."""
        project_path = Path(project_path)

        # Generic Python project structure detection
        source_dirs = []
        test_dirs = []
        config_files = []
        doc_files = []
        build_files = []

        for item in project_path.rglob("*"):
            if item.is_file():
                if item.suffix == ".py":
                    if "test" in item.name.lower() or "test" in str(item.parent).lower():
                        if str(item.parent) not in test_dirs:
                            test_dirs.append(str(item.parent))
                    else:
                        if str(item.parent) not in source_dirs:
                            source_dirs.append(str(item.parent))
                elif item.name in [
                    "setup.py",
                    "pyproject.toml",
                    "requirements.txt",
                    "Pipfile",
                ]:
                    config_files.append(str(item))
                elif item.suffix in [".md", ".rst", ".txt"] and "doc" in item.name.lower():
                    doc_files.append(str(item))
                elif item.name in ["Makefile", "tox.ini", ".github"]:
                    build_files.append(str(item))

        # Count total files and lines
        total_files = len(list(project_path.rglob("*.py")))
        total_lines = 0

        for py_file in project_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    total_lines += len(f.readlines())
            except (UnicodeDecodeError, PermissionError):
                continue

        return GenericProjectStructure(
            root_path=str(project_path),
            source_directories=source_dirs,
            test_directories=test_dirs,
            config_files=config_files,
            documentation_files=doc_files,
            build_files=build_files,
            total_files=total_files,
            total_lines=total_lines,
        )

    def identify_source_files(self, project_path: Union[str, Path]) -> List[str]:
        """Default implementation for identifying source files."""
        project_path = Path(project_path)
        source_files = []

        for py_file in project_path.rglob("*.py"):
            # Skip common non-source files but be more selective about test exclusion
            file_str = str(py_file).lower()
            skip_patterns = ["__pycache__", ".git", "build", "dist"]

            # Only skip if it's actually a test file within the project being analyzed
            relative_path = py_file.relative_to(project_path)
            is_test_file = (
                py_file.name.startswith("test_")
                or py_file.name.endswith("_test.py")
                or "tests" in relative_path.parts[:-1]  # tests in relative path, not absolute
            )

            should_skip = any(skip in file_str for skip in skip_patterns) or is_test_file

            if not should_skip:
                source_files.append(str(py_file))

        return source_files


class BaseFileAnalyzer(ABC):
    """Base class for file analyzers with generic interface implementation."""

    @abstractmethod
    def analyze_file(
        self, file_path: Union[str, Path], external_context: Optional[Any] = None
    ) -> GenericAnalysisResult:
        """Analyze a single file and return generic results."""
        pass

    def calculate_file_metrics(self, file_path: Union[str, Path]) -> Dict[str, float]:
        """Default implementation for calculating file metrics."""
        file_path = Path(file_path)

        if not file_path.exists() or file_path.suffix != ".py":
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic metrics
            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]

            # Parse AST for more detailed metrics
            try:
                tree = ast.parse(content)
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                imports = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]

                return {
                    "total_lines": len(lines),
                    "non_empty_lines": len(non_empty_lines),
                    "classes_count": len(classes),
                    "functions_count": len(functions),
                    "imports_count": len(imports),
                    "complexity_estimate": len(functions) + len(classes) * 2,  # Simple estimate
                }
            except SyntaxError:
                return {
                    "total_lines": len(lines),
                    "non_empty_lines": len(non_empty_lines),
                    "parse_error": True,
                }

        except (UnicodeDecodeError, PermissionError):
            return {"error": "Cannot read file"}


class BaseRefactoringSystem(ABC):
    """Base class for refactoring systems with generic interface implementation."""

    @abstractmethod
    def identify_opportunities(
        self, analysis_data: Dict[str, Any]
    ) -> List[GenericRefactoringOpportunity]:
        """Identify refactoring opportunities from analysis data."""
        pass

    def assess_opportunity_quality(self, opportunity: GenericRefactoringOpportunity) -> float:
        """Default implementation for assessing opportunity quality."""
        # Simple quality assessment based on priority and automation confidence
        priority_weight = opportunity.priority / 10.0  # Normalize priority
        confidence_weight = opportunity.automation_confidence

        # Risk adjustment
        risk_adjustment = 1.0
        if opportunity.risk_level == "high":
            risk_adjustment = 0.7
        elif opportunity.risk_level == "medium":
            risk_adjustment = 0.85

        return (priority_weight * 0.4 + confidence_weight * 0.6) * risk_adjustment


class BaseKnowledgeManager(ABC):
    """Base class for knowledge managers with generic interface implementation."""

    @abstractmethod
    def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        pass

    @abstractmethod
    def add_knowledge(self, knowledge_item: Dict[str, Any]) -> bool:
        """Add a knowledge item to the knowledge base."""
        pass

    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """Default implementation for updating knowledge."""
        # This would typically be overridden by concrete implementations
        return False

    def delete_knowledge(self, knowledge_id: str) -> bool:
        """Default implementation for deleting knowledge."""
        # This would typically be overridden by concrete implementations
        return False

    def learn_from_result(self, result: Dict[str, Any]) -> None:
        """Default implementation for learning from results."""
        # This would typically be overridden by concrete implementations
        pass
