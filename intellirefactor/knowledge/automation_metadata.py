"""
Refactoring Automation Metadata for IntelliRefactor.

This module extracts code transformation rules, dependency injection patterns,
interface extraction templates, and testing strategies to enable intelligent
automation of refactoring efforts for any Python project.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


class TransformationRuleType(Enum):
    """Types of code transformation rules."""

    EXTRACT_METHOD_TO_CLASS = "extract_method_to_class"
    EXTRACT_INTERFACE = "extract_interface"
    INTRODUCE_DEPENDENCY_INJECTION = "introduce_dependency_injection"
    SPLIT_CONFIGURATION = "split_configuration"
    CREATE_FACADE = "create_facade"
    EXTRACT_SERVICE_LAYER = "extract_service_layer"
    CENTRALIZE_ERROR_HANDLING = "centralize_error_handling"
    ABSTRACT_CACHING = "abstract_caching"


class DIPatternType(Enum):
    """Types of dependency injection patterns."""

    CONSTRUCTOR_INJECTION = "constructor_injection"
    SETTER_INJECTION = "setter_injection"
    INTERFACE_INJECTION = "interface_injection"
    SERVICE_LOCATOR = "service_locator"
    FACTORY_PATTERN = "factory_pattern"


class TestingStrategyType(Enum):
    """Types of testing strategies for refactored code."""

    UNIT_TESTING = "unit_testing"
    PROPERTY_BASED_TESTING = "property_based_testing"
    INTEGRATION_TESTING = "integration_testing"
    CONTRACT_TESTING = "contract_testing"
    PERFORMANCE_TESTING = "performance_testing"


@dataclass
class CodeMetrics:
    """Code metrics for refactoring decision making."""

    file_size: int
    cyclomatic_complexity: float
    number_of_responsibilities: int
    coupling_level: float
    cohesion_level: float
    test_coverage: float
    number_of_dependencies: int = 0
    lines_per_method: float = 0.0
    number_of_methods: int = 0


@dataclass
class RefactoringContext:
    """Context information for refactoring decisions."""

    project_type: str = "generic"
    has_existing_clients: bool = False
    backward_compatibility_required: bool = True
    testing_infrastructure: str = "basic"  # "none", "basic", "good", "excellent"
    performance_critical: bool = False
    team_size: str = "small"  # "small", "medium", "large"
    maintenance_priority: str = "medium"  # "low", "medium", "high"


@dataclass
class CodeTransformationRule:
    """A reusable code transformation rule extracted from refactoring."""

    rule_id: str
    name: str
    type: TransformationRuleType
    description: str

    # Pattern matching
    source_pattern: str  # Regex or AST pattern to match
    target_pattern: str  # Template for replacement
    preconditions: List[str]  # Conditions that must be met
    postconditions: List[str]  # Conditions that should be true after

    # Context requirements
    required_imports: List[str]
    required_interfaces: List[str]
    dependency_requirements: List[str]

    # Transformation details
    transformation_steps: List[str]
    code_examples: Dict[str, str]  # before -> after examples

    # Quality metrics
    complexity_impact: float  # Change in cyclomatic complexity
    coupling_impact: float  # Change in coupling
    cohesion_impact: float  # Change in cohesion

    # Automation metadata
    automation_confidence: float  # 0.0 to 1.0
    manual_review_required: bool
    risk_level: str  # "low", "medium", "high"

    # Usage tracking
    applied_count: int = 0
    success_rate: float = 0.0
    common_failure_modes: List[str] = field(default_factory=list)


@dataclass
class DependencyInjectionPattern:
    """A dependency injection pattern template."""

    pattern_id: str
    name: str
    type: DIPatternType
    description: str

    # Pattern structure
    interface_template: str
    implementation_template: str
    registration_template: str
    usage_template: str

    # Configuration
    container_configuration: Dict[str, Any]
    lifecycle_management: str  # "singleton", "transient", "scoped"

    # Quality attributes
    testability_improvement: float
    coupling_reduction: float
    flexibility_increase: float

    # Implementation guidance
    implementation_steps: List[str]
    best_practices: List[str]
    common_pitfalls: List[str]

    # Automation support
    can_auto_generate: bool
    generation_confidence: float
    manual_verification_points: List[str]


@dataclass
class InterfaceExtractionTemplate:
    """Template for extracting interfaces from concrete classes."""

    template_id: str
    name: str
    description: str

    # Analysis patterns
    method_analysis_rules: List[str]
    dependency_analysis_rules: List[str]
    cohesion_analysis_rules: List[str]

    # Generation templates
    interface_naming_pattern: str
    method_signature_template: str
    documentation_template: str

    # Quality criteria
    interface_quality_metrics: List[str]
    segregation_principles: List[str]

    # Automation metadata
    extraction_confidence: float
    requires_human_review: bool
    validation_steps: List[str]


@dataclass
class TestingStrategyTemplate:
    """Template for testing strategies applied to refactored components."""

    strategy_id: str
    name: str
    type: TestingStrategyType
    description: str

    # Test structure
    test_file_template: str
    test_method_template: str
    setup_template: str
    teardown_template: str

    # Test patterns
    test_patterns: List[str]
    assertion_patterns: List[str]
    mock_patterns: List[str]

    # Quality metrics
    coverage_expectations: float
    test_quality_metrics: List[str]

    # Automation support
    can_auto_generate: bool
    generation_accuracy: float
    human_review_required: bool

    # Property-based testing specifics (with defaults must come last)
    property_templates: List[str] = field(default_factory=list)
    generator_templates: List[str] = field(default_factory=list)
    invariant_patterns: List[str] = field(default_factory=list)


@dataclass
class RefactoringAutomationMetadata:
    """Complete automation metadata for a refactoring project."""

    project_id: str
    project_name: str
    refactoring_date: str

    # Transformation rules
    transformation_rules: List[CodeTransformationRule]

    # Dependency injection patterns
    di_patterns: List[DependencyInjectionPattern]

    # Interface extraction templates
    interface_templates: List[InterfaceExtractionTemplate]

    # Testing strategies
    testing_strategies: List[TestingStrategyTemplate]

    # Project-level metrics
    overall_success_metrics: Dict[str, float]
    automation_potential_score: float
    reusability_score: float

    # Knowledge base integration
    related_patterns: List[str]
    applicable_contexts: List[str]
    contraindications: List[str]


class AutomationMetadata:
    """Generates automation metadata from refactoring projects."""

    def __init__(self, project_config: Optional[Dict[str, Any]] = None):
        """Initialize with optional project configuration."""
        self.project_config = project_config or {}
        self.metadata_cache: Dict[str, RefactoringAutomationMetadata] = {}

    def generate_metadata(
        self, project_id: str, project_name: str
    ) -> RefactoringAutomationMetadata:
        """Generate automation metadata for a project."""

        # Extract transformation rules
        transformation_rules = self._extract_transformation_rules()

        # Extract DI patterns
        di_patterns = self._extract_di_patterns()

        # Extract interface templates
        interface_templates = self._extract_interface_templates()

        # Extract testing strategies
        testing_strategies = self._extract_testing_strategies()

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics()

        metadata = RefactoringAutomationMetadata(
            project_id=project_id,
            project_name=project_name,
            refactoring_date=datetime.now().strftime("%Y-%m-%d"),
            transformation_rules=transformation_rules,
            di_patterns=di_patterns,
            interface_templates=interface_templates,
            testing_strategies=testing_strategies,
            overall_success_metrics=overall_metrics,
            automation_potential_score=self._calculate_automation_score(),
            reusability_score=self._calculate_reusability_score(),
            related_patterns=self._get_related_patterns(),
            applicable_contexts=self._get_applicable_contexts(),
            contraindications=self._get_contraindications(),
        )

        self.metadata_cache[metadata.project_id] = metadata
        return metadata

    def _extract_transformation_rules(self) -> List[CodeTransformationRule]:
        """Extract reusable transformation rules."""
        rules = []

        # Rule 1: Extract Method to Component Class
        extract_method_rule = CodeTransformationRule(
            rule_id="extract_method_to_component",
            name="Extract Method to Component Class",
            type=TransformationRuleType.EXTRACT_METHOD_TO_CLASS,
            description="Extract a method with its related state into a separate component class",
            source_pattern=r"class\s+(\w+):\s*.*?def\s+(\w+)\(self[^)]*\):\s*(.*?)(?=def|\Z)",
            target_pattern="class {ComponentName}({InterfaceName}):\n    def {method_name}(self{params}):\n        {body}",
            preconditions=[
                "Method has clear single responsibility",
                "Method uses subset of class instance variables",
                "Method has low coupling with other methods",
                "Method can be tested independently",
            ],
            postconditions=[
                "New component class created with single responsibility",
                "Interface extracted for the component",
                "Original class updated to use component via DI",
                "Tests updated to test component independently",
            ],
            required_imports=[
                "from abc import ABC, abstractmethod",
                "from typing import Protocol",
            ],
            required_interfaces=[
                "Component interface definition",
                "Dependency injection container registration",
            ],
            dependency_requirements=[
                "Dependency injection container",
                "Interface definitions",
                "Component lifecycle management",
            ],
            transformation_steps=[
                "1. Identify method and its dependencies",
                "2. Create interface for the component",
                "3. Extract method to new component class",
                "4. Update original class to use component via DI",
                "5. Register component in DI container",
                "6. Update tests",
            ],
            code_examples={
                "before": """
class ServiceClass:
    def process_data(self, data):
        # Complex data processing logic
        result = []
        # ... 200 lines of code
        return result
""",
                "after": """
class IDataProcessor(Protocol):
    def process_data(self, data: List[Any]) -> List[Any]:
        pass

class DataProcessor(IDataProcessor):
    def process_data(self, data: List[Any]) -> List[Any]:
        # Complex data processing logic
        result = []
        # ... 200 lines of code
        return result

class ServiceClass:
    def __init__(self, data_processor: IDataProcessor):
        self.data_processor = data_processor
        
    def process_data(self, data):
        return self.data_processor.process_data(data)
""",
            },
            complexity_impact=-0.7,
            coupling_impact=-0.5,
            cohesion_impact=0.8,
            automation_confidence=0.8,
            manual_review_required=True,
            risk_level="medium",
            applied_count=0,
            success_rate=1.0,
            common_failure_modes=[
                "Incorrect dependency identification",
                "Missing interface methods",
                "Circular dependency creation",
            ],
        )
        rules.append(extract_method_rule)

        # Rule 2: Create Facade for Backward Compatibility
        facade_rule = CodeTransformationRule(
            rule_id="create_backward_compatible_facade",
            name="Create Backward Compatible Facade",
            type=TransformationRuleType.CREATE_FACADE,
            description="Create facade to maintain API compatibility while using refactored components",
            source_pattern=r"class\s+(\w+):\s*(.*?)(?=class|\Z)",
            target_pattern="class {ClassName}:\n    def __init__(self, {dependencies}):\n        {initialization}\n    \n    {facade_methods}",
            preconditions=[
                "Major refactoring completed",
                "Existing clients must continue working",
                "Internal architecture significantly changed",
                "Configuration format may have changed",
            ],
            postconditions=[
                "Facade maintains identical public API",
                "All existing tests pass without modification",
                "Internal components properly orchestrated",
                "Configuration conversion handled transparently",
            ],
            required_imports=["from typing import Any, Dict, Optional"],
            required_interfaces=[
                "All internal service interfaces",
                "Configuration conversion utilities",
            ],
            dependency_requirements=[
                "Dependency injection container",
                "All refactored internal services",
                "Configuration management system",
            ],
            transformation_steps=[
                "1. Analyze existing public API surface",
                "2. Map old API methods to new service operations",
                "3. Create configuration conversion utilities",
                "4. Implement facade with dependency injection",
                "5. Add backward compatibility validation",
                "6. Performance test facade overhead",
            ],
            code_examples={
                "before": """
class ComplexService:
    def __init__(self, config):
        # Monolithic initialization
        self.config = config
        # ... complex internal state
        
    def execute_operation(self, params):
        # Monolithic implementation
        # ... 500 lines of code
        return result
""",
                "after": """
class ComplexService:
    def __init__(self, config: Optional[Dict] = None):
        # Convert old config format to new
        service_config = self._convert_config(config)
        
        # Initialize with DI container
        self.container = DIContainer.create_default(service_config)
        self.operation_service = self.container.get(IOperationService)
        self.validation_service = self.container.get(IValidationService)
        
    def execute_operation(self, params: Dict):
        # Facade orchestrates internal services
        return self.operation_service.execute(params)
        
    def _convert_config(self, old_config):
        # Convert old config format to new structure
        return ServiceConfig.from_legacy(old_config)
""",
            },
            complexity_impact=0.1,
            coupling_impact=0.2,
            cohesion_impact=0.7,
            automation_confidence=0.7,
            manual_review_required=True,
            risk_level="medium",
            applied_count=0,
            success_rate=1.0,
            common_failure_modes=[
                "Incomplete API mapping",
                "Configuration conversion errors",
                "Performance overhead issues",
            ],
        )
        rules.append(facade_rule)

        # Rule 3: Split Monolithic Configuration
        config_split_rule = CodeTransformationRule(
            rule_id="split_monolithic_configuration",
            name="Split Monolithic Configuration",
            type=TransformationRuleType.SPLIT_CONFIGURATION,
            description="Split large configuration class into domain-specific configuration classes",
            source_pattern=r"class\s+(\w+Config):\s*(.*?)(?=class|\Z)",
            target_pattern="{domain_configs}\n\nclass {MainConfig}:\n    {domain_config_fields}",
            preconditions=[
                "Configuration class has multiple concerns",
                "Configuration is difficult to understand",
                "Different parts used by different components",
                "Validation is complex and mixed",
            ],
            postconditions=[
                "Each domain has its own configuration class",
                "Main configuration composes domain configs",
                "Validation is separated by domain",
                "Configuration is easier to understand and maintain",
            ],
            required_imports=[
                "from dataclasses import dataclass",
                "from typing import Optional",
            ],
            required_interfaces=["Configuration validation interfaces"],
            dependency_requirements=["Configuration validation framework"],
            transformation_steps=[
                "1. Identify distinct configuration domains",
                "2. Create separate config classes for each domain",
                "3. Move related fields to appropriate domain configs",
                "4. Create main config that composes domain configs",
                "5. Update validation logic",
                "6. Update all usage sites",
            ],
            code_examples={
                "before": """
@dataclass
class AppConfig:
    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # API settings
    api_timeout: float = 30.0
    api_retries: int = 3
""",
                "after": """
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    timeout: float = 30.0

@dataclass
class CacheConfig:
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000

@dataclass
class APIConfig:
    timeout: float = 30.0
    retries: int = 3
    rate_limit: int = 100

@dataclass
class AppConfig:
    database: DatabaseConfig
    cache: CacheConfig
    api: APIConfig
""",
            },
            complexity_impact=-0.3,
            coupling_impact=-0.4,
            cohesion_impact=0.6,
            automation_confidence=0.9,
            manual_review_required=False,
            risk_level="low",
            applied_count=0,
            success_rate=1.0,
            common_failure_modes=[
                "Incorrect domain boundary identification",
                "Missing field dependencies",
                "Validation logic errors",
            ],
        )
        rules.append(config_split_rule)

        return rules

    def _extract_di_patterns(self) -> List[DependencyInjectionPattern]:
        """Extract dependency injection patterns."""
        patterns = []

        # Constructor Injection Pattern
        constructor_injection = DependencyInjectionPattern(
            pattern_id="constructor_injection_pattern",
            name="Constructor Injection with Interfaces",
            type=DIPatternType.CONSTRUCTOR_INJECTION,
            description="Inject dependencies through constructor using interface types",
            interface_template='''
from typing import Protocol

class I{ServiceName}(Protocol):
    def {method_name}(self, {parameters}) -> {return_type}:
        """Interface method documentation."""
        pass
''',
            implementation_template='''
class {ServiceName}(I{ServiceName}):
    def __init__(self, {dependencies}):
        {dependency_assignments}
        
    def {method_name}(self, {parameters}) -> {return_type}:
        """Implementation of interface method."""
        {implementation}
''',
            registration_template="""
# In DI container configuration
container.register_singleton(I{ServiceName}, {ServiceName})
container.register_transient(I{DependencyName}, {DependencyName})
""",
            usage_template="""
class {ClientClass}:
    def __init__(self, {service_name}: I{ServiceName}):
        self.{service_name} = {service_name}
        
    def {client_method}(self):
        return self.{service_name}.{method_name}({arguments})
""",
            container_configuration={
                "singleton_services": [
                    "IDataService",
                    "IConfigService",
                    "ILoggingService",
                ],
                "transient_components": ["IProcessor", "IValidator", "ITransformer"],
                "factory_services": ["ICacheManager", "IMetricsCollector"],
            },
            lifecycle_management="mixed",
            testability_improvement=0.8,
            coupling_reduction=0.7,
            flexibility_increase=0.9,
            implementation_steps=[
                "1. Define interface with Protocol or ABC",
                "2. Implement concrete class with interface",
                "3. Update constructor to accept interface type",
                "4. Register in DI container with appropriate lifecycle",
                "5. Update all creation sites to use container",
                "6. Create test utilities for easy mocking",
            ],
            best_practices=[
                "Use Protocol for structural typing when possible",
                "Keep interfaces focused and cohesive",
                "Avoid circular dependencies",
                "Use factory pattern for complex object creation",
                "Register interfaces, not concrete types",
                "Use appropriate lifecycle management",
            ],
            common_pitfalls=[
                "Creating interfaces that are too broad",
                "Circular dependency issues",
                "Incorrect lifecycle management",
                "Missing interface registrations",
                "Over-engineering simple dependencies",
            ],
            can_auto_generate=True,
            generation_confidence=0.85,
            manual_verification_points=[
                "Interface method signatures are correct",
                "Lifecycle management is appropriate",
                "No circular dependencies created",
                "All registrations are present",
            ],
        )
        patterns.append(constructor_injection)

        return patterns

    def _extract_interface_templates(self) -> List[InterfaceExtractionTemplate]:
        """Extract interface extraction templates."""
        templates = []

        # Service Interface Template
        service_interface_template = InterfaceExtractionTemplate(
            template_id="service_interface_extraction",
            name="Service Interface Extraction",
            description="Extract interface from service class with multiple public methods",
            method_analysis_rules=[
                "Include all public methods",
                "Exclude private and protected methods",
                "Include methods with clear business purpose",
                "Exclude infrastructure/utility methods unless they're part of the contract",
            ],
            dependency_analysis_rules=[
                "Analyze method parameters for interface types",
                "Identify return types that should be interfaces",
                "Check for dependencies that should be injected",
                "Validate no concrete type leakage in interface",
            ],
            cohesion_analysis_rules=[
                "Methods should serve related purposes",
                "Interface should have single responsibility",
                "Methods should operate at same abstraction level",
                "Avoid mixing different concerns in same interface",
            ],
            interface_naming_pattern="I{ServiceName}",
            method_signature_template='''
def {method_name}(self, {parameters}) -> {return_type}:
    """
    {method_description}
    
    Args:
        {parameter_docs}
        
    Returns:
        {return_description}
        
    Raises:
        {exception_docs}
    """
    pass
''',
            documentation_template='''
class I{ServiceName}(Protocol):
    """
    Interface for {service_description}.
    
    This interface defines the contract for {service_purpose}.
    Implementations should {implementation_guidance}.
    """
''',
            interface_quality_metrics=[
                "Interface Segregation Principle compliance",
                "Method cohesion score",
                "Dependency direction correctness",
                "Documentation completeness",
            ],
            segregation_principles=[
                "No client should depend on methods it doesn't use",
                "Prefer multiple specific interfaces over one general interface",
                "Group related methods that change together",
                "Separate read and write operations when appropriate",
            ],
            extraction_confidence=0.8,
            requires_human_review=True,
            validation_steps=[
                "Verify all public methods are included",
                "Check method signatures are correct",
                "Validate documentation is complete",
                "Ensure no concrete type leakage",
                "Confirm interface segregation principles",
            ],
        )
        templates.append(service_interface_template)

        return templates

    def _extract_testing_strategies(self) -> List[TestingStrategyTemplate]:
        """Extract testing strategies for refactored components."""
        strategies = []

        # Property-Based Testing Strategy
        pbt_strategy = TestingStrategyTemplate(
            strategy_id="component_property_testing",
            name="Component Property-Based Testing",
            type=TestingStrategyType.PROPERTY_BASED_TESTING,
            description="Property-based testing strategy for refactored components",
            test_file_template='''
import pytest
from hypothesis import given, strategies as st
from {module_path} import {ComponentClass}, I{ComponentClass}

class Test{ComponentClass}Properties:
    """Property-based tests for {ComponentClass}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        {setup_code}
        
    {test_methods}
''',
            test_method_template='''
@given({generators})
def test_{property_name}(self, {parameters}):
    """
    **Property {property_number}: {property_description}**
    **Validates: Requirements {requirements}**
    """
    # Arrange
    {arrange_code}
    
    # Act
    {act_code}
    
    # Assert
    {assert_code}
''',
            setup_template="""
self.{component_name} = {ComponentClass}({dependencies})
self.mock_{dependency} = Mock(spec=I{DependencyClass})
""",
            teardown_template="""
# Clean up any resources
{cleanup_code}
""",
            test_patterns=[
                "Round-trip properties for serialization/deserialization",
                "Idempotence properties",
                "Metamorphic properties",
                "Error condition properties",
            ],
            assertion_patterns=[
                "assert result is not None",
                "assert len(result) > 0",
                "assert all(condition for item in result)",
                "assert result.property == expected_value",
                "assert_that(result, has_properties(expected_properties))",
            ],
            mock_patterns=[
                "mock_{dependency}.{method}.return_value = {value}",
                "mock_{dependency}.{method}.side_effect = {exception}",
                "assert mock_{dependency}.{method}.called_once_with({args})",
            ],
            property_templates=[
                "For any {input_type}, {operation} should {expected_behavior}",
                "For all valid {input_type}, {operation} then {inverse_operation} should return original",
                "For any {input_type}, applying {operation} twice should equal applying it once",
                "For any {input_type}, {operation} should preserve {invariant}",
            ],
            generator_templates=[
                "st.text(min_size=1, max_size=100)",
                "st.integers(min_value=1, max_value=1000)",
                "st.lists(st.text(), min_size=0, max_size=10)",
                "st.dictionaries(st.text(), st.text(), min_size=0, max_size=5)",
            ],
            invariant_patterns=[
                "len(result) >= 0",
                "result.start_time <= result.end_time",
                "all(item.is_valid() for item in result)",
                "result.total == sum(item.value for item in result.items)",
            ],
            coverage_expectations=0.9,
            test_quality_metrics=[
                "Property coverage completeness",
                "Generator quality and diversity",
                "Assertion strength and specificity",
                "Error condition coverage",
            ],
            can_auto_generate=True,
            generation_accuracy=0.75,
            human_review_required=True,
        )
        strategies.append(pbt_strategy)

        # Unit Testing Strategy
        unit_testing_strategy = TestingStrategyTemplate(
            strategy_id="component_unit_testing",
            name="Component Unit Testing",
            type=TestingStrategyType.UNIT_TESTING,
            description="Unit testing strategy for individual refactored components",
            test_file_template='''
import pytest
from unittest.mock import Mock, patch
from {module_path} import {ComponentClass}

class Test{ComponentClass}:
    """Unit tests for {ComponentClass}."""
    
    def setup_method(self):
        """Set up test fixtures."""
        {setup_code}
        
    {test_methods}
''',
            test_method_template='''
def test_{method_name}_{scenario}(self):
    """Test {method_name} when {scenario}."""
    # Arrange
    {arrange_code}
    
    # Act
    {act_code}
    
    # Assert
    {assert_code}
''',
            setup_template="""
self.mock_{dependency} = Mock(spec=I{DependencyClass})
self.{component_name} = {ComponentClass}(self.mock_{dependency})
""",
            teardown_template="# No teardown needed for unit tests",
            test_patterns=[
                "Happy path scenarios",
                "Edge cases and boundary conditions",
                "Error conditions and exception handling",
                "Dependency interaction verification",
                "State change validation",
            ],
            assertion_patterns=[
                "assert result == expected",
                "assert result is not None",
                "assert len(result) == expected_count",
                "assert mock.called_once_with(expected_args)",
                "with pytest.raises(ExpectedException):",
            ],
            mock_patterns=[
                "Mock(spec=IInterface)",
                "mock.method.return_value = value",
                "mock.method.side_effect = exception",
                "assert mock.method.call_count == expected",
            ],
            coverage_expectations=0.85,
            test_quality_metrics=[
                "Branch coverage percentage",
                "Edge case coverage",
                "Error condition coverage",
                "Mock usage appropriateness",
            ],
            can_auto_generate=True,
            generation_accuracy=0.8,
            human_review_required=False,
        )
        strategies.append(unit_testing_strategy)

        return strategies

    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall success metrics."""
        base_metrics = {
            "complexity_reduction": 0.5,
            "coupling_reduction": 0.4,
            "cohesion_improvement": 0.6,
            "testability_improvement": 0.5,
            "maintainability_improvement": 0.5,
            "test_coverage_improvement": 0.3,
            "file_count_increase": 2.0,
            "lines_per_file_reduction": 0.4,
            "interface_extraction_success": 0.8,
            "backward_compatibility_maintained": 1.0,
            "performance_impact": 0.05,
            "automation_rule_extraction": 0.7,
        }

        # Apply project-specific adjustments
        if self.project_config.get("expected_metrics"):
            base_metrics.update(self.project_config["expected_metrics"])

        return base_metrics

    def _calculate_automation_score(self) -> float:
        """Calculate automation potential score."""
        base_score = 0.7

        # Adjust based on project configuration
        if self.project_config.get("has_tests", False):
            base_score += 0.1
        if self.project_config.get("has_interfaces", False):
            base_score += 0.1
        if self.project_config.get("follows_patterns", False):
            base_score += 0.1

        return min(base_score, 1.0)

    def _calculate_reusability_score(self) -> float:
        """Calculate reusability score."""
        base_score = 0.8

        # Adjust based on project type
        project_type = self.project_config.get("project_type", "generic")
        if project_type == "library":
            base_score += 0.1
        elif project_type == "framework":
            base_score += 0.15

        return min(base_score, 1.0)

    def _get_related_patterns(self) -> List[str]:
        """Get related design patterns."""
        return [
            "monolith_decomposition",
            "facade_pattern",
            "dependency_injection",
            "layered_architecture",
            "strategy_pattern",
            "factory_pattern",
        ]

    def _get_applicable_contexts(self) -> List[str]:
        """Get applicable contexts."""
        return [
            "large_monolithic_classes",
            "god_objects",
            "tightly_coupled_systems",
            "legacy_modernization",
            "microservice_extraction",
            "api_modernization",
        ]

    def _get_contraindications(self) -> List[str]:
        """Get contraindications."""
        return [
            "small_simple_classes",
            "performance_critical_tight_loops",
            "stable_apis_without_clients",
            "prototype_code",
            "single_use_scripts",
        ]

    def export_metadata(
        self, metadata: RefactoringAutomationMetadata, filepath: str
    ) -> None:
        """Export automation metadata to JSON file."""
        export_data = asdict(metadata)

        # Convert enums to strings for JSON serialization
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif hasattr(obj, "value"):  # Enum
                return obj.value
            else:
                return obj

        export_data = convert_enums(export_data)
        export_data["export_timestamp"] = datetime.now().isoformat()

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported automation metadata to {filepath}")

    def get_automation_recommendations(
        self, target_metrics: CodeMetrics, target_context: RefactoringContext
    ) -> List[str]:
        """Get automation recommendations based on target code metrics."""
        recommendations = []

        # Generate metadata if not cached
        if not self.metadata_cache:
            self.generate_metadata("default", "Default Project")

        # Get first available metadata
        metadata = next(iter(self.metadata_cache.values()))

        # Analyze which transformation rules apply
        for rule in metadata.transformation_rules:
            if self._rule_applies(rule, target_metrics, target_context):
                recommendations.append(f"Apply {rule.name}: {rule.description}")

        # Analyze which DI patterns apply
        for pattern in metadata.di_patterns:
            if self._pattern_applies(pattern, target_metrics, target_context):
                recommendations.append(f"Use {pattern.name}: {pattern.description}")

        return recommendations

    def _rule_applies(
        self,
        rule: CodeTransformationRule,
        metrics: CodeMetrics,
        context: RefactoringContext,
    ) -> bool:
        """Check if a transformation rule applies to the target code."""
        if rule.type == TransformationRuleType.EXTRACT_METHOD_TO_CLASS:
            return (
                metrics.file_size > 500
                and metrics.number_of_responsibilities > 2
                and metrics.cohesion_level < 0.6
            )
        elif rule.type == TransformationRuleType.CREATE_FACADE:
            return (
                context.backward_compatibility_required and context.has_existing_clients
            )
        elif rule.type == TransformationRuleType.SPLIT_CONFIGURATION:
            return metrics.number_of_responsibilities > 3

        return False

    def _pattern_applies(
        self,
        pattern: DependencyInjectionPattern,
        metrics: CodeMetrics,
        context: RefactoringContext,
    ) -> bool:
        """Check if a DI pattern applies to the target code."""
        if pattern.type == DIPatternType.CONSTRUCTOR_INJECTION:
            return (
                metrics.number_of_dependencies > 2
                and context.testing_infrastructure in ["good", "excellent"]
            )
        return False


# Global instance for easy access
_automation_metadata: Optional[AutomationMetadata] = None


def get_automation_metadata() -> AutomationMetadata:
    """Get the global automation metadata instance."""
    global _automation_metadata
    if _automation_metadata is None:
        _automation_metadata = AutomationMetadata()
    return _automation_metadata


def generate_and_export_metadata(
    export_path: str = "automation_metadata.json",
    project_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate and export automation metadata."""
    metadata_generator = AutomationMetadata(project_config)
    metadata = metadata_generator.generate_metadata(
        "intellirefactor", "IntelliRefactor Project"
    )
    metadata_generator.export_metadata(metadata, export_path)

    logger.info("Generated automation metadata with:")
    logger.info(f"  - {len(metadata.transformation_rules)} transformation rules")
    logger.info(f"  - {len(metadata.di_patterns)} dependency injection patterns")
    logger.info(
        f"  - {len(metadata.interface_templates)} interface extraction templates"
    )
    logger.info(f"  - {len(metadata.testing_strategies)} testing strategy templates")
    logger.info(
        f"  - Automation potential score: {metadata.automation_potential_score}"
    )
    logger.info(f"  - Reusability score: {metadata.reusability_score}")
