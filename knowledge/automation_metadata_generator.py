#!/usr/bin/env python3
"""
Automation Metadata Generator for IntelliRefactor.

This module generates automation metadata for refactoring operations,
making it configurable for different project types and contexts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .automation_metadata import AutomationMetadata


class AutomationMetadataGenerator:
    """Generates automation metadata for refactoring operations."""
    
    def __init__(self, project_config: Optional[Dict[str, Any]] = None):
        """Initialize the generator with optional project configuration."""
        self.project_config = project_config or {}
        self.logger = logging.getLogger(__name__)
    
    def generate_metadata(self, project_id: str, project_name: str) -> Dict[str, Any]:
        """Generate comprehensive automation metadata for a project."""
        
        metadata = {
            "project_id": project_id,
            "project_name": project_name,
            "refactoring_date": datetime.now().strftime("%Y-%m-%d"),
            "generation_timestamp": datetime.now().isoformat(),
            
            "transformation_rules": self._generate_transformation_rules(),
            "dependency_injection_patterns": self._generate_di_patterns(),
            "interface_extraction_templates": self._generate_interface_templates(),
            "testing_strategies": self._generate_testing_strategies(),
            "overall_success_metrics": self._generate_success_metrics(),
            
            "automation_potential_score": self._calculate_automation_score(),
            "reusability_score": self._calculate_reusability_score(),
            
            "related_patterns": self._get_related_patterns(),
            "applicable_contexts": self._get_applicable_contexts(),
            "contraindications": self._get_contraindications(),
            
            "code_transformation_examples": self._generate_transformation_examples(),
            "decision_criteria": self._generate_decision_criteria(),
            "automation_recommendations": self._generate_automation_recommendations()
        }
        
        return metadata
    
    def _generate_transformation_rules(self) -> List[Dict[str, Any]]:
        """Generate transformation rules based on project configuration."""
        base_rules = [
            {
                "rule_id": "extract_method_to_component",
                "name": "Extract Method to Component Class",
                "type": "extract_method_to_class",
                "description": "Extract a method with its related state into a separate component class",
                "source_pattern": r"class\s+(\w+):\s*.*?def\s+(\w+)\(self[^)]*\):\s*(.*?)(?=def|\Z)",
                "target_pattern": "class {ComponentName}({InterfaceName}):\n    def {method_name}(self{params}):\n        {body}",
                "preconditions": [
                    "Method has clear single responsibility",
                    "Method uses subset of class instance variables",
                    "Method has low coupling with other methods",
                    "Method can be tested independently"
                ],
                "postconditions": [
                    "New component class created with single responsibility",
                    "Interface extracted for the component",
                    "Original class updated to use component via DI",
                    "Tests updated to test component independently"
                ],
                "complexity_impact": -0.7,
                "coupling_impact": -0.5,
                "cohesion_impact": 0.8,
                "automation_confidence": 0.8,
                "manual_review_required": True,
                "risk_level": "medium",
                "applied_count": 0,
                "success_rate": 1.0
            },
            {
                "rule_id": "create_backward_compatible_facade",
                "name": "Create Backward Compatible Facade",
                "type": "create_facade",
                "description": "Create facade to maintain API compatibility while using refactored components",
                "source_pattern": r"class\s+(\w+):\s*(.*?)(?=class|\Z)",
                "target_pattern": "class {ClassName}:\n    def __init__(self, {dependencies}):\n        {initialization}\n    \n    {facade_methods}",
                "preconditions": [
                    "Major refactoring completed",
                    "Existing clients must continue working",
                    "Internal architecture significantly changed",
                    "Configuration format may have changed"
                ],
                "postconditions": [
                    "Facade maintains identical public API",
                    "All existing tests pass without modification",
                    "Internal components properly orchestrated",
                    "Configuration conversion handled transparently"
                ],
                "complexity_impact": 0.1,
                "coupling_impact": 0.2,
                "cohesion_impact": 0.7,
                "automation_confidence": 0.7,
                "manual_review_required": True,
                "risk_level": "medium",
                "applied_count": 0,
                "success_rate": 1.0
            },
            {
                "rule_id": "split_monolithic_configuration",
                "name": "Split Monolithic Configuration",
                "type": "split_configuration",
                "description": "Split large configuration class into domain-specific configuration classes",
                "source_pattern": r"class\s+(\w+Config):\s*(.*?)(?=class|\Z)",
                "target_pattern": "{domain_configs}\n\nclass {MainConfig}:\n    {domain_config_fields}",
                "preconditions": [
                    "Configuration class has multiple concerns",
                    "Configuration is difficult to understand",
                    "Different parts used by different components",
                    "Validation is complex and mixed"
                ],
                "postconditions": [
                    "Each domain has its own configuration class",
                    "Main configuration composes domain configs",
                    "Validation is separated by domain",
                    "Configuration is easier to understand and maintain"
                ],
                "complexity_impact": -0.3,
                "coupling_impact": -0.4,
                "cohesion_impact": 0.6,
                "automation_confidence": 0.9,
                "manual_review_required": False,
                "risk_level": "low",
                "applied_count": 0,
                "success_rate": 1.0
            }
        ]
        
        # Apply project-specific customizations
        if self.project_config.get("custom_rules"):
            base_rules.extend(self.project_config["custom_rules"])
        
        return base_rules
    
    def _generate_di_patterns(self) -> List[Dict[str, Any]]:
        """Generate dependency injection patterns."""
        return [
            {
                "pattern_id": "constructor_injection_pattern",
                "name": "Constructor Injection with Interfaces",
                "type": "constructor_injection",
                "description": "Inject dependencies through constructor using interface types",
                "interface_template": "from typing import Protocol\n\nclass I{ServiceName}(Protocol):\n    def {method_name}(self, {parameters}) -> {return_type}:\n        pass",
                "implementation_template": "class {ServiceName}(I{ServiceName}):\n    def __init__(self, {dependencies}):\n        {dependency_assignments}",
                "registration_template": "container.register_singleton(I{ServiceName}, {ServiceName})",
                "usage_template": "class {ClientClass}:\n    def __init__(self, {service_name}: I{ServiceName}):\n        self.{service_name} = {service_name}",
                "testability_improvement": 0.8,
                "coupling_reduction": 0.7,
                "flexibility_increase": 0.9,
                "can_auto_generate": True,
                "generation_confidence": 0.85
            }
        ]
    
    def _generate_interface_templates(self) -> List[Dict[str, Any]]:
        """Generate interface extraction templates."""
        return [
            {
                "template_id": "service_interface_extraction",
                "name": "Service Interface Extraction",
                "description": "Extract interface from service class with multiple public methods",
                "interface_naming_pattern": "I{ServiceName}",
                "method_signature_template": "def {method_name}(self, {parameters}) -> {return_type}:\n    '''Method documentation'''\n    pass",
                "extraction_confidence": 0.8,
                "requires_human_review": True
            }
        ]
    
    def _generate_testing_strategies(self) -> List[Dict[str, Any]]:
        """Generate testing strategies."""
        return [
            {
                "strategy_id": "component_property_testing",
                "name": "Component Property-Based Testing",
                "type": "property_based_testing",
                "description": "Property-based testing strategy for refactored components",
                "test_file_template": "import pytest\nfrom hypothesis import given, strategies as st\n\nclass Test{ComponentClass}Properties:\n    {test_methods}",
                "test_method_template": "@given({generators})\ndef test_{property_name}(self, {parameters}):\n    '''Property test'''\n    {test_body}",
                "coverage_expectations": 0.9,
                "can_auto_generate": True,
                "generation_accuracy": 0.75,
                "human_review_required": True
            },
            {
                "strategy_id": "component_unit_testing",
                "name": "Component Unit Testing",
                "type": "unit_testing",
                "description": "Unit testing strategy for individual refactored components",
                "test_file_template": "import pytest\nfrom unittest.mock import Mock\n\nclass Test{ComponentClass}:\n    {test_methods}",
                "test_method_template": "def test_{method_name}_{scenario}(self):\n    '''Unit test'''\n    {test_body}",
                "coverage_expectations": 0.85,
                "can_auto_generate": True,
                "generation_accuracy": 0.8,
                "human_review_required": False
            }
        ]
    
    def _generate_success_metrics(self) -> Dict[str, float]:
        """Generate success metrics based on project configuration."""
        default_metrics = {
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
            "automation_rule_extraction": 0.7
        }
        
        # Apply project-specific metric overrides
        if self.project_config.get("expected_metrics"):
            default_metrics.update(self.project_config["expected_metrics"])
        
        return default_metrics
    
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
            "factory_pattern"
        ]
    
    def _get_applicable_contexts(self) -> List[str]:
        """Get applicable contexts for the refactoring patterns."""
        return [
            "large_monolithic_classes",
            "god_objects",
            "tightly_coupled_systems",
            "legacy_modernization",
            "microservice_extraction",
            "api_modernization"
        ]
    
    def _get_contraindications(self) -> List[str]:
        """Get contraindications for applying these patterns."""
        return [
            "small_simple_classes",
            "performance_critical_tight_loops",
            "stable_apis_without_clients",
            "prototype_code",
            "single_use_scripts"
        ]
    
    def _generate_transformation_examples(self) -> Dict[str, str]:
        """Generate code transformation examples."""
        return {
            "extract_component_before": '''
class MonolithicService:
    def process_data(self, data):
        # Complex data processing logic
        processed = []
        # ... 200 lines of code
        return processed
''',
            "extract_component_after": '''
class IDataProcessor(Protocol):
    def process_data(self, data: List[Any]) -> List[Any]:
        pass

class DataProcessor(IDataProcessor):
    def process_data(self, data: List[Any]) -> List[Any]:
        # Complex data processing logic
        processed = []
        # ... 200 lines of code
        return processed

class MonolithicService:
    def __init__(self, data_processor: IDataProcessor):
        self.data_processor = data_processor
        
    def process_data(self, data):
        return self.data_processor.process_data(data)
''',
            "facade_pattern_before": '''
class ComplexService:
    def __init__(self, config):
        # Monolithic initialization
        self.config = config
        # ... complex internal state
        
    def execute_operation(self, params):
        # Monolithic implementation
        # ... 500 lines of code
        return result
''',
            "facade_pattern_after": '''
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
'''
        }
    
    def _generate_decision_criteria(self) -> Dict[str, Any]:
        """Generate decision criteria for refactoring patterns."""
        return {
            "extract_component": {
                "file_size_threshold": self.project_config.get("file_size_threshold", 500),
                "complexity_threshold": self.project_config.get("complexity_threshold", 10),
                "responsibility_count_threshold": self.project_config.get("responsibility_threshold", 3),
                "coupling_indicators": [
                    "Multiple import statements for unrelated functionality",
                    "Methods accessing unrelated instance variables",
                    "Conditional logic based on component availability"
                ],
                "trigger_conditions": [
                    "Class violates Single Responsibility Principle",
                    "Difficulty in unit testing specific functionality",
                    "Need to mock only part of the class functionality"
                ]
            },
            "apply_facade": {
                "preconditions": [
                    "Major refactoring completed",
                    "Existing clients must continue working",
                    "Internal interfaces different from original"
                ]
            },
            "dependency_injection": {
                "preconditions": [
                    "Multiple components with dependencies",
                    "Testing requires extensive mocking",
                    "Configuration management is complex"
                ]
            }
        }
    
    def _generate_automation_recommendations(self) -> Dict[str, List[str]]:
        """Generate automation recommendations."""
        return {
            "high_confidence_automatable": [
                "Configuration splitting",
                "Interface extraction from concrete classes",
                "Basic dependency injection setup",
                "Unit test template generation"
            ],
            "medium_confidence_automatable": [
                "Component extraction from monolithic classes",
                "Facade pattern implementation",
                "Property-based test generation"
            ],
            "requires_human_oversight": [
                "Complex component boundary decisions",
                "API compatibility validation",
                "Performance impact assessment",
                "Business logic correctness verification"
            ]
        }
    
    def export_metadata(self, metadata: Dict[str, Any], export_path: str) -> None:
        """Export metadata to a JSON file."""
        export_file = Path(export_path)
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Automation metadata exported to: {export_path}")


def generate_simple_metadata(project_id: str = "generic-project", 
                            project_name: str = "Generic Project",
                            project_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate simple automation metadata without complex dependencies."""
    generator = AutomationMetadataGenerator(project_config)
    return generator.generate_metadata(project_id, project_name)


def main():
    """Generate and export automation metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Generating refactoring automation metadata...")
    
    try:
        # Generate metadata with default configuration
        generator = AutomationMetadataGenerator()
        metadata = generator.generate_metadata("intellirefactor-project", "IntelliRefactor Project")
        
        # Export to JSON
        export_path = "data/refactoring_automation_metadata.json"
        generator.export_metadata(metadata, export_path)
        
        logger.info(f"Generated automation metadata with:")
        logger.info(f"  - {len(metadata['transformation_rules'])} transformation rules")
        logger.info(f"  - {len(metadata['dependency_injection_patterns'])} dependency injection patterns")
        logger.info(f"  - {len(metadata['interface_extraction_templates'])} interface extraction templates")
        logger.info(f"  - {len(metadata['testing_strategies'])} testing strategy templates")
        logger.info(f"  - Automation potential score: {metadata['automation_potential_score']}")
        logger.info(f"  - Reusability score: {metadata['reusability_score']}")
        
        # Display key insights
        logger.info("\nKey Automation Insights:")
        logger.info("High confidence automatable patterns:")
        for pattern in metadata['automation_recommendations']['high_confidence_automatable']:
            logger.info(f"  ✓ {pattern}")
        
        logger.info("\n✅ Refactoring automation metadata generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating automation metadata: {e}")
        raise


if __name__ == "__main__":
    main()