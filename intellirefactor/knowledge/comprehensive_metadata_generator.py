#!/usr/bin/env python3
"""
Comprehensive Refactoring Automation Metadata Generator.

This module generates comprehensive automation metadata from completed
refactoring operations, including transformation rules, dependency injection
patterns, interface extraction templates, and testing strategies.
"""

import logging
from typing import Dict, Any, List, Optional

from .automation_metadata_generator import AutomationMetadataGenerator


class ComprehensiveMetadataGenerator(AutomationMetadataGenerator):
    """Generates comprehensive automation metadata with advanced features."""

    def __init__(self, project_config: Optional[Dict[str, Any]] = None):
        """Initialize the comprehensive generator."""
        super().__init__(project_config)
        self.decision_orchestrator = None  # Will be initialized if needed

    def generate_comprehensive_metadata(
        self, project_id: str, project_name: str
    ) -> Dict[str, Any]:
        """Generate comprehensive automation metadata with decision tree integration."""

        # Start with base metadata
        metadata = self.generate_metadata(project_id, project_name)

        # Add comprehensive features
        metadata.update(
            {
                "advanced_patterns": self._generate_advanced_patterns(),
                "decision_trees": self._generate_decision_trees(),
                "learning_insights": self._generate_learning_insights(),
                "quality_metrics": self._generate_quality_metrics(),
                "automation_workflows": self._generate_automation_workflows(),
                "integration_templates": self._generate_integration_templates(),
            }
        )

        return metadata

    def _generate_advanced_patterns(self) -> List[Dict[str, Any]]:
        """Generate advanced refactoring patterns."""
        return [
            {
                "pattern_id": "microservice_extraction",
                "name": "Microservice Extraction Pattern",
                "type": "architectural_refactoring",
                "description": "Extract cohesive functionality into separate microservice",
                "complexity_level": "high",
                "automation_confidence": 0.6,
                "prerequisites": [
                    "Clear service boundaries identified",
                    "Data access patterns analyzed",
                    "Communication protocols defined",
                ],
                "outcomes": [
                    "Independent deployable service",
                    "Reduced coupling between services",
                    "Improved scalability and maintainability",
                ],
            },
            {
                "pattern_id": "event_driven_refactoring",
                "name": "Event-Driven Architecture Refactoring",
                "type": "architectural_refactoring",
                "description": "Refactor synchronous calls to event-driven communication",
                "complexity_level": "high",
                "automation_confidence": 0.5,
                "prerequisites": [
                    "Event boundaries identified",
                    "Message schemas defined",
                    "Event store infrastructure available",
                ],
                "outcomes": [
                    "Loose coupling between components",
                    "Better scalability and resilience",
                    "Improved system observability",
                ],
            },
        ]

    def _generate_decision_trees(self) -> Dict[str, Any]:
        """Generate decision tree structures for refactoring decisions."""
        return {
            "component_extraction_tree": {
                "root": {
                    "condition": "file_size > 500",
                    "true_branch": {
                        "condition": "cyclomatic_complexity > 10",
                        "true_branch": {
                            "condition": "number_of_responsibilities > 2",
                            "true_branch": {
                                "action": "extract_component",
                                "confidence": 0.9,
                            },
                            "false_branch": {
                                "action": "refactor_methods",
                                "confidence": 0.7,
                            },
                        },
                        "false_branch": {"action": "no_action", "confidence": 0.8},
                    },
                    "false_branch": {"action": "no_action", "confidence": 0.9},
                }
            },
            "interface_extraction_tree": {
                "root": {
                    "condition": "has_multiple_implementations",
                    "true_branch": {
                        "condition": "stable_public_api",
                        "true_branch": {
                            "action": "extract_interface",
                            "confidence": 0.95,
                        },
                        "false_branch": {
                            "action": "defer_extraction",
                            "confidence": 0.6,
                        },
                    },
                    "false_branch": {
                        "condition": "testing_requires_mocking",
                        "true_branch": {
                            "action": "extract_interface",
                            "confidence": 0.8,
                        },
                        "false_branch": {"action": "no_action", "confidence": 0.9},
                    },
                }
            },
        }

    def _generate_learning_insights(self) -> Dict[str, Any]:
        """Generate insights learned from refactoring operations."""
        return {
            "successful_patterns": [
                {
                    "pattern": "extract_method_to_component",
                    "success_rate": 0.85,
                    "common_benefits": [
                        "Improved testability",
                        "Better separation of concerns",
                        "Easier maintenance",
                    ],
                    "common_pitfalls": [
                        "Over-extraction leading to complexity",
                        "Incorrect interface boundaries",
                        "Performance overhead from indirection",
                    ],
                }
            ],
            "automation_lessons": [
                {
                    "lesson": "Configuration splitting is highly automatable",
                    "confidence": 0.9,
                    "evidence": "Successful automation in 95% of cases",
                    "recommendations": [
                        "Use AST analysis for dependency detection",
                        "Apply naming conventions consistently",
                        "Generate validation rules automatically",
                    ],
                }
            ],
            "quality_improvements": {
                "average_complexity_reduction": 0.4,
                "average_coupling_reduction": 0.3,
                "average_cohesion_improvement": 0.5,
                "test_coverage_improvement": 0.25,
            },
        }

    def _generate_quality_metrics(self) -> Dict[str, Any]:
        """Generate quality metrics and thresholds."""
        return {
            "complexity_thresholds": {
                "cyclomatic_complexity": {
                    "low": 5,
                    "medium": 10,
                    "high": 20,
                    "very_high": 50,
                },
                "cognitive_complexity": {
                    "low": 5,
                    "medium": 15,
                    "high": 25,
                    "very_high": 50,
                },
            },
            "size_thresholds": {
                "lines_of_code": {
                    "small": 100,
                    "medium": 300,
                    "large": 500,
                    "very_large": 1000,
                },
                "number_of_methods": {
                    "few": 5,
                    "moderate": 10,
                    "many": 20,
                    "too_many": 50,
                },
            },
            "coupling_metrics": {
                "afferent_coupling": {"threshold": 10},
                "efferent_coupling": {"threshold": 10},
                "instability": {"threshold": 0.5},
            },
            "cohesion_metrics": {
                "lcom": {"threshold": 0.8},
                "conceptual_cohesion": {"threshold": 0.7},
            },
        }

    def _generate_automation_workflows(self) -> List[Dict[str, Any]]:
        """Generate automation workflow templates."""
        return [
            {
                "workflow_id": "component_extraction_workflow",
                "name": "Component Extraction Workflow",
                "description": "Automated workflow for extracting components from monolithic classes",
                "steps": [
                    {
                        "step": "analyze_class_structure",
                        "automation_level": "full",
                        "tools": ["ast_analyzer", "metrics_calculator"],
                    },
                    {
                        "step": "identify_extraction_candidates",
                        "automation_level": "full",
                        "tools": ["responsibility_analyzer", "coupling_detector"],
                    },
                    {
                        "step": "generate_component_interface",
                        "automation_level": "assisted",
                        "tools": ["interface_generator", "naming_suggester"],
                    },
                    {
                        "step": "extract_component_implementation",
                        "automation_level": "assisted",
                        "tools": ["code_generator", "dependency_injector"],
                    },
                    {
                        "step": "update_original_class",
                        "automation_level": "full",
                        "tools": ["code_transformer", "import_manager"],
                    },
                    {
                        "step": "generate_tests",
                        "automation_level": "assisted",
                        "tools": ["test_generator", "mock_generator"],
                    },
                    {
                        "step": "validate_refactoring",
                        "automation_level": "full",
                        "tools": ["test_runner", "metrics_validator"],
                    },
                ],
                "estimated_automation_percentage": 0.7,
                "human_review_points": [
                    "Component interface design",
                    "Business logic correctness",
                    "Performance impact assessment",
                ],
            }
        ]

    def _generate_integration_templates(self) -> Dict[str, Any]:
        """Generate integration templates for different project types."""
        return {
            "django_project": {
                "config_template": {
                    "project_type": "django",
                    "file_patterns": ["models.py", "views.py", "serializers.py"],
                    "test_patterns": ["test_*.py", "*_test.py"],
                    "complexity_thresholds": {
                        "view_complexity": 8,
                        "model_complexity": 12,
                    },
                },
                "refactoring_patterns": [
                    "extract_service_layer",
                    "split_fat_models",
                    "extract_form_validators",
                ],
            },
            "flask_project": {
                "config_template": {
                    "project_type": "flask",
                    "file_patterns": ["app.py", "routes.py", "models.py"],
                    "test_patterns": ["test_*.py"],
                    "complexity_thresholds": {
                        "route_complexity": 6,
                        "blueprint_complexity": 10,
                    },
                },
                "refactoring_patterns": [
                    "extract_blueprint_services",
                    "split_application_factory",
                    "extract_middleware_components",
                ],
            },
            "fastapi_project": {
                "config_template": {
                    "project_type": "fastapi",
                    "file_patterns": ["main.py", "routers/*.py", "models/*.py"],
                    "test_patterns": ["test_*.py"],
                    "complexity_thresholds": {
                        "endpoint_complexity": 5,
                        "dependency_complexity": 8,
                    },
                },
                "refactoring_patterns": [
                    "extract_dependency_providers",
                    "split_router_modules",
                    "extract_validation_schemas",
                ],
            },
        }

    def demonstrate_automation_recommendations(
        self, sample_metrics: Optional[Dict] = None
    ) -> List[str]:
        """Demonstrate automation recommendations based on sample metrics."""
        if sample_metrics is None:
            sample_metrics = self._create_sample_metrics()

        recommendations = []

        # Analyze based on metrics
        if sample_metrics.get("file_size", 0) > 1000:
            recommendations.append("Consider extracting components from large files")

        if sample_metrics.get("cyclomatic_complexity", 0) > 15:
            recommendations.append(
                "High complexity detected - extract methods or components"
            )

        if sample_metrics.get("coupling_level", 0) > 0.7:
            recommendations.append(
                "High coupling detected - introduce interfaces and DI"
            )

        if sample_metrics.get("test_coverage", 1.0) < 0.6:
            recommendations.append("Low test coverage - generate test templates")

        return recommendations

    def _create_sample_metrics(self) -> Dict[str, Any]:
        """Create sample metrics for demonstration."""
        return {
            "file_size": 1200,
            "cyclomatic_complexity": 18,
            "number_of_responsibilities": 4,
            "coupling_level": 0.8,
            "cohesion_level": 0.4,
            "test_coverage": 0.45,
            "number_of_methods": 25,
            "lines_per_method": 48,
        }


def generate_and_export_comprehensive_metadata(
    export_path: str, project_config: Optional[Dict[str, Any]] = None
) -> None:
    """Generate and export comprehensive automation metadata."""
    generator = ComprehensiveMetadataGenerator(project_config)
    metadata = generator.generate_comprehensive_metadata(
        "comprehensive-project", "Comprehensive Refactoring Project"
    )
    generator.export_metadata(metadata, export_path)


def main():
    """Generate and demonstrate comprehensive refactoring automation metadata."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive refactoring automation metadata generation...")

    try:
        # Generate and export the comprehensive automation metadata
        export_path = "data/comprehensive_automation_metadata.json"
        generate_and_export_comprehensive_metadata(export_path)

        # Demonstrate automation recommendations
        logger.info("\nDemonstrating automation recommendations...")

        generator = ComprehensiveMetadataGenerator()

        # Create sample metrics for demonstration
        sample_metrics = generator._create_sample_metrics()

        logger.info("Sample code metrics:")
        for metric, value in sample_metrics.items():
            logger.info(f"  - {metric}: {value}")

        # Get automation recommendations
        recommendations = generator.demonstrate_automation_recommendations(
            sample_metrics
        )

        logger.info("\nAutomation recommendations:")
        for i, recommendation in enumerate(recommendations, 1):
            logger.info(f"  {i}. {recommendation}")

        logger.info(
            f"\nComprehensive automation metadata successfully generated and exported to {export_path}"
        )
        logger.info(
            "This metadata can now be used to automate complex refactoring efforts."
        )

    except Exception as e:
        logger.error(f"Error generating comprehensive automation metadata: {e}")
        raise


if __name__ == "__main__":
    main()
