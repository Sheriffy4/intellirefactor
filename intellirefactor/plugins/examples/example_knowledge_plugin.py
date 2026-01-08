"""
Example knowledge management plugin for IntelliRefactor

Demonstrates how to create custom knowledge extraction and management rules using the hook system.
This plugin shows how to capture and utilize refactoring knowledge.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from ..plugin_interface import KnowledgePlugin, PluginMetadata, PluginType
from ..hook_system import HookSystem, HookType, HookPriority


class ExampleKnowledgePlugin(KnowledgePlugin):
    """
    Example plugin demonstrating custom knowledge management.

    This plugin implements several knowledge management features:
    1. Extract patterns from successful refactorings
    2. Learn from refactoring failures
    3. Build a database of project-specific patterns
    4. Provide recommendations based on historical data
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_knowledge",
            version="1.0.0",
            description="Example plugin demonstrating knowledge extraction and management",
            author="IntelliRefactor Team",
            plugin_type=PluginType.KNOWLEDGE,
            dependencies=[],
            config_schema={
                "knowledge_db_path": {
                    "type": "string",
                    "default": "knowledge.json",
                    "description": "Path to knowledge database",
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.7,
                    "description": "Minimum confidence for recommendations",
                },
                "max_knowledge_items": {
                    "type": "integer",
                    "default": 1000,
                    "description": "Maximum knowledge items to store",
                },
                "enable_pattern_learning": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable automatic pattern learning",
                },
            },
        )

    def initialize(self) -> bool:
        """Initialize the plugin and set up knowledge storage."""
        try:
            # Get hook system instance
            self.hook_system = getattr(self, "_hook_system", None)
            if not self.hook_system:
                self.logger.warning("Hook system not available, creating local instance")
                self.hook_system = HookSystem()

            # Initialize knowledge storage
            self.knowledge_db_path = self.config.get("knowledge_db_path", "knowledge.json")
            self.knowledge_db = self._load_knowledge_db()

            # Register our knowledge hooks
            self._register_hooks()

            self.logger.info("Example knowledge plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge plugin: {e}")
            return False

    def _load_knowledge_db(self) -> Dict[str, Any]:
        """Load knowledge database from file."""
        if os.path.exists(self.knowledge_db_path):
            try:
                with open(self.knowledge_db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading knowledge database: {e}")

        # Return empty database structure
        return {
            "patterns": {},
            "refactoring_history": [],
            "success_metrics": {},
            "failure_patterns": {},
            "project_characteristics": {},
            "recommendations": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
            },
        }

    def _save_knowledge_db(self) -> None:
        """Save knowledge database to file."""
        try:
            self.knowledge_db["metadata"]["last_updated"] = datetime.now().isoformat()
            with open(self.knowledge_db_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_db, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving knowledge database: {e}")

    def _register_hooks(self) -> None:
        """Register all knowledge management hooks."""
        # Knowledge extraction hooks
        self.hook_system.register_hook(
            hook_type=HookType.PRE_KNOWLEDGE_EXTRACTION,
            callback=self._pre_knowledge_extraction,
            name="example_pre_knowledge_extraction",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Prepare knowledge extraction context",
        )

        self.hook_system.register_hook(
            hook_type=HookType.POST_KNOWLEDGE_EXTRACTION,
            callback=self._post_knowledge_extraction,
            name="example_post_knowledge_extraction",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Process extracted knowledge",
        )

        # Refactoring success/failure learning hooks
        self.hook_system.register_hook(
            hook_type=HookType.POST_REFACTORING,
            callback=self._learn_from_refactoring,
            name="learn_from_refactoring",
            priority=HookPriority.LOW,  # Run after other post-refactoring hooks
            plugin_name=self.metadata.name,
            description="Learn from refactoring results",
        )

        # Custom knowledge hooks
        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._extract_code_patterns,
            name="extract_code_patterns",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Extract code patterns from analysis",
            custom_key="extract_code_patterns",
        )

        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._generate_recommendations,
            name="generate_recommendations",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Generate recommendations based on knowledge",
            custom_key="generate_recommendations",
        )

    def _pre_knowledge_extraction(
        self,
        analysis_results: Dict[str, Any],
        refactoring_results: List[Dict[str, Any]],
    ) -> None:
        """Prepare knowledge extraction context."""
        self.logger.debug("Pre-knowledge extraction hook")

        # Set up extraction context
        self.hook_system.set_context(
            "knowledge_extraction",
            {
                "start_time": datetime.now(),
                "analysis_files": len(analysis_results.get("files", {})),
                "refactoring_count": len(refactoring_results),
                "patterns_found": [],
            },
        )

    def _post_knowledge_extraction(
        self,
        analysis_results: Dict[str, Any],
        refactoring_results: List[Dict[str, Any]],
        knowledge_items: List[Dict[str, Any]],
    ) -> None:
        """Process extracted knowledge."""
        self.logger.debug("Post-knowledge extraction hook")

        # Update knowledge database with new items
        for item in knowledge_items:
            self._store_knowledge_item(item)

        # Update extraction statistics
        context = self.hook_system.get_context("knowledge_extraction", {})
        extraction_time = (
            datetime.now() - context.get("start_time", datetime.now())
        ).total_seconds()

        self.knowledge_db["metadata"]["last_extraction"] = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": extraction_time,
            "items_extracted": len(knowledge_items),
            "files_processed": context.get("analysis_files", 0),
        }

        self._save_knowledge_db()

    def _learn_from_refactoring(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Learn from refactoring results."""
        if not self.config.get("enable_pattern_learning", True):
            return

        refactoring_record = {
            "timestamp": datetime.now().isoformat(),
            "opportunity_type": opportunity.get("type", "unknown"),
            "success": result.get("success", False),
            "file_path": result.get("file_path", ""),
            "transformation_type": result.get("transformation_type", ""),
            "changes_made": result.get("changes_made", 0),
            "error": result.get("error", None),
        }

        # Add to refactoring history
        self.knowledge_db["refactoring_history"].append(refactoring_record)

        # Update success metrics
        refactoring_type = refactoring_record["opportunity_type"]
        if refactoring_type not in self.knowledge_db["success_metrics"]:
            self.knowledge_db["success_metrics"][refactoring_type] = {
                "total_attempts": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
            }

        metrics = self.knowledge_db["success_metrics"][refactoring_type]
        metrics["total_attempts"] += 1

        if refactoring_record["success"]:
            metrics["successful"] += 1
        else:
            metrics["failed"] += 1
            # Store failure pattern
            self._store_failure_pattern(refactoring_record)

        metrics["success_rate"] = metrics["successful"] / metrics["total_attempts"]

        # Limit history size
        max_items = self.config.get("max_knowledge_items", 1000)
        if len(self.knowledge_db["refactoring_history"]) > max_items:
            self.knowledge_db["refactoring_history"] = self.knowledge_db["refactoring_history"][
                -max_items:
            ]

        self._save_knowledge_db()

    def _store_failure_pattern(self, refactoring_record: Dict[str, Any]) -> None:
        """Store patterns from failed refactorings."""
        failure_type = refactoring_record["opportunity_type"]
        error_msg = refactoring_record.get("error", "Unknown error")

        if failure_type not in self.knowledge_db["failure_patterns"]:
            self.knowledge_db["failure_patterns"][failure_type] = {}

        # Categorize error
        error_category = self._categorize_error(error_msg)

        if error_category not in self.knowledge_db["failure_patterns"][failure_type]:
            self.knowledge_db["failure_patterns"][failure_type][error_category] = {
                "count": 0,
                "examples": [],
                "common_characteristics": [],
            }

        pattern = self.knowledge_db["failure_patterns"][failure_type][error_category]
        pattern["count"] += 1

        # Store example (limit to 5 examples per pattern)
        if len(pattern["examples"]) < 5:
            pattern["examples"].append(
                {
                    "file_path": refactoring_record["file_path"],
                    "error": error_msg,
                    "timestamp": refactoring_record["timestamp"],
                }
            )

    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error messages into common types."""
        error_lower = error_msg.lower()

        if "syntax" in error_lower:
            return "syntax_error"
        elif "import" in error_lower or "module" in error_lower:
            return "import_error"
        elif "name" in error_lower and "not defined" in error_lower:
            return "name_error"
        elif "type" in error_lower:
            return "type_error"
        elif "attribute" in error_lower:
            return "attribute_error"
        elif "file" in error_lower or "path" in error_lower:
            return "file_error"
        else:
            return "other_error"

    def _extract_code_patterns(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract code patterns from analysis results."""
        patterns = {
            "common_imports": {},
            "function_patterns": {},
            "class_patterns": {},
            "architectural_patterns": [],
        }

        # Analyze files for common patterns
        for file_path, file_results in analysis_results.get("files", {}).items():
            if not file_path.endswith(".py"):
                continue

            try:
                # Extract import patterns
                imports = file_results.get("imports", [])
                for imp in imports:
                    module = imp.get("module", "")
                    if module:
                        patterns["common_imports"][module] = (
                            patterns["common_imports"].get(module, 0) + 1
                        )

                # Extract function patterns
                functions = file_results.get("functions", [])
                for func in functions:
                    func_name = func.get("name", "")
                    param_count = func.get("parameter_count", 0)
                    complexity = func.get("complexity", 0)

                    pattern_key = f"params_{param_count}_complexity_{complexity // 5 * 5}"  # Group by complexity ranges
                    if pattern_key not in patterns["function_patterns"]:
                        patterns["function_patterns"][pattern_key] = {
                            "count": 0,
                            "examples": [],
                            "avg_lines": 0,
                            "common_names": {},
                        }

                    pattern = patterns["function_patterns"][pattern_key]
                    pattern["count"] += 1

                    # Track common function names
                    name_parts = func_name.split("_")
                    for part in name_parts:
                        if len(part) > 2:  # Skip very short parts
                            pattern["common_names"][part] = pattern["common_names"].get(part, 0) + 1

                # Extract class patterns
                classes = file_results.get("classes", [])
                for cls in classes:
                    class_name = cls.get("name", "")
                    method_count = cls.get("method_count", 0)

                    pattern_key = f"methods_{method_count // 5 * 5}"  # Group by method count ranges
                    if pattern_key not in patterns["class_patterns"]:
                        patterns["class_patterns"][pattern_key] = {
                            "count": 0,
                            "examples": [],
                            "common_suffixes": {},
                        }

                    pattern = patterns["class_patterns"][pattern_key]
                    pattern["count"] += 1

                    # Track common class name suffixes
                    if class_name.endswith(
                        ("Manager", "Handler", "Service", "Controller", "Factory")
                    ):
                        suffix = (
                            class_name[-7:]
                            if class_name.endswith("Controller")
                            else class_name[-7:]
                        )
                        for common_suffix in [
                            "Manager",
                            "Handler",
                            "Service",
                            "Controller",
                            "Factory",
                        ]:
                            if class_name.endswith(common_suffix):
                                suffix = common_suffix
                                break
                        pattern["common_suffixes"][suffix] = (
                            pattern["common_suffixes"].get(suffix, 0) + 1
                        )

            except Exception as e:
                self.logger.error(f"Error extracting patterns from {file_path}: {e}")

        return patterns

    def _generate_recommendations(
        self, project_path: str, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate recommendations based on accumulated knowledge."""
        recommendations = {
            "refactoring_suggestions": [],
            "pattern_suggestions": [],
            "risk_warnings": [],
            "best_practices": [],
        }

        min_confidence = self.config.get("min_confidence", 0.7)

        # Generate refactoring recommendations based on success rates
        for refactoring_type, metrics in self.knowledge_db["success_metrics"].items():
            success_rate = metrics["success_rate"]
            total_attempts = metrics["total_attempts"]

            if success_rate >= min_confidence and total_attempts >= 5:
                recommendations["refactoring_suggestions"].append(
                    {
                        "type": refactoring_type,
                        "confidence": success_rate,
                        "evidence": f"Success rate: {success_rate:.1%} ({metrics['successful']}/{total_attempts})",
                        "recommendation": f"Consider applying {refactoring_type} refactoring",
                        "priority": "high" if success_rate > 0.9 else "medium",
                    }
                )
            elif success_rate < 0.5 and total_attempts >= 3:
                recommendations["risk_warnings"].append(
                    {
                        "type": refactoring_type,
                        "risk_level": "high" if success_rate < 0.3 else "medium",
                        "evidence": f"Low success rate: {success_rate:.1%} ({metrics['successful']}/{total_attempts})",
                        "warning": f"Be cautious with {refactoring_type} refactoring",
                        "common_failures": self._get_common_failures(refactoring_type),
                    }
                )

        # Generate pattern-based recommendations
        patterns = self.knowledge_db.get("patterns", {})
        for pattern_type, pattern_data in patterns.items():
            if (
                isinstance(pattern_data, dict)
                and pattern_data.get("confidence", 0) >= min_confidence
            ):
                recommendations["pattern_suggestions"].append(
                    {
                        "pattern": pattern_type,
                        "confidence": pattern_data["confidence"],
                        "suggestion": pattern_data.get(
                            "suggestion", f"Consider applying {pattern_type} pattern"
                        ),
                        "benefits": pattern_data.get("benefits", []),
                    }
                )

        # Generate best practice recommendations
        recommendations["best_practices"] = self._generate_best_practices(analysis_results)

        return recommendations

    def _get_common_failures(self, refactoring_type: str) -> List[str]:
        """Get common failure reasons for a refactoring type."""
        failures = self.knowledge_db["failure_patterns"].get(refactoring_type, {})
        common_failures = []

        for error_category, pattern in failures.items():
            if pattern["count"] >= 2:  # At least 2 occurrences
                common_failures.append(
                    {
                        "category": error_category,
                        "count": pattern["count"],
                        "description": f"Common {error_category.replace('_', ' ')} issues",
                    }
                )

        return common_failures

    def _generate_best_practices(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate best practice recommendations."""
        practices = []

        # Analyze project characteristics
        total_files = len(analysis_results.get("files", {}))
        avg_complexity = self._calculate_average_complexity(analysis_results)

        if avg_complexity > 10:
            practices.append(
                {
                    "category": "complexity",
                    "recommendation": "Consider breaking down complex functions",
                    "rationale": f"Average complexity is {avg_complexity:.1f}, which is above recommended threshold",
                    "priority": "high",
                }
            )

        if total_files > 50:
            practices.append(
                {
                    "category": "architecture",
                    "recommendation": "Consider modularizing the codebase",
                    "rationale": f"Project has {total_files} files, consider organizing into packages",
                    "priority": "medium",
                }
            )

        return practices

    def _calculate_average_complexity(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate average complexity across all functions."""
        total_complexity = 0
        function_count = 0

        for file_results in analysis_results.get("files", {}).values():
            functions = file_results.get("functions", [])
            for func in functions:
                complexity = func.get("complexity", 1)
                total_complexity += complexity
                function_count += 1

        return total_complexity / function_count if function_count > 0 else 0

    def _store_knowledge_item(self, item: Dict[str, Any]) -> None:
        """Store a knowledge item in the database."""
        item_type = item.get("type", "unknown")
        item_id = item.get("id", f"{item_type}_{len(self.knowledge_db['patterns'])}")

        self.knowledge_db["patterns"][item_id] = {
            "type": item_type,
            "content": item.get("content", {}),
            "confidence": item.get("confidence", 0.5),
            "created": datetime.now().isoformat(),
            "usage_count": 0,
            "last_used": None,
            "metadata": item.get("metadata", {}),
        }

    def extract_knowledge(
        self,
        analysis_results: Dict[str, Any],
        refactoring_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract knowledge from analysis and refactoring results."""
        # Execute pre-extraction hooks
        self.hook_system.execute_hooks(
            HookType.PRE_KNOWLEDGE_EXTRACTION, analysis_results, refactoring_results
        )

        knowledge_items = []

        # Extract code patterns
        pattern_results = self.hook_system.execute_custom_hooks(
            "extract_code_patterns", analysis_results
        )
        for result in pattern_results:
            if result:
                for pattern_type, pattern_data in result.items():
                    if pattern_data and isinstance(pattern_data, dict):
                        knowledge_items.append(
                            {
                                "id": f"pattern_{pattern_type}_{datetime.now().timestamp()}",
                                "type": "code_pattern",
                                "content": {
                                    "pattern_type": pattern_type,
                                    "data": pattern_data,
                                },
                                "confidence": self._calculate_pattern_confidence(pattern_data),
                                "metadata": {
                                    "extracted_from": "analysis_results",
                                    "extraction_method": "custom_hook",
                                },
                            }
                        )

        # Extract refactoring patterns
        for refactoring_result in refactoring_results:
            if refactoring_result.get("success", False):
                knowledge_items.append(
                    {
                        "id": f"refactoring_{refactoring_result.get('transformation_type', 'unknown')}_{datetime.now().timestamp()}",
                        "type": "successful_refactoring",
                        "content": {
                            "transformation_type": refactoring_result.get("transformation_type"),
                            "file_path": refactoring_result.get("file_path"),
                            "changes_made": refactoring_result.get("changes_made", 0),
                        },
                        "confidence": 0.8,  # High confidence for successful refactorings
                        "metadata": {
                            "extracted_from": "refactoring_results",
                            "success": True,
                        },
                    }
                )

        # Execute post-extraction hooks
        self.hook_system.execute_hooks(
            HookType.POST_KNOWLEDGE_EXTRACTION,
            analysis_results,
            refactoring_results,
            knowledge_items,
        )

        return knowledge_items

    def _calculate_pattern_confidence(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate confidence score for a pattern."""
        if not isinstance(pattern_data, dict):
            return 0.1

        # Simple heuristic based on frequency
        total_count = 0
        for item in pattern_data.values():
            if isinstance(item, dict) and "count" in item:
                total_count += item["count"]
            elif isinstance(item, int):
                total_count += item

        # Normalize confidence based on frequency
        if total_count >= 10:
            return 0.9
        elif total_count >= 5:
            return 0.7
        elif total_count >= 2:
            return 0.5
        else:
            return 0.3

    def query_knowledge(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        query_type = query.get("type", "all")
        min_confidence = query.get("min_confidence", self.config.get("min_confidence", 0.7))
        limit = query.get("limit", 10)

        results = []

        # Search patterns
        for pattern_id, pattern in self.knowledge_db["patterns"].items():
            if query_type == "all" or pattern["type"] == query_type:
                if pattern["confidence"] >= min_confidence:
                    results.append(
                        {
                            "id": pattern_id,
                            "type": pattern["type"],
                            "content": pattern["content"],
                            "confidence": pattern["confidence"],
                            "usage_count": pattern["usage_count"],
                            "last_used": pattern["last_used"],
                        }
                    )

        # Sort by confidence and usage
        results.sort(key=lambda x: (x["confidence"], x["usage_count"]), reverse=True)

        return results[:limit]

    def update_knowledge(self, knowledge_items: List[Dict[str, Any]]) -> bool:
        """Update knowledge base with new items."""
        try:
            for item in knowledge_items:
                self._store_knowledge_item(item)

            self._save_knowledge_db()
            return True

        except Exception as e:
            self.logger.error(f"Error updating knowledge: {e}")
            return False

    def get_knowledge_types(self) -> List[str]:
        """Get list of knowledge types this plugin handles."""
        return [
            "code_pattern",
            "successful_refactoring",
            "failure_pattern",
            "architectural_pattern",
            "best_practice",
        ]

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Save knowledge database
        self._save_knowledge_db()

        if hasattr(self, "hook_system"):
            # Unregister our hooks
            hooks_to_remove = [
                "example_pre_knowledge_extraction",
                "example_post_knowledge_extraction",
                "learn_from_refactoring",
                "extract_code_patterns",
                "generate_recommendations",
            ]

            for hook_name in hooks_to_remove:
                self.hook_system.unregister_hook(hook_name)

        self.logger.info("Example knowledge plugin cleaned up")
