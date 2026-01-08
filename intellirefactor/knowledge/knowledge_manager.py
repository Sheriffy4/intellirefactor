"""
Knowledge Base Manager for IntelliRefactor

Extracted and generalized from recon project.
Manages refactoring knowledge base with CRUD operations and recommendations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..config import KnowledgeConfig


class KnowledgeItem:
    """Represents a single knowledge item in the knowledge base."""

    def __init__(
        self,
        filename: str,
        item_type: str,
        project_name: str,
        date_created: str,
        metadata: Dict[str, Any],
    ):
        self.filename = filename
        self.type = item_type
        self.project_name = project_name
        self.date_created = date_created
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge item to dictionary representation."""
        return {
            "filename": self.filename,
            "type": self.type,
            "project_name": self.project_name,
            "date_created": self.date_created,
            **self.metadata,
        }


class KnowledgeManager:
    """
    High-level interface for knowledge base operations.

    Manages refactoring knowledge with full CRUD operations, indexing,
    search, and recommendation capabilities.
    """

    def __init__(
        self, config: Optional[KnowledgeConfig] = None, knowledge_dir: str = None
    ):
        """Initialize the knowledge manager with configuration."""
        self.config = config or KnowledgeConfig()
        self.knowledge_dir = Path(knowledge_dir) if knowledge_dir else Path("knowledge")
        self.index_file = self.knowledge_dir / "knowledge_index.json"

        # Ensure knowledge directory exists
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def initialize_project_knowledge(self, project_path: str) -> Dict[str, Any]:
        """
        Initialize knowledge base for a specific project.

        Args:
            project_path: Path to the project

        Returns:
            Initialization status
        """
        # Ensure project-specific knowledge directory exists if needed
        # For now, just return success as the main KB is global
        return {
            "status": "initialized",
            "knowledge_base": str(self.knowledge_dir),
            "project": project_path,
        }

    def add_knowledge(self, knowledge_item: Dict[str, Any]) -> bool:
        """
        Add a generic knowledge item to the knowledge base.

        Args:
            knowledge_item: Dictionary containing knowledge data

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Generate a filename if not provided
            item_type = knowledge_item.get("type", "generic_knowledge")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{item_type}_{timestamp}.json"

            # Ensure metadata fields
            if "date_created" not in knowledge_item:
                knowledge_item["date_created"] = datetime.now().isoformat()

            # Save to file
            file_path = self.knowledge_dir / filename
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(knowledge_item, f, indent=2, ensure_ascii=False)

            # Create index entry
            project_name = knowledge_item.get("project_name", "Unknown Project")
            if "module_name" in knowledge_item:
                project_name = knowledge_item["module_name"]

            index_entry = {
                "filename": filename,
                "type": item_type,
                "project_name": project_name,
                "date_created": knowledge_item["date_created"],
                "metadata": knowledge_item,
            }

            # Update index
            self._update_index(index_entry)
            return True

        except Exception as e:
            print(f"Error adding knowledge item: {e}")
            return False

    def add_refactoring_metadata(
        self, metadata_file: str, project_name: str = None
    ) -> bool:
        """
        Add new refactoring metadata file to the knowledge base.

        Args:
            metadata_file: Path to the metadata file
            project_name: Optional project name override

        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Determine project name
            if not project_name:
                project_name = metadata.get("project_name", "Unknown Project")

            # Create knowledge entry
            knowledge_entry = {
                "filename": Path(metadata_file).name,
                "type": "refactoring_metadata",
                "project_name": project_name,
                "date_created": metadata.get(
                    "refactoring_date", datetime.now().strftime("%Y-%m-%d")
                ),
                "transformation_rules_count": len(
                    metadata.get("transformation_rules", [])
                ),
                "di_patterns_count": len(metadata.get("di_patterns", [])),
                "interface_templates_count": len(
                    metadata.get("interface_templates", [])
                ),
                "testing_strategies_count": len(metadata.get("testing_strategies", [])),
                "automation_potential_score": metadata.get(
                    "automation_potential_score", 0.0
                ),
                "reusability_score": metadata.get("reusability_score", 0.0),
                "key_metrics": metadata.get("overall_success_metrics", {}),
                "applicable_contexts": metadata.get("applicable_contexts", []),
            }

            # Update index
            self._update_index(knowledge_entry)
            return True

        except Exception as e:
            print(f"Error adding refactoring metadata: {e}")
            return False

    def _update_index(self, new_entry: Dict[str, Any]) -> None:
        """Update the knowledge base index file."""

        # Load existing index or create new one
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except json.JSONDecodeError:
                # Handle corrupted index file
                index = self._create_empty_index()
        else:
            index = self._create_empty_index()

        # Check if file already exists
        existing_files = [f.get("filename") for f in index.get("knowledge_files", [])]
        if new_entry["filename"] in existing_files:
            # Update existing entry
            for i, entry in enumerate(index["knowledge_files"]):
                if entry.get("filename") == new_entry["filename"]:
                    index["knowledge_files"][i] = new_entry
                    break
        else:
            # Add new entry
            if "knowledge_files" not in index:
                index["knowledge_files"] = []
            index["knowledge_files"].append(new_entry)
            index["total_refactoring_projects"] = (
                index.get("total_refactoring_projects", 0) + 1
            )

        # Update statistics
        self._recalculate_statistics(index)

        # Update timestamp
        index["last_updated"] = datetime.now().isoformat()

        # Save index
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _create_empty_index(self) -> Dict[str, Any]:
        """Create an empty index structure."""
        return {
            "knowledge_base_version": "1.0",
            "last_updated": "",
            "total_refactoring_projects": 0,
            "knowledge_files": [],
            "statistics": {
                "total_transformation_rules": 0,
                "total_di_patterns": 0,
                "total_interface_templates": 0,
                "total_testing_strategies": 0,
                "average_automation_confidence": 0.0,
                "average_reusability_score": 0.0,
            },
        }

    def _recalculate_statistics(self, index: Dict[str, Any]) -> None:
        """Recalculate knowledge base statistics."""

        refactoring_files = [
            f
            for f in index.get("knowledge_files", [])
            if f.get("type") == "refactoring_metadata"
        ]

        if not refactoring_files:
            return

        # Sum totals
        total_rules = sum(
            f.get("transformation_rules_count", 0) for f in refactoring_files
        )
        total_di = sum(f.get("di_patterns_count", 0) for f in refactoring_files)
        total_interfaces = sum(
            f.get("interface_templates_count", 0) for f in refactoring_files
        )
        total_testing = sum(
            f.get("testing_strategies_count", 0) for f in refactoring_files
        )

        # Calculate averages
        avg_automation = sum(
            f.get("automation_potential_score", 0) for f in refactoring_files
        ) / len(refactoring_files)
        avg_reusability = sum(
            f.get("reusability_score", 0) for f in refactoring_files
        ) / len(refactoring_files)

        index["statistics"] = {
            "total_transformation_rules": total_rules,
            "total_di_patterns": total_di,
            "total_interface_templates": total_interfaces,
            "total_testing_strategies": total_testing,
            "average_automation_confidence": round(avg_automation, 2),
            "average_reusability_score": round(avg_reusability, 2),
        }

    def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for refactoring patterns and insights.

        Args:
            query: Knowledge query string

        Returns:
            List of knowledge items matching the query
        """
        if not self.index_file.exists():
            return []

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            results = []
            query_lower = query.lower()

            for file_info in index.get("knowledge_files", []):
                # Build searchable text from various fields
                searchable_parts = [
                    file_info.get("project_name", ""),
                    file_info.get("type", ""),
                    " ".join(file_info.get("applicable_contexts", [])),
                ]

                # Add metadata values to search
                metadata = file_info.get("metadata", {})
                if isinstance(metadata, dict):
                    for v in metadata.values():
                        if isinstance(v, str):
                            searchable_parts.append(v)
                        elif isinstance(v, list):
                            searchable_parts.append(" ".join(str(x) for x in v))

                searchable_text = " ".join(searchable_parts).lower()

                if query_lower in searchable_text:
                    results.append(file_info)

            return results

        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []

    def get_recommendations(self, context: str) -> List[str]:
        """
        Get recommendations based on context.

        Args:
            context: Context for recommendations

        Returns:
            List of recommendation strings
        """
        if not self.index_file.exists():
            return []

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            recommendations = []

            for file_info in index.get("knowledge_files", []):
                if file_info.get("type") == "refactoring_metadata":
                    applicable_contexts = file_info.get("applicable_contexts", [])
                    if context in applicable_contexts:
                        recommendations.append(
                            f"Use patterns from {file_info.get('project_name', 'Unknown')} "
                            f"(automation score: {file_info.get('automation_potential_score', 0)})"
                        )

            return recommendations

        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []

    def list_knowledge(self) -> Dict[str, Any]:
        """
        List all knowledge files and statistics.

        Returns:
            Dictionary containing knowledge base information
        """
        if not self.index_file.exists():
            return {"error": "Knowledge base not found"}

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            return {
                "knowledge_base_version": index.get("knowledge_base_version", "1.0"),
                "last_updated": index.get("last_updated", ""),
                "total_projects": index.get("total_refactoring_projects", 0),
                "knowledge_files": index.get("knowledge_files", []),
                "statistics": index.get("statistics", {}),
            }

        except Exception as e:
            return {"error": f"Error reading knowledge base: {e}"}

    def learn_from_result(self, refactoring_result: Dict[str, Any]) -> None:
        """
        Learn from a refactoring result to improve future recommendations.

        Args:
            refactoring_result: Result of a refactoring operation
        """
        # Extract learning data from refactoring result
        learning_data = {
            "operation_type": refactoring_result.get("operation_type", "unknown"),
            "success": refactoring_result.get("success", False),
            "metrics_improvement": refactoring_result.get("metrics_improvement", {}),
            "patterns_used": refactoring_result.get("patterns_used", []),
            "context": refactoring_result.get("context", ""),
            "timestamp": datetime.now().isoformat(),
        }

        # Store learning data (simplified implementation)
        learning_file = self.knowledge_dir / "learning_data.json"

        try:
            if learning_file.exists():
                with open(learning_file, "r", encoding="utf-8") as f:
                    learning_history = json.load(f)
            else:
                learning_history = {"learning_entries": []}

            learning_history["learning_entries"].append(learning_data)

            with open(learning_file, "w", encoding="utf-8") as f:
                json.dump(learning_history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error storing learning data: {e}")

    def get_knowledge_item(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific knowledge item by filename.

        Args:
            filename: Name of the knowledge file

        Returns:
            Knowledge item dictionary or None if not found
        """
        if not self.index_file.exists():
            return None

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            for file_info in index.get("knowledge_files", []):
                if file_info.get("filename") == filename:
                    return file_info

            return None

        except Exception as e:
            print(f"Error getting knowledge item: {e}")
            return None

    def delete_knowledge_item(self, filename: str) -> bool:
        """
        Delete a knowledge item from the knowledge base.

        Args:
            filename: Name of the knowledge file to delete

        Returns:
            True if successfully deleted, False otherwise
        """
        if not self.index_file.exists():
            return False

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                index = json.load(f)

            # Find and remove the item
            original_count = len(index.get("knowledge_files", []))
            index["knowledge_files"] = [
                f
                for f in index.get("knowledge_files", [])
                if f.get("filename") != filename
            ]

            if len(index["knowledge_files"]) < original_count:
                # Update project count
                index["total_refactoring_projects"] = len(
                    [
                        f
                        for f in index["knowledge_files"]
                        if f.get("type") == "refactoring_metadata"
                    ]
                )

                # Recalculate statistics
                self._recalculate_statistics(index)

                # Update timestamp
                index["last_updated"] = datetime.now().isoformat()

                # Save updated index
                with open(self.index_file, "w", encoding="utf-8") as f:
                    json.dump(index, f, indent=2, ensure_ascii=False)

                return True

            return False

        except Exception as e:
            print(f"Error deleting knowledge item: {e}")
            return False
