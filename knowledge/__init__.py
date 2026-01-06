"""
Knowledge management module for IntelliRefactor

Provides knowledge base and learning capabilities including:
- Refactoring pattern storage and retrieval
- Machine learning from refactoring outcomes
- Decision tree management
- Automation metadata generation
"""

from .knowledge_manager import KnowledgeManager
from .knowledge_base import KnowledgeBase
from .automation_metadata import AutomationMetadata

__all__ = ["KnowledgeManager", "KnowledgeBase", "AutomationMetadata"]