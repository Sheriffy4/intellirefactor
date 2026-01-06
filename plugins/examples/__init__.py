"""
Example plugins for IntelliRefactor

This package contains example plugins that demonstrate how to extend
IntelliRefactor's functionality using the hook system and plugin architecture.
"""

from .example_custom_rules_plugin import CustomRulesPlugin
from .example_refactoring_plugin import ExampleRefactoringPlugin
from .example_knowledge_plugin import ExampleKnowledgePlugin

__all__ = [
    'CustomRulesPlugin',
    'ExampleRefactoringPlugin', 
    'ExampleKnowledgePlugin'
]