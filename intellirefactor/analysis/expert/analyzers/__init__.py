"""
Specialized analyzers for expert refactoring analysis.

Each analyzer focuses on a specific aspect of code analysis needed
for safe and precise refactoring decisions.
"""

from .call_graph_analyzer import CallGraphAnalyzer
from .caller_analyzer import CallerAnalyzer
from .cohesion_analyzer import CohesionMatrixAnalyzer
from .contract_analyzer import BehavioralContractAnalyzer
from .dependency_analyzer import DependencyInterfaceAnalyzer
from .test_analyzer import TestDiscoveryAnalyzer
from .characterization_generator import CharacterizationTestGenerator
from .duplicate_analyzer import ConcreteDeduplicationAnalyzer
from .git_analyzer import GitHistoryAnalyzer
from .compatibility_analyzer import CompatibilityAnalyzer

__all__ = [
    'CallGraphAnalyzer',
    'CallerAnalyzer', 
    'CohesionMatrixAnalyzer',
    'BehavioralContractAnalyzer',
    'DependencyInterfaceAnalyzer',
    'TestDiscoveryAnalyzer',
    'CharacterizationTestGenerator',
    'ConcreteDeduplicationAnalyzer',
    'GitHistoryAnalyzer',
    'CompatibilityAnalyzer',
]