"""
Specialized analyzers for expert refactoring analysis.

Each analyzer focuses on a specific aspect of code analysis needed
for safe and precise refactoring decisions.
"""

from __future__ import annotations

import importlib

__all__ = [
    "CallGraphAnalyzer",
    "CallerAnalyzer",
    "CohesionMatrixAnalyzer",
    "DependencyInterfaceAnalyzer",
    "ConcreteDeduplicationAnalyzer",
    "CharacterizationTestGenerator",
    "GitHistoryAnalyzer",
    "ExceptionContractAnalyzer",
    "DataSchemaAnalyzer",
    "GoldenTracesExtractor",
]


def __getattr__(name: str):
    # Lazy imports to avoid heavy/missing analyzer modules breaking package import.
    mapping = {
        "CallGraphAnalyzer": (".call_graph_analyzer", "CallGraphAnalyzer"),
        "CallerAnalyzer": (".caller_analyzer", "CallerAnalyzer"),
        "CohesionMatrixAnalyzer": (".cohesion_analyzer", "CohesionMatrixAnalyzer"),
        "DependencyInterfaceAnalyzer": (".dependency_analyzer", "DependencyInterfaceAnalyzer"),
        "ConcreteDeduplicationAnalyzer": (".duplicate_analyzer", "ConcreteDeduplicationAnalyzer"),
        "CharacterizationTestGenerator": (".characterization_generator", "CharacterizationTestGenerator"),
        "GitHistoryAnalyzer": (".git_analyzer", "GitHistoryAnalyzer"),
        "ExceptionContractAnalyzer": (".exception_contract_analyzer", "ExceptionContractAnalyzer"),
        "DataSchemaAnalyzer": (".data_schema_analyzer", "DataSchemaAnalyzer"),
        "GoldenTracesExtractor": (".golden_traces_extractor", "GoldenTracesExtractor"),
    }
    if name not in mapping:
        raise AttributeError(name)

    mod_name, attr = mapping[name]
    try:
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, attr)
    except Exception as e:
        # Important: should not raise ImportError here, otherwise package becomes fragile.
        raise AttributeError(f"{name} (optional analyzer unavailable: {e})") from None