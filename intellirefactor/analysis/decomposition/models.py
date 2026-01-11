"""
Data models for functional decomposition and consolidation.

Implements the core data structures from ref.md for representing
functional blocks, capabilities, similarity clusters, and consolidation plans.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RecommendationType(str, Enum):
    """Types of consolidation recommendations."""
    MERGE = "MERGE"
    EXTRACT_BASE = "EXTRACT_BASE"
    KEEP_SEPARATE = "KEEP_SEPARATE"
    WRAP_ONLY = "WRAP_ONLY"


class RiskLevel(str, Enum):
    """Risk levels for refactoring operations."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class UnresolvedReason(str, Enum):
    """Reasons why a call couldn't be resolved (normalized, lowercase values)."""
    ambiguous = "ambiguous"
    unknown_alias = "unknown_alias"
    star_import = "star_import"
    external = "external"
    dynamic_attribute = "dynamic_attribute"      # v5: Method call on unknown object
    baseclass_method = "baseclass_method"        # v6: Inherited method call
    class_instantiation = "class_instantiation"  # v6.1: Dataclass/attrs instantiation
    self_call = "self_call"                      # v6.3: Self-recursive calls
    dynamic_callback = "dynamic_callback"        # v6.6: Higher-order function calls (callbacks/injected functions)
    nested_local = "nested_local"                # v6.7: Nested local helper functions
    not_found = "not_found"

    @classmethod
    def is_actionable(cls, r: "UnresolvedReason") -> bool:
        # “actionable” = можно потенциально улучшить резолвинг
        return r not in {
            cls.external,
            cls.star_import,
            cls.dynamic_attribute,
            cls.baseclass_method,
            cls.class_instantiation,
            cls.self_call,
            cls.dynamic_callback,
            cls.nested_local,
        }

    @classmethod
    def is_internal(cls, r: "UnresolvedReason") -> bool:
        # “internal” = не внешний вызов и не star import
        return r not in {cls.external, cls.star_import}


def as_reason(x: str | UnresolvedReason | None) -> UnresolvedReason:
    """Normalize various reason representations into UnresolvedReason."""
    if isinstance(x, UnresolvedReason):
        return x
    s = str(x or UnresolvedReason.not_found.value).strip().lower()
    try:
        return UnresolvedReason(s)
    except ValueError:
        return UnresolvedReason.not_found


@dataclass
class FileSymbolTable:
    """
    Per-file symbol table for import alias resolution.

    Tracks module aliases and symbol aliases to enable proper
    call resolution across import boundaries.
    """
    # alias -> fully qualified module (import numpy as np)
    module_aliases: Dict[str, str] = field(default_factory=dict)

    # alias -> fully qualified symbol (from json import dumps as d)
    symbol_aliases: Dict[str, str] = field(default_factory=dict)

    # Track star imports which make resolution unreliable
    has_star_import: bool = False

    # Current module path for relative import resolution (kept for pipeline usage)
    current_module: str = ""

    def resolve_call(self, raw_call: str, project_roots: Optional[List[str]] = None) -> str:
        """
        Resolve a raw call using the symbol table.

        Args:
            raw_call: Raw call string like 'dumps' or 'np.array'
            project_roots: List of project package roots to normalize

        Returns:
            Normalized call string
        """
        project_roots = project_roots or []

        # 1) Direct symbol aliases: f(...) where f is from-import alias
        if raw_call in self.symbol_aliases:
            return self._normalize_project_path(self.symbol_aliases[raw_call], project_roots)

        # 2) Attribute calls: alias.attr
        if "." in raw_call:
            head, rest = raw_call.split(".", 1)

            # Module alias: np.array -> numpy.array
            if head in self.module_aliases:
                full_path = f"{self.module_aliases[head]}.{rest}"
                return self._normalize_project_path(full_path, project_roots)

            # Symbol alias: from pkg import sub as s; s.func -> pkg.sub.func
            if head in self.symbol_aliases:
                full_path = f"{self.symbol_aliases[head]}.{rest}"
                return self._normalize_project_path(full_path, project_roots)

        # 3) Module alias used as callable (rare)
        if raw_call in self.module_aliases:
            return self._normalize_project_path(self.module_aliases[raw_call], project_roots)

        # Even if we did not resolve anything, still normalize project root if present
        return self._normalize_project_path(raw_call, project_roots)

    def _normalize_project_path(self, path: str, project_roots: List[str]) -> str:
        """Normalize project paths by removing known project roots."""
        for root in project_roots:
            if root and path.startswith(f"{root}."):
                return path[len(root) + 1:]
        return path


class EffortClass(str, Enum):
    """Effort classification for refactoring operations."""
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"


class PatchStepKind(str, Enum):
    """Types of patch operations."""
    ADD_NEW_MODULE = "ADD_NEW_MODULE"
    ADD_WRAPPER = "ADD_WRAPPER"
    UPDATE_IMPORTS = "UPDATE_IMPORTS"
    DEPRECATE = "DEPRECATE"
    DELETE_DEAD = "DELETE_DEAD"


class ApplicationMode(str, Enum):
    """Modes for applying changes."""
    ANALYZE_ONLY = "analyze_only"
    PLAN_ONLY = "plan_only"
    APPLY_SAFE = "apply_safe"
    APPLY_ASSISTED = "apply_assisted"


@dataclass
class FunctionalBlock:
    """
    Atomic unit of analysis at method/function level.

    Represents a single function or method with all its characteristics
    needed for functional decomposition and similarity analysis.
    """
    # Identity (required fields first, id becomes optional and generated)
    module: str
    file_path: str
    qualname: str  # "Class.method" or "function"
    lineno: int
    end_lineno: int

    # Stable ID: f"{module}:{qualname}:{lineno}"
    id: str = ""

    # Categorization
    category: str = ""
    subcategory: str = ""
    tags: List[str] = field(default_factory=list)

    # Interface
    signature: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Dependencies and context
    raw_calls: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    imports_used: List[str] = field(default_factory=list)
    imports_context: List[str] = field(default_factory=list)
    globals_used: List[str] = field(default_factory=list)
    literals: List[str] = field(default_factory=list)
    local_defs: List[str] = field(default_factory=list)
    local_assigned: List[str] = field(default_factory=list)

    # NEW: simple local type inference results: var -> ClassName or module.ClassName
    local_type_hints: Dict[str, str] = field(default_factory=dict)

    # Metrics/fingerprints
    loc: int = 0
    cyclomatic: int = 0
    cognitive: Optional[int] = None
    ast_hash: str = ""
    token_fingerprint: str = ""
    semantic_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            module_part = self.module or Path(self.file_path).stem
            self.id = f"{module_part}:{self.qualname}:{self.lineno}"

    @property
    def is_method(self) -> bool:
        return "." in self.qualname

    @property
    def class_name(self) -> Optional[str]:
        if self.is_method:
            return self.qualname.split(".")[0]
        return None

    @property
    def method_name(self) -> str:
        if self.is_method:
            return self.qualname.split(".")[-1]
        return self.qualname


@dataclass
class Capability:
    """
    Group of blocks that solve the same task (even in different modules).
    """
    name: str
    description: str
    blocks: List[str] = field(default_factory=list)  # FunctionalBlock.id
    owners: List[str] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.blocks)


@dataclass
class SimilarityCluster:
    """
    Cluster of similar blocks within capability/category.
    """
    id: str
    category: str
    subcategory: str

    blocks: List[str] = field(default_factory=list)  # block ids
    similarity: Dict[Tuple[str, str], float] = field(default_factory=dict)
    avg_similarity: float = 0.0

    recommendation: RecommendationType = RecommendationType.KEEP_SEPARATE
    canonical_candidate: str = ""
    proposed_target: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    effort_class: EffortClass = EffortClass.M
    notes: List[str] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    def get_similarity(self, block1: str, block2: str) -> float:
        key = tuple(sorted([block1, block2]))
        return self.similarity.get(key, 0.0)


@dataclass
class PatchStep:
    """
    Minimal applicable step that can be rolled back and validated separately.
    """
    kind: PatchStepKind
    id: str = ""

    files_touched: List[str] = field(default_factory=list)
    description: str = ""
    preconditions: List[str] = field(default_factory=list)
    validations: List[str] = field(default_factory=list)

    # Additional metadata
    target_module: str = ""
    target_symbol: str = ""
    source_blocks: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            # sha256 is stable and avoids md5/OpenSSL/FIPS issues.
            payload = "|".join([
                self.kind.value,
                self.target_module,
                self.target_symbol,
                self.description,
                ",".join(sorted(self.source_blocks)),
            ]).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()[:12]
            self.id = f"{self.kind.value}_{digest}"


@dataclass
class CanonicalizationPlan:
    """
    Plan for canonicalizing a cluster.
    """
    cluster_id: str
    target_module: str
    target_symbol: str

    steps: List[PatchStep] = field(default_factory=list)
    removal_criteria: Dict[str, Any] = field(default_factory=dict)

    estimated_effort: EffortClass = EffortClass.M
    risk_assessment: RiskLevel = RiskLevel.LOW
    dependencies: List[str] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class ProjectFunctionalMap:
    """
    Complete functional map of the project built from atomic blocks.
    """
    project_root: str
    timestamp: str

    # Core data
    blocks: Dict[str, FunctionalBlock] = field(default_factory=dict)
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    clusters: Dict[str, SimilarityCluster] = field(default_factory=dict)

    # Graph data
    call_edges: List[Tuple[str, str]] = field(default_factory=list)
    unresolved_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_blocks: int = 0
    total_capabilities: int = 0
    total_clusters: int = 0

    # Rates are fractions in [0..1], not percents
    resolution_rate: float = 0.0
    resolution_rate_internal: float = 0.0
    resolution_rate_actionable: float = 0.0

    external_calls_count: int = 0
    dynamic_attribute_calls_count: int = 0

    def __post_init__(self) -> None:
        self.recompute_stats()

    def recompute_stats(self) -> None:
        self.total_blocks = len(self.blocks)
        self.total_capabilities = len(self.capabilities)
        self.total_clusters = len(self.clusters)

        total_calls = len(self.call_edges) + len(self.unresolved_calls)
        if total_calls <= 0:
            self.resolution_rate = 0.0
            self.resolution_rate_internal = 0.0
            self.resolution_rate_actionable = 0.0
            self.external_calls_count = 0
            self.dynamic_attribute_calls_count = 0
            return

        self.resolution_rate = len(self.call_edges) / total_calls

        internal_unresolved = [
            c for c in self.unresolved_calls
            if UnresolvedReason.is_internal(as_reason(c.get("reason")))
        ]
        actionable_unresolved = [
            c for c in self.unresolved_calls
            if UnresolvedReason.is_actionable(as_reason(c.get("reason")))
        ]

        self.external_calls_count = sum(
            1 for c in self.unresolved_calls
            if as_reason(c.get("reason")) in {UnresolvedReason.external, UnresolvedReason.star_import}
        )

        internal_total = len(self.call_edges) + len(internal_unresolved)
        self.resolution_rate_internal = (len(self.call_edges) / internal_total) if internal_total > 0 else 0.0

        self.dynamic_attribute_calls_count = sum(
            1 for c in self.unresolved_calls
            if as_reason(c.get("reason")) == UnresolvedReason.dynamic_attribute
        )

        actionable_total = len(self.call_edges) + len(actionable_unresolved)
        self.resolution_rate_actionable = (len(self.call_edges) / actionable_total) if actionable_total > 0 else 0.0

    def get_block(self, block_id: str) -> Optional[FunctionalBlock]:
        return self.blocks.get(block_id)

    def get_blocks_by_category(self, category: str) -> List[FunctionalBlock]:
        return [block for block in self.blocks.values() if block.category == category]

    def get_blocks_by_module(self, module: str) -> List[FunctionalBlock]:
        return [block for block in self.blocks.values() if block.module == module]


@dataclass
class DecompositionConfig:
    """
    Configuration for functional decomposition analysis.
    """
    category_rules: List[Dict[str, Any]] = field(default_factory=list)

    similarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "ast_shape": 0.30,
        "token": 0.20,
        "signature": 0.15,
        "dependency": 0.15,
        "literals": 0.10,
        "name": 0.10,
    })

    merge_threshold: float = 0.85
    extract_threshold: float = 0.70
    separate_threshold: float = 0.70

    max_component_size: int = 100

    extract_nested_functions: bool = False
    exclude_docstrings_from_literals: bool = True

    require_name_similarity_for_merge: bool = True
    min_name_similarity_for_merge: float = 0.6
    exclude_dunder_methods_from_merge: bool = True

    project_package_roots: List[str] = field(default_factory=lambda: ["intellirefactor"])
    enable_symbol_table_resolution: bool = True
    enable_relative_import_resolution: bool = True

    default_mode: ApplicationMode = ApplicationMode.ANALYZE_ONLY

    validation_steps: List[str] = field(default_factory=lambda: [
        "parse_check",
        "import_check",
        "static_arch_check",
        "unit_tests",
        "smoke_scenarios",
    ])

    output_formats: List[str] = field(default_factory=lambda: ["json", "markdown", "mermaid"])

    @classmethod
    def default(cls) -> "DecompositionConfig":
        return cls(
            category_rules=[
                {"match_name": "^parse_", "category": "parsing", "subcategory": "generic"},
                {"match_name": "^validate_", "category": "validation", "subcategory": "generic"},
                {"match_name": "^log_", "category": "telemetry", "subcategory": "logging"},
                {"match_import": "^logging$", "category": "telemetry", "subcategory": "logging"},
                {"match_import": "^re$", "category": "parsing", "subcategory": "regex"},
                {"match_path": r"[\\/](test_|tests?)[\\/]", "category": "testing", "subcategory": "unit"},
                {"match_path": r"[\\/](config|settings)[\\/]", "category": "configuration", "subcategory": "files"},
                {"match_path": r"[\\/]models?[\\/]", "category": "domain", "subcategory": "models"},
                {"match_module": r"\.(test_|tests?)\.", "category": "testing", "subcategory": "unit"},
                {"match_module": r"\.(config|settings)\.", "category": "configuration", "subcategory": "modules"},
                {"match_module": r"\.models?\.", "category": "domain", "subcategory": "models"},
            ]
        )