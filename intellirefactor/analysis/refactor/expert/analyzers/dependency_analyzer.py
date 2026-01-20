"""
Dependency Interface Analyzer for expert refactoring analysis.

Analyzes external dependencies and their interfaces to understand
the "стыки" (boundaries) between modules for safe refactoring.
"""

from __future__ import annotations

import ast
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models import DependencyInterface, InterfaceUsage, RiskLevel

logger = logging.getLogger(__name__)


class DependencyInterfaceAnalyzer:
    """Analyzes interfaces of dependencies and import-level relationships."""

    def __init__(self, project_root: str, target_module: str, enable_introspection: bool = True):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        self.enable_introspection = enable_introspection

        # Cache for import resolution
        self._resolve_cache: Dict[Tuple[str, int, Optional[str]], List[Path]] = {}

    # ---------------------------
    # Public API
    # ---------------------------

    def extract_dependency_interfaces(self, module_ast: ast.Module) -> List[DependencyInterface]:
        """
        Extract interfaces of external dependencies (stdlib + third-party).

        Internal project imports are excluded from "dependency interface" list
        to avoid noise; they are handled separately in import dependency analysis.
        """
        logger.info("Extracting dependency interfaces...")

        imports = self._extract_imports(module_ast)
        parents = self._build_parent_map(module_ast)

        interfaces: List[DependencyInterface] = []
        for imp in imports:
            # Skip internal imports here (analyze them in import graph)
            if self._is_import_internal(imp):
                continue

            interface = self._analyze_dependency_interface(imp, module_ast, parents)
            if interface:
                interfaces.append(interface)

        logger.info("Extracted %d dependency interfaces", len(interfaces))
        return interfaces

    def analyze_import_dependencies(self, module_ast: ast.Module) -> Dict[str, Any]:
        """
        Analyze import-level dependencies and attempt to detect real cycles
        for INTERNAL modules by building a local import graph from files.

        Returns dict with:
          - external_imports
          - internal_imports
          - cycles
        """
        logger.info("Analyzing import dependencies and cycles...")

        imports = self._extract_imports(module_ast)

        internal_imports: List[Dict[str, Any]] = []
        external_imports: List[Dict[str, Any]] = []

        for imp in imports:
            if self._is_import_internal(imp):
                internal_imports.append(imp)
            else:
                external_imports.append(imp)

        cycles = self._detect_internal_import_cycles()

        return {
            "external_imports": [
                {
                    "module": self._import_display_name(imp),
                    "type": imp["type"],
                    "line": imp["line"],
                    "is_standard_library": self._is_standard_library(imp.get("module", "")),
                }
                for imp in external_imports
            ],
            "internal_imports": [
                {
                    "module": self._import_display_name(imp),
                    "type": imp["type"],
                    "line": imp["line"],
                }
                for imp in internal_imports
            ],
            "cycles": cycles,
            "cycle_type_clarification": "These are import-level cycles, different from call-level cycles",
        }

    def extract_external_dependency_contracts(self, module_ast: ast.Module) -> Dict[str, Any]:
        """
        Extract contracts of external dependencies (third-party only; stdlib skipped).
        """
        logger.info("Extracting external dependency contracts...")

        interfaces = self.extract_dependency_interfaces(module_ast)
        contracts: Dict[str, Any] = {}

        for interface in interfaces:
            module_name = interface.module_name
            if self._is_standard_library(module_name):
                continue

            contract_info = self._extract_dependency_contract(interface, module_name)
            if contract_info:
                contracts[module_name] = contract_info

        return {
            "dependency_contracts": contracts,
            "contract_summary": {
                "total_external_dependencies": len(
                    [i for i in interfaces if not self._is_standard_library(i.module_name)]
                ),
                "contracts_extracted": len(contracts),
                "high_risk_dependencies": [
                    name for name, contract in contracts.items() if contract.get("risk_level") == "high"
                ],
            },
            "recommendations": self._generate_contract_recommendations(contracts),
        }

    # ---------------------------
    # Import extraction & classification
    # ---------------------------

    def _extract_imports(self, module_ast: ast.Module) -> List[Dict[str, Any]]:
        """Extract all import statements from the module (no duplicates removal here)."""
        imports: List[Dict[str, Any]] = []

        for node in ast.walk(module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": getattr(node, "lineno", 0),
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                level = getattr(node, "level", 0)

                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module_name,  # module part
                            "name": alias.name,     # imported symbol
                            "alias": alias.asname,
                            "level": level,
                            "line": getattr(node, "lineno", 0),
                        }
                    )

        return imports

    def _import_display_name(self, imp: Dict[str, Any]) -> str:
        if imp["type"] == "import":
            return imp.get("module", "")
        # from_import
        level = imp.get("level", 0) or 0
        dots = "." * level
        mod = imp.get("module", "") or ""
        name = imp.get("name", "") or ""
        if mod:
            return f"{dots}{mod}.{name}"
        return f"{dots}{name}"

    def _is_import_internal(self, import_info: Dict[str, Any]) -> bool:
        """
        Decide whether an import is internal to this project.

        Rules:
        - Any relative import (level > 0) => internal
        - Absolute import is internal if it can be resolved to a file under project_root
        """
        level = int(import_info.get("level", 0) or 0)
        if level > 0:
            return True

        # Absolute import
        resolved = self._resolve_import_to_paths(import_info)
        return len(resolved) > 0 and all(self._is_path_under_project(p) for p in resolved)

    def _is_path_under_project(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.project_root.resolve())
            return True
        except Exception:
            return False

    def _resolve_import_to_paths(self, import_info: Dict[str, Any]) -> List[Path]:
        """
        Resolve import to candidate file paths.

        Returns list of paths (could be empty).
        Cached.
        """
        key = (
            str(import_info.get("module", "")),
            int(import_info.get("level", 0) or 0),
            str(import_info.get("name")) if import_info.get("type") == "from_import" else None,
        )
        if key in self._resolve_cache:
            return self._resolve_cache[key]

        level = int(import_info.get("level", 0) or 0)
        imp_type = import_info["type"]
        module_part = import_info.get("module", "") or ""

        candidates: List[Path] = []

        if imp_type == "import":
            # import a.b.c
            candidates = self._resolve_module_to_file_candidates(module_part)

        else:
            # from X import Y
            if level > 0:
                # relative: base_dir = target_module.parent up N levels
                base_dir = self.target_module.parent
                for _ in range(level - 1):
                    base_dir = base_dir.parent

                if module_part:
                    base_dir = base_dir.joinpath(*module_part.split("."))

                # Try module package/file itself
                candidates.extend(self._resolve_dir_as_module_candidates(base_dir))

                # Try imported name as submodule if exists (from pkg import submodule)
                imported_name = import_info.get("name", "") or ""
                if imported_name:
                    candidates.extend(self._resolve_dir_child_as_module_candidates(base_dir, imported_name))

            else:
                # absolute from_import: from a.b import c
                # often c is attribute, but could be submodule; try both.
                if module_part:
                    candidates.extend(self._resolve_module_to_file_candidates(module_part))

                    imported_name = import_info.get("name", "") or ""
                    if imported_name:
                        # try a/b/c.py
                        candidates.extend(self._resolve_module_to_file_candidates(f"{module_part}.{imported_name}"))
                else:
                    # from import without module part is rare in absolute mode; ignore
                    candidates = []

        # filter existing
        candidates = [p for p in candidates if p.exists()]

        # store
        self._resolve_cache[key] = candidates
        return candidates

    def _resolve_module_to_file_candidates(self, dotted: str) -> List[Path]:
        """
        Convert dotted module path to possible file paths under project_root:
        - <root>/a/b/c.py
        - <root>/a/b/c/__init__.py
        """
        if not dotted:
            return []
        rel = Path(*dotted.split("."))
        return [
            self.project_root / (str(rel) + ".py"),
            self.project_root / rel / "__init__.py",
        ]

    def _resolve_dir_as_module_candidates(self, base_dir: Path) -> List[Path]:
        """Given a directory path, try interpret as package or module file."""
        return [
            base_dir.with_suffix(".py"),
            base_dir / "__init__.py",
        ]

    def _resolve_dir_child_as_module_candidates(self, base_dir: Path, child: str) -> List[Path]:
        """Try resolve base_dir/child as module or package."""
        return [
            base_dir / f"{child}.py",
            base_dir / child / "__init__.py",
        ]

    # ---------------------------
    # Usage analysis (call vs attribute)
    # ---------------------------

    def _build_parent_map(self, root: ast.AST) -> Dict[ast.AST, ast.AST]:
        """Build parent pointers for AST nodes."""
        parents: Dict[ast.AST, ast.AST] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            for child in ast.iter_child_nodes(node):
                parents[child] = node
                stack.append(child)
        return parents

    def _is_call_site(self, node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> bool:
        """
        True if node is used as Call.func (directly).
        Handles Name/Attribute being the callable.
        """
        parent = parents.get(node)
        return isinstance(parent, ast.Call) and parent.func is node

    def _find_usage_in_ast(
        self,
        import_info: Dict[str, Any],
        module_ast: ast.Module,
        parents: Dict[ast.AST, ast.AST],
    ) -> Tuple[List[str], List[str]]:
        """Find actual usage of imported symbols in the AST."""
        used_methods: Set[str] = set()
        used_attributes: Set[str] = set()

        imp_type = import_info["type"]
        module_name = import_info.get("module", "") or ""

        if imp_type == "import":
            alias = import_info.get("alias") or module_name
            m, a = self._find_attribute_usage(module_ast, alias, parents)
            used_methods.update(m)
            used_attributes.update(a)

        else:
            imported_name = import_info.get("name", "") or ""
            alias = import_info.get("alias") or imported_name

            for node in ast.walk(module_ast):
                # Direct usage: alias(...)
                if isinstance(node, ast.Name) and node.id == alias:
                    if self._is_call_site(node, parents):
                        used_methods.add(imported_name)
                    else:
                        used_attributes.add(imported_name)

                # Attribute usage: alias.attr or alias.attr(...)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == alias:
                    full = f"{imported_name}.{node.attr}"
                    if self._is_call_site(node, parents):
                        used_methods.add(full)
                    else:
                        used_attributes.add(full)

        return sorted(used_methods), sorted(used_attributes)

    def _find_attribute_usage(
        self, module_ast: ast.Module, module_alias: str, parents: Dict[ast.AST, ast.AST]
    ) -> Tuple[List[str], List[str]]:
        used_methods: Set[str] = set()
        used_attributes: Set[str] = set()

        for node in ast.walk(module_ast):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == module_alias:
                if self._is_call_site(node, parents):
                    used_methods.add(node.attr)
                else:
                    used_attributes.add(node.attr)

        return sorted(used_methods), sorted(used_attributes)

    # ---------------------------
    # Dependency interface extraction
    # ---------------------------

    def _analyze_dependency_interface(
        self,
        import_info: Dict[str, Any],
        module_ast: ast.Module,
        parents: Dict[ast.AST, ast.AST],
    ) -> Optional[DependencyInterface]:
        module_name = import_info.get("module", "") or ""
        imp_type = import_info["type"]

        # For from_import, dependency is module_part (not imported symbol).
        # If module_part is empty in absolute imports, skip.
        if imp_type == "from_import" and not module_name and (import_info.get("level", 0) or 0) == 0:
            return None

        if self._is_standard_library(module_name):
            criticality = RiskLevel.LOW
        else:
            criticality = RiskLevel.MEDIUM

        used_methods, used_attributes = self._find_usage_in_ast(import_info, module_ast, parents)

        interface_info: Dict[str, Any] = {}
        if self.enable_introspection and module_name:
            interface_info = self._introspect_module_interface(module_name)

        import_style = self._format_import_style(import_info)

        return DependencyInterface(
            module_name=module_name or self._import_display_name(import_info),
            used_methods=used_methods,
            used_attributes=used_attributes,
            import_style=import_style,
            criticality=criticality,
            version_constraints=interface_info.get("version"),
        )

    def _format_import_style(self, import_info: Dict[str, Any]) -> str:
        if import_info["type"] == "import":
            return f"import {import_info.get('module', '')}"
        level = int(import_info.get("level", 0) or 0)
        dots = "." * level
        mod = import_info.get("module", "") or ""
        name = import_info.get("name", "") or ""
        if mod:
            return f"from {dots}{mod} import {name}"
        return f"from {dots} import {name}"

    def _introspect_module_interface(self, module_name: str) -> Dict[str, Any]:
        interface_info: Dict[str, Any] = {}

        try:
            module = importlib.import_module(module_name)

            if hasattr(module, "__version__"):
                interface_info["version"] = getattr(module, "__version__")

            if hasattr(module, "__all__"):
                interface_info["public_interface"] = getattr(module, "__all__")
            else:
                interface_info["public_interface"] = [n for n in dir(module) if not n.startswith("_")]

            doc = getattr(module, "__doc__", None)
            if doc:
                interface_info["docstring"] = doc[:200]

        except ImportError as e:
            logger.warning("Could not import %s: %s", module_name, e)
            interface_info["import_error"] = str(e)
        except Exception as e:
            logger.warning("Error introspecting %s: %s", module_name, e)
            interface_info["introspection_error"] = str(e)

        return interface_info

    # ---------------------------
    # Stdlib detection
    # ---------------------------

    def _is_standard_library(self, module_name: str) -> bool:
        if not module_name:
            return False
        base = module_name.split(".")[0]

        # Best effort: Python 3.10+
        stdlib = getattr(sys, "stdlib_module_names", None)
        if isinstance(stdlib, set):
            return base in stdlib

        # Fallback list (your original list trimmed)
        fallback = {
            "ast", "hashlib", "sqlite3", "logging", "pathlib", "typing",
            "datetime", "dataclasses", "collections", "json", "re",
            "sys", "os", "time", "enum",
        }
        return base in fallback

    # ---------------------------
    # Interface usage / violations
    # ---------------------------

    def analyze_interface_usage(self, interfaces: List[DependencyInterface]) -> InterfaceUsage:
        logger.info("Analyzing interface usage patterns...")

        critical_interfaces: List[DependencyInterface] = []
        unused_imports: List[str] = []
        potential_violations: List[str] = []

        for interface in interfaces:
            total_usage = len(interface.used_methods) + len(interface.used_attributes)

            if total_usage == 0:
                unused_imports.append(interface.module_name)
            elif total_usage > 5 or interface.criticality == RiskLevel.HIGH:
                critical_interfaces.append(interface)

            # Heuristics: private usage
            if interface.module_name.startswith("_"):
                potential_violations.append(f"Using private module: {interface.module_name}")

            for m in interface.used_methods:
                if m.split(".")[-1].startswith("_"):
                    potential_violations.append(f"Using private method: {interface.module_name}.{m}")

            for a in interface.used_attributes:
                if a.split(".")[-1].startswith("_"):
                    potential_violations.append(f"Using private attribute: {interface.module_name}.{a}")

        usage = InterfaceUsage(
            total_dependencies=len(interfaces),
            critical_interfaces=critical_interfaces,
            unused_imports=unused_imports,
            potential_violations=potential_violations,
        )

        logger.info(
            "Interface usage: %d critical, %d unused", len(critical_interfaces), len(unused_imports)
        )
        return usage

    def detect_interface_violations(self, usage: InterfaceUsage) -> List[str]:
        violations = list(usage.potential_violations)

        for interface in usage.critical_interfaces:
            if not interface.version_constraints and not self._is_standard_library(interface.module_name):
                violations.append(f"No version constraint for critical dependency: {interface.module_name}")

            if len(interface.used_methods) > 10:
                violations.append(
                    f"High coupling with {interface.module_name}: {len(interface.used_methods)} methods used"
                )

        return violations

    def suggest_interface_improvements(self, interfaces: List[DependencyInterface]) -> List[str]:
        suggestions: List[str] = []

        usage = self.analyze_interface_usage(interfaces)

        if usage.unused_imports:
            suggestions.append(f"Remove unused imports: {', '.join(sorted(set(usage.unused_imports)))}")

        for interface in usage.critical_interfaces:
            if len(interface.used_methods) > 8:
                suggestions.append(
                    f"Consider creating an adapter/facade for {interface.module_name} "
                    f"(uses {len(interface.used_methods)} methods)"
                )

        for interface in usage.critical_interfaces:
            if (
                not interface.version_constraints
                and not self._is_standard_library(interface.module_name)
            ):
                suggestions.append(f"Pin version for critical dependency: {interface.module_name}")

        return suggestions

    # ---------------------------
    # Import cycle detection (internal graph)
    # ---------------------------

    def _detect_internal_import_cycles(self) -> List[Dict[str, Any]]:
        """
        Build internal import graph by reading internal module files, then detect cycles.

        This avoids the previous "typing <-> typing" noise and repeated pairs.
        """
        # Build graph nodes as file paths relative to project_root
        max_depth = 6
        max_nodes = 500

        start = self.target_module.resolve()
        graph: Dict[Path, Set[Path]] = {}
        seen: Set[Path] = set()

        def parse_file(p: Path) -> Optional[ast.Module]:
            try:
                txt = p.read_text(encoding="utf-8-sig")
                return ast.parse(txt)
            except Exception:
                return None

        def collect_edges(file_path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            if file_path in seen:
                return
            if len(seen) >= max_nodes:
                return

            seen.add(file_path)

            mod_ast = parse_file(file_path)
            if not mod_ast:
                return

            imports = self._extract_imports(mod_ast)
            outs: Set[Path] = set()

            for imp in imports:
                if not self._is_import_internal(imp):
                    continue
                targets = self._resolve_import_to_paths(imp)
                for t in targets:
                    if self._is_path_under_project(t):
                        outs.add(t.resolve())

            graph[file_path] = outs

            for nxt in outs:
                collect_edges(nxt, depth + 1)

        collect_edges(start, 0)

        # DFS cycle detection
        cycles: List[List[Path]] = []
        stack: List[Path] = []
        onstack: Set[Path] = set()
        visited: Set[Path] = set()

        def dfs(v: Path) -> None:
            visited.add(v)
            stack.append(v)
            onstack.add(v)

            for w in graph.get(v, set()):
                if w not in visited:
                    dfs(w)
                elif w in onstack:
                    # found cycle: w ... v -> w
                    idx = stack.index(w)
                    cyc = stack[idx:] + [w]
                    cycles.append(cyc)

            stack.pop()
            onstack.remove(v)

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node)

        # Deduplicate cycles by normalized signature
        uniq: Dict[str, List[Path]] = {}
        for cyc in cycles:
            # represent by relative path strings
            rels = [str(p.relative_to(self.project_root)) for p in cyc if self._is_path_under_project(p)]
            if len(rels) < 2:
                continue
            sig = " -> ".join(rels)
            uniq[sig] = cyc

        out: List[Dict[str, Any]] = []
        for sig, cyc in uniq.items():
            rels = [str(p.relative_to(self.project_root)) for p in cyc if self._is_path_under_project(p)]
            out.append(
                {
                    "modules": rels,
                    "type": "cycle",
                    "description": f"Import cycle detected: {sig}",
                }
            )

        return out

    # ---------------------------
    # Contract extraction (оставил твой подход)
    # ---------------------------

    def _extract_dependency_contract(
        self, interface: DependencyInterface, module_name: str
    ) -> Optional[Dict[str, Any]]:
        contract: Dict[str, Any] = {
            "module_name": module_name,
            "used_methods": interface.used_methods,
            "used_attributes": interface.used_attributes,
            "import_style": interface.import_style,
            "risk_level": "medium",
            "contract_details": {},
        }

        # Example heuristics (оставил, но теперь это реально для third-party будет применяться)
        contract["contract_details"] = {
            "description": f"External dependency: {module_name}",
            "key_methods": interface.used_methods[:20],
            "expected_structure": "Unknown - requires investigation",
            "critical_invariants": [
                "API compatibility must be maintained",
                "Error handling must be preserved",
            ],
        }
        return contract

    def _generate_contract_recommendations(self, contracts: Dict[str, Any]) -> List[str]:
        recommendations: List[str] = []

        high_risk_count = len([c for c in contracts.values() if c.get("risk_level") == "high"])
        if high_risk_count > 0:
            recommendations.append(f"Document and test {high_risk_count} high-risk dependency contracts")

        recommendations.extend(
            [
                "Create integration tests that verify dependency contracts",
                "Document expected behavior of external dependencies",
                "Consider dependency injection for easier testing",
            ]
        )
        return recommendations