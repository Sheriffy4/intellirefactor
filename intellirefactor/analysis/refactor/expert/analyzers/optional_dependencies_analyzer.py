"""
Optional Dependencies Analyzer for expert refactoring analysis.

Purpose:
- Identify optional dependencies in Python modules:
  - conditional imports (try/except ImportError|ModuleNotFoundError)
  - if TYPE_CHECKING guarded imports
  - importlib-based availability checks (find_spec)
- Identify real "feature flags":
  - module-level configuration variables (usually bool/int/str constants)
  - used in conditional branches (if/elif)
- Build execution modes and dual-mode test scenario suggestions.

Notes:
- This analyzer intentionally avoids treating computed per-symbol attributes
  like `is_static`, `is_classmethod` etc. as "feature flags".
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


_IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError"}


@dataclass(frozen=True)
class _ImportSymbol:
    name: str
    asname: Optional[str] = None


def _safe_unparse(node: ast.AST) -> str:
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
    except Exception:
        pass
    return "<expr>"


def _is_type_checking_guard(test: ast.expr) -> bool:
    """
    Detect `if TYPE_CHECKING:` or `if typing.TYPE_CHECKING:`.
    """
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _collect_module_level_assignments(module_ast: ast.Module) -> List[ast.Assign]:
    return [n for n in module_ast.body if isinstance(n, ast.Assign)]


def _collect_module_level_annassign(module_ast: ast.Module) -> List[ast.AnnAssign]:
    return [n for n in module_ast.body if isinstance(n, ast.AnnAssign)]


def _is_constant_like(node: ast.AST) -> bool:
    # constants / simple literals / Name constants
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return True
    # simple unary ops: -1, +1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        return True
    return False


def _extract_constant_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        if isinstance(node.op, ast.USub):
            try:
                return -node.operand.value
            except Exception:
                return None
        if isinstance(node.op, ast.UAdd):
            return node.operand.value
    return None

def _root_module_name(module_name: str) -> str:
    """
    Normalize "a.b.c" -> "a".
    """
    s = (module_name or "").strip()
    if not s:
        return ""
    return s.split(".", 1)[0]

class OptionalDependenciesAnalyzer:
    """Analyzes optional dependencies and module-level feature flags."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_optional_dependencies(self, module_ast: ast.Module) -> Dict[str, Any]:
        logger.info("Analyzing optional dependencies and feature flags...")

        conditional_imports = self._find_conditional_imports(module_ast)
        feature_flags = self._find_feature_flags(module_ast)
        optional_modules = self._derive_optional_modules(conditional_imports, module_ast)

        dependency_branches = self._map_dependency_branches(
            module_ast, conditional_imports, feature_flags
        )
        test_scenarios = self._generate_test_scenarios(
            conditional_imports, feature_flags, optional_modules
        )

        return {
            "conditional_imports": conditional_imports,
            "feature_flags": feature_flags,
            "optional_modules": optional_modules,
            "dependency_branches": dependency_branches,
            "test_scenarios": test_scenarios,
        }

    def export_detailed_optional_dependencies(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # feature flags map
        feature_flags_map: Dict[str, Any] = {}
        for flag in analysis.get("feature_flags", []):
            feature_flags_map[flag["name"]] = {
                "type": flag["type"],
                "default_value": flag["default_value"],
                "dependent_methods": flag["dependent_methods"],
                "dependent_branches": flag["dependent_branches"],
                "usage_locations": flag["usage_locations"],
            }

        # module availability map derived from optional_modules
        module_availability_map: Dict[str, Any] = {}
        for module in analysis.get("optional_modules", []):
            module_availability_map[module["name"]] = {
                "import_method": module.get("import_method", "unknown"),
                "fallback_behavior": module.get("fallback_behavior", "unknown"),
                "dependent_methods": module.get("dependent_methods", []),
                "usage_patterns": module.get("usage_patterns", []),
            }

        dual_mode_tests = []
        for scenario in analysis.get("test_scenarios", []):
            dual_mode_tests.append(
                {
                    "scenario_name": scenario["name"],
                    "available_mode": {
                        "setup": scenario["available_setup"],
                        "expected_behavior": scenario["available_behavior"],
                        "test_methods": scenario["available_tests"],
                    },
                    "unavailable_mode": {
                        "setup": scenario["unavailable_setup"],
                        "expected_behavior": scenario["unavailable_behavior"],
                        "test_methods": scenario["unavailable_tests"],
                    },
                }
            )

        return {
            "feature_flags_map": feature_flags_map,
            "module_availability_map": module_availability_map,
            "dual_mode_test_plans": dual_mode_tests,
            "execution_modes": self._identify_execution_modes(analysis),
            "recommendations": self._generate_optional_dependency_recommendations(analysis),
        }

    # ---------------------------------------------------------------------
    # Core extractors
    # ---------------------------------------------------------------------

    def _find_conditional_imports(self, module_ast: ast.Module) -> List[Dict[str, Any]]:
        """
        Find conditional imports of forms:
          - try: import X ... except ImportError: ...
          - if TYPE_CHECKING: import X
          - if <condition>: import X
        """
        results: List[Dict[str, Any]] = []

        for node in ast.walk(module_ast):
            # try/except conditional imports
            if isinstance(node, ast.Try):
                imports_in_try = [s for s in node.body if isinstance(s, (ast.Import, ast.ImportFrom))]
                if not imports_in_try:
                    continue

                except_names = self._except_handler_type_names(node.handlers)
                is_import_optional = bool(except_names & _IMPORT_ERROR_NAMES)

                for stmt in imports_in_try:
                    info = self._extract_import_info(stmt)
                    if not info:
                        continue
                    info.update(
                        {
                            "condition_type": "try_except",
                            "line": getattr(stmt, "lineno", getattr(node, "lineno", 0)),
                            "except_types": sorted(except_names),
                            "is_optional_dependency": is_import_optional,
                            "fallback_behavior": self._extract_except_behavior(node.handlers),
                        }
                    )
                    results.append(info)

            # if-based imports
            if isinstance(node, ast.If):
                imports_in_body = [s for s in node.body if isinstance(s, (ast.Import, ast.ImportFrom))]
                if not imports_in_body:
                    continue

                condition_str = _safe_unparse(node.test)
                condition_type = "type_checking" if _is_type_checking_guard(node.test) else "if_statement"

                for stmt in imports_in_body:
                    info = self._extract_import_info(stmt)
                    if not info:
                        continue
                    info.update(
                        {
                            "condition_type": condition_type,
                            "line": getattr(stmt, "lineno", getattr(node, "lineno", 0)),
                            "condition": condition_str,
                            "is_optional_dependency": condition_type != "type_checking",
                        }
                    )
                    results.append(info)

        return results

    def _find_feature_flags(self, module_ast: ast.Module) -> List[Dict[str, Any]]:
        """
        Feature flags here are *module-level* variables that look like configuration:
          - NAME = True/False/0/"x"
          - NAME: bool = True
        And are referenced in `if` tests somewhere.
        """
        candidates: Dict[str, Tuple[Any, int, str]] = {}  # name -> (value, line, type)
        flags: List[Dict[str, Any]] = []

        # module-level Assign
        for node in _collect_module_level_assignments(module_ast):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not _is_constant_like(node.value):
                continue
            name = target.id
            if not self._is_likely_feature_flag(name):
                continue
            val = _extract_constant_value(node.value)
            if val is None and not (isinstance(node.value, ast.Name) and node.value.id == "None"):
                # ignore unknown constants
                continue
            typ = self._determine_flag_type(node.value)
            candidates[name] = (val, getattr(node, "lineno", 0), typ)

        # module-level AnnAssign
        for node in _collect_module_level_annassign(module_ast):
            if not isinstance(node.target, ast.Name):
                continue
            if node.value is None or not _is_constant_like(node.value):
                continue
            name = node.target.id
            if not self._is_likely_feature_flag(name):
                continue
            val = _extract_constant_value(node.value)
            typ = self._determine_flag_type(node.value)
            candidates[name] = (val, getattr(node, "lineno", 0), typ)

        if not candidates:
            return []

        # keep only those that are used in conditions or referenced in module at all
        referenced = self._find_name_references(module_ast, set(candidates.keys()))

        for name, (val, line, typ) in candidates.items():
            if name not in referenced:
                continue

            flags.append(
                {
                    "name": name,
                    "type": typ,
                    "default_value": val,
                    "line": line,
                    "dependent_methods": self._find_flag_usage_in_functions(module_ast, name),
                    "dependent_branches": self._find_flag_branches(module_ast, name),
                    "usage_locations": self._find_name_usage_locations(module_ast, name),
                }
            )

        return flags

    def _derive_optional_modules(self, conditional_imports: List[Dict[str, Any]], module_ast: ast.Module) -> List[Dict[str, Any]]:
        """
        Build optional_modules list from conditional imports and importlib-based checks.
        """
        optional: Dict[str, Dict[str, Any]] = {}

        # 1) From conditional imports
        for imp in conditional_imports:
            mod_full = imp.get("module_name_full") or imp.get("module_name") or ""
            mod_root = imp.get("root_module") or _root_module_name(mod_full)
            if not mod_root:
                continue
            if not imp.get("is_optional_dependency", False):
                # TYPE_CHECKING imports are not runtime optionals
                continue

            entry = optional.setdefault(
                mod_root,
                {
                    "name": mod_root,
                    "full_imports": [],
                    "is_used": True,
                    "import_method": imp.get("condition_type", "conditional_import"),
                    "fallback_behavior": imp.get("fallback_behavior", "unknown"),
                    "dependent_methods": [],
                    "usage_patterns": [],
                },
            )
            if mod_full and mod_full not in entry["full_imports"]:
                entry["full_imports"].append(mod_full)

        # 2) importlib.util.find_spec("X") pattern
        for node in ast.walk(module_ast):
            if not isinstance(node, ast.Call):
                continue
            # match importlib.util.find_spec("x")
            func = node.func
            func_str = _safe_unparse(func)
            if "find_spec" not in func_str:
                continue
            if not node.args:
                continue
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                mod_full = node.args[0].value
                mod_root = _root_module_name(mod_full)
                entry = optional.setdefault(
                    mod_root,
                    {
                        "name": mod_root,
                        "full_imports": [],
                        "is_used": True,
                        "import_method": "importlib_find_spec",
                        "fallback_behavior": "checked via importlib.util.find_spec",
                        "dependent_methods": [],
                        "usage_patterns": [],
                    },
                )
                if mod_full and mod_full not in entry["full_imports"]:
                    entry["full_imports"].append(mod_full)
                entry["usage_patterns"].append(
                    {"line": getattr(node, "lineno", 0), "context": f"importlib.util.find_spec({mod_full!r})"}
                )

        # Fill dependent methods for optional modules
        for mod_root in list(optional.keys()):
            # Best-effort: search by root module string (availability is at package level).
            optional[mod_root]["dependent_methods"] = self._find_methods_referencing_module(module_ast, mod_root)

        return list(optional.values())

    def _map_dependency_branches(
        self,
        module_ast: ast.Module,
        conditional_imports: List[Dict[str, Any]],
        feature_flags: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        branches: List[Dict[str, Any]] = []

        for imp in conditional_imports:
            dep_full = imp.get("module_name_full") or imp.get("module_name", "") or ""
            dep_root = imp.get("root_module") or _root_module_name(dep_full)
            branches.append(
                {
                    "type": "conditional_import",
                    "dependency": dep_root,
                    "dependency_full": dep_full,
                    "condition": imp.get("condition", "import_available"),
                    "affected_methods": self._find_methods_using_symbols(module_ast, imp.get("symbols", [])),
                    "branch_complexity": self._calculate_branch_complexity(module_ast, imp.get("symbols", [])),
                    "is_optional_dependency": imp.get("is_optional_dependency", False),
                }
            )

        for flag in feature_flags:
            branches.append(
                {
                    "type": "feature_flag",
                    "dependency": flag["name"],
                    "condition": f"{flag['name']} == {flag['default_value']}",
                    "affected_methods": flag.get("dependent_methods", []),
                    "branch_complexity": len(flag.get("dependent_branches", [])),
                }
            )

        return branches

    def _generate_test_scenarios(
        self,
        conditional_imports: List[Dict[str, Any]],
        feature_flags: List[Dict[str, Any]],
        optional_modules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scenarios: List[Dict[str, Any]] = []

        seen: Set[str] = set()

        for imp in conditional_imports:
            mod_full = imp.get("module_name_full") or imp.get("module_name", "") or ""
            mod_root = imp.get("root_module") or _root_module_name(mod_full)
            if not mod_root or mod_root in seen:
                continue
            seen.add(mod_root)

            if not imp.get("is_optional_dependency", False):
                # TYPE_CHECKING or other non-runtime condition
                continue

            scenarios.append(
                {
                    "name": f"test_{mod_root}_availability",
                    "available_setup": f"Ensure import {mod_root} succeeds",
                    "available_behavior": f"Uses {mod_root} code path",
                    "available_tests": [f"test_{mod_root}_available"],
                    "unavailable_setup": f"Mock import of {mod_root} to raise ImportError",
                    "unavailable_behavior": imp.get("fallback_behavior", "Uses fallback behavior"),
                    "unavailable_tests": [f"test_{mod_root}_unavailable"],
                }
            )

        for flag in feature_flags:
            if flag.get("type") != "boolean":
                continue
            name = flag["name"]
            scenarios.append(
                {
                    "name": f"test_{name}_modes",
                    "available_setup": f"Set {name} = True",
                    "available_behavior": f"Enabled behavior for {name}",
                    "available_tests": [f"test_{name}_enabled"],
                    "unavailable_setup": f"Set {name} = False",
                    "unavailable_behavior": f"Disabled behavior for {name}",
                    "unavailable_tests": [f"test_{name}_disabled"],
                }
            )

        # optional_modules not already covered by conditional imports (importlib checks)
        for mod in optional_modules:
            name = mod.get("name", "")
            if not name or name in seen:
                continue
            scenarios.append(
                {
                    "name": f"test_{name}_availability",
                    "available_setup": f"Ensure {name} is installed / find_spec returns not None",
                    "available_behavior": f"Uses {name} related behavior",
                    "available_tests": [f"test_{name}_available"],
                    "unavailable_setup": f"Simulate {name} missing (find_spec None / import error)",
                    "unavailable_behavior": mod.get("fallback_behavior", "Uses fallback behavior"),
                    "unavailable_tests": [f"test_{name}_unavailable"],
                }
            )

        return scenarios

    def _identify_execution_modes(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        modes: List[Dict[str, Any]] = []

        modes.append(
            {
                "name": "basic_mode",
                "description": "Core functionality without optional deps",
                "required_setup": "Optional deps unavailable; feature flags default",
                "expected_behavior": "Core behavior works with graceful degradation",
            }
        )

        if analysis.get("optional_modules") or analysis.get("conditional_imports"):
            modes.append(
                {
                    "name": "full_mode",
                    "description": "All optional dependencies installed and enabled",
                    "required_setup": "Optional deps available; enable feature flags",
                    "expected_behavior": "All optional code paths available",
                }
            )

        flags = [f for f in analysis.get("feature_flags", []) if f.get("type") == "boolean"]
        if len(flags) >= 2:
            modes.append(
                {
                    "name": "mixed_mode",
                    "description": "Subset of optionals and/or flags enabled",
                    "required_setup": "Enable/disable a subset of feature flags or deps",
                    "expected_behavior": "Gracefully handles partial availability",
                }
            )

        return modes

    def _generate_optional_dependency_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recs: List[str] = []

        conditional_imports = [i for i in analysis.get("conditional_imports", []) if i.get("is_optional_dependency", False)]
        feature_flags = analysis.get("feature_flags", [])
        optional_modules = analysis.get("optional_modules", [])

        if conditional_imports:
            recs.append(
                f"Test {len(conditional_imports)} runtime-conditional imports in both available/unavailable states"
            )
        if feature_flags:
            bool_flags = [f for f in feature_flags if f.get("type") == "boolean"]
            if bool_flags:
                recs.append(f"Create test matrix for {len(bool_flags)} boolean feature flags (at least pairwise)")
        if optional_modules:
            recs.append(f"Verify graceful degradation when optional modules are unavailable ({len(optional_modules)})")

        if len(feature_flags) > 6:
            recs.append("Too many module-level flags detected; consider consolidating configuration")

        if len(conditional_imports) > 8:
            recs.append("Many conditional imports; consider dependency injection or adapter abstraction")

        return recs

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _except_handler_type_names(self, handlers: Sequence[ast.ExceptHandler]) -> Set[str]:
        names: Set[str] = set()
        for h in handlers:
            t = h.type
            if t is None:
                continue
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, ast.Attribute):
                names.add(t.attr)
            elif isinstance(t, ast.Tuple):
                for elt in t.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
                    elif isinstance(elt, ast.Attribute):
                        names.add(elt.attr)
        return names

    def _extract_import_info(self, import_node: ast.stmt) -> Dict[str, Any]:
        if isinstance(import_node, ast.Import):
            syms = [_ImportSymbol(a.name, a.asname) for a in import_node.names]
            module_full = syms[0].name if syms else ""
            module_root = _root_module_name(module_full)
            return {
                "module_name": module_root,
                "module_name_full": module_full,
                "root_module": module_root,
                "symbols": [s.asname or s.name for s in syms],
                "import_type": "import",
            }
        if isinstance(import_node, ast.ImportFrom):
            syms = [_ImportSymbol(a.name, a.asname) for a in import_node.names]
            module_full = import_node.module or ""
            module_root = _root_module_name(module_full)
            return {
                # keep module_name as root for consistency with import normalization
                "module_name": module_root,
                "module_name_full": module_full,
                "root_module": module_root,
                "symbols": [s.asname or s.name for s in syms],
                "import_type": "from_import",
                "level": getattr(import_node, "level", 0),
            }
        return {}

    def _extract_except_behavior(self, handlers: List[ast.ExceptHandler]) -> str:
        if not handlers:
            return "No fallback behavior"

        for handler in handlers:
            # very small heuristic: look at first meaningful stmt
            for stmt in handler.body:
                if isinstance(stmt, ast.Assign):
                    return "Assigns fallback value"
                if isinstance(stmt, ast.Raise):
                    return "Re-raises error"
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    return "Calls fallback function"
                if isinstance(stmt, ast.Pass):
                    return "Pass (silent fallback)"
        return "Has fallback behavior"

    def _is_likely_feature_flag(self, var_name: str) -> bool:
        """
        Only treat as feature flag if it looks like config.
        Avoid grabbing ubiquitous local names by requiring strong indicators.
        """
        indicators = [
            "enable", "disable", "use_", "has_", "allow_", "debug", "verbose", "strict",
            "feature", "experimental", "opt_", "flag", "fallback", "compat",
        ]
        lower = var_name.lower()
        return any(ind in lower for ind in indicators)

    def _determine_flag_type(self, value_node: ast.AST) -> str:
        val = _extract_constant_value(value_node)
        if isinstance(val, bool):
            return "boolean"
        if isinstance(val, str):
            return "string"
        if isinstance(val, (int, float)):
            return "numeric"
        if val is None:
            return "optional"
        return "unknown"

    def _find_name_references(self, module_ast: ast.Module, names: Set[str]) -> Set[str]:
        seen: Set[str] = set()
        for n in ast.walk(module_ast):
            if isinstance(n, ast.Name) and n.id in names:
                seen.add(n.id)
        return seen

    def _find_flag_usage_in_functions(self, module_ast: ast.Module, flag_name: str) -> List[str]:
        using: List[str] = []
        for node in module_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(child, ast.Name) and child.id == flag_name for child in ast.walk(node)):
                    using.append(node.name)
        return using

    def _find_flag_branches(self, module_ast: ast.Module, flag_name: str) -> List[Dict[str, Any]]:
        branches: List[Dict[str, Any]] = []
        for node in ast.walk(module_ast):
            if isinstance(node, ast.If):
                cond = _safe_unparse(node.test)
                if flag_name in cond:
                    branches.append(
                        {
                            "line": getattr(node, "lineno", 0),
                            "condition": cond,
                            "branch_size": len(node.body),
                            "has_else": bool(node.orelse),
                        }
                    )
        return branches

    def _find_name_usage_locations(self, module_ast: ast.Module, name: str) -> List[Dict[str, Any]]:
        locs: List[Dict[str, Any]] = []
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Name) and node.id == name:
                locs.append({"line": getattr(node, "lineno", 0), "context": f"Name reference: {name}"})
        return locs

    def _find_methods_using_symbols(self, module_ast: ast.Module, symbols: List[str]) -> List[str]:
        using: List[str] = []
        symset = set(symbols)
        for node in module_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(isinstance(child, ast.Name) and child.id in symset for child in ast.walk(node)):
                    using.append(node.name)
        return using

    def _calculate_branch_complexity(self, module_ast: ast.Module, symbols: List[str]) -> int:
        symset = set(symbols)
        complexity = 0
        for node in ast.walk(module_ast):
            if isinstance(node, ast.If):
                if any(isinstance(child, ast.Name) and child.id in symset for child in ast.walk(node)):
                    complexity += 1
        return complexity

    def _find_methods_referencing_module(self, module_ast: ast.Module, module_name: str) -> List[str]:
        """
        Approx: search for Name/Attribute usage that matches module prefix.
        E.g. `import numpy as np` would not be resolved here — this analyzer is best-effort.
        """
        hits: List[str] = []
        for node in module_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                text = _safe_unparse(node)
                if module_name in text:
                    hits.append(node.name)
        return hits