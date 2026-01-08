from __future__ import annotations

import ast as python_ast
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ..config import AnalysisConfig
from ..interfaces import BaseFileAnalyzer, GenericAnalysisResult


@dataclass
class MethodInfo:
    """Information about a method or function."""

    name: str
    line_start: int
    line_end: int
    complexity: int
    parameters_count: int
    calls_count: int
    is_public: bool
    responsibility_group: str
    is_async: bool = False


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    line_start: int
    line_end: int
    methods: List[MethodInfo]
    responsibilities_count: int
    is_god_object: bool

    @property
    def public_methods_count(self) -> int:
        return sum(1 for m in self.methods if m.is_public)

    @property
    def total_complexity(self) -> int:
        return sum(m.complexity for m in self.methods)


@dataclass
class NestedDefinitionsSummary:
    nested_classes_count: int
    nested_functions_count: int
    max_nesting_depth: int
    total_nested_complexity: int

    @property
    def total_count(self) -> int:
        return self.nested_classes_count + self.nested_functions_count

    def has_deep_nesting(self, threshold: int = 3) -> bool:
        return self.max_nesting_depth >= threshold


class FileAnalyzer(BaseFileAnalyzer):
    """
    File analyzer for IntelliRefactor.

    Main metrics are computed ONLY from:
    - top-level classes (methods counted, nested defs excluded)
    - top-level functions (nested defs excluded)

    Nested class/function definitions are collected separately and reported,
    without polluting main metrics.
    """

    # Heuristics for nested definitions (kept as constants; can be moved to config later)
    NESTING_DEPTH_WARN_THRESHOLD = 3
    MANY_NESTED_DEFS_THRESHOLD = 10

    def __init__(self, config: Optional[AnalysisConfig] = None):
        # Accept either an AnalysisConfig or a full IntelliRefactorConfig
        if config is None:
            self.config = AnalysisConfig()
        else:
            # If passed the top-level IntelliRefactorConfig, extract analysis_settings
            if hasattr(config, "analysis_settings"):
                self.config = config.analysis_settings
            else:
                self.config = config
        self.logger = logging.getLogger(__name__)

        # Ordered by specificity: earlier groups win on substring match.
        self.responsibility_keywords: Dict[str, List[str]] = {
            "validation": ["validate", "sanitize", "normalize", "clean", "format"],
            "testing": ["test", "verify", "assert", "mock", "stub", "fake", "check"],
            "security": [
                "auth",
                "secure",
                "encrypt",
                "decrypt",
                "hash",
                "token",
                "login",
            ],
            "network": [
                "connect",
                "request",
                "response",
                "http",
                "tcp",
                "socket",
                "fetch",
                "url",
            ],
            "storage": [
                "store",
                "save",
                "load",
                "persist",
                "retrieve",
                "cache",
                "db",
                "repo",
            ],
            "config": ["config", "setting", "option", "parameter", "preference", "env"],
            "logging": ["log", "debug", "info", "warn", "warning", "error", "trace"],
            "monitoring": ["monitor", "track", "measure", "metric", "stat", "observe"],
            "api": ["api", "endpoint", "route", "handler", "controller"],
            "analysis": ["analyze", "parse", "process", "examine", "inspect", "scan"],
            "transformation": [
                "transform",
                "convert",
                "modify",
                "change",
                "update",
                "map",
            ],
            "coordination": [
                "coordinate",
                "orchestrate",
                "manage",
                "control",
                "handle",
                "dispatch",
            ],
            "business": ["business", "domain", "service", "logic", "rule", "calc"],
            "data": ["data", "model", "entity", "record", "schema", "dto"],
            "ui": ["render", "display", "show", "view", "interface", "gui", "print"],
        }

    def analyze_file(
        self,
        file_path: Union[str, Path],
        external_context: Optional[Any] = None,
    ) -> GenericAnalysisResult:
        filepath = Path(file_path)
        analysis_time = datetime.now()

        try:
            content = self._read_file_safe(filepath)
            tree = python_ast.parse(content, filename=str(filepath))

            lines_count = len(content.splitlines())

            imports_count = sum(
                1
                for node in python_ast.walk(tree)
                if isinstance(node, (python_ast.Import, python_ast.ImportFrom))
            )

            # Main (top-level) structure only
            classes: List[ClassInfo] = []
            functions: List[MethodInfo] = []

            for node in getattr(tree, "body", []):
                if isinstance(node, python_ast.ClassDef):
                    classes.append(self._analyze_class(node))
                elif isinstance(node, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
                    functions.append(self._analyze_method(node))

            # Main metrics exclude nested defs by construction
            complexity_score = sum(c.total_complexity for c in classes) + sum(
                f.complexity for f in functions
            )

            # Nested defs: separate, do NOT affect main metrics/issues/priority
            nested_classes, nested_functions, nested_summary = self._collect_nested_definitions(
                tree
            )

            issues = self._identify_issues(
                lines_count, classes, functions, complexity_score, nested_summary
            )
            recommendations = self._generate_recommendations(classes, issues, nested_summary)
            priority = self._calculate_priority(lines_count, classes, len(issues), nested_summary)
            automation_potential = self._calculate_automation_potential(classes, issues)

            classes_data = [
                {
                    "name": cls.name,
                    "line_start": cls.line_start,
                    "line_end": cls.line_end,
                    "methods": [asdict(m) for m in cls.methods],
                    "public_methods_count": cls.public_methods_count,
                    "total_complexity": cls.total_complexity,
                    "responsibilities_count": cls.responsibilities_count,
                    "is_god_object": cls.is_god_object,
                }
                for cls in classes
            ]
            functions_data = [asdict(fn) for fn in functions]

            nested_data = {
                "nested_classes": nested_classes,
                "nested_functions": nested_functions,
                "max_nesting_depth": nested_summary.max_nesting_depth,
                "total_nested_complexity": nested_summary.total_nested_complexity,
                "total_nested_count": nested_summary.total_count,
            }

            legacy_result: Dict[str, Any] = {
                "filepath": str(filepath),
                "lines_count": lines_count,
                "classes": classes_data,
                "functions": functions_data,
                "imports_count": imports_count,
                "complexity_score": complexity_score,
                "issues": issues,
                "recommendations": recommendations,
                "refactoring_priority": priority,
                "automation_potential": automation_potential,
                "nested_definitions": nested_data,
            }

            analysis_data: Dict[str, Any] = {
                "legacy_result": legacy_result,
                **legacy_result,
            }

            # Base metrics: do not fail analysis if base metrics fail
            metrics: Dict[str, Any] = {}
            calc = getattr(self, "calculate_file_metrics", None)
            if callable(calc):
                try:
                    base_metrics = calc(filepath)
                    if isinstance(base_metrics, dict):
                        metrics.update(base_metrics)
                except Exception as e:
                    self.logger.debug(f"Base metrics calculation failed: {e}")

            metrics.update(
                {
                    "lines_count": float(lines_count),
                    "imports_count": float(imports_count),
                    "classes_count": float(len(classes)),
                    "functions_count": float(len(functions)),
                    "complexity_score": float(complexity_score),
                    "refactoring_priority": float(priority),
                    "automation_potential": float(automation_potential),
                    "issues_count": float(len(issues)),
                    # nested (informational only)
                    "nested_classes_count": float(nested_summary.nested_classes_count),
                    "nested_functions_count": float(nested_summary.nested_functions_count),
                    "max_nesting_depth": float(nested_summary.max_nesting_depth),
                    "nested_complexity": float(nested_summary.total_nested_complexity),
                }
            )

            metadata: Dict[str, Any] = {
                "analyzer_version": "3.1.0",
                "file_size_bytes": filepath.stat().st_size,
                "analysis_timestamp": analysis_time.isoformat(),
                "external_context_used": external_context is not None,
                "has_nested_definitions": nested_summary.total_count > 0,
            }

            if external_context is not None:
                self._enrich_with_context(filepath, external_context, analysis_data, metadata)

            return GenericAnalysisResult(
                success=True,
                project_path=str(filepath.parent),
                analysis_type="file_analysis",
                data=analysis_data,
                metrics=metrics,
                issues=issues,
                recommendations=recommendations,
                metadata=metadata,
                timestamp=analysis_time,
            )

        except SyntaxError as e:
            self.logger.error(f"Syntax error in file {filepath}: {e}")
            return self._create_error_result(
                filepath=filepath,
                timestamp=analysis_time,
                error_msg=f"Syntax error at line {getattr(e, 'lineno', '?')}: {getattr(e, 'msg', str(e))}",
                recommendations=["Fix syntax errors before analysis"],
            )
        except Exception as e:
            self.logger.error(f"Error analyzing file {filepath}: {e}", exc_info=True)
            return self._create_error_result(
                filepath=filepath,
                timestamp=analysis_time,
                error_msg=f"Analysis error: {e}",
                recommendations=["Check file syntax and encoding"],
            )

    def _create_error_result(
        self,
        filepath: Path,
        timestamp: datetime,
        error_msg: str,
        recommendations: List[str],
    ) -> GenericAnalysisResult:
        return GenericAnalysisResult(
            success=False,
            project_path=str(filepath.parent),
            analysis_type="file_analysis",
            data={},
            metrics={},
            issues=[error_msg],
            recommendations=recommendations,
            metadata={"error": error_msg},
            timestamp=timestamp,
        )

    def _read_file_safe(self, filepath: Path) -> str:
        try:
            return filepath.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            self.logger.warning(f"UTF-8 decode failed for {filepath}, using replacement")
            return filepath.read_text(encoding="utf-8", errors="replace")

    def _iter_nodes_excluding_nested(self, node: python_ast.AST) -> Iterable[python_ast.AST]:
        """
        Iterate child nodes without descending into nested function/class/lambda definitions.
        Prevents nested defs from affecting parent's complexity/calls.
        """
        for child in python_ast.iter_child_nodes(node):
            yield child
            if not isinstance(
                child,
                (
                    python_ast.FunctionDef,
                    python_ast.AsyncFunctionDef,
                    python_ast.ClassDef,
                    python_ast.Lambda,
                ),
            ):
                yield from self._iter_nodes_excluding_nested(child)

    def _get_node_end(self, node: python_ast.AST, fallback_start: int) -> int:
        end = getattr(node, "end_lineno", None)
        if isinstance(end, int) and end > 0:
            return end

        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            return self._get_node_end(body[-1], fallback_start)

        lineno = getattr(node, "lineno", None)
        if isinstance(lineno, int) and lineno > 0:
            return lineno

        return fallback_start

    def _analyze_class(self, class_node: python_ast.ClassDef) -> ClassInfo:
        methods: List[MethodInfo] = []

        for node in class_node.body:
            if isinstance(node, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
                methods.append(self._analyze_method(node))

        responsibilities = {m.responsibility_group for m in methods}
        total_complexity = sum(m.complexity for m in methods)

        is_god_object = (
            len(methods) > self.config.god_object_threshold
            or len(responsibilities) > self.config.responsibilities_threshold
            or total_complexity > (self.config.complexity_threshold * 10)
        )

        return ClassInfo(
            name=class_node.name,
            line_start=class_node.lineno,
            line_end=self._get_node_end(class_node, class_node.lineno),
            methods=methods,
            responsibilities_count=len(responsibilities),
            is_god_object=is_god_object,
        )

    def _analyze_method(
        self, node: Union[python_ast.FunctionDef, python_ast.AsyncFunctionDef]
    ) -> MethodInfo:
        complexity = self._calculate_complexity(node)
        parameters_count = self._count_parameters(node)
        calls_count = sum(
            1 for n in self._iter_nodes_excluding_nested(node) if isinstance(n, python_ast.Call)
        )

        return MethodInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=self._get_node_end(node, node.lineno),
            complexity=complexity,
            parameters_count=parameters_count,
            calls_count=calls_count,
            is_public=not node.name.startswith("_"),
            responsibility_group=self._determine_responsibility(node.name),
            is_async=isinstance(node, python_ast.AsyncFunctionDef),
        )

    def _calculate_complexity(self, node: python_ast.AST) -> int:
        """Cyclomatic complexity (McCabe-like). Excludes nested defs."""
        complexity = 1
        Match = getattr(python_ast, "Match", None)

        for child in self._iter_nodes_excluding_nested(node):
            if isinstance(
                child,
                (python_ast.If, python_ast.While, python_ast.For, python_ast.AsyncFor),
            ):
                complexity += 1
            elif isinstance(child, python_ast.BoolOp):
                complexity += max(0, len(child.values) - 1)
            elif isinstance(child, python_ast.Try):
                complexity += len(child.handlers)
            elif isinstance(
                child,
                (
                    python_ast.ListComp,
                    python_ast.SetComp,
                    python_ast.DictComp,
                    python_ast.GeneratorExp,
                ),
            ):
                for gen in child.generators:
                    complexity += 1 + len(gen.ifs)
            elif isinstance(child, python_ast.IfExp):
                complexity += 1
            elif Match is not None and isinstance(child, Match):
                complexity += max(0, len(child.cases) - 1)

        return complexity

    def _count_parameters(
        self, node: Union[python_ast.FunctionDef, python_ast.AsyncFunctionDef]
    ) -> int:
        args = node.args
        count = len(args.args) + len(args.kwonlyargs)

        if getattr(args, "vararg", None):
            count += 1
        if getattr(args, "kwarg", None):
            count += 1

        posonly = list(getattr(args, "posonlyargs", []))
        count += len(posonly)

        all_pos = posonly + list(args.args)
        if all_pos and getattr(all_pos[0], "arg", None) in ("self", "cls"):
            count -= 1

        return max(0, count)

    def _determine_responsibility(self, name: str) -> str:
        lower = name.lower()
        for group, keywords in self.responsibility_keywords.items():
            if any(k in lower for k in keywords):
                return group
        return "other"

    # ---------- Nested definitions (robust traversal; does not pollute metrics) ----------

    def _collect_nested_definitions(
        self, tree: python_ast.AST
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], NestedDefinitionsSummary]:
        """
        Collect nested ClassDef and FunctionDef/AsyncFunctionDef definitions.

        Excludes:
        - top-level defs (empty stack)
        - methods (FunctionDef/AsyncFunctionDef directly inside any ClassDef),
          because class methods are covered in class analysis.

        Traverses into blocks (if/for/try/with/...) via NodeVisitor, so it does NOT miss nested defs.
        """
        analyzer = self

        def stack_names(stack: List[python_ast.AST]) -> List[str]:
            parts: List[str] = []
            for s in stack:
                if isinstance(s, python_ast.ClassDef):
                    parts.append(s.name)
                elif isinstance(s, (python_ast.FunctionDef, python_ast.AsyncFunctionDef)):
                    parts.append(s.name)
            return parts

        def qualname(stack: List[python_ast.AST], leaf: str) -> str:
            parts = stack_names(stack)
            parts.append(leaf)
            return ".".join(parts) if parts else leaf

        def parent_qualname(stack: List[python_ast.AST]) -> Optional[str]:
            parts = stack_names(stack)
            return ".".join(parts) if parts else None

        nested_classes: List[Dict[str, Any]] = []
        nested_functions: List[Dict[str, Any]] = []

        max_depth = 0
        total_nested_complexity = 0

        class Collector(python_ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: List[python_ast.AST] = []

            def visit_ClassDef(self, node: python_ast.ClassDef) -> None:
                nonlocal max_depth, total_nested_complexity

                if self.stack:
                    depth = len(self.stack)
                    max_depth = max(max_depth, depth)

                    ci = analyzer._analyze_class(node)
                    total_nested_complexity += ci.total_complexity

                    nested_classes.append(
                        {
                            "kind": "class",
                            "name": node.name,
                            "qualname": qualname(self.stack, node.name),
                            "parent_qualname": parent_qualname(self.stack),
                            "depth": depth,
                            "line_start": node.lineno,
                            "line_end": ci.line_end,
                            "methods_count": len(ci.methods),
                            "public_methods_count": ci.public_methods_count,
                            "total_complexity": ci.total_complexity,
                            "responsibilities_count": ci.responsibilities_count,
                            "is_god_object": ci.is_god_object,
                        }
                    )

                self.stack.append(node)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: python_ast.FunctionDef) -> None:
                self._handle_function(node, kind="function")

            def visit_AsyncFunctionDef(self, node: python_ast.AsyncFunctionDef) -> None:
                self._handle_function(node, kind="async_function")

            def _handle_function(
                self,
                node: Union[python_ast.FunctionDef, python_ast.AsyncFunctionDef],
                kind: str,
            ) -> None:
                nonlocal max_depth, total_nested_complexity

                if self.stack:
                    # exclude methods: function directly inside class body
                    if isinstance(self.stack[-1], python_ast.ClassDef):
                        pass
                    else:
                        depth = len(self.stack)
                        max_depth = max(max_depth, depth)

                        mi = analyzer._analyze_method(node)
                        total_nested_complexity += mi.complexity

                        nested_functions.append(
                            {
                                "kind": kind,
                                "name": node.name,
                                "qualname": qualname(self.stack, node.name),
                                "parent_qualname": parent_qualname(self.stack),
                                "depth": depth,
                                "line_start": node.lineno,
                                "line_end": mi.line_end,
                                "complexity": mi.complexity,
                                "parameters_count": mi.parameters_count,
                                "calls_count": mi.calls_count,
                                "is_public": mi.is_public,
                                "responsibility_group": mi.responsibility_group,
                                "is_async": isinstance(node, python_ast.AsyncFunctionDef),
                            }
                        )

                self.stack.append(node)
                self.generic_visit(node)
                self.stack.pop()

        Collector().visit(tree)

        summary = NestedDefinitionsSummary(
            nested_classes_count=len(nested_classes),
            nested_functions_count=len(nested_functions),
            max_nesting_depth=max_depth,
            total_nested_complexity=total_nested_complexity,
        )
        return nested_classes, nested_functions, summary

    # ---------- Issues / recommendations / scoring ----------

    def _identify_issues(
        self,
        lines_count: int,
        classes: List[ClassInfo],
        functions: List[MethodInfo],
        complexity_score: int,
        nested_summary: NestedDefinitionsSummary,
    ) -> List[str]:
        issues: List[str] = []
        # The FileAnalyzer expects an AnalysisConfig instance. If an IntelliRefactorConfig
        # was incorrectly passed, try to adapt by selecting the analysis settings.
        if hasattr(self.config, "analysis_settings"):
            cfg = self.config.analysis_settings
        else:
            cfg = self.config

        if lines_count > cfg.large_file_threshold * 2:
            issues.append(
                f"Very large file: {lines_count} lines (recommended <{cfg.large_file_threshold})"
            )
        elif lines_count > cfg.large_file_threshold:
            issues.append(
                f"Large file: {lines_count} lines (recommended <{cfg.large_file_threshold})"
            )

        file_complexity_limit = self.config.complexity_threshold * 10
        if complexity_score > file_complexity_limit * 2:
            issues.append(
                f"Critical complexity: {complexity_score} (recommended <{file_complexity_limit})"
            )
        elif complexity_score > file_complexity_limit:
            issues.append(
                f"High complexity: {complexity_score} (recommended <{file_complexity_limit})"
            )

        for cls in classes:
            if cls.is_god_object:
                issues.append(
                    f"God Object: class '{cls.name}' has {cls.public_methods_count} public methods "
                    f"and {cls.responsibilities_count} responsibilities"
                )

        limit = self.config.complexity_threshold
        complex_items: List[str] = []

        for cls in classes:
            for m in cls.methods:
                if m.complexity > limit:
                    complex_items.append(f"{cls.name}.{m.name}")

        for fn in functions:
            if fn.complexity > limit:
                complex_items.append(fn.name)

        if complex_items:
            preview = ", ".join(complex_items[:3]) + ("..." if len(complex_items) > 3 else "")
            issues.append(f"Complex methods/functions ({len(complex_items)}): {preview}")

        # Nested heuristics (informational; do not affect complexity_score)
        if nested_summary.has_deep_nesting(threshold=self.NESTING_DEPTH_WARN_THRESHOLD):
            issues.append(
                f"Deep nesting detected: max depth {nested_summary.max_nesting_depth} "
                f"(recommended <{self.NESTING_DEPTH_WARN_THRESHOLD})"
            )

        if nested_summary.total_count > self.MANY_NESTED_DEFS_THRESHOLD:
            issues.append(
                f"Many nested definitions: {nested_summary.total_count} "
                f"(consider extracting to module level)"
            )

        return issues

    def _generate_recommendations(
        self,
        classes: List[ClassInfo],
        issues: List[str],
        nested_summary: NestedDefinitionsSummary,
    ) -> List[str]:
        recs: List[str] = []
        issue_text = " ".join(issues).lower()

        has_god_object = any(c.is_god_object for c in classes)
        has_large_file = "large file" in issue_text
        has_complexity = "complex" in issue_text
        has_deep_nesting = "deep nesting" in issue_text
        has_many_nested = "many nested" in issue_text

        if has_god_object:
            recs.append(
                "Extract Component: split God Objects into smaller single-responsibility classes"
            )
            recs.append("Introduce Interfaces/Protocols to decouple extracted components")

        if has_large_file:
            recs.append("Split Module: move cohesive classes/functions into separate files")

        if has_complexity:
            recs.append("Extract Method: break down complex logic into helpers")
            recs.append("Replace Conditional with Polymorphism (Strategy) where applicable")

        if has_deep_nesting:
            recs.append(
                "Flatten Structure: move deeply nested classes/functions to module level where possible"
            )

        if has_many_nested:
            recs.append(
                "Extract Nested: promote frequently-used nested definitions to module level"
            )

        if has_god_object or has_complexity:
            recs.append("Add characterization tests before refactoring")

        automation = self._calculate_automation_potential(classes, issues)
        if automation >= 0.7:
            recs.append(
                f"Automation potential is high ({automation:.0%}): consider automated refactoring tools"
            )
        elif automation >= 0.4:
            recs.append(
                f"Automation potential is medium ({automation:.0%}): partial automation is feasible"
            )

        return recs

    def _calculate_priority(
        self,
        lines_count: int,
        classes: List[ClassInfo],
        issues_count: int,
        nested_summary: NestedDefinitionsSummary,
    ) -> int:
        score = 1
        thr = getattr(self.config, "large_file_threshold", 500)

        if lines_count > thr * 4:
            score += 4
        elif lines_count > thr * 2:
            score += 3
        elif lines_count > thr:
            score += 2

        score += min(sum(1 for c in classes if c.is_god_object) * 2, 4)
        score += min(issues_count, 3)

        # small bump for deep nesting
        if nested_summary.has_deep_nesting(threshold=self.NESTING_DEPTH_WARN_THRESHOLD):
            score += 1

        return min(score, 10)

    def _calculate_automation_potential(self, classes: List[ClassInfo], issues: List[str]) -> float:
        potential = 0.0
        issues_text = " ".join(issues).lower()

        if any(c.is_god_object for c in classes):
            potential += 0.4
        if "large file" in issues_text:
            potential += 0.3
        if "complex" in issues_text:
            potential += 0.2
        if any(c.responsibilities_count > 3 for c in classes):
            potential += 0.2

        return min(potential, 1.0)

    # ---------- External context ----------

    def _guess_module_qualnames(self, filepath: Path) -> List[str]:
        p = filepath.with_suffix("")
        parts = list(p.parts)

        if filepath.is_absolute() and parts:
            anchor = filepath.anchor
            if parts[0] == anchor or parts[0] == "/":
                parts = parts[1:]

        candidates: List[str] = []
        for n in range(1, min(6, len(parts)) + 1):
            candidates.append(".".join(parts[-n:]))

        if parts:
            candidates.append(".".join(parts))

        seen = set()
        out: List[str] = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _enrich_with_context(
        self,
        filepath: Path,
        context: Any,
        data: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> None:
        LazyProjectContext = None
        try:
            from .lazy_loader import LazyProjectContext as _LPC  # optional

            LazyProjectContext = _LPC
        except Exception:
            LazyProjectContext = None

        importers: Any = None

        finder = getattr(context, "find_importers_of_module", None)
        if callable(finder):
            for key in [str(filepath), *self._guess_module_qualnames(filepath)]:
                try:
                    importers = finder(key)
                    if importers:
                        break
                except Exception:
                    continue

        if (not importers) and LazyProjectContext is not None:
            try:
                lazy = LazyProjectContext(filepath.parent, context)
                finder2 = getattr(lazy, "find_importers_of_module", None)
                if callable(finder2):
                    importers = finder2(str(filepath))
            except Exception:
                importers = None

        paths: List[str] = []
        if isinstance(importers, list):
            for imp in importers:
                if isinstance(imp, dict):
                    p = imp.get("importer_path")
                    if isinstance(p, str) and p:
                        paths.append(p)
                elif isinstance(imp, str) and imp:
                    paths.append(imp)

        data["has_external_usage"] = bool(paths)
        if paths:
            data["importers"] = paths
            metadata["importer_count"] = len(paths)

    # ---------- Report ----------

    def generate_detailed_report(self, result: GenericAnalysisResult) -> str:
        if not result.success:
            err = (result.metadata or {}).get("error", "Unknown error")
            return f"# Analysis Failed\n\n**Error**: {err}"

        data = result.data.get("legacy_result", result.data)
        classes = data.get("classes", [])
        functions = data.get("functions", [])
        nested_data = data.get("nested_definitions", {}) or {}

        priority = data.get("refactoring_priority", 0)
        automation = data.get("automation_potential", 0.0)
        lines_count = data.get("lines_count", 0)
        complexity_score = data.get("complexity_score", 0)

        large_thr = getattr(self.config, "large_file_threshold", 500)
        complexity_thr = getattr(self.config, "complexity_threshold", 15.0) * 10

        def status(value: float, warn: float, crit: float) -> str:
            if value > crit:
                return "🔴"
            if value > warn:
                return "🟡"
            return "🟢"

        report: List[str] = [
            "# Detailed File Analysis",
            "",
            f"**File**: `{Path(data.get('filepath', '')).name}`",
            f"**Date**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overview Metrics",
            "",
            "| Metric | Value | Threshold | Status |",
            "|--------|------:|----------:|:------:|",
            f"| Lines of Code | {lines_count:,} | {large_thr} | {status(lines_count, large_thr, large_thr * 2)} |",
            f"| Total Complexity | {complexity_score} | {complexity_thr} | {status(complexity_score, complexity_thr, complexity_thr * 2)} |",
            f"| Refactoring Priority | {priority}/10 | 5 | {status(priority, 4, 7)} |",
            f"| Automation Potential | {automation:.1%} | - | {'🟢' if automation > 0.5 else '🟡' if automation > 0.3 else '⚪'} |",
            "",
            "## Structure Summary",
            f"- **Top-level Classes**: {len(classes)}",
            f"- **Top-level Functions**: {len(functions)}",
            f"- **Imports**: {data.get('imports_count', 0)}",
        ]

        if data.get("has_external_usage"):
            importers = data.get("importers", [])
            report.append(f"- **External Usage**: Yes ({len(importers)} importer(s))")
        elif "has_external_usage" in data:
            report.append("- **External Usage**: No (potentially unused)")

        if classes:
            report.extend(["", "### Classes"])
            for cls in classes:
                icon = "🔴" if cls.get("is_god_object") else "🟢"
                report.append(
                    f"- {icon} **{cls.get('name', 'Unknown')}** "
                    f"(lines {cls.get('line_start', '?')}-{cls.get('line_end', '?')}): "
                    f"{len(cls.get('methods', []))} methods, "
                    f"complexity {cls.get('total_complexity', 0)}, "
                    f"{cls.get('responsibilities_count', 0)} responsibilities"
                )

                # Optional: group methods by responsibility for God Objects
                if cls.get("is_god_object") and cls.get("methods"):
                    groups: Dict[str, List[str]] = defaultdict(list)
                    for m in cls.get("methods", []):
                        groups[m.get("responsibility_group", "other")].append(m.get("name", "?"))

                    for group, names in sorted(groups.items()):
                        preview = ", ".join(names[:3]) + ("..." if len(names) > 3 else "")
                        report.append(f"  - {group}: {preview}")

        if functions:
            report.extend(["", "### Top-level Functions"])
            for fn in functions:
                cplx = fn.get("complexity", 0)
                icon = "🔴" if cplx > self.config.complexity_threshold else "🟢"
                async_marker = "async " if fn.get("is_async") else ""
                report.append(
                    f"- {icon} **{async_marker}{fn.get('name', 'Unknown')}**: complexity {cplx}"
                )

        # Nested section
        nested_classes = nested_data.get("nested_classes", []) or []
        nested_functions = nested_data.get("nested_functions", []) or []
        report.extend(["", "## Nested Definitions"])

        if nested_classes or nested_functions:
            report.extend(
                [
                    f"- Nested classes: {len(nested_classes)}",
                    f"- Nested functions: {len(nested_functions)}",
                    f"- Max nesting depth: {nested_data.get('max_nesting_depth', 0)}",
                    f"- Total nested complexity: {nested_data.get('total_nested_complexity', 0)}",
                    "",
                    "| Kind | Qualname | Lines | Complexity | Params | Calls |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )

            limit = 40
            items: List[Dict[str, Any]] = []
            items.extend(nested_classes)
            items.extend(nested_functions)
            # stable-ish order: by line_start then depth
            items.sort(key=lambda d: (d.get("line_start", 10**9), d.get("depth", 0)))

            for d in items[:limit]:
                kind = d.get("kind", "?")
                qn = d.get("qualname", d.get("name", "?"))
                ls, le = d.get("line_start", "?"), d.get("line_end", "?")

                if kind == "class":
                    cplx = d.get("total_complexity", 0)
                    params = "-"
                    calls = "-"
                else:
                    cplx = d.get("complexity", 0)
                    params = d.get("parameters_count", 0)
                    calls = d.get("calls_count", 0)

                report.append(f"| {kind} | `{qn}` | {ls}-{le} | {cplx} | {params} | {calls} |")

            if len(items) > limit:
                report.append(f"\n*Showing first {limit} nested definitions out of {len(items)}.*")
        else:
            report.append("*No nested definitions found*")

        report.extend(["", "## Issues"])
        if result.issues:
            report.extend([f"- ⚠️ {i}" for i in result.issues])
        else:
            report.append("*No issues found*")

        if result.recommendations:
            report.extend(["", "## Recommendations"])
            report.extend([f"- 💡 {r}" for r in result.recommendations])

        if priority > 6:
            report.extend(
                [
                    "",
                    "## Action Plan (High Priority)",
                    "### Stage 1: Preparation",
                    "1. Create backup of the file",
                    "2. Ensure tests exist (add characterization tests if needed)",
                    "3. Commit current state to version control",
                    "",
                    "### Stage 2: Refactoring",
                    f"- Automation potential: {automation:.1%}",
                    "",
                    "### Stage 3: Validation",
                    "1. Run all tests",
                    "2. Check for performance regressions",
                    "3. Verify backward compatibility",
                ]
            )

        report.extend(["", "---", "*Report generated by FileAnalyzer v3.1.0*"])
        return "\n".join(report)
