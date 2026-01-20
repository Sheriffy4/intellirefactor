"""
Test Quality Analyzer for expert refactoring analysis.

Fixes vs previous version:
- Correct parent detection (test functions vs class methods)
- Better assertion detection: pytest.raises, unittest asserts, pytest.fail
- Fix distribution counters (count tests, not files)
- Less misleading "edge case" penalty: now it's a positive indicator rather than a penalty
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _ParentMapBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.parents: Dict[int, ast.AST] = {}

    def generic_visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            self.parents[id(child)] = node
        super().generic_visit(node)


def _get_parent_chain(parents: Dict[int, ast.AST], node: ast.AST) -> List[ast.AST]:
    chain: List[ast.AST] = []
    cur = node
    while id(cur) in parents:
        cur = parents[id(cur)]
        chain.append(cur)
    return chain


def _is_inside_class(parents: Dict[int, ast.AST], node: ast.AST) -> bool:
    return any(isinstance(p, ast.ClassDef) for p in _get_parent_chain(parents, node))


def _safe_unparse(node: ast.AST) -> str:
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
    except Exception:
        pass
    return "<expr>"


class TestQualityAnalyzer:
    """Analyzes test quality and identifies signal vs noise."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_test_quality(self, test_files: List[str]) -> Dict[str, Any]:
        logger.info("Analyzing test quality and signal vs noise...")

        quality_analysis: Dict[str, Any] = {
            "overall_score": 0.0,
            "file_analyses": [],
            "quality_issues": [],
            "signal_vs_noise": {},
            "recommendations": [],
        }

        total_score = 0.0
        total_tests = 0

        for test_file in test_files:
            fa = self._analyze_test_file_quality(test_file)
            if not fa:
                continue
            quality_analysis["file_analyses"].append(fa)
            # weight by number of tests
            ntests = len(fa["test_analyses"])
            if ntests:
                total_score += fa["quality_score"] * ntests
                total_tests += ntests

        if total_tests > 0:
            quality_analysis["overall_score"] = total_score / total_tests

        quality_analysis["quality_issues"] = self._aggregate_quality_issues(quality_analysis["file_analyses"])
        quality_analysis["signal_vs_noise"] = self._calculate_signal_vs_noise(quality_analysis["file_analyses"])
        quality_analysis["recommendations"] = self._generate_quality_recommendations(quality_analysis)

        return quality_analysis

    def export_detailed_test_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        high_quality_tests: List[Dict[str, Any]] = []
        low_quality_tests: List[Dict[str, Any]] = []
        noisy_tests: List[Dict[str, Any]] = []

        all_tests: List[Dict[str, Any]] = []

        for file_analysis in analysis.get("file_analyses", []):
            file_name = file_analysis["file"]
            for t in file_analysis["test_analyses"]:
                test_info = {
                    "file": file_name,
                    "test_name": t["name"],
                    "quality_score": t["quality_score"],
                    "issues": t["issues"],
                    "signal_strength": t["signal_strength"],
                }
                all_tests.append(test_info)

                if t["quality_score"] >= 80:
                    high_quality_tests.append(test_info)
                if t["quality_score"] <= 40:
                    low_quality_tests.append(test_info)
                if t["signal_strength"] <= 30:
                    noisy_tests.append(test_info)

        problematic_patterns = self._identify_problematic_patterns(analysis.get("file_analyses", []))

        # distribution should be per-test
        total = len(all_tests)
        medium = max(0, total - len(high_quality_tests) - len(low_quality_tests))

        quality_metrics = self._generate_quality_metrics(analysis)

        return {
            "test_quality_distribution": {
                "high_quality": len(high_quality_tests),
                "medium_quality": medium,
                "low_quality": len(low_quality_tests),
                "detailed_high_quality": high_quality_tests,
                "detailed_low_quality": low_quality_tests,
            },
            "signal_vs_noise_analysis": {
                "overall_signal_ratio": analysis.get("signal_vs_noise", {}).get("signal_ratio", 0.0),
                "noisy_tests": noisy_tests,
                "signal_strength_distribution": analysis.get("signal_vs_noise", {}).get("distribution", {}),
                "noise_sources": analysis.get("signal_vs_noise", {}).get("noise_sources", []),
            },
            "problematic_patterns": problematic_patterns,
            "quality_metrics": quality_metrics,
            "improvement_recommendations": self._generate_improvement_recommendations(
                {**analysis, "problematic_patterns": problematic_patterns, "quality_metrics": quality_metrics}
            ),
        }

    # ------------------------------------------------------------------
    # File analysis
    # ------------------------------------------------------------------

    def _analyze_test_file_quality(self, test_file: str) -> Optional[Dict[str, Any]]:
        test_path = (self.project_root / test_file).resolve()
        if not test_path.exists():
            return None

        try:
            content = test_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (OSError, SyntaxError) as e:
            logger.warning("Could not parse %s: %s", test_file, e)
            return None

        parent_builder = _ParentMapBuilder()
        parent_builder.visit(tree)
        parents = parent_builder.parents

        file_analysis: Dict[str, Any] = {
            "file": test_file,
            "quality_score": 0.0,
            "test_analyses": [],
            "file_issues": [],
            "metrics": {},
        }

        # test functions and class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if _is_inside_class(parents, node):
                    # will be counted via its own analysis anyway
                    file_analysis["test_analyses"].append(self._analyze_test_function_quality(node))
                else:
                    file_analysis["test_analyses"].append(self._analyze_test_function_quality(node))

        file_analysis["metrics"] = self._calculate_file_metrics(tree, content)
        if file_analysis["test_analyses"]:
            file_analysis["quality_score"] = sum(t["quality_score"] for t in file_analysis["test_analyses"]) / len(
                file_analysis["test_analyses"]
            )

        file_analysis["file_issues"] = self._identify_file_level_issues(tree, content, file_analysis["metrics"])
        return file_analysis

    # ------------------------------------------------------------------
    # Test analysis
    # ------------------------------------------------------------------

    def _analyze_test_function_quality(self, test_func: ast.FunctionDef) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "name": test_func.name,
            "line": getattr(test_func, "lineno", 0),
            "quality_score": 0.0,
            "signal_strength": 0.0,
            "issues": [],
            "positive_indicators": [],
            "metrics": {},
        }

        analysis["metrics"] = self._calculate_test_metrics(test_func)

        quality = 50.0
        signal = 50.0

        if ast.get_docstring(test_func):
            quality += 8
            signal += 6
            analysis["positive_indicators"].append("Has docstring")

        assertions = self._find_assertions(test_func)
        if assertions:
            bonus = min(len(assertions) * 6, 25)
            quality += bonus
            signal += bonus
            analysis["positive_indicators"].append(f"Has {len(assertions)} assertions/checks")
        else:
            quality -= 35
            signal -= 35
            analysis["issues"].append("No assertions found")

        if self._uses_mocking(test_func):
            quality += 8
            signal += 8
            analysis["positive_indicators"].append("Uses mocking")

        if self._has_trivial_assertions(test_func):
            quality -= 20
            signal -= 30
            analysis["issues"].append("Has trivial assertions (e.g., assert True)")

        if self._has_exception_swallowing(test_func):
            quality -= 30
            signal -= 30
            analysis["issues"].append("Swallows exceptions (except: pass)")

        if self._has_suspicious_inputs(test_func):
            quality -= 12
            signal -= 15
            analysis["issues"].append("Uses suspicious constant inputs (likely meaningless edge cases)")

        if self._is_too_complex(test_func):
            quality -= 10
            signal -= 10
            analysis["issues"].append("Test is too complex")

        # Edge-case coverage should be a POSITIVE indicator, not a penalty
        if self._mentions_edge_case(test_func):
            quality += 6
            signal += 4
            analysis["positive_indicators"].append("Mentions edge cases (name/docstring)")

        analysis["quality_score"] = float(max(0, min(100, quality)))
        analysis["signal_strength"] = float(max(0, min(100, signal)))
        return analysis

    def _calculate_test_metrics(self, test_func: ast.FunctionDef) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "lines_of_code": 0,
            "cyclomatic_complexity": 1,
            "assertion_count": 0,
            "mock_count": 0,
            "parameter_count": len(test_func.args.args),
        }

        if hasattr(test_func, "end_lineno") and hasattr(test_func, "lineno"):
            metrics["lines_of_code"] = (test_func.end_lineno - test_func.lineno + 1)  # type: ignore[attr-defined]

        for node in ast.walk(test_func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                metrics["cyclomatic_complexity"] += 1

        metrics["assertion_count"] = len(self._find_assertions(test_func))
        metrics["mock_count"] = len(self._find_mocks(test_func))
        return metrics

    def _find_assertions(self, test_func: ast.FunctionDef) -> List[ast.AST]:
        """
        What counts as "assertion/check":
        - `assert ...`
        - unittest style: self.assertEqual / self.assertTrue / etc
        - pytest.fail(...)
        - pytest.raises(...) context manager or call
        """
        found: List[ast.AST] = []

        for node in ast.walk(test_func):
            if isinstance(node, ast.Assert):
                found.append(node)
                continue

            if isinstance(node, ast.Call):
                # unittest asserts
                if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("assert"):
                    found.append(node)
                    continue

                # pytest.fail
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pytest" and node.func.attr == "fail":
                        found.append(node)
                        continue

                # bare assert* style
                if isinstance(node.func, ast.Name) and node.func.id.startswith("assert"):
                    found.append(node)
                    continue

                # pytest.raises(...) used as a call (not as context) – still a check
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "pytest" and node.func.attr == "raises":
                        found.append(node)
                        continue

            # pytest.raises as context manager: with pytest.raises(...)
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Attribute):
                        if isinstance(ctx.func.value, ast.Name) and ctx.func.value.id == "pytest" and ctx.func.attr == "raises":
                            found.append(node)
                            break

        return found

    def _find_mocks(self, test_func: ast.FunctionDef) -> List[ast.Call]:
        mocks: List[ast.Call] = []
        for node in ast.walk(test_func):
            if not isinstance(node, ast.Call):
                continue

            func = node.func
            if isinstance(func, ast.Name) and "mock" in func.id.lower():
                mocks.append(node)
            elif isinstance(func, ast.Attribute) and "mock" in func.attr.lower():
                mocks.append(node)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in {"patch", "Mock", "MagicMock"}:
                mocks.append(node)

        return mocks

    def _uses_mocking(self, test_func: ast.FunctionDef) -> bool:
        return bool(self._find_mocks(test_func))

    def _has_trivial_assertions(self, test_func: ast.FunctionDef) -> bool:
        for node in ast.walk(test_func):
            if isinstance(node, ast.Assert):
                # assert True / assert 1
                if isinstance(node.test, ast.Constant) and node.test.value in (True, 1):
                    return True

            if isinstance(node, ast.Call):
                # self.assertTrue(True)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "assertTrue":
                    if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value is True:
                        return True
        return False

    def _has_exception_swallowing(self, test_func: ast.FunctionDef) -> bool:
        for node in ast.walk(test_func):
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    return True
        return False

    def _has_suspicious_inputs(self, test_func: ast.FunctionDef) -> bool:
        """
        Detect patterns like:
          visit_Import(node=None)
          build_index(project_path=0)
          progress_callback=42 (truthy non-callable)
        """
        suspicious_kw = {"node", "project_path", "progress_callback"}

        def is_suspicious_value(v: ast.AST) -> bool:
            if isinstance(v, ast.Constant) and v.value in (None, 0, "", [], {}):
                return True
            if isinstance(v, (ast.List, ast.Dict)) and len(getattr(v, "elts", getattr(v, "keys", []))) == 0:
                return True
            return False

        for node in ast.walk(test_func):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg in suspicious_kw and kw.value and is_suspicious_value(kw.value):
                    return True
            # positional "node=None" style can't be reliably detected without signature resolution
        return False

    def _is_too_complex(self, test_func: ast.FunctionDef) -> bool:
        complexity = 1
        for node in ast.walk(test_func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity > 6

    def _mentions_edge_case(self, test_func: ast.FunctionDef) -> bool:
        indicators = ["edge", "none", "null", "empty", "zero", "negative", "min", "max", "invalid"]
        name = test_func.name.lower()
        doc = (ast.get_docstring(test_func) or "").lower()
        return any(i in name or i in doc for i in indicators)

    # ------------------------------------------------------------------
    # File metrics
    # ------------------------------------------------------------------

    def _calculate_file_metrics(self, tree: ast.Module, content: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "total_lines": len(content.splitlines()),
            "test_function_count": 0,
            "test_class_count": 0,
            "import_count": 0,
            "assertion_density": 0.0,
        }

        total_checks = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                metrics["test_function_count"] += 1
                total_checks += len(self._find_assertions(node))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                metrics["test_class_count"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["import_count"] += 1

        if metrics["test_function_count"] > 0:
            metrics["assertion_density"] = total_checks / metrics["test_function_count"]

        return metrics

    def _identify_file_level_issues(self, tree: ast.Module, content: str, metrics: Dict[str, Any]) -> List[str]:
        issues: List[str] = []

        if metrics["test_function_count"] == 0:
            issues.append("No test functions found")

        if metrics["assertion_density"] < 0.8:
            issues.append("Low check density (assert/raises) - tests may not verify behavior")

        if metrics["total_lines"] > 1200:
            issues.append("Very large test file - consider splitting")

        if metrics["import_count"] > 25:
            issues.append("Too many imports - may indicate complex dependencies")

        return issues

    # ------------------------------------------------------------------
    # Aggregation + reporting
    # ------------------------------------------------------------------

    def _aggregate_quality_issues(self, file_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issue_counts: Dict[str, int] = {}

        for fa in file_analyses:
            for issue in fa.get("file_issues", []):
                key = f"file_level: {issue}"
                issue_counts[key] = issue_counts.get(key, 0) + 1
            for ta in fa.get("test_analyses", []):
                for issue in ta.get("issues", []):
                    key = f"test_level: {issue}"
                    issue_counts[key] = issue_counts.get(key, 0) + 1

        aggregated = [
            {"issue": issue, "count": count, "severity": self._determine_issue_severity(issue)}
            for issue, count in issue_counts.items()
        ]
        aggregated.sort(key=lambda x: (x["severity"], -x["count"]))
        return aggregated

    def _determine_issue_severity(self, issue: str) -> int:
        text = issue.split(": ", 1)[-1]
        high = ["No assertions found", "Swallows exceptions"]
        med = ["trivial assertions", "suspicious constant inputs"]

        if any(h in text for h in high):
            return 1
        if any(m in text for m in med):
            return 2
        return 3

    def _calculate_signal_vs_noise(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = 0
        total_signal = 0.0
        dist = {"high": 0, "medium": 0, "low": 0}
        noise_sources: List[Dict[str, Any]] = []

        for fa in file_analyses:
            for t in fa.get("test_analyses", []):
                total += 1
                s = float(t.get("signal_strength", 0.0))
                total_signal += s
                if s >= 70:
                    dist["high"] += 1
                elif s >= 40:
                    dist["medium"] += 1
                else:
                    dist["low"] += 1
                    if s < 30:
                        noise_sources.append(
                            {
                                "test": t["name"],
                                "file": fa["file"],
                                "signal_strength": s,
                                "issues": t.get("issues", []),
                            }
                        )

        return {
            "signal_ratio": (total_signal / total) if total else 0.0,
            "distribution": dist,
            "noise_sources": noise_sources,
            "total_tests_analyzed": total,
        }

    def _identify_problematic_patterns(self, file_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []

        def collect(issue_substr: str) -> List[str]:
            out: List[str] = []
            for fa in file_analyses:
                for t in fa.get("test_analyses", []):
                    if any(issue_substr in i for i in t.get("issues", [])):
                        out.append(f"{fa['file']}::{t['name']}")
            return out

        no_assert = collect("No assertions found")
        if no_assert:
            patterns.append(
                {
                    "pattern": "Tests without assertions/checks",
                    "count": len(no_assert),
                    "severity": "high",
                    "examples": no_assert[:5],
                    "recommendation": "Add asserts/pytest.raises/unittest asserts verifying behavior",
                }
            )

        swallow = collect("Swallows exceptions")
        if swallow:
            patterns.append(
                {
                    "pattern": "Exception swallowing (except: pass)",
                    "count": len(swallow),
                    "severity": "high",
                    "examples": swallow[:5],
                    "recommendation": "Assert expected exception type or fail explicitly",
                }
            )

        trivial = collect("trivial assertions")
        if trivial:
            patterns.append(
                {
                    "pattern": "Trivial assertions (assert True)",
                    "count": len(trivial),
                    "severity": "medium",
                    "examples": trivial[:5],
                    "recommendation": "Replace with meaningful assertions on outputs/side-effects",
                }
            )

        return patterns

    def _generate_quality_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "overall_quality_score": analysis.get("overall_score", 0.0),
            "total_files_analyzed": len(analysis.get("file_analyses", [])),
            "total_tests_analyzed": 0,
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "common_issues": [],
            "best_practices_adherence": 0.0,
        }

        issue_counts: Dict[str, int] = {}

        for fa in analysis.get("file_analyses", []):
            for t in fa.get("test_analyses", []):
                metrics["total_tests_analyzed"] += 1
                score = float(t.get("quality_score", 0.0))
                if score >= 70:
                    metrics["quality_distribution"]["high"] += 1
                elif score >= 40:
                    metrics["quality_distribution"]["medium"] += 1
                else:
                    metrics["quality_distribution"]["low"] += 1

                for issue in t.get("issues", []):
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

        metrics["common_issues"] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        total = metrics["total_tests_analyzed"]
        high = metrics["quality_distribution"]["high"]
        metrics["best_practices_adherence"] = (high / total * 100) if total else 0.0

        return metrics

    def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recs: List[str] = []

        overall = float(analysis.get("overall_score", 0.0))
        signal = float(analysis.get("signal_vs_noise", {}).get("signal_ratio", 0.0))

        if overall < 50:
            recs.append("Overall test quality is low - review tests for meaningful assertions/checks")

        if signal < 50:
            recs.append("High noise-to-signal ratio - remove trivial tests and add behavioral assertions")

        for issue_info in (analysis.get("quality_issues") or [])[:3]:
            text = issue_info.get("issue", "")
            cnt = issue_info.get("count", 0)
            if "No assertions found" in text:
                recs.append(f"Add assertions/checks to {cnt} tests")
            if "Swallows exceptions" in text:
                recs.append(f"Fix {cnt} tests that swallow exceptions")
            if "trivial assertions" in text:
                recs.append(f"Replace trivial assertions in {cnt} tests")

        return recs

    def _generate_improvement_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        noise_sources = analysis.get("signal_vs_noise", {}).get("noise_sources", [])
        if noise_sources:
            recs.append(f"Refactor/remove {len(noise_sources)} noisy tests with low signal strength")

        for pat in analysis.get("problematic_patterns", []) or []:
            sev = pat.get("severity", "medium")
            cnt = pat.get("count", 0)
            name = pat.get("pattern", "")
            if sev == "high":
                recs.append(f"High priority: fix {cnt} tests with pattern '{name}'")
            else:
                recs.append(f"Medium priority: improve {cnt} tests with pattern '{name}'")

        qm = analysis.get("quality_metrics", {}) or {}
        if float(qm.get("best_practices_adherence", 0.0)) < 60:
            recs.append("Improve adherence to testing best practices (assertions, pytest.raises, clear fixtures)")

        return recs