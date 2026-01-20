"""
Caller Analyzer for expert refactoring analysis.

Finds and analyzes external dependencies on the target module,
helping assess the impact of potential changes.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import (
    ExternalCaller,
    UsageAnalysis,
    ImpactAssessment,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class CallerAnalyzer:
    """Analyzes external usage of the target module."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        
        # Derive module name candidates for import detection
        self.module_candidates = self._derive_module_name_candidates()

    def _derive_module_name_candidates(self) -> List[str]:
        """Derive possible module names for import detection."""
        candidates = []
        
        try:
            # Relative to project root
            rel_path = self.target_module.relative_to(self.project_root)
            module_path = rel_path.as_posix().replace('/', '.')
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            # package __init__.py -> package name
            if module_path.endswith(".__init__"):
                module_path = module_path[: -len(".__init__")]
            candidates.append(module_path)
        except ValueError:
            pass
        
        # Just the filename
        stem = self.target_module.stem
        if stem not in candidates:
            candidates.append(stem)
        
        # Remove duplicates while preserving order
        unique_candidates = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates

    def find_external_callers(self, target_module: str) -> List[ExternalCaller]:
        """
        Find all external files that import and use the target module.
        
        Args:
            target_module: Path to the target module
            
        Returns:
            List of ExternalCaller objects
        """
        logger.info("Finding external callers for module candidates: %s", self.module_candidates)
        
        external_callers = []
        
        # Search all Python files in the project
        for py_file in self.project_root.rglob("*.py"):
            if py_file == self.target_module:
                continue
            
            try:
                callers = self._analyze_file_for_usage(py_file)
                external_callers.extend(callers)
            except Exception as e:
                logger.warning("Error analyzing %s: %s", py_file, e)
                continue
        
        logger.info("Found %d external callers", len(external_callers))
        return external_callers

    def export_detailed_external_usage(self, callers: List[ExternalCaller]) -> Dict[str, Any]:
        """
        Export detailed external usage data as requested by experts.
        
        Returns:
            Dictionary with specific file locations and method usage
        """
        # Group by file for better organization
        files_usage = {}
        symbol_usage = {}
        
        for caller in callers:
            file_path = caller.file_path
            if file_path not in files_usage:
                files_usage[file_path] = {
                    "imports": [],
                    "usages": [],
                    "total_usage_count": 0
                }
            
            # Categorize as import or usage
            if "import" in caller.import_statement.lower():
                files_usage[file_path]["imports"].append({
                    "line": caller.line_number,
                    "statement": caller.import_statement,
                    "symbols": caller.used_symbols,
                    "context": caller.context
                })
            else:
                files_usage[file_path]["usages"].append({
                    "line": caller.line_number,
                    "statement": caller.import_statement,
                    "symbols": caller.used_symbols,
                    "context": caller.context
                })
            
            files_usage[file_path]["total_usage_count"] += caller.usage_frequency
            
            # Track symbol usage across all files
            for symbol in caller.used_symbols:
                if symbol not in symbol_usage:
                    symbol_usage[symbol] = {
                        "total_uses": 0,
                        "files": []
                    }
                symbol_usage[symbol]["total_uses"] += caller.usage_frequency
                
                # Add file reference if not already present
                file_ref = {
                    "file": file_path,
                    "line": caller.line_number,
                    "context": caller.context
                }
                if file_ref not in symbol_usage[symbol]["files"]:
                    symbol_usage[symbol]["files"].append(file_ref)
        
        # Create detailed external usage list as requested by experts
        external_usage_list = []
        for caller in callers:
            if "usage" in caller.import_statement.lower():
                for symbol in caller.used_symbols:
                    external_usage_list.append({
                        "file": caller.file_path,
                        "line": caller.line_number,
                        "symbol": symbol,
                        "context": caller.context
                    })
        
        # Sort by most used symbols
        most_used_symbols = sorted(symbol_usage.items(), key=lambda x: x[1]["total_uses"], reverse=True)
        
        return {
            "external_usage": external_usage_list,
            "files_summary": {
                "total_files": len(files_usage),
                "files": list(files_usage.keys()),
                "detailed_usage": files_usage
            },
            "symbol_usage": {
                "most_used_symbols": [
                    {
                        "symbol": symbol,
                        "uses": data["total_uses"],
                        "files": len(data["files"]),
                        "locations": data["files"]
                    }
                    for symbol, data in most_used_symbols
                ],
                "total_symbols": len(symbol_usage)
            },
            "summary": {
                "total_external_files": len(files_usage),
                "total_callers": len(callers),
                "total_usage_instances": sum(caller.usage_frequency for caller in callers)
            }
        }

    def _analyze_file_for_usage(self, py_file: Path) -> List[ExternalCaller]:
        """Analyze a single file for usage of the target module."""
        callers = []
        
        try:
            content = py_file.read_text(encoding="utf-8-sig", errors="replace")
        except (OSError, UnicodeDecodeError):
            return callers
        
        # Quick pre-filter to avoid parsing files that don't reference our module
        if not any(candidate in content for candidate in self.module_candidates):
            return callers
        
        try:
            tree = ast.parse(content, filename=str(py_file))
        except SyntaxError:
            return callers
        
        # Find import statements
        imports = self._find_imports(tree, py_file)
        callers.extend(imports)
        
        # Find usage of imported symbols
        if imports:
            usage_callers = self._find_symbol_usage(tree, py_file, imports)
            callers.extend(usage_callers)
        
        return callers

    def _find_imports(self, tree: ast.AST, py_file: Path) -> List[ExternalCaller]:
        """Find import statements that reference the target module."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if self._matches_target_module(alias.name):
                        # local name used in code (import pkg.mod -> pkg ; import pkg.mod as pm -> pm)
                        local_name = alias.asname or alias.name.split(".", 1)[0]
                        caller = ExternalCaller(
                            file_path=py_file.relative_to(self.project_root).as_posix(),
                            line_number=getattr(node, 'lineno', 0),
                            import_statement=f"import {alias.name}",
                            used_symbols=[local_name],
                            context=self._get_import_context(node)
                        )
                        imports.append(caller)
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                level = getattr(node, 'level', 0)
                
                if level == 0:  # Absolute import
                    if self._matches_target_module(module_name):
                        symbols = [a.asname or a.name for a in node.names]
                        caller = ExternalCaller(
                            file_path=py_file.relative_to(self.project_root).as_posix(),
                            line_number=getattr(node, 'lineno', 0),
                            import_statement=f"from {module_name} import {', '.join(symbols)}",
                            used_symbols=symbols,
                            context=self._get_import_context(node)
                        )
                        imports.append(caller)
                else:  # Relative import
                    # Handle relative imports by resolving the actual module
                    resolved_module = self._resolve_relative_import(py_file, module_name, level)
                    if resolved_module and self._matches_target_module(resolved_module):
                        symbols = [a.asname or a.name for a in node.names]
                        caller = ExternalCaller(
                            file_path=py_file.relative_to(self.project_root).as_posix(),
                            line_number=getattr(node, 'lineno', 0),
                            import_statement=f"from {'.' * level}{module_name} import {', '.join(symbols)}",
                            used_symbols=symbols,
                            context=self._get_import_context(node)
                        )
                        imports.append(caller)
        
        return imports

    def _matches_target_module(self, module_name: str) -> bool:
        """Check if a module name matches our target module."""
        if not module_name:
            return False
        
        for candidate in self.module_candidates:
            if module_name == candidate or module_name.startswith(candidate + '.'):
                return True
        
        return False

    def _resolve_relative_import(self, py_file: Path, module_name: str, level: int) -> Optional[str]:
        """Resolve a relative import to an absolute module name."""
        try:
            # module path of importing file (package context)
            rel_path = py_file.relative_to(self.project_root).as_posix()
            file_module = rel_path.replace("/", ".")
            if file_module.endswith(".py"):
                file_module = file_module[:-3]
            # base package = drop current module name
            parts = file_module.split(".")[:-1]
            # level=1 => current package; level=2 => parent package, etc.
            up = max(0, (level or 0) - 1)
            if up:
                parts = parts[:-up] if up < len(parts) else []

            if module_name:
                parts.extend(module_name.split('.'))
            
            return '.'.join(parts) if parts else None
        except Exception:
            return None

    def _get_import_context(self, node: ast.AST) -> Optional[str]:
        """Get context information about the import."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return f"import at line {getattr(node, 'lineno', 0)}"
        except Exception:
            return None

    def _find_symbol_usage(self, tree: ast.AST, py_file: Path, imports: List[ExternalCaller]) -> List[ExternalCaller]:
        """Find actual usage of imported symbols."""
        usage_callers = []
        
        # Extract imported symbols and their aliases
        imported_symbols = set()
        for imp in imports:
            imported_symbols.update(imp.used_symbols)
        
        # Find usage of these symbols
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in imported_symbols:
                # Found usage of an imported symbol
                caller = ExternalCaller(
                    file_path=py_file.relative_to(self.project_root).as_posix(),
                    line_number=getattr(node, 'lineno', 0),
                    import_statement=f"usage of {node.id}",
                    used_symbols=[node.id],
                    usage_frequency=1,
                    context=self._get_usage_context(node)
                )
                usage_callers.append(caller)
            
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like module.function
                if isinstance(node.value, ast.Name) and node.value.id in imported_symbols:
                    symbol = f"{node.value.id}.{node.attr}"
                    caller = ExternalCaller(
                        file_path=py_file.relative_to(self.project_root).as_posix(),
                        line_number=getattr(node, 'lineno', 0),
                        import_statement=f"usage of {symbol}",
                        used_symbols=[symbol],
                        usage_frequency=1,
                        context=self._get_usage_context(node)
                    )
                    usage_callers.append(caller)
        
        return usage_callers

    def _get_usage_context(self, node: ast.AST) -> Optional[str]:
        """Get context information about symbol usage."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return f"usage at line {getattr(node, 'lineno', 0)}"
        except Exception:
            return None

    def analyze_usage_patterns(self, callers: List[ExternalCaller]) -> UsageAnalysis:
        """
        Analyze patterns in how the module is used externally.
        
        Args:
            callers: List of external callers
            
        Returns:
            UsageAnalysis with patterns and statistics
        """
        logger.info("Analyzing usage patterns...")
        
        # Count symbol usage
        symbol_usage = {}
        usage_patterns = {}
        critical_dependencies = []
        
        for caller in callers:
            for symbol in caller.used_symbols:
                symbol_usage[symbol] = symbol_usage.get(symbol, 0) + caller.usage_frequency
        
        # Sort by usage frequency
        most_used_symbols = sorted(symbol_usage.items(), key=lambda x: x[1], reverse=True)
        
        # Identify usage patterns
        file_usage = {}
        for caller in callers:
            file_path = caller.file_path
            file_usage[file_path] = file_usage.get(file_path, 0) + 1
        
        # Files with high usage are critical dependencies
        for file_path, usage_count in file_usage.items():
            if usage_count > 3:  # Threshold for critical dependency
                critical_dependencies.append(file_path)
        
        # Categorize usage patterns
        for pattern_type in ['import', 'from_import', 'usage']:
            pattern_count = sum(1 for caller in callers if pattern_type in caller.import_statement)
            if pattern_count > 0:
                usage_patterns[pattern_type] = pattern_count
        
        analysis = UsageAnalysis(
            total_callers=len(set(caller.file_path for caller in callers)),
            most_used_symbols=most_used_symbols[:10],  # Top 10
            usage_patterns=usage_patterns,
            critical_dependencies=critical_dependencies
        )
        
        logger.info(f"Usage analysis: {analysis.total_callers} callers, {len(most_used_symbols)} symbols used")
        return analysis

    def assess_breaking_change_impact(self, changes: List[str]) -> ImpactAssessment:
        """
        Assess the impact of potential breaking changes.
        
        Args:
            changes: List of proposed changes (method renames, removals, etc.)
            
        Returns:
            ImpactAssessment with risk level and recommendations
        """
        logger.info("Assessing breaking change impact...")
        
        # Find all external callers first
        callers = self.find_external_callers(str(self.target_module))
        
        affected_files = set()
        breaking_changes = []
        
        # Analyze impact of each change
        for change in changes:
            for caller in callers:
                # Check if this caller would be affected by the change
                if any(change in symbol for symbol in caller.used_symbols):
                    affected_files.add(caller.file_path)
                    breaking_changes.append(f"{change} affects {caller.file_path}")
        
        # Assess risk level
        risk_level = RiskLevel.LOW
        if len(affected_files) > 10:
            risk_level = RiskLevel.CRITICAL
        elif len(affected_files) > 5:
            risk_level = RiskLevel.HIGH
        elif len(affected_files) > 2:
            risk_level = RiskLevel.MEDIUM
        
        # Generate recommendations
        recommendations = []
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Create deprecation warnings before removing functionality")
            recommendations.append("Provide migration guide for affected code")
            recommendations.append("Consider phased rollout of changes")
        
        if len(affected_files) > 0:
            recommendations.append("Update all affected files in the same commit")
            recommendations.append("Run comprehensive tests after changes")
        
        # Estimate migration effort
        migration_effort = "low"
        if len(affected_files) > 10:
            migration_effort = "high"
        elif len(affected_files) > 3:
            migration_effort = "medium"
        
        assessment = ImpactAssessment(
            affected_files=list(affected_files),
            risk_level=risk_level,
            breaking_changes=breaking_changes,
            migration_effort=migration_effort,
            recommendations=recommendations
        )
        
        logger.info(f"Impact assessment: {len(affected_files)} files affected, risk level {risk_level.value}")
        return assessment