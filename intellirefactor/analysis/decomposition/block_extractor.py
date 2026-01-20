"""
Functional Block Extractor

Extracts FunctionalBlock instances from AST using direct AST parsing.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Tuple

from .models import FunctionalBlock, DecompositionConfig
from .ast_helpers import calculate_cyclomatic_complexity
from .signature_builder import SignatureBuilder
from .dependency_extractor import DependencyExtractor
from .type_collectors import ModuleTypeHintCollector, LocalTypeHintVisitor
from .visitors import FunctionVisitor, AssignedNameVisitor
from .import_resolver import ImportResolver

logger = logging.getLogger(__name__)


# -----------------------------
# Extractor
# -----------------------------


class FunctionalBlockExtractor:
    """
    Extracts FunctionalBlock instances from Python source files.

    Uses direct AST parsing to extract functions and methods.
    """

    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.logger = logger
        self.config = config or DecompositionConfig.default()

        # reuse one generator (instead of per-function)
        from .fingerprints import FingerprintGenerator

        self._fingerprint_gen = FingerprintGenerator()

        # reuse dependency extractor
        self._dependency_extractor = DependencyExtractor(self.config)

    def _extract_module_type_hints(self, tree: ast.Module) -> Dict[str, str]:
        return ModuleTypeHintCollector().collect(tree)

    def extract_from_file(
        self, file_path: str, module_name: str = ""
    ) -> List[FunctionalBlock]:
        """Extract functional blocks from a Python file."""
        try:
            file_path = str(Path(file_path).resolve())
            source = Path(file_path).read_text(encoding="utf-8")

            if not source.strip():
                return []

            tree = ast.parse(source, filename=file_path)

            # pass module_name for relative import resolution
            file_imports = ImportResolver.extract_file_imports(
                tree, current_module=module_name
            )

            module_type_hints = self._extract_module_type_hints(tree)
            visitor = FunctionVisitor(
                file_path, module_name, source, file_imports, module_type_hints, self
            )
            visitor.visit(tree)

            self.logger.info(
                f"Extracted {len(visitor.blocks)} functional blocks from {file_path}"
            )
            return visitor.blocks

        except Exception as e:
            self.logger.error(
                f"Failed to extract blocks from {file_path}: {e}", exc_info=True
            )
            return []

    def _function_to_block(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        file_path: str,
        module_name: str,
        source: str,
        qualname: str,
        file_imports: List[str],
        module_type_hints: Optional[Dict[str, str]] = None,
        is_nested: bool = False,
    ) -> Optional[FunctionalBlock]:
        """Convert function AST node to FunctionalBlock."""
        try:
            lines = source.split("\n")
            start_line, end_line, func_source = self._extract_function_source(
                node, lines
            )

            # Fingerprints
            ast_hash = self._fingerprint_gen.generate_ast_hash_from_node(node)
            token_fingerprint = self._fingerprint_gen.generate_token_fingerprint(
                func_source
            )

            # Dependencies from AST node (fast path)
            calls, block_imports, globals_used, literals = self._extract_dependencies(
                func_source, node
            )

            # Nested local defs (names only)
            local_defs = sorted(
                {
                    n.name
                    for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n is not node
                }
            )

            # Local assigned names in this scope
            arg_names = self._collect_function_arguments(node)
            av = AssignedNameVisitor(node)
            av.visit(node)
            local_assigned = sorted(
                n for n in av.names if n not in arg_names and n not in {"self", "cls"}
            )

            # Local type hints with heuristics
            local_type_hints = self._build_type_hints_with_heuristics(
                node, module_type_hints, file_imports, calls
            )

            # Evidence-based import usage
            used_from_file_imports = self._calculate_used_imports(file_imports, calls)

            imports_used = sorted(set(block_imports) | set(used_from_file_imports))
            imports_context = sorted(set(file_imports))

            signature = SignatureBuilder.build_signature(node)
            inputs, outputs = SignatureBuilder.extract_signature_tokens(node)

            return FunctionalBlock(
                id="",
                module=module_name,
                file_path=file_path,
                qualname=qualname,
                lineno=start_line,
                end_lineno=end_line,
                signature=signature,
                inputs=inputs,
                outputs=outputs,
                raw_calls=calls,
                calls=[],
                imports_used=imports_used,
                imports_context=imports_context,
                globals_used=globals_used,
                literals=literals,
                local_defs=local_defs,
                local_assigned=local_assigned,
                local_type_hints=local_type_hints,
                loc=max(0, end_line - start_line + 1),
                cyclomatic=self._calculate_cyclomatic(node),
                ast_hash=ast_hash,
                token_fingerprint=token_fingerprint,
                tags=["nested"] if is_nested else [],
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to convert function {getattr(node, 'name', '<unknown>')}: {e}",
                exc_info=True,
            )
            return None

    def _extract_function_source(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        lines: List[str],
    ) -> Tuple[int, int, str]:
        """
        Extract function source code including decorators.

        Returns:
            (start_line, end_line, func_source)
        """
        # include decorators into the function slice (lineno points to 'def', not '@decorator')
        decorator_lines = [
            getattr(d, "lineno", node.lineno)
            for d in getattr(node, "decorator_list", [])
        ]
        start_line = (
            min([node.lineno, *decorator_lines]) if decorator_lines else node.lineno
        )

        end_line = getattr(node, "end_lineno", None)
        if end_line is None:
            if getattr(node, "body", None):
                last = node.body[-1]
                end_line = getattr(last, "end_lineno", node.lineno)
            else:
                end_line = node.lineno

        if 1 <= start_line <= len(lines):
            func_source = "\n".join(lines[start_line - 1 : min(end_line, len(lines))])
            import textwrap

            func_source = textwrap.dedent(func_source).strip()
        else:
            func_source = ""

        return start_line, end_line, func_source

    def _collect_function_arguments(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Set[str]:
        """
        Collect all argument names from function signature.

        Returns:
            Set of argument names (including *args, **kwargs)
        """
        arg_names = {a.arg for a in node.args.args} | {
            a.arg for a in node.args.kwonlyargs
        }
        for a in getattr(node.args, "posonlyargs", []):
            arg_names.add(a.arg)
        if node.args.vararg:
            arg_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            arg_names.add(node.args.kwarg.arg)
        return arg_names

    def _build_type_hints_with_heuristics(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        module_type_hints: Optional[Dict[str, str]],
        file_imports: List[str],
        calls: List[str],
    ) -> Dict[str, str]:
        """
        Build local type hints with logging heuristics.

        Merges module-level type hints with function-level hints,
        and applies heuristics for common patterns (e.g., logging.Logger).

        Returns:
            Dictionary mapping variable names to type hints
        """
        tv = LocalTypeHintVisitor(node)
        tv.visit(node)
        merged_type_hints: Dict[str, str] = dict(module_type_hints or {})
        merged_type_hints.update(tv.hints)

        # Heuristic (compact, low-risk):
        # If we see self.logger.* (or logger.*) and the file imports logging,
        # treat it as logging.Logger to reduce dynamic_attribute noise.
        logging_in_file = any(
            imp == "logging" or imp.startswith("logging.")
            for imp in (file_imports or [])
        )
        if logging_in_file:
            if any(c.startswith("self.logger.") for c in calls):
                merged_type_hints.setdefault("self.logger", "logging.Logger")
            # optional but usually safe/helpful too:
            if any(c.startswith("logger.") for c in calls):
                merged_type_hints.setdefault("logger", "logging.Logger")

        return merged_type_hints

    def _extract_dependencies(
        self,
        source: str,
        func_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract calls/imports/globals/literals from AST node (preferred)."""
        if func_node is None:
            return self._dependency_extractor.extract_from_source(source, None)
        return self._dependency_extractor.extract_from_node(func_node, source)

    def _calculate_cyclomatic(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> int:
        """Calculate cyclomatic complexity."""
        return calculate_cyclomatic_complexity(node)

    def _calculate_used_imports(
        self, file_imports: List[str], raw_calls: List[str]
    ) -> List[str]:
        """Evidence-based import usage detection based on raw_calls."""
        return self._dependency_extractor.calculate_used_imports(
            file_imports, raw_calls
        )
