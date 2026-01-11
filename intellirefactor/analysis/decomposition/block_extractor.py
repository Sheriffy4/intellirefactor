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
from .utils import iter_toplevel_import_nodes

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------

_BUILTIN_TYPES = {"list", "dict", "set", "tuple", "str", "bytes", "int", "float", "bool"}

def _looks_like_class_ref(ref: str) -> bool:
    """
    Heuristic: accept class/type references where last segment starts with uppercase.
    Examples:
      - "QualityAnalyzer" -> True
      - "analysis.QualityAnalyzer" -> True
      - "create_analyzer" -> False
      - "typing.Optional" -> False
    """
    if not ref:
        return False
    last = ref.split(".")[-1]
    return (last and last[0].isupper()) or (last in _BUILTIN_TYPES)
    
# -----------------------------
# AST function visitor
# -----------------------------

class FunctionVisitor(ast.NodeVisitor):
    """
    AST visitor to extract functions with proper class context.

    Uses stacks to track current class context and function nesting.
    """

    def __init__(self, file_path: str, module_name: str, source: str, file_imports: List[str], module_type_hints: Dict[str, str], extractor):
        self.file_path = file_path
        self.module_name = module_name
        self.source = source
        self.file_imports = file_imports
        self.module_type_hints = module_type_hints or {}
        self.extractor = extractor

        self.blocks: List[FunctionalBlock] = []
        self.class_stack: List[str] = []
        self.function_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_nested = len(self.function_stack) > 0

        # do not extract nested defs unless configured
        if is_nested and not self.extractor.config.extract_nested_functions:
            return

        self._process_function(node, is_nested)

        self.function_stack.append(node.name)
        if self.extractor.config.extract_nested_functions:
            self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        is_nested = len(self.function_stack) > 0

        if is_nested and not self.extractor.config.extract_nested_functions:
            return

        self._process_function(node, is_nested)

        self.function_stack.append(node.name)
        if self.extractor.config.extract_nested_functions:
            self.generic_visit(node)
        self.function_stack.pop()

    def _process_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_nested: bool = False,
    ) -> None:
        # keep full nesting for classes: Outer.Inner.method
        if self.class_stack:
            class_prefix = ".".join(self.class_stack)
            qualname = f"{class_prefix}.{node.name}"
        else:
            qualname = node.name

        block = self.extractor._function_to_block(
            node=node,
            file_path=self.file_path,
            module_name=self.module_name,
            source=self.source,
            qualname=qualname,
            file_imports=self.file_imports,
            module_type_hints=self.module_type_hints,
            is_nested=is_nested,
        )
        if block:
            self.blocks.append(block)


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

    def _extract_module_type_hints(self, tree: ast.Module) -> Dict[str, str]:
        return _ModuleTypeHintCollector().collect(tree)
    
    def extract_from_file(self, file_path: str, module_name: str = "") -> List[FunctionalBlock]:
        """Extract functional blocks from a Python file."""
        try:
            file_path = str(Path(file_path).resolve())
            source = Path(file_path).read_text(encoding="utf-8")

            if not source.strip():
                return []

            tree = ast.parse(source, filename=file_path)

            # pass module_name for relative import resolution
            file_imports = self._extract_file_imports(tree, current_module=module_name)

            module_type_hints = self._extract_module_type_hints(tree)
            visitor = FunctionVisitor(file_path, module_name, source, file_imports, module_type_hints, self)
            visitor.visit(tree)

            self.logger.info(f"Extracted {len(visitor.blocks)} functional blocks from {file_path}")
            return visitor.blocks

        except Exception as e:
            self.logger.error(f"Failed to extract blocks from {file_path}: {e}", exc_info=True)
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

            # include decorators into the function slice (lineno points to 'def', not '@decorator')
            decorator_lines = [getattr(d, "lineno", node.lineno) for d in getattr(node, "decorator_list", [])]
            start_line = min([node.lineno, *decorator_lines]) if decorator_lines else node.lineno

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

            # Fingerprints
            ast_hash = self._fingerprint_gen.generate_ast_hash_from_node(node)
            token_fingerprint = self._fingerprint_gen.generate_token_fingerprint(func_source)

            # Dependencies from AST node (fast path)
            calls, block_imports, globals_used, literals = self._extract_dependencies(func_source, node)

            # Nested local defs (names only)
            local_defs = sorted(
                {
                    n.name
                    for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not node
                }
            )

            # Local assigned names in this scope
            arg_names = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            for a in getattr(node.args, "posonlyargs", []):
                arg_names.add(a.arg)
            if node.args.vararg:
                arg_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                arg_names.add(node.args.kwarg.arg)

            av = _AssignedNameVisitor(node)
            av.visit(node)
            local_assigned = sorted(n for n in av.names if n not in arg_names and n not in {"self", "cls"})

            # NEW: local type hints (maximally enhanced)
            tv = _LocalTypeHintVisitor(node)
            tv.visit(node)
            merged_type_hints: Dict[str, str] = dict(module_type_hints or {})
            merged_type_hints.update(tv.hints)
            local_type_hints = merged_type_hints
            
            # Heuristic (compact, low-risk):
            # If we see self.logger.* (or logger.*) and the file imports logging,
            # treat it as logging.Logger to reduce dynamic_attribute noise.
            logging_in_file = any(
                imp == "logging" or imp.startswith("logging.")
                for imp in (file_imports or [])
            )
            if logging_in_file:
                if any(c.startswith("self.logger.") for c in calls):
                    local_type_hints.setdefault("self.logger", "logging.Logger")
                # optional but usually safe/helpful too:
                if any(c.startswith("logger.") for c in calls):
                    local_type_hints.setdefault("logger", "logging.Logger")

            # Evidence-based import usage
            used_from_file_imports = self._calculate_used_imports(file_imports, calls)

            imports_used = sorted(set(block_imports) | set(used_from_file_imports))
            imports_context = sorted(set(file_imports))

            signature = self._build_function_signature(node)
            inputs, outputs = self._extract_signature_tokens(node)

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
            self.logger.warning(f"Failed to convert function {getattr(node, 'name', '<unknown>')}: {e}", exc_info=True)
            return None

    def _extract_dependencies(
        self,
        source: str,
        func_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract calls/imports/globals/literals from AST node (preferred)."""
        if func_node is None:
            return self._extract_dependencies_from_source(source, None)

        try:
            visitor = _DependencyVisitor(self, func_node)
            visitor.visit(func_node)
            return visitor.calls, visitor.imports_used, visitor.globals_used, visitor.literals
        except Exception as e:
            self.logger.debug(f"Failed to extract dependencies from AST: {e}")
            return self._extract_dependencies_from_source(source, func_node)

    def _extract_dependencies_from_source(
        self,
        source: str,
        func_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Fallback method: extract dependencies by parsing function source."""
        calls: List[str] = []
        imports_used: List[str] = []
        globals_used: List[str] = []
        literals: List[str] = []

        try:
            import textwrap
            dedented_source = textwrap.dedent(source)

            if not dedented_source.strip():
                return calls, imports_used, globals_used, literals

            tree = ast.parse(dedented_source)

            function_docstring = None
            if func_node and self.config.exclude_docstrings_from_literals:
                function_docstring = ast.get_docstring(func_node)

            tree_docstring = None
            if self.config.exclude_docstrings_from_literals and tree.body:
                first_node = tree.body[0]
                if isinstance(first_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    tree_docstring = ast.get_docstring(first_node)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_key = self._call_key(node.func)
                    if call_key:
                        calls.append(call_key)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_used.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_used.append(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                imports_used.append(f"{node.module}.{alias.name}")

                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if len(node.value) > 2:
                        if self.config.exclude_docstrings_from_literals:
                            if (function_docstring and node.value == function_docstring) or (
                                tree_docstring and node.value == tree_docstring
                            ):
                                continue
                        literals.append(node.value)

                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id.isupper():
                        globals_used.append(node.id)

        except Exception as e:
            self.logger.debug(f"Failed to extract dependencies: {e}")

        return calls, imports_used, globals_used, literals

    def _call_key(self, func: ast.AST) -> str:
        """
        Generate normalized call key from AST func node.

        FIX: keep dot even for complex receivers, so callers like x[i].append(...)
        become "<subscript>.append" (dynamic_attribute), not "append" (not_found).
        """
        if isinstance(func, ast.Name):
            return func.id

        if isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Call):
                base = base.func

            if isinstance(base, ast.Subscript):
                base_s = self._call_key(base.value) or "<subscript>"
            elif isinstance(base, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare)):
                base_s = "<expr>"
            else:
                try:
                    base_s = self._call_key(base) or "<expr>"
                except Exception:
                    base_s = "<expr>"

            return f"{base_s}.{func.attr}"

        try:
            return ast.unparse(func)
        except Exception:
            return ""

    def _build_function_signature(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """Build function signature from AST node."""
        parts: List[str] = []

        def fmt_arg(arg: ast.arg) -> str:
            s = arg.arg
            if arg.annotation:
                try:
                    s += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            return s

        posonly = [fmt_arg(a) for a in getattr(node.args, "posonlyargs", [])]
        regular = [fmt_arg(a) for a in node.args.args]
        kwonly = [fmt_arg(a) for a in node.args.kwonlyargs]

        if posonly:
            parts.extend(posonly)
            parts.append("/")

        parts.extend(regular)

        if node.args.vararg:
            parts.append("*" + fmt_arg(node.args.vararg))
        elif kwonly:
            parts.append("*")

        parts.extend(kwonly)

        if node.args.kwarg:
            parts.append("**" + fmt_arg(node.args.kwarg))

        ret = ""
        if node.returns:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret = ""

        return f"{node.name}({', '.join(parts)}){ret}"

    def _calculate_cyclomatic(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += max(1, len(getattr(child, "values", [])) - 1)
            elif isinstance(child, ast.IfExp):
                complexity += 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
                for gen in child.generators:
                    complexity += len(getattr(gen, "ifs", []))
        return complexity

    def _extract_file_imports(self, tree: ast.Module, current_module: str = "") -> List[str]:
        """
        Extract top-level imports (including within top-level Try/If).
        Handles relative imports like `from . import x`.
        """
        imports: List[str] = []

        def resolve_absolute_module(node: ast.ImportFrom) -> Optional[str]:
            if node.level and node.level > 0:
                if not current_module:
                    return None
                parts = current_module.split(".")
                if len(parts) < node.level:
                    return None
                base = ".".join(parts[:-node.level])
                if node.module:
                    return f"{base}.{node.module}" if base else node.module
                return base or None
            return node.module

        for node in iter_toplevel_import_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                abs_mod = resolve_absolute_module(node)
                if not abs_mod:
                    continue

                imports.append(abs_mod)
                for alias in node.names:
                    if alias.name != "*":
                        imports.append(f"{abs_mod}.{alias.name}")

        return imports

    def _extract_signature_tokens(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Tuple[List[str], List[str]]:
        """Extract input/output tokens from signature."""
        inputs: List[str] = []
        outputs: List[str] = []

        all_args: List[ast.arg] = []
        all_args.extend(getattr(node.args, "posonlyargs", []))
        all_args.extend(node.args.args)
        all_args.extend(node.args.kwonlyargs)

        for idx, arg in enumerate(all_args):
            if idx == 0 and arg.arg in {"self", "cls"}:
                continue
            inputs.append(arg.arg)
            if arg.annotation:
                try:
                    inputs.append(ast.unparse(arg.annotation))
                except Exception:
                    pass

        if node.args.vararg:
            inputs.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            inputs.append("**" + node.args.kwarg.arg)

        if node.returns:
            try:
                outputs.append(ast.unparse(node.returns))
            except Exception:
                pass

        return inputs, outputs

    def _calculate_used_imports(self, file_imports: List[str], raw_calls: List[str]) -> List[str]:
        """Evidence-based import usage detection based on raw_calls."""
        used_imports: List[str] = []
        raw_calls_set = set(raw_calls)

        for imp in file_imports:
            parts = imp.split(".")
            head = parts[0]
            tail = parts[-1]

            is_used = False
            if tail in raw_calls_set:
                is_used = True
            elif any(call.startswith(head + ".") for call in raw_calls_set):
                is_used = True
            elif head in raw_calls_set:
                is_used = True
            elif any(call.startswith(tail + ".") or call.endswith("." + tail) for call in raw_calls_set):
                is_used = True

            if is_used:
                used_imports.append(imp)

        return used_imports


# -----------------------------
# Dependency visitor
# -----------------------------

class _DependencyVisitor(ast.NodeVisitor):
    """
    Collect dependencies from the function scope WITHOUT descending into nested defs/classes/lambdas.
    """

    __slots__ = ("extractor", "root", "calls", "imports_used", "globals_used", "literals", "_docstring")

    def __init__(self, extractor: FunctionalBlockExtractor, root: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        self.extractor = extractor
        self.root = root
        self.calls: List[str] = []
        self.imports_used: List[str] = []
        self.globals_used: List[str] = []
        self.literals: List[str] = []
        self._docstring = ast.get_docstring(root) if extractor.config.exclude_docstrings_from_literals else None

    def visit_FunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_Call(self, node):
        call_key = self._call_key(node.func)
        if call_key:
            self.calls.append(call_key)
        self.generic_visit(node)

    def _call_key(self, func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Call):
                base = base.func

            # keep dot even for complex receivers
            if isinstance(base, ast.Subscript):
                base_s = self._call_key(base.value) or "<subscript>"
            elif isinstance(base, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare)):
                base_s = "<expr>"
            else:
                base_s = self._call_key(base) or "<expr>"

            return f"{base_s}.{func.attr}"
        return ""

    def visit_Import(self, node):
        for alias in node.names:
            self.imports_used.append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports_used.append(node.module)
            for alias in node.names:
                if alias.name != "*":
                    self.imports_used.append(f"{node.module}.{alias.name}")

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 2:
            if not (self._docstring and node.value == self._docstring):
                self.literals.append(node.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id.isupper():
            self.globals_used.append(node.id)


# -----------------------------
# Assigned names visitor
# -----------------------------

class _AssignedNameVisitor(ast.NodeVisitor):
    """Collect names assigned in the current function scope (no nested defs/classes/lambdas/comprehensions)."""

    __slots__ = ("root", "names")

    def __init__(self, root: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        self.root = root
        self.names: Set[str] = set()

    def visit_FunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ListComp(self, node):
        return

    def visit_SetComp(self, node):
        return

    def visit_DictComp(self, node):
        return

    def visit_GeneratorExp(self, node):
        return

    def visit_Assign(self, node):
        for t in node.targets:
            self._collect_target(t)

    def visit_AnnAssign(self, node):
        self._collect_target(node.target)

    def visit_AugAssign(self, node):
        self._collect_target(node.target)

    def visit_For(self, node):
        self._collect_target(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_AsyncFor(self, node):
        self._collect_target(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        for s in node.body:
            self.visit(s)

    def visit_AsyncWith(self, node):
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        for s in node.body:
            self.visit(s)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        for s in node.body:
            self.visit(s)

    def visit_NamedExpr(self, node):
        self._collect_target(node.target)

    def _collect_target(self, t):
        if isinstance(t, ast.Name):
            self.names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._collect_target(e)

# -----------------------------
# Module-level type hints (simple, high-impact)
# -----------------------------
class _ModuleTypeHintCollector:
    """
    Collect very lightweight module-scope type hints.

    Primary goal: capture patterns like:
      logger = logging.getLogger(__name__)
      logger = getLogger(__name__)   # from logging import getLogger
      console = Console()

    These hints are then merged into each function's local_type_hints for the file.
    """

    __slots__ = ("hints",)

    _WRAPPERS = {"Optional", "Union", "Annotated", "Final", "ClassVar"}
    _CAST_NAMES = {"cast"}  # also supports typing.cast

    def __init__(self):
        self.hints: Dict[str, str] = {}

    def collect(self, tree: ast.Module) -> Dict[str, str]:
        self.hints = {}
        self._scan_nodes(list(getattr(tree, "body", []) or []))
        return dict(self.hints)

    def _scan_nodes(self, nodes: List[ast.stmt]) -> None:
        for n in nodes:
            # Do not descend into defs/classes
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            if isinstance(n, ast.Assign):
                self._handle_assign(n)
                continue

            if isinstance(n, ast.AnnAssign):
                self._handle_annassign(n)
                continue

            # Scan inside top-level If/Try (common in logging/config)
            if isinstance(n, ast.If):
                self._scan_nodes(n.body)
                self._scan_nodes(n.orelse)
                continue

            if isinstance(n, ast.Try):
                self._scan_nodes(n.body)
                self._scan_nodes(n.orelse)
                self._scan_nodes(n.finalbody)
                for h in n.handlers:
                    self._scan_nodes(h.body)
                continue

    def _handle_assign(self, node: ast.Assign) -> None:
        class_ref = self._type_from_value(node.value)

        # alias propagation: x = y (module scope)
        if not class_ref:
            if isinstance(node.value, ast.Name) and node.value.id in self.hints:
                class_ref = self.hints[node.value.id]

        if class_ref and _looks_like_class_ref(class_ref):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.hints[t.id] = class_ref

    def _handle_annassign(self, node: ast.AnnAssign) -> None:
        # Prefer constructor-based type, else annotation.
        class_ref = self._type_from_value(node.value) if node.value else ""
        if not class_ref:
            class_ref = self._type_from_annotation(node.annotation)

        # alias propagation: x: T = y
        if not class_ref and node.value and isinstance(node.value, ast.Name) and node.value.id in self.hints:
            class_ref = self.hints[node.value.id]

        if class_ref and _looks_like_class_ref(class_ref) and isinstance(node.target, ast.Name):
            self.hints[node.target.id] = class_ref

    def _type_from_value(self, v: Optional[ast.AST]) -> str:
        if v is None:
            return ""

        # builtins from literals
        if isinstance(v, ast.List):
            return "list"
        if isinstance(v, ast.Dict):
            return "dict"
        if isinstance(v, ast.Set):
            return "set"
        if isinstance(v, ast.Tuple):
            return "tuple"
        if isinstance(v, ast.Constant):
            if isinstance(v.value, str):
                return "str"
            if isinstance(v.value, bool):
                return "bool"
            if isinstance(v.value, int):
                return "int"
            if isinstance(v.value, float):
                return "float"

        # logging.getLogger(...) / getLogger(...) -> logging.Logger
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Attribute) and f.attr == "getLogger":
                return "logging.Logger"
            if isinstance(f, ast.Name) and f.id == "getLogger":
                return "logging.Logger"

        # cast(Type, expr) / typing.cast(Type, expr)
        if isinstance(v, ast.Call) and self._is_cast_call(v):
            if v.args:
                t = self._type_from_annotation(v.args[0])
                return t or ""

        # x = ClassName(...)
        if isinstance(v, ast.Call):
            ref = self._class_ref_from_callee(v.func)
            if ref and _looks_like_class_ref(ref):
                return ref
            return ""

        return ""

    def _is_cast_call(self, call: ast.Call) -> bool:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id in self._CAST_NAMES
        if isinstance(f, ast.Attribute):
            return f.attr in self._CAST_NAMES
        return False

    def _class_ref_from_callee(self, f: ast.AST) -> str:
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return self._attr_path(f)
        return ""

    def _type_from_annotation(self, ann: Optional[ast.AST]) -> str:
        if ann is None:
            return ""

        # PEP604: T | None
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            cands = self._collect_union_binop(ann)
            cands = [c for c in cands if c and c != "None"]
            for c in cands:
                if _looks_like_class_ref(c):
                    return c
            return cands[0] if cands else ""

        if isinstance(ann, ast.Name):
            return ann.id

        if isinstance(ann, ast.Attribute):
            return self._attr_path(ann)

        if isinstance(ann, ast.Subscript):
            base = self._annotation_base_name(ann.value)
            base_last = base.split(".")[-1] if base else ""

            if base_last in self._WRAPPERS:
                inner = self._collect_subscript_args(ann.slice)
                inner = [x for x in inner if x and x != "None"]
                for x in inner:
                    if _looks_like_class_ref(x):
                        return x
                return inner[0] if inner else ""

            return ""

        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            s = ann.value.strip()
            return s if _looks_like_class_ref(s) else ""

        return ""

    def _annotation_base_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._attr_path(node)
        return ""

    def _collect_subscript_args(self, sl: ast.AST) -> List[str]:
        if isinstance(sl, ast.Tuple):
            return [self._type_from_annotation(e) for e in sl.elts]
        return [self._type_from_annotation(sl)]

    def _collect_union_binop(self, node: ast.AST) -> List[str]:
        out: List[str] = []

        def walk(n: ast.AST):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                walk(n.left)
                walk(n.right)
            else:
                out.append(self._type_from_annotation(n))

        walk(node)
        return out

    def _attr_path(self, node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            return ""
        return ".".join(reversed(parts))
# -----------------------------
# Local type hints (maximized)
# -----------------------------

class _LocalTypeHintVisitor(ast.NodeVisitor):
    """
    Collect robust-ish local type hints within a function scope.

    Produces mapping:
      - "x" -> "QualityAnalyzer"
      - "self.analyzer" -> "QualityAnalyzer"
      - "self.ctx.store" -> "analysis.index_store.IndexStore" (if it can be inferred)
    """

    __slots__ = ("root", "hints")

    _WRAPPERS = {"Optional", "Union", "Annotated", "Final", "ClassVar"}
    _CAST_NAMES = {"cast"}  # also supports typing.cast

    def __init__(self, root: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        self.root = root
        self.hints: Dict[str, str] = {}
        self._seed_from_params(root)

    # ---- scope control ----

    def visit_FunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ListComp(self, node):
        return
    def visit_SetComp(self, node):
        return
    def visit_DictComp(self, node):
        return
    def visit_GeneratorExp(self, node):
        return

    # ---- control-flow (promote only intersection) ----

    def visit_If(self, node: ast.If):
        base = dict(self.hints)

        # body
        self.hints = dict(base)
        for s in node.body:
            self.visit(s)
        body_h = dict(self.hints)

        # else
        self.hints = dict(base)
        for s in node.orelse:
            self.visit(s)
        else_h = dict(self.hints)

        # restore base + intersection( body, else )
        self.hints = dict(base)
        if node.orelse:
            for k in (set(body_h.keys()) & set(else_h.keys())):
                if body_h[k] == else_h[k]:
                    self.hints[k] = body_h[k]

    def visit_Try(self, node: ast.Try):
        base = dict(self.hints)

        # try branch
        self.hints = dict(base)
        for s in node.body:
            self.visit(s)
        try_h = dict(self.hints)

        # except branches (each)
        except_h_list: List[Dict[str, str]] = []
        for h in node.handlers:
            self.hints = dict(base)
            for s in h.body:
                self.visit(s)
            except_h_list.append(dict(self.hints))

        # else branch
        self.hints = dict(base)
        for s in node.orelse:
            self.visit(s)
        else_h = dict(self.hints)

        # finally branch (executed always, but may execute after partial failure)
        # We treat it as additional constraints only if it assigns same types across all paths,
        # but simplest is: compute it and intersect like others.
        self.hints = dict(base)
        for s in node.finalbody:
            self.visit(s)
        fin_h = dict(self.hints)

        # Merge logic:
        # We promote only keys which are equal in:
        #   - try_h and else_h (else runs only if try succeeded)
        #   - and in all except handlers (if any)
        #   - and in finally (if present)
        paths = [try_h]
        if node.orelse:
            paths.append(else_h)
        paths.extend(except_h_list)
        if node.finalbody:
            paths.append(fin_h)

        self.hints = dict(base)
        if paths:
            common_keys = set(paths[0].keys())
            for p in paths[1:]:
                common_keys &= set(p.keys())
            for k in common_keys:
                v0 = paths[0][k]
                if all(p.get(k) == v0 for p in paths[1:]):
                    self.hints[k] = v0

    # ---- assignments ----

    def visit_Assign(self, node: ast.Assign):
        class_ref = self._type_from_value(node.value)

        # alias propagation: x = y
        if not class_ref:
            alias_ref = self._ref_from_expr(node.value)
            if alias_ref and alias_ref in self.hints:
                class_ref = self.hints[alias_ref]

        if class_ref and _looks_like_class_ref(class_ref):
            for t in node.targets:
                self._record_target(t, class_ref)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        # Prefer constructor-based type, else annotation.
        class_ref = self._type_from_value(node.value) if node.value else ""

        if not class_ref:
            class_ref = self._type_from_annotation(node.annotation)

        # alias propagation for annotated assignment: x: T = y
        if not class_ref and node.value:
            alias_ref = self._ref_from_expr(node.value)
            if alias_ref and alias_ref in self.hints:
                class_ref = self.hints[alias_ref]

        if class_ref and _looks_like_class_ref(class_ref):
            self._record_target(node.target, class_ref)

    def visit_With(self, node: ast.With):
        for item in node.items:
            class_ref = self._type_from_value(item.context_expr)
            if class_ref and _looks_like_class_ref(class_ref) and item.optional_vars:
                self._record_target(item.optional_vars, class_ref)
        for s in node.body:
            self.visit(s)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for item in node.items:
            class_ref = self._type_from_value(item.context_expr)
            if class_ref and _looks_like_class_ref(class_ref) and item.optional_vars:
                self._record_target(item.optional_vars, class_ref)
        for s in node.body:
            self.visit(s)

    # ---- extraction helpers ----

    def _seed_from_params(self, fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        """
        Seed hints from parameter annotations (safe, very useful).
        """
        all_args: List[ast.arg] = []
        all_args.extend(getattr(fn.args, "posonlyargs", []))
        all_args.extend(fn.args.args)
        all_args.extend(fn.args.kwonlyargs)
        if fn.args.vararg:
            all_args.append(fn.args.vararg)
        if fn.args.kwarg:
            all_args.append(fn.args.kwarg)

        for i, a in enumerate(all_args):
            name = a.arg
            if i == 0 and name in {"self", "cls"}:
                continue
            if a.annotation:
                t = self._type_from_annotation(a.annotation)
                if t and _looks_like_class_ref(t):
                    self.hints[name] = t

    def _record_target(self, t: ast.AST, class_ref: str) -> None:
        """
        Record a target -> class_ref in hints.
        Also stores "self.<top>" for deep attribute chains.
        """
        if isinstance(t, ast.Name):
            self.hints[t.id] = class_ref
            return

        if isinstance(t, ast.Attribute):
            path = self._attr_path(t)
            if path:
                self.hints[path] = class_ref
                # additionally store only first attr for self.<attr>
                if path.startswith("self."):
                    parts = path.split(".")
                    if len(parts) >= 2:
                        self.hints[f"self.{parts[1]}"] = class_ref
            return

        # do not try to infer destructuring targets safely
        if isinstance(t, (ast.Tuple, ast.List)):
            return

    def _type_from_value(self, v: Optional[ast.AST]) -> str:
        """
        Infer type reference from expression (best-effort).
        """
        if v is None:
            return ""

        # --- builtins from literals (big share of dynamic_attribute receivers) ---
        # x = [] / {} / set() / () / "..." / 1 / True ...
        if isinstance(v, ast.List):
            return "list"
        if isinstance(v, ast.Dict):
            return "dict"
        if isinstance(v, ast.Set):
            return "set"
        if isinstance(v, ast.Tuple):
            return "tuple"
        if isinstance(v, ast.Constant):
            if isinstance(v.value, str):
                return "str"
            if isinstance(v.value, bool):
                return "bool"
            if isinstance(v.value, int):
                return "int"
            if isinstance(v.value, float):
                return "float"

        # logging.getLogger(...) / getLogger(...) -> logging.Logger
        # (works even if imported as: from logging import getLogger)
        if isinstance(v, ast.Call):
            f = v.func
            if isinstance(f, ast.Attribute) and f.attr == "getLogger":
                return "logging.Logger"
            if isinstance(f, ast.Name) and f.id == "getLogger":
                return "logging.Logger"

        # cast(Type, expr) / typing.cast(Type, expr)
        if isinstance(v, ast.Call) and self._is_cast_call(v):
            if v.args:
                t = self._type_from_annotation(v.args[0])
                return t or ""

        # x = ClassName(...)
        if isinstance(v, ast.Call):
            ref = self._class_ref_from_callee(v.func)
            if ref and _looks_like_class_ref(ref):
                return ref
            return ""

        # x = y (alias propagation handled by caller)
        return ""

    def _is_cast_call(self, call: ast.Call) -> bool:
        f = call.func
        if isinstance(f, ast.Name):
            return f.id in self._CAST_NAMES
        if isinstance(f, ast.Attribute):
            # typing.cast / t.cast
            return f.attr in self._CAST_NAMES
        return False

    def _class_ref_from_callee(self, f: ast.AST) -> str:
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return self._attr_path(f)
        return ""

    def _type_from_annotation(self, ann: Optional[ast.AST]) -> str:
        """
        Parse annotation and return a single "best" type reference.
        Conservative: ignores container generics like list[T] (not useful for internal call resolution),
        but unwraps Optional/Union/Annotated and "| None".
        """
        if ann is None:
            return ""

        # PEP604: T | None  (BinOp BitOr)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            cands = self._collect_union_binop(ann)
            cands = [c for c in cands if c and c != "None"]
            for c in cands:
                if _looks_like_class_ref(c):
                    return c
            return cands[0] if cands else ""

        if isinstance(ann, ast.Name):
            return ann.id

        if isinstance(ann, ast.Attribute):
            return self._attr_path(ann)

        # Optional[T], Union[A,B], Annotated[T, ...]
        if isinstance(ann, ast.Subscript):
            base = self._annotation_base_name(ann.value)
            base_last = base.split(".")[-1] if base else ""

            # unwrap wrappers
            if base_last in self._WRAPPERS:
                inner = self._collect_subscript_args(ann.slice)
                inner = [x for x in inner if x and x != "None"]
                for x in inner:
                    if _looks_like_class_ref(x):
                        return x
                return inner[0] if inner else ""

            # For non-wrapper generics (list[T], dict[K,V], etc.) do NOT infer element type
            # because it causes false positives (list methods != T methods)
            return ""

        # string annotations not handled (need runtime eval) -> ignore
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            # could try parse "Foo" -> Foo, but risky; do minimal:
            s = ann.value.strip()
            return s if _looks_like_class_ref(s) else ""

        return ""

    def _annotation_base_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self._attr_path(node)
        return ""

    def _collect_subscript_args(self, sl: ast.AST) -> List[str]:
        # Python versions differ: slice can be Tuple or Name/Attribute
        if isinstance(sl, ast.Tuple):
            return [self._type_from_annotation(e) for e in sl.elts]
        return [self._type_from_annotation(sl)]

    def _collect_union_binop(self, node: ast.AST) -> List[str]:
        # recursively flatten (A | B | C)
        out: List[str] = []

        def walk(n: ast.AST):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr):
                walk(n.left)
                walk(n.right)
            else:
                out.append(self._type_from_annotation(n))

        walk(node)
        return out

    def _ref_from_expr(self, v: ast.AST) -> str:
        """
        Produce a key compatible with our hints:
          - Name -> "x"
          - Attribute -> "self.attr" / "obj.attr"
        """
        if isinstance(v, ast.Name):
            return v.id
        if isinstance(v, ast.Attribute):
            return self._attr_path(v)
        return ""

    def _attr_path(self, node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            return ""
        return ".".join(reversed(parts))
