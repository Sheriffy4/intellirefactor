"""SAFE_EXACT evaluation for functional decomposition.

Extracted from DecompositionAnalyzer to reduce god class complexity.
Evaluates whether blocks in a cluster are safe to unify exactly.
"""
import ast
import builtins
import copy
from pathlib import Path
from typing import Set, Tuple, Dict, Optional

from .models import (
    CanonicalizationPlan,
    SimilarityCluster,
    ProjectFunctionalMap,
)


# Module-level constants
NOISE_FUNCS = {"print", "debug", "info", "warning", "error", "critical", "exception", "log"}


def is_logger_like(expr: ast.AST) -> bool:
    """Check if expression is logger-like (logger.info, logging.info, etc.)."""
    if isinstance(expr, ast.Name) and expr.id in ("logger", "logging"):
        return True
    if isinstance(expr, ast.Attribute) and expr.attr == "logger":
        return True
    return False


def is_noise_call(call: ast.Call) -> bool:
    """Check if call is noise (print, logger calls) that can be ignored."""
    # print(...)
    if isinstance(call.func, ast.Name) and call.func.id == "print":
        return True

    # logger.info(...), self.logger.info(...), logging.info(...)
    if isinstance(call.func, ast.Attribute) and call.func.attr in NOISE_FUNCS:
        return is_logger_like(call.func.value)
    return False


def collect_locals(fn: ast.AST) -> Set[str]:
    """Collect local variable names for alpha-renaming.
    
    Collects:
    - Function arguments
    - Assigned targets (Store context)
    
    Excludes:
    - Names declared global/nonlocal
    
    Does not descend into nested defs/classes/lambdas.
    """
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()

    locals_: Set[str] = set()

    # args are locals
    a = fn.args
    for x in getattr(a, "posonlyargs", []):
        locals_.add(x.arg)
    for x in a.args:
        locals_.add(x.arg)
    for x in a.kwonlyargs:
        locals_.add(x.arg)
    if a.vararg:
        locals_.add(a.vararg.arg)
    if a.kwarg:
        locals_.add(a.kwarg.arg)

    globalish: Set[str] = set()

    def add_target(t: ast.AST) -> None:
        """Handle tuple/list destructuring."""
        if isinstance(t, ast.Name):
            locals_.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                add_target(e)
        elif isinstance(t, ast.Starred):
            add_target(t.value)

    class LocalCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            return  # no nested scopes
        def visit_AsyncFunctionDef(self, node):
            return
        def visit_ClassDef(self, node):
            return
        def visit_Lambda(self, node):
            return

        def visit_Global(self, node: ast.Global):
            for n in node.names:
                globalish.add(n)

        def visit_Nonlocal(self, node: ast.Nonlocal):
            for n in node.names:
                globalish.add(n)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Store):
                locals_.add(node.id)

        def visit_Assign(self, node: ast.Assign):
            for t in node.targets:
                add_target(t)
            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign):
            add_target(node.target)
            if node.value:
                self.generic_visit(node.value)

        def visit_AugAssign(self, node: ast.AugAssign):
            add_target(node.target)
            self.generic_visit(node.value)

        def visit_For(self, node: ast.For):
            add_target(node.target)
            self.generic_visit(node.iter)
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)

        def visit_AsyncFor(self, node: ast.AsyncFor):
            add_target(node.target)
            self.generic_visit(node.iter)
            for s in node.body:
                self.visit(s)
            for s in node.orelse:
                self.visit(s)

        def visit_With(self, node: ast.With):
            for item in node.items:
                if item.optional_vars:
                    add_target(item.optional_vars)
            for s in node.body:
                self.visit(s)

        def visit_AsyncWith(self, node: ast.AsyncWith):
            for item in node.items:
                if item.optional_vars:
                    add_target(item.optional_vars)
            for s in node.body:
                self.visit(s)

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            if node.name:
                locals_.add(node.name)
            for s in node.body:
                self.visit(s)

        def visit_comprehension(self, node: ast.comprehension):
            add_target(node.target)
            self.generic_visit(node.iter)
            for if_ in node.ifs:
                self.visit(if_)

    v = LocalCollector()
    for stmt in fn.body:
        v.visit(stmt)

    # if declared global/nonlocal - treat as not-local (do not rename)
    locals_ -= globalish
    return locals_


def signature_shape(fn: ast.AST) -> Tuple[int, int, Tuple[str, ...], bool, bool]:
    """Extract signature shape for compatibility checking.
    
    Returns:
        Tuple of (posonly_count, args_count, kwonly_names, has_vararg, has_kwarg)
    """
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
    a = fn.args
    posonly_n = len(getattr(a, "posonlyargs", []))
    args_n = len(a.args)
    kwonly_names = tuple(x.arg for x in a.kwonlyargs)
    has_vararg = bool(a.vararg)
    has_kwarg = bool(a.kwarg)
    return (posonly_n, args_n, kwonly_names, has_vararg, has_kwarg)


def create_normalizer(locals_set: Set[str]) -> type:
    """Create AST normalizer class for alpha-renaming.
    
    Args:
        locals_set: Set of local variable names to normalize
        
    Returns:
        Normalizer class (ast.NodeTransformer subclass)
    """
    class Normalizer(ast.NodeTransformer):
        def __init__(self):
            self._i = 0
            self._map: Dict[str, str] = {}

        def _m(self, name: str, prefix: str = "v") -> str:
            if name not in self._map:
                self._map[name] = f"{prefix}{self._i}"
                self._i += 1
            return self._map[name]

        def visit(self, node):
            # wipe positions for stable dump
            for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
                if hasattr(node, attr):
                    setattr(node, attr, 0)
            return super().visit(node)

        def visit_Expr(self, node: ast.Expr):
            # IMPORTANT: only safe to replace a noise call if it's a standalone statement
            if isinstance(node.value, ast.Call) and is_noise_call(node.value):
                return ast.Pass()
            return self.generic_visit(node)

        def visit_arg(self, node: ast.arg):
            if node.arg in locals_set and node.arg not in ("self", "cls"):
                node.arg = self._m(node.arg, prefix="a")
            node.annotation = None
            return node

        def visit_Name(self, node: ast.Name):
            if node.id in ("True", "False", "None", "self", "cls"):
                return node
            # rename ONLY locals; keep globals/free names intact
            if node.id in locals_set:
                node.id = self._m(node.id, prefix="v")
            return node

        def visit_Call(self, node: ast.Call):
            # do not treat call itself as noise here; noise handled in visit_Expr
            node.func = self.visit(node.func)
            node.args = [self.visit(a) for a in node.args]
            node.keywords = [self.visit(k) for k in node.keywords]
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef):
            return node  # do not descend into nested defs

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            return node

        def visit_ClassDef(self, node: ast.ClassDef):
            return node

        def visit_Lambda(self, node: ast.Lambda):
            return node

    return Normalizer


def safe_exact_fingerprint(fn: ast.AST, *, is_method: bool, dec_kind: str) -> str:
    """Generate normalized fingerprint for exact comparison.
    
    Alpha-normalizes locals only; keeps free/global names and literals.
    Optionally drops logger/print statements (as standalone statements only).
    
    Args:
        fn: Function/method AST node
        is_method: Whether this is a method
        dec_kind: Decorator kind (classmethod, staticmethod, etc.)
        
    Returns:
        Normalized fingerprint string
    """
    assert isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))

    locals_ = collect_locals(fn)
    fn2 = copy.deepcopy(fn)

    # drop docstring (noise)
    if (
        fn2.body
        and isinstance(fn2.body[0], ast.Expr)
        and isinstance(getattr(fn2.body[0], "value", None), ast.Constant)
        and isinstance(fn2.body[0].value.value, str)
    ):
        fn2.body = fn2.body[1:]

    # remove decorators in AST; semantics tracked in prefix + checks
    fn2.decorator_list = []

    # drop annotations
    for a in getattr(fn2.args, "posonlyargs", []):
        a.annotation = None
    for a in fn2.args.args:
        a.annotation = None
    for a in fn2.args.kwonlyargs:
        a.annotation = None
    if fn2.args.vararg:
        fn2.args.vararg.annotation = None
    if fn2.args.kwarg:
        fn2.args.kwarg.annotation = None
    fn2.returns = None

    # normalize name
    fn2.name = "__X__"

    Normalizer = create_normalizer(locals_)
    fn2 = Normalizer().visit(fn2)
    ast.fix_missing_locations(fn2)

    kind_prefix = (
        f"{'method' if is_method else 'function'}|"
        f"{dec_kind}|"
        f"{'async' if isinstance(fn, ast.AsyncFunctionDef) else 'sync'}"
    )
    return kind_prefix + "|" + ast.dump(fn2, include_attributes=False)


class SafeExactEvaluator:
    """Evaluates whether blocks in a cluster are safe to unify exactly."""
    
    def __init__(self, file_ops, ast_utils_module, safe_decorators_allowlist: Set[str]):
        """Initialize evaluator.
        
        Args:
            file_ops: FileOperations instance
            ast_utils_module: ast_utils module for utility functions
            safe_decorators_allowlist: Set of allowed decorator names
        """
        self.file_ops = file_ops
        self.ast_utils = ast_utils_module
        self.safe_decorators = safe_decorators_allowlist
    
    def evaluate_safe_exact(
        self,
        plan: CanonicalizationPlan,
        cluster: Optional[SimilarityCluster],
        fm: ProjectFunctionalMap,
        package_root: Path,
    ) -> Tuple[str, str]:
        """Evaluate if cluster blocks are safe to unify exactly.
        
        Args:
            plan: Canonicalization plan
            cluster: Similarity cluster
            fm: Project functional map
            package_root: Package root path
            
        Returns:
            Tuple of (status, reason) where status is "SAFE_EXACT_OK" or "SAFE_EXACT_FAIL"
        """
        if not cluster:
            return "SAFE_EXACT_FAIL", "cluster not found"

        block_ids = list(cluster.blocks or [])
        if len(block_ids) < 2:
            return "SAFE_EXACT_FAIL", "cluster has <2 blocks"

        # Collect data from all blocks
        fps: Set[str] = set()
        method_names: Set[str] = set()
        kinds: Set[str] = set()
        dec_kinds: Set[str] = set()
        sig_shapes: Set[Tuple[int, int, Tuple[str, ...], bool, bool]] = set()
        free_sets: Set[Tuple[str, ...]] = set()

        for bid in block_ids:
            b = fm.blocks.get(bid)
            if not b:
                return "SAFE_EXACT_FAIL", f"missing block id {bid}"

            kinds.add("method" if bool(getattr(b, "is_method", False)) else "function")
            if getattr(b, "is_method", False):
                method_names.add(str(getattr(b, "method_name", "")))

            file_path = self.file_ops.resolve_block_file_path(b, package_root)
            if not file_path.exists():
                return "SAFE_EXACT_FAIL", f"file not found for {b.qualname}: {file_path}"

            try:
                tree = self.file_ops.parse_file(file_path)
                node = self.ast_utils.find_def_node(tree, b.qualname, lineno=b.lineno)
            except Exception as e:
                return "SAFE_EXACT_FAIL", f"cannot locate callable node for {b.qualname}: {e}"

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return "SAFE_EXACT_FAIL", f"node is not a function def: {b.qualname}"

            # decorator safety: wrappers support only allowlisted decorators
            dec_names = [self.ast_utils.get_decorator_name(d) for d in (node.decorator_list or [])]
            bad = [d for d in dec_names if d and d not in self.safe_decorators]
            if bad:
                return "SAFE_EXACT_FAIL", f"unsupported decorators on {b.qualname}: {bad}"

            dec_kind = self.ast_utils.get_decorator_kind(node)
            dec_kinds.add(dec_kind)

            sig_shapes.add(signature_shape(node))

            # free-name safety: if global deps differ, do NOT safe-unify
            free = self.ast_utils.collect_free_names(node)
            free = {n for n in free if not hasattr(builtins, n)}
            free_sets.add(tuple(sorted(free)))

            fp = safe_exact_fingerprint(
                node, 
                is_method=bool(getattr(b, "is_method", False)), 
                dec_kind=dec_kind
            )
            fps.add(fp)

        # Validation checks
        # 1) Do not mix methods and functions in safe exact
        if len(kinds) > 1:
            return "SAFE_EXACT_FAIL", f"mixed callable kinds in cluster: {sorted(kinds)}"

        # 2) Methods: names must match (API surface)
        if "method" in kinds and len({n for n in method_names if n}) > 1:
            return "SAFE_EXACT_FAIL", f"method names differ in cluster: {sorted(method_names)}"

        # 3) Decorator kind must match (classmethod vs method, property, etc.)
        if len({k for k in dec_kinds if k}) > 1:
            return "SAFE_EXACT_FAIL", f"decorator kinds differ in cluster: {sorted(dec_kinds)}"

        # 4) Signature shape compatibility (kwonly names are important!)
        if len(sig_shapes) > 1:
            return "SAFE_EXACT_FAIL", f"signature shapes differ ({len(sig_shapes)} variants)"

        # 5) Free-name sets must match (avoid different globals/imports/constants)
        if len(free_sets) > 1:
            return "SAFE_EXACT_FAIL", f"free-name sets differ ({len(free_sets)} variants)"

        # 6) Finally: normalized fingerprints must match
        if len(fps) == 1:
            return "SAFE_EXACT_OK", "alpha-normalized bodies match (locals normalized, globals/literals preserved)"

        return "SAFE_EXACT_FAIL", f"normalized body fingerprints differ ({len(fps)} variants)"
