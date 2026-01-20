"""AST utility functions for decomposition analysis.

Extracted from DecompositionAnalyzer to reduce god class complexity.
"""
import ast
from pathlib import Path
from typing import Optional, List, Set


def find_def_node(tree: ast.AST, qualname: str, lineno: Optional[int] = None) -> ast.AST:
    """Find a function/method definition node in an AST by qualified name.
    
    Args:
        tree: AST module to search
        qualname: Qualified name (e.g., "MyClass.my_method" or "my_function")
        lineno: Optional line number hint for disambiguation
        
    Returns:
        The FunctionDef or AsyncFunctionDef node
        
    Raises:
        RuntimeError: If the callable is not found
    """
    parts = qualname.split(".")
    body = getattr(tree, "body", [])

    def pick(cands: List[ast.AST]) -> Optional[ast.AST]:
        if not cands:
            return None
        if lineno is None:
            return cands[0]
        exact = [n for n in cands if getattr(n, "lineno", None) == lineno]
        if exact:
            return exact[0]
        return sorted(cands, key=lambda n: abs(int(getattr(n, "lineno", 10**9)) - int(lineno)))[0]

    if len(parts) == 1:
        name = parts[0]
        cands = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        node = pick(cands)
        if node:
            return node
        raise RuntimeError(f"Callable not found: {qualname}")

    method = parts[-1]
    class_chain = parts[:-1]

    cur_body = body
    for cls_name in class_chain:
        cls_candidates = [n for n in cur_body if isinstance(n, ast.ClassDef) and n.name == cls_name]
        if not cls_candidates:
            raise RuntimeError(f"Class not found in qualname: {qualname}")
        cur_body = cls_candidates[0].body

    cands = [n for n in cur_body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method]
    node = pick(cands)
    if node:
        return node
    raise RuntimeError(f"Callable not found: {qualname}")


def get_decorator_name(decorator: ast.AST) -> str:
    """Extract decorator name from AST node.
    
    Args:
        decorator: Decorator AST node
        
    Returns:
        Decorator name as string (e.g., "staticmethod", "functools.lru_cache")
    """
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        left = ast.unparse(decorator.value) if hasattr(ast, "unparse") else ""
        return f"{left}.{decorator.attr}" if left else decorator.attr
    return ""


def get_decorator_kind(fn_node: ast.AST) -> str:
    """Determine the decorator kind for a function node.
    
    Args:
        fn_node: FunctionDef or AsyncFunctionDef node
        
    Returns:
        Decorator kind: "classmethod", "staticmethod", "property", 
        "cached_property", "functools.cached_property", or ""
    """
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    names = [get_decorator_name(d) for d in (fn_node.decorator_list or [])]
    for k in ("classmethod", "staticmethod", "property", "cached_property", "functools.cached_property"):
        if k in names:
            return k
    return ""


def collect_free_names(fn_node: ast.AST) -> Set[str]:
    """Collect free (non-local) variable names used in a function.
    
    Args:
        fn_node: FunctionDef or AsyncFunctionDef node
        
    Returns:
        Set of free variable names (excludes parameters and local assignments)
    """
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()

    params: Set[str] = set()
    for a in getattr(fn_node.args, "posonlyargs", []):
        params.add(a.arg)
    for a in fn_node.args.args:
        params.add(a.arg)
    for a in fn_node.args.kwonlyargs:
        params.add(a.arg)
    if fn_node.args.vararg:
        params.add(fn_node.args.vararg.arg)
    if fn_node.args.kwarg:
        params.add(fn_node.args.kwarg.arg)

    loads: Set[str] = set()
    stores: Set[str] = set()

    class NameCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node): 
            return  # Don't descend into nested functions
        def visit_AsyncFunctionDef(self, node): 
            return
        def visit_ClassDef(self, node): 
            return
        def visit_Lambda(self, node): 
            return
        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                loads.add(node.id)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                stores.add(node.id)

    visitor = NameCollector()
    for stmt in fn_node.body:
        visitor.visit(stmt)

    locals_ = params | stores
    return {n for n in loads if n not in locals_}
