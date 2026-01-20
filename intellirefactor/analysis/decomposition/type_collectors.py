"""
Type Hint Collectors

Collects type hints from module-level and function-level AST nodes.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Union

from .ast_helpers import looks_like_class_ref, attr_path_from_ast


class ModuleTypeHintCollector:
    """
    Collect lightweight module-scope type hints.

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

        if class_ref and looks_like_class_ref(class_ref):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.hints[t.id] = class_ref

    def _handle_annassign(self, node: ast.AnnAssign) -> None:
        # Prefer constructor-based type, else annotation.
        class_ref = self._type_from_value(node.value) if node.value else ""
        if not class_ref:
            class_ref = self._type_from_annotation(node.annotation)

        # alias propagation: x: T = y
        if (
            not class_ref
            and node.value
            and isinstance(node.value, ast.Name)
            and node.value.id in self.hints
        ):
            class_ref = self.hints[node.value.id]

        if (
            class_ref
            and looks_like_class_ref(class_ref)
            and isinstance(node.target, ast.Name)
        ):
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
            if ref and looks_like_class_ref(ref):
                return ref
            return ""

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
            return attr_path_from_ast(f)
        return ""

    def _type_from_annotation(self, ann: Optional[ast.AST]) -> str:
        if ann is None:
            return ""

        # PEP604: T | None
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            cands = self._collect_union_binop(ann)
            cands = [c for c in cands if c and c != "None"]
            for c in cands:
                if looks_like_class_ref(c):
                    return c
            return cands[0] if cands else ""

        if isinstance(ann, ast.Name):
            return ann.id

        if isinstance(ann, ast.Attribute):
            return attr_path_from_ast(ann)

        if isinstance(ann, ast.Subscript):
            base = self._annotation_base_name(ann.value)
            base_last = base.split(".")[-1] if base else ""

            if base_last in self._WRAPPERS:
                inner = self._collect_subscript_args(ann.slice)
                inner = [x for x in inner if x and x != "None"]
                for x in inner:
                    if looks_like_class_ref(x):
                        return x
                return inner[0] if inner else ""

            return ""

        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            s = ann.value.strip()
            return s if looks_like_class_ref(s) else ""

        return ""

    def _annotation_base_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return attr_path_from_ast(node)
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


class LocalTypeHintVisitor(ast.NodeVisitor):
    """
    Collect robust local type hints within a function scope.

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
            for k in set(body_h.keys()) & set(else_h.keys()):
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

        # finally branch
        self.hints = dict(base)
        for s in node.finalbody:
            self.visit(s)
        fin_h = dict(self.hints)

        # Merge logic: promote only keys equal in all paths
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

        if class_ref and looks_like_class_ref(class_ref):
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

        if class_ref and looks_like_class_ref(class_ref):
            self._record_target(node.target, class_ref)

    def visit_With(self, node: ast.With):
        for item in node.items:
            class_ref = self._type_from_value(item.context_expr)
            if class_ref and looks_like_class_ref(class_ref) and item.optional_vars:
                self._record_target(item.optional_vars, class_ref)
        for s in node.body:
            self.visit(s)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for item in node.items:
            class_ref = self._type_from_value(item.context_expr)
            if class_ref and looks_like_class_ref(class_ref) and item.optional_vars:
                self._record_target(item.optional_vars, class_ref)
        for s in node.body:
            self.visit(s)

    # ---- extraction helpers ----

    def _seed_from_params(
        self, fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        """Seed hints from parameter annotations (safe, very useful)."""
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
                if t and looks_like_class_ref(t):
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
            path = attr_path_from_ast(t)
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
        """Infer type reference from expression (best-effort)."""
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
            if ref and looks_like_class_ref(ref):
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
            return attr_path_from_ast(f)
        return ""

    def _type_from_annotation(self, ann: Optional[ast.AST]) -> str:
        """
        Parse annotation and return a single "best" type reference.
        Conservative: ignores container generics, unwraps Optional/Union/Annotated.
        """
        if ann is None:
            return ""

        # PEP604: T | None  (BinOp BitOr)
        if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
            cands = self._collect_union_binop(ann)
            cands = [c for c in cands if c and c != "None"]
            for c in cands:
                if looks_like_class_ref(c):
                    return c
            return cands[0] if cands else ""

        if isinstance(ann, ast.Name):
            return ann.id

        if isinstance(ann, ast.Attribute):
            return attr_path_from_ast(ann)

        # Optional[T], Union[A,B], Annotated[T, ...]
        if isinstance(ann, ast.Subscript):
            base = self._annotation_base_name(ann.value)
            base_last = base.split(".")[-1] if base else ""

            # unwrap wrappers
            if base_last in self._WRAPPERS:
                inner = self._collect_subscript_args(ann.slice)
                inner = [x for x in inner if x and x != "None"]
                for x in inner:
                    if looks_like_class_ref(x):
                        return x
                return inner[0] if inner else ""

            return ""

        # string annotations
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            s = ann.value.strip()
            return s if looks_like_class_ref(s) else ""

        return ""

    def _annotation_base_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return attr_path_from_ast(node)
        return ""

    def _collect_subscript_args(self, sl: ast.AST) -> List[str]:
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
            return attr_path_from_ast(v)
        return ""
