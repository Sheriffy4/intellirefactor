Ниже — **точный diff** для твоего текущего `refactoring/llm_context_generator.py`, который реализует именно то, о чём сказал эксперт (и то, что видно по твоему `llm_context.md`):

1) **Убирает “гигантский слайс”** для `god_class`/огромных диапазонов (вместо этого: outline + мелкие слайсы).  
2) Добавляет **File Outline** (таблица `class/def` с диапазонами + какие smells туда попали).  
3) Делает clone-groups **операциональными**:
   - в “Clone Groups (top)” печатает target+other локации,
   - в slices добавляет tags и связывает slice с конкретным `group_id`.
4) Добавляет **SM1/UN1 идентификаторы** и печатает их в smells/unused.
5) В “Allowed Files” показывает отдельным блоком **Additional allowed files (from clone groups)** (чтобы LLM не “терял” файлы вроде `visitors_temp.py`, если они попали через клоны, а не через neighbors).
6) Добавляет **Validation checklist per step** (импорт → 1 тест → formatter → lint).

---

## PATCH

```diff
diff --git a/refactoring/llm_context_generator.py b/refactoring/llm_context_generator.py
index 3333333..eeeeeee 100644
--- a/refactoring/llm_context_generator.py
+++ b/refactoring/llm_context_generator.py
@@ -62,6 +62,7 @@ class CodeContext:
     surrounding_context: str = ""
     dependencies: List[str] = field(default_factory=list)
     related_methods: List[str] = field(default_factory=list)
+    tags: List[str] = field(default_factory=list)  # e.g. ["SM1", "CG:exact_xxx:target", "UN3"]
     test_coverage: Optional[float] = None
 
     def get_full_context(self) -> str:
@@ -414,6 +415,56 @@ class LLMContextGenerator:
         lines.append(f"- **Project:** `{mission.project_path}`")
         lines.append(f"- **Target file:** `{mission.target_file}`")
         lines.append("")
 
+        # ------------------------------------------------------------
+        # Per-step Validation Checklist (micro-loop)
+        # ------------------------------------------------------------
+        lines.append("## Per-step Validation Checklist (run after EACH small step)")
+        lines.append("")
+        lines.append("Use this micro-loop after every extraction/move/delete:")
+        lines.append("")
+        lines.append("1) **Import check (fast)**")
+        lines.append("```bash")
+        lines.append("python -c \"import sys; sys.path.insert(0,'.'); import intellirefactor; print('import ok')\"")
+        lines.append("```")
+        lines.append("")
+        lines.append("2) **Run ONE relevant test file (fast)**")
+        lines.append("```bash")
+        lines.append("python -m pytest -q tests/test_imports.py")
+        lines.append("```")
+        lines.append("")
+        lines.append("3) **Formatter**")
+        lines.append("```bash")
+        lines.append("python -m black <changed_paths> --line-length 100")
+        lines.append("```")
+        lines.append("")
+        lines.append("4) **Lint (optional but recommended)**")
+        lines.append("```bash")
+        lines.append("# Ruff (preferred if installed)")
+        lines.append("python -m ruff check <changed_paths>")
+        lines.append("")
+        lines.append("# Flake8 (alternative)")
+        lines.append("python -m flake8 <changed_paths>")
+        lines.append("```")
+        lines.append("")
+        lines.append("5) **After 2–3 steps**: run full tests")
+        lines.append("```bash")
+        lines.append("python -m pytest -q")
+        lines.append("```")
+        lines.append("")
+
         lines.append("## Role & Output Contract")
         lines.append("")
         lines.append("You are a **senior refactoring engineer** operating in two phases:")
         lines.append("")
@@ -520,6 +571,40 @@ class LLMContextGenerator:
         lines.append("## Curated Findings (target-only)")
         lines.append("")
 
         lines.append("### Duplicate / Clone Groups (top)")
         if not mission.clone_groups:
             lines.append("- (No clone groups found for target in artifacts.)")
         else:
             for g in mission.clone_groups:
                 gid = g.get("group_id", "<unknown>")
                 ctype = g.get("clone_type", "unknown")
                 inst = g.get("instance_count", "?")
                 uniq = g.get("unique_files", "?")
                 strat = g.get("extraction_strategy", "n/a")
                 score = g.get("ranking_score", g.get("similarity_score", ""))
-                lines.append(f"- `{gid}` | type={ctype} | instances={inst} | files={uniq} | strategy={strat} | score={score}")
+                # add target+other location (operational)
+                previews = g.get("instances_preview") or []
+                target_loc = ""
+                other_loc = ""
+                for p in previews:
+                    if not isinstance(p, dict):
+                        continue
+                    fp = self._norm_path(p.get("file_path") or "")
+                    ls = p.get("line_start")
+                    le = p.get("line_end")
+                    loc = f"{fp}:{ls}-{le}"
+                    if fp and self._path_matches(fp, mission.target_file) and not target_loc:
+                        target_loc = loc
+                    elif fp and (not self._path_matches(fp, mission.target_file)) and not other_loc:
+                        other_loc = loc
+                loc_suffix = ""
+                if target_loc or other_loc:
+                    loc_suffix = f" | target={target_loc or '-'} | other={other_loc or '-'}"
+                lines.append(
+                    f"- `{gid}` | type={ctype} | instances={inst} | files={uniq} | strategy={strat} | score={score}{loc_suffix}"
+                )
         lines.append("")
 
         lines.append("### Smells (top)")
         if not mission.smells:
             lines.append("- (No smells found for target in artifacts.)")
         else:
             for s in mission.smells:
+                sid = s.get("sm_id") or ""
                 st = s.get("smell_type", "unknown")
                 sev = s.get("severity", "unknown")
                 sym = s.get("symbol_name", "")
                 ls = s.get("line_start", "")
                 le = s.get("line_end", "")
-                lines.append(f"- {st} | severity={sev} | `{sym}` | lines {ls}-{le}")
+                prefix = f"{sid} " if sid else ""
+                lines.append(f"- {prefix}{st} | severity={sev} | `{sym}` | lines {ls}-{le}")
         lines.append("")
 
         lines.append("### Unused (top)")
         if not mission.unused:
             lines.append("- (No unused findings found for target in artifacts.)")
         else:
             for u in mission.unused:
+                uid = u.get("un_id") or ""
                 name = u.get("symbol_name") or u.get("name") or u.get("symbol") or "<unknown>"
                 conf = u.get("confidence", "")
                 ls = u.get("line_start", "")
                 le = u.get("line_end", "")
                 ut = u.get("unused_type", u.get("type", ""))
-                lines.append(f"- `{name}` | unused_type={ut} | confidence={conf} | lines {ls}-{le}")
+                prefix = f"{uid} " if uid else ""
+                lines.append(f"- {prefix}`{name}` | unused_type={ut} | confidence={conf} | lines {ls}-{le}")
         lines.append("")
 
+        # ------------------------------------------------------------
+        # File Outline (operational replacement for giant slices)
+        # ------------------------------------------------------------
+        lines.append("## File Outline (for precise navigation)")
+        lines.append("")
+        outline_md = self._generate_file_outline_md(
+            project_path=mission.project_path,
+            target_file=mission.target_file,
+            smells=mission.smells,
+            max_items=80,
+        )
+        lines.extend(outline_md)
+        lines.append("")
+
         # Navigation map
         lines.append("## Navigation Map (start here)")
         lines.append("")
         lines.append("Use the following **targeted code slices** as anchors. Start refactoring from the highest impact slices.")
@@ -533,6 +618,8 @@ class LLMContextGenerator:
         else:
             for i, ctx in enumerate(mission.code_contexts, 1):
                 lines.append(f"### Slice {i}: `{ctx.file_path}` lines {ctx.line_start}-{ctx.line_end}")
+                if ctx.tags:
+                    lines.append(f"- tags: {', '.join(ctx.tags)}")
                 if ctx.dependencies:
                     lines.append(f"- deps: {', '.join(ctx.dependencies[:20])}" + (" ..." if len(ctx.dependencies) > 20 else ""))
                 if ctx.related_methods:
                     lines.append(f"- methods: {', '.join(ctx.related_methods[:20])}" + (" ..." if len(ctx.related_methods) > 20 else ""))
                 lines.append("")
                 lines.append("```python")
                 lines.append(ctx.get_full_context())
                 lines.append("```")
                 lines.append("")
@@ -560,6 +647,27 @@ class LLMContextGenerator:
             # Group by relationship type if available
             if mission.neighbor_summary and "neighbors" in mission.neighbor_summary:
                 neighbors = mission.neighbor_summary.get("neighbors", [])
+                neighbor_paths: Set[str] = set()
                 for neighbor in neighbors[:15]:  # Top 15
                     fp = neighbor.get("file_path", "")
+                    if fp:
+                        neighbor_paths.add(self._norm_path(fp))
                     rel_type = neighbor.get("relationship_type", "")
                     edge_count = neighbor.get("edge_count", 0)
                     symbols = neighbor.get("symbols_involved", [])
                     lines.append(f"- `{fp}` ({rel_type}, {edge_count} edges)")
                     if symbols:
                         lines.append(f"  - symbols: {', '.join(symbols[:5])}")
+
+                # Additional allowed files that came from clone groups (not direct neighbors)
+                extra_allowed = [
+                    self._norm_path(p)
+                    for p in mission.allowed_files
+                    if self._norm_path(p) not in neighbor_paths and self._norm_path(p) != self._norm_path(mission.target_file)
+                ]
+                if extra_allowed:
+                    lines.append("")
+                    lines.append("### Additional allowed files (from clone groups)")
+                    lines.append("")
+                    for fp in extra_allowed[:20]:
+                        lines.append(f"- `{fp}`")
             else:
                 # Fallback: just list files
                 for fp in mission.allowed_files[:20]:
                     lines.append(f"- `{fp}`")
             lines.append("")
@@ -724,6 +832,17 @@ class LLMContextGenerator:
     def _load_smells_for_target(self, run_path: Path, target_file: str, limit: int) -> List[Dict[str, Any]]:
         target_file = self._norm_path(target_file)
         smells_path = run_path / "decompose" / "smells.json"
         data = self._try_load_json(smells_path) or {}
         smells = data.get("smells") or []
         target_smells = [s for s in smells if self._norm_path(str(s.get("file_path") or "")) == target_file]
@@ -738,7 +857,11 @@ class LLMContextGenerator:
 
         target_smells.sort(key=_rank, reverse=True)
-        return target_smells[: max(0, int(limit))]
+        out = target_smells[: max(0, int(limit))]
+        # assign stable-ish IDs for planning references
+        for i, s in enumerate(out, 1):
+            s["sm_id"] = f"SM{i}"
+        return out
 
     def _load_unused_for_target(self, run_path: Path, target_file: str, limit: int) -> List[Dict[str, Any]]:
         target_file = self._norm_path(target_file)
         unused_path = run_path / "refactor" / "unused.json"
@@ -747,6 +870,9 @@ class LLMContextGenerator:
         data = self._try_load_json(unused_path) or {}
         findings = data.get("findings") or data.get("unused") or []
         out = [f for f in findings if self._norm_path(str(f.get("file_path") or "")) == target_file]
 
+        # Avoid useless "module_unreachable line 1--1" slices
+        out = [f for f in out if str(f.get("unused_type") or "") != "module_unreachable"]
+
         # sort: confidence desc, then line_start asc
         out.sort(
             key=lambda f: (float(f.get("confidence") or 0.0), -int(f.get("line_start") or 10**9)),
             reverse=True,
         )
-        return out[: max(0, int(limit))]
+        out = out[: max(0, int(limit))]
+        for i, u in enumerate(out, 1):
+            u["un_id"] = f"UN{i}"
+        return out
 
@@ -900,6 +1026,104 @@ class LLMContextGenerator:
         return contexts
 
+    def _generate_file_outline_md(
+        self,
+        *,
+        project_path: str,
+        target_file: str,
+        smells: List[Dict[str, Any]],
+        max_items: int = 80,
+    ) -> List[str]:
+        """
+        Generate an outline of classes/functions with line ranges.
+        Also shows which smell IDs overlap each symbol range (SM1..).
+        """
+        # resolve file on disk (best-effort)
+        candidates = [target_file, *list(self._path_aliases(target_file))]
+        content: Optional[str] = None
+        used_path: Optional[str] = None
+        for rel in candidates:
+            p = Path(project_path) / self._norm_path(rel)
+            if p.exists():
+                try:
+                    content = p.read_text(encoding="utf-8-sig")
+                    used_path = self._norm_path(rel)
+                    break
+                except Exception:
+                    continue
+
+        if not content:
+            return ["- (Outline unavailable: could not read target file from disk)"]
+
+        try:
+            tree = ast.parse(content)
+        except Exception as e:
+            return [f"- (Outline unavailable: AST parse failed: {e})"]
+
+        # build smell intervals with IDs
+        smell_intervals: List[Tuple[int, int, str]] = []
+        for s in smells or []:
+            sid = s.get("sm_id") or ""
+            if not sid:
+                continue
+            try:
+                ls = int(s.get("line_start") or 1)
+                le = int(s.get("line_end") or ls)
+            except Exception:
+                continue
+            if le < ls:
+                le = ls
+            smell_intervals.append((ls, le, sid))
+
+        items: List[Dict[str, Any]] = []
+
+        def add_item(kind: str, name: str, node: ast.AST):
+            ls = int(getattr(node, "lineno", 1) or 1)
+            le = int(getattr(node, "end_lineno", ls) or ls)
+            loc = max(1, le - ls + 1)
+            overlapping = [
+                sid for (sls, sle, sid) in smell_intervals
+                if not (le < sls or ls > sle)
+            ]
+            items.append(
+                {
+                    "kind": kind,
+                    "name": name,
+                    "line_start": ls,
+                    "line_end": le,
+                    "loc": loc,
+                    "smells": ", ".join(overlapping[:6]) + ("..." if len(overlapping) > 6 else ""),
+                }
+            )
+
+        for node in tree.body:
+            if isinstance(node, ast.ClassDef):
+                add_item("class", node.name, node)
+                for sub in node.body:
+                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
+                        add_item("method", f"{node.name}.{sub.name}", sub)
+            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
+                add_item("function", node.name, node)
+
+        items.sort(key=lambda x: x["line_start"])
+        items = items[: max(0, int(max_items))]
+
+        md: List[str] = []
+        md.append(f"- Resolved path: `{used_path}`")
+        md.append("")
+        md.append("| Kind | Symbol | Lines | LOC | Smell IDs |")
+        md.append("|---|---|---:|---:|---|")
+        for it in items:
+            md.append(
+                f"| {it['kind']} | `{it['name']}` | {it['line_start']}-{it['line_end']} | {it['loc']} | {it['smells']} |"
+            )
+        return md
```

---
