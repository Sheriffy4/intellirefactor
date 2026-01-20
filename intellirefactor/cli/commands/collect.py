from __future__ import annotations

import fnmatch
import csv
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from intellirefactor.cli.artifacts import ArtifactWriter
from intellirefactor.config import load_config
from intellirefactor.api import IntelliRefactor

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".intellirefactor",
    "intellirefactor_out",
}

_DEFAULT_EXCLUDE_GLOBS: List[str] = [
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.intellirefactor/**",
    "**/intellirefactor_out/**",
    "**/archive/**",
    # local scratch / patch artifacts (avoid polluting reports)
    "**/patched_*.py",
    "**/temp_*.py",
]


class _SkipAnalysis(Exception):
    """Internal control-flow: used to skip expensive/duplicate steps deterministically."""


def _normpath(p: str) -> str:
    return str(p or "").replace("\\", "/")


def _relposix(project_root: Path, p: Any) -> str:
    """
    Canonical path for artifacts: POSIX, relative to project_root whenever possible.
    """
    try:
        if isinstance(p, Path):
            pp = p
        else:
            pp = Path(str(p or ""))
        # if relative - interpret under project root
        if not pp.is_absolute():
            pp = (project_root / pp)
        try:
            pp = pp.resolve()
        except Exception:
            pass
        try:
            return pp.relative_to(project_root.resolve()).as_posix()
        except Exception:
            return pp.as_posix()
    except Exception:
        return _normpath(str(p or ""))

def _canonicalize_audit_payload(project_root: Path, audit_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort normalize file_path fields inside audit payload so correlation works.
    """
    if not isinstance(audit_payload, dict):
        return audit_payload
    findings = audit_payload.get("findings")
    if isinstance(findings, list):
        for f in findings:
            if not isinstance(f, dict):
                continue
            if f.get("file_path"):
                f["file_path"] = _relposix(project_root, f["file_path"])
            ev = f.get("evidence")
            if isinstance(ev, dict):
                for key in ("file_references", "locations"):
                    lst = ev.get(key)
                    if isinstance(lst, list):
                        for loc in lst:
                            if isinstance(loc, dict) and loc.get("file_path"):
                                loc["file_path"] = _relposix(project_root, loc["file_path"])
    return audit_payload

def _same_path(a: str, b: str) -> bool:
    """
    Best-effort path comparison for cases where one side is absolute and the other is relative.
    """
    aa = _normpath(a)
    bb = _normpath(b)
    if not aa or not bb:
        return False
    return aa == bb or aa.endswith(bb) or bb.endswith(aa)


def _merge_exclude_globs(user_globs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for g in list(user_globs or []) + _DEFAULT_EXCLUDE_GLOBS:
        if not g:
            continue
        gg = str(g).replace("\\", "/")
        if gg not in seen:
            seen.add(gg)
            out.append(gg)
    return out


def _as_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, list, dict)):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return getattr(obj, "__dict__", str(obj))


def _is_excluded(path: Path, project_root: Path, exclude_globs: Sequence[str]) -> bool:
    rel = None
    try:
        rel = path.relative_to(project_root).as_posix()
    except Exception:
        rel = path.as_posix()
    for pat in exclude_globs:
        # normalize patterns to posix-style matching
        pat2 = pat.replace("\\", "/")
        if fnmatch.fnmatch(rel, pat2):
            return True
    return False


def _discover_py_files(
    project_root: Path, out_base_dir: Path, extra_excludes: Sequence[str]
) -> List[Path]:
    """
    Collect .py files, excluding:
      - output base directory (collect artifacts, all runs)
      - project_root/.intellirefactor
      - extra excludes (glob patterns)
    """
    files: List[Path] = []
    out_base_dir = out_base_dir.resolve()
    ir_dir = (project_root / ".intellirefactor").resolve()

    for p in project_root.rglob("*.py"):
        try:
            rp = p.resolve()
        except Exception:
            rp = p

        # skip artifacts and internal db/logs
        if out_base_dir in rp.parents or rp == out_base_dir:
            continue
        if ir_dir in rp.parents or rp == ir_dir:
            continue

        # skip common non-source dirs even if user forgot to exclude them
        try:
            rel_parts = set(p.relative_to(project_root).parts)
        except Exception:
            rel_parts = set(p.parts)
        if rel_parts.intersection(_DEFAULT_SKIP_DIRS):
            continue

        if _is_excluded(p, project_root, extra_excludes):
            continue

        files.append(p)
    # deterministic order for stable artifacts
    files.sort(key=lambda x: x.as_posix())
    return files


def _md_kv(title: str, items: List[Tuple[str, Any]]) -> str:
    lines = [f"# {title}", ""]
    for k, v in items:
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    return "\n".join(lines)


def _md_table(title: str, headers: List[str], rows: List[List[Any]]) -> str:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    lines.append("")
    return "\n".join(lines)


_SEVERITY_WEIGHT: Dict[str, float] = {
    "critical": 10.0,
    "high": 6.0,
    "medium": 3.0,
    "low": 1.0,
    "info": 0.25,
}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _severity_weight(sev: str) -> float:
    return _SEVERITY_WEIGHT.get(str(sev).lower(), 1.0)


def _compute_health_score(
    *,
    audit_payload: Dict[str, Any],
    index_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simple, deterministic health score (0..100).
    Uses only audit statistics + optional index stats.
    """
    stats = (audit_payload or {}).get("statistics", {}) if isinstance(audit_payload, dict) else {}
    by_sev = (stats.get("findings_by_severity") or {}) if isinstance(stats, dict) else {}
    by_type = (stats.get("findings_by_type") or {}) if isinstance(stats, dict) else {}

    critical = _safe_int(by_sev.get("critical", 0))
    high = _safe_int(by_sev.get("high", 0))
    medium = _safe_int(by_sev.get("medium", 0))
    low = _safe_int(by_sev.get("low", 0))

    # Weighted penalty (tunable later)
    penalty = 0.0
    penalty += critical * 25.0
    penalty += high * 10.0
    penalty += medium * 4.0
    penalty += low * 1.5

    # Extra mild penalties by category (keeps score sensitive to scale)
    penalty += _safe_int(by_type.get("duplicate_block", 0)) * 0.8
    penalty += _safe_int(by_type.get("unused_code", 0)) * 0.4
    penalty += _safe_int(by_type.get("quality_issue", 0)) * 0.3

    # Normalize by project size to avoid always hitting 0 on real projects.
    files_count = 0
    if isinstance(index_status, dict):
        files_count = _safe_int(index_status.get("files_count", 0))
    if files_count <= 0:
        files_count = max(1, _safe_int(stats.get("files_analyzed", 0), 1))

    # Logistic-ish scaling (deterministic, stable, not collapsing to 0)
    scale = max(1.0, float(files_count) * 15.0)
    penalty_norm = float(penalty) / scale
    score = 100.0 / (1.0 + penalty_norm)
    score_int = int(round(score))

    return {
        "score": score_int,
        "score_raw": score,
        "inputs": {
            "findings_by_severity": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
            },
            "findings_by_type": dict(by_type) if isinstance(by_type, dict) else {},
            "index": dict(index_status) if isinstance(index_status, dict) else None,
        },
        "weights": {
            "critical": 25.0,
            "high": 10.0,
            "medium": 4.0,
            "low": 1.5,
            "duplicate_block": 0.8,
            "unused_code": 0.4,
            "quality_issue": 0.3,
        },
        "computed_at": datetime.now().isoformat(),
    }


def _compute_hotspots(audit_payload: Dict[str, Any], *, top_n: int = 25) -> Dict[str, Any]:
    findings = (audit_payload or {}).get("findings", []) if isinstance(audit_payload, dict) else []
    per_file: Dict[str, Dict[str, Any]] = {}

    for f in findings or []:
        if not isinstance(f, dict):
            continue
        file_path = str(f.get("file_path") or "")
        if not file_path:
            continue

        sev = str(f.get("severity") or "low").lower()
        conf = _safe_float(f.get("confidence", 1.0), 1.0)
        ftype = str(f.get("finding_type") or "unknown")

        entry = per_file.setdefault(
            file_path,
            {
                "file_path": file_path,
                "score": 0.0,
                "findings_total": 0,
                "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
                "by_type": {},
                "top_examples": [],
            },
        )

        w = _severity_weight(sev) * conf
        entry["score"] += w
        entry["findings_total"] += 1
        entry["by_severity"][sev] = entry["by_severity"].get(sev, 0) + 1
        entry["by_type"][ftype] = entry["by_type"].get(ftype, 0) + 1

        # Keep a few examples for UI previews
        if len(entry["top_examples"]) < 5:
            entry["top_examples"].append(
                {
                    "finding_id": f.get("finding_id"),
                    "finding_type": ftype,
                    "severity": sev,
                    "confidence": conf,
                    "title": f.get("title"),
                    "line_start": f.get("line_start"),
                    "line_end": f.get("line_end"),
                }
            )

    hotspots = sorted(per_file.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    hotspots = hotspots[: int(top_n)]

    return {
        "top_n": int(top_n),
        "total_files_with_findings": len(per_file),
        "hotspots": hotspots,
        "computed_at": datetime.now().isoformat(),
        "scoring": {
            "severity_weight": dict(_SEVERITY_WEIGHT),
            "formula": "sum(severity_weight[sev] * confidence) per finding",
        },
    }


def _compute_dependency_hubs(
    *,
    db_path: Path,
    hotspots_payload: Optional[Dict[str, Any]] = None,
    top_n_files: int = 25,
    top_n_targets: int = 50,
) -> Dict[str, Any]:
    """
    Compute dependency hubs from SQLite index.

    Canonical schema assumed (v3): dependencies.dependency_kind + dependencies.usage_count.
    """
    from intellirefactor.analysis.index.store import IndexStore
    store = IndexStore(db_path)

    hotspots = (hotspots_payload or {}).get("hotspots", []) if isinstance(hotspots_payload, dict) else []
    hotspot_scores: List[float] = []
    for h in hotspots:
        if isinstance(h, dict):
            hotspot_scores.append(_safe_float(h.get("score", 0.0), 0.0))
    max_hotspot = max(hotspot_scores) if hotspot_scores else 0.0

    def hotspot_score_for_db_path(db_file_path: str) -> float:
        """
        Match DB relative file paths to hotspot absolute paths by suffix.
        Example:
          db: "intellirefactor/analysis/index/store.py"
          hotspot: "C:\\...\\intellirefactor\\analysis\\index\\store.py"
        """
        db_norm = db_file_path.replace("\\", "/")
        for h in hotspots:
            if not isinstance(h, dict):
                continue
            hp = str(h.get("file_path") or "").replace("\\", "/")
            if hp.endswith(db_norm):
                return _safe_float(h.get("score", 0.0), 0.0)
        return 0.0

    top_files: List[Dict[str, Any]] = []
    top_targets: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {}

    with store._get_connection() as conn:
        # --- totals
        try:
            cur = conn.execute("SELECT COUNT(*) FROM dependencies")
            totals["dependencies_total"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["dependencies_total"] = None

        # --- top files by outgoing dependencies + incoming (resolved only)
        rows = conn.execute(
            """
            SELECT
                f.file_path as file_path,
                SUM(COALESCE(d.usage_count, 1)) as deps_total,
                SUM(CASE WHEN d.dependency_kind = 'imports' THEN COALESCE(d.usage_count, 1) ELSE 0 END) as imports_count,
                SUM(CASE WHEN d.dependency_kind = 'calls' THEN COALESCE(d.usage_count, 1) ELSE 0 END) as calls_count,
                COUNT(DISTINCT NULLIF(d.target_external, '')) as unique_targets,
                (
                    SELECT SUM(COALESCE(din.usage_count, 1))
                    FROM dependencies din
                    JOIN symbols ts ON din.target_symbol_id = ts.symbol_id
                    JOIN files tf ON ts.file_id = tf.file_id
                    WHERE tf.file_path = f.file_path
                ) as fanin_total
            FROM dependencies d
            JOIN symbols s ON d.source_symbol_id = s.symbol_id
            JOIN files f ON s.file_id = f.file_id
            GROUP BY f.file_path
            ORDER BY deps_total DESC
            LIMIT ?
            """,
            (int(top_n_files),),
        ).fetchall()

        for (file_path, deps_total, imports_count, calls_count, unique_targets, fanin_total) in rows:
            hs = hotspot_score_for_db_path(str(file_path))
            hs_norm = (hs / max_hotspot) if max_hotspot > 0 else 0.0

            # Keystone score: fan-out * (1 + normalized hotspot)
            # This is intentionally simple and deterministic for MVP.
            keystone = float(deps_total or 0) * (1.0 + hs_norm)

            top_files.append(
                {
                    "file_path": str(file_path),
                    "deps_total": int(deps_total or 0),
                    "imports_count": int(imports_count or 0),
                    "calls_count": int(calls_count or 0),
                    "unique_targets": int(unique_targets or 0),
                    "fanin_total": int(fanin_total or 0),
                    "hotspot_score": hs,
                    "hotspot_score_norm": hs_norm,
                    "keystone_score": keystone,
                }
            )

        # --- top external targets (what is depended on most)
        rows = conn.execute(
            """
            SELECT
                d.target_external as target_external,
                d.dependency_kind as kind,
                SUM(COALESCE(d.usage_count, 1)) as cnt
            FROM dependencies d
            WHERE d.target_external IS NOT NULL AND d.target_external != ''
            GROUP BY d.target_external, d.dependency_kind
            ORDER BY cnt DESC
            LIMIT ?
            """,
            (int(top_n_targets),),
        ).fetchall()

        for (target_external, kind, cnt) in rows:
            top_targets.append(
                {
                    "target_external": str(target_external),
                    "kind": str(kind),
                    "count": int(cnt or 0),
                }
            )

    # Sort by keystone_score for UI convenience
    top_files_sorted = sorted(top_files, key=lambda x: x.get("keystone_score", 0.0), reverse=True)

    return {
        "db_path": str(db_path),
        "computed_at": datetime.now().isoformat(),
        "limits": {"top_n_files": int(top_n_files), "top_n_targets": int(top_n_targets)},
        "totals": totals,
        "top_files_by_fanout": top_files_sorted,
        "top_external_targets": top_targets,
        "notes": {
            "incoming_dependencies": "Computed only for resolved internal targets (target_symbol_id IS NOT NULL).",
            "keystone_score": "deps_total * (1 + hotspot_score_norm)",
        },
    }


def _compute_dependency_graph(
    *,
    db_path: Path,
    hotspots_payload: Optional[Dict[str, Any]] = None,
    hubs_payload: Optional[Dict[str, Any]] = None,
    max_external_nodes: int = 200,
) -> Dict[str, Any]:
    """
    Build a file-level dependency graph:
      - nodes: internal files + selected external targets
      - edges: aggregated dependencies (imports/calls)
      - node weights: hotspot_score, fanout, fanin, keystone_score
    """
    from intellirefactor.analysis.index.store import IndexStore
    store = IndexStore(db_path)

    hotspots = (hotspots_payload or {}).get("hotspots", []) if isinstance(hotspots_payload, dict) else []
    hubs = (hubs_payload or {}).get("top_files_by_fanout", []) if isinstance(hubs_payload, dict) else []

    # map relative file_path -> hub row
    hub_by_file: Dict[str, Dict[str, Any]] = {}
    for h in hubs:
        if isinstance(h, dict) and h.get("file_path"):
            hub_by_file[str(h["file_path"])] = h

    # hotspot score mapping by suffix match
    hotspot_by_suffix: Dict[str, float] = {}
    for h in hotspots:
        if isinstance(h, dict) and h.get("file_path"):
            hotspot_by_suffix[str(h["file_path"]).replace("\\", "/")] = _safe_float(h.get("score", 0.0), 0.0)

    def hotspot_for_db_file(rel_file: str) -> float:
        rel_norm = str(rel_file).replace("\\", "/")
        for full, sc in hotspot_by_suffix.items():
            if full.endswith(rel_norm):
                return sc
        return 0.0

    max_hotspot = max((_safe_float(h.get("score", 0.0), 0.0) for h in hotspots if isinstance(h, dict)), default=0.0)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges_agg: Dict[tuple, int] = {}
    external_counts: Dict[str, int] = {}

    with store._get_connection() as conn:
        # fanout for ALL files (not only top hubs)
        fanout_by_file: Dict[str, int] = {}
        fanin_by_file: Dict[str, int] = {}
        try:
            rows = conn.execute(
                """
                SELECT f.file_path, SUM(COALESCE(d.usage_count, 1)) as deps_total
                FROM dependencies d
                JOIN symbols s ON d.source_symbol_id = s.symbol_id
                JOIN files f ON s.file_id = f.file_id
                GROUP BY f.file_path
                """
            ).fetchall()
            for fp, cnt in rows:
                fanout_by_file[str(fp)] = int(cnt or 0)
        except Exception:
            fanout_by_file = {}

        try:
            rows = conn.execute(
                """
                SELECT tf.file_path, SUM(COALESCE(d.usage_count, 1)) as deps_total
                FROM dependencies d
                JOIN symbols ts ON d.target_symbol_id = ts.symbol_id
                JOIN files tf ON ts.file_id = tf.file_id
                WHERE d.target_symbol_id IS NOT NULL
                GROUP BY tf.file_path
                """
            ).fetchall()
            for fp, cnt in rows:
                fanin_by_file[str(fp)] = int(cnt or 0)
        except Exception:
            fanin_by_file = {}

        # internal nodes = all files
        rows = conn.execute("SELECT file_path, is_test_file, lines_of_code FROM files").fetchall()
        for file_path, is_test_file, loc in rows:
            fp = str(file_path)
            hs = hotspot_for_db_file(fp)
            hs_norm = (hs / max_hotspot) if max_hotspot > 0 else 0.0
            fanout = fanout_by_file.get(fp, _safe_int(hub_by_file.get(fp, {}).get("deps_total"), 0))
            fanin = fanin_by_file.get(fp, _safe_int(hub_by_file.get(fp, {}).get("fanin_total"), 0))
            keystone = float(fanout) * (1.0 + hs_norm)
            nodes[fp] = {
                "id": fp,
                "type": "file",
                "file_path": fp,
                "is_test_file": bool(is_test_file),
                "lines_of_code": _safe_int(loc, 0),
                "hotspot_score": hs,
                "fanout": int(fanout),
                "fanin": int(fanin),
                "keystone_score": keystone,
            }

        # map module qualified_name -> file_path using module symbols (works after builder patch)
        mod_to_file: Dict[str, str] = {}
        try:
            mrows = conn.execute(
                """
                SELECT s.qualified_name, f.file_path
                FROM symbols s
                JOIN files f ON s.file_id = f.file_id
                WHERE s.kind = 'module'
                """
            ).fetchall()
            for qn, fp in mrows:
                if qn and fp:
                    mod_to_file[str(qn)] = str(fp)
        except Exception:
            pass

        # 1) resolved internal edges via target_symbol_id
        rows = conn.execute(
            """
            SELECT sf.file_path, tf.file_path, d.dependency_kind, SUM(COALESCE(d.usage_count, 1))
            FROM dependencies d
            JOIN symbols ss ON d.source_symbol_id = ss.symbol_id
            JOIN files sf ON ss.file_id = sf.file_id
            JOIN symbols ts ON d.target_symbol_id = ts.symbol_id
            JOIN files tf ON ts.file_id = tf.file_id
            WHERE d.target_symbol_id IS NOT NULL
            GROUP BY sf.file_path, tf.file_path, d.dependency_kind
            """
        ).fetchall()

        for src_fp, dst_fp, kind, cnt in rows:
            key = (str(src_fp), str(dst_fp), str(kind))
            edges_agg[key] = edges_agg.get(key, 0) + int(cnt or 0)

        # 2) unresolved imports: try map target_external -> module symbol -> file_path
        urows = conn.execute(
            """
            SELECT sf.file_path, d.dependency_kind, d.target_external, SUM(COALESCE(d.usage_count, 1))
            FROM dependencies d
            JOIN symbols ss ON d.source_symbol_id = ss.symbol_id
            JOIN files sf ON ss.file_id = sf.file_id
            WHERE d.target_symbol_id IS NULL
              AND d.target_external IS NOT NULL AND d.target_external != ''
              AND d.dependency_kind = 'imports'
            GROUP BY sf.file_path, d.dependency_kind, d.target_external
            """
        ).fetchall()
        kind_idx = 1
        target_idx = 2
        cnt_idx = 3

        for row in urows:
            src_fp = str(row[0])
            kind = str(row[kind_idx] or "imports")
            target = str(row[target_idx] or "")
            cnt = int(row[cnt_idx] or 0)
            if not target:
                continue

            # progressive shortening: a.b.c -> a.b -> a
            cand = target
            dst_fp = None
            for _ in range(6):
                if cand in mod_to_file:
                    dst_fp = mod_to_file[cand]
                    break
                if "." not in cand:
                    break
                cand = cand.rsplit(".", 1)[0]

            if dst_fp:
                key = (src_fp, dst_fp, kind)
                edges_agg[key] = edges_agg.get(key, 0) + cnt
            else:
                ext_id = f"external:{target}"
                external_counts[ext_id] = external_counts.get(ext_id, 0) + cnt
                key = (src_fp, ext_id, kind)
                edges_agg[key] = edges_agg.get(key, 0) + cnt

        # external nodes (limited)
        ext_sorted = sorted(external_counts.items(), key=lambda kv: kv[1], reverse=True)[: int(max_external_nodes)]
        ext_allow = {k for k, _ in ext_sorted}
        for ext_id, cnt in ext_sorted:
            nodes[ext_id] = {"id": ext_id, "type": "external", "label": ext_id.replace("external:", ""), "count": int(cnt)}

    edges: List[Dict[str, Any]] = []
    for (src, dst, kind), cnt in edges_agg.items():
        if str(dst).startswith("external:") and dst not in nodes:
            continue
        edges.append({"source": src, "target": dst, "kind": kind, "count": int(cnt)})

    return {
        "db_path": str(db_path),
        "computed_at": datetime.now().isoformat(),
        "nodes": list(nodes.values()),
        "edges": edges,
        "summary": {
            "nodes_total": len(nodes),
            "edges_total": len(edges),
            "internal_edges": sum(1 for e in edges if not str(e["target"]).startswith("external:")),
            "external_edges": sum(1 for e in edges if str(e["target"]).startswith("external:")),
        },
    }

def _generate_refactoring_path_md(
    *,
    audit_payload: Dict[str, Any],
    hotspots_payload: Dict[str, Any],
    run_id: str,
) -> str:
    """
    Simple refactoring path generator based on correlation of:
    - duplicates -> dedup step
    - unused -> cleanup step
    - smells -> decomposition/refactor step
    """
    findings = (audit_payload or {}).get("findings", []) if isinstance(audit_payload, dict) else []
    hotspots = (hotspots_payload or {}).get("hotspots", []) if isinstance(hotspots_payload, dict) else []

    lines: List[str] = []
    lines.append("# Refactoring Path (auto-generated)")
    lines.append("")
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Generated:** `{datetime.now().isoformat()}`")
    lines.append("")

    if not hotspots:
        lines.append("No hotspots were detected (no findings).")
        lines.append("")
        return "\n".join(lines)

    target_file = str(hotspots[0].get("file_path") or "")
    lines.append("## Target hotspot")
    lines.append("")
    lines.append(f"- **File:** `{target_file}`")
    lines.append(f"- **Hotspot score:** `{hotspots[0].get('score')}`")
    lines.append(f"- **Findings:** `{hotspots[0].get('findings_total')}`")
    lines.append("")

    file_findings = [
        f for f in findings if isinstance(f, dict) and _same_path(str(f.get("file_path") or ""), target_file)
    ]
    dup = [f for f in file_findings if f.get("finding_type") == "duplicate_block"]
    unused = [f for f in file_findings if f.get("finding_type") == "unused_code"]
    quality = [f for f in file_findings if f.get("finding_type") == "quality_issue"]

    lines.append("## Path")
    lines.append("")

    step = 1
    # Step 1: Dedup first (prepares codebase for safer refactor)
    lines.append(f"### Step {step}: Deduplication (reduce repetition)")
    if dup:
        lines.append(f"Found **{len(dup)}** duplicate-block findings in this file.")
        for f in dup[:5]:
            md = f.get("metadata") or {}
            clone_type = md.get("clone_type") or md.get("detection_method") or "unknown"
            lines.append(f"- {f.get('title')} — clone_type=`{clone_type}`, confidence={f.get('confidence')}")
    else:
        lines.append("No duplicate-block findings in this file.")
    lines.append("")
    step += 1

    # Step 2: Unused cleanup
    lines.append(f"### Step {step}: Cleanup unused code (quick wins)")
    if unused:
        lines.append(f"Found **{len(unused)}** unused-code findings in this file.")
        for f in unused[:8]:
            md = f.get("metadata") or {}
            sym = md.get("symbol_name") or "unknown_symbol"
            lines.append(f"- `{sym}` @ lines {f.get('line_start')}-{f.get('line_end')} (confidence={f.get('confidence')})")
    else:
        lines.append("No unused-code findings in this file.")
    lines.append("")
    step += 1

    # Step 3: Smells/quality issues
    lines.append(f"### Step {step}: Address smells / quality issues (structural improvements)")
    if quality:
        lines.append(f"Found **{len(quality)}** quality findings in this file.")
        for f in quality[:8]:
            md = f.get("metadata") or {}
            smell_type = md.get("smell_type")
            tag = f"smell_type=`{smell_type}`" if smell_type else "quality_issue"
            lines.append(f"- {f.get('title')} — {tag} (severity={f.get('severity')}, confidence={f.get('confidence')})")
    else:
        lines.append("No quality/smell findings in this file.")
    lines.append("")

    lines.append("## References (artifacts)")
    lines.append("")
    lines.append("- `audit/audit.json` (unified findings)")
    lines.append("- `dedup/block_clones.json` (extractor-based clones)")
    lines.append("- `dedup/index_duplicates.json` (DB-based duplicates)")
    lines.append("- `refactor/unused.json` (unused findings)")
    lines.append("- `decompose/smells.json` (smells)")
    lines.append("")
    lines.append("## Next iteration ideas")
    lines.append("")
    lines.append("- Add dependency hub scoring from index (keystone components).")
    lines.append("- Correlate duplicates across multiple files into a single 'extract common module' step.")
    lines.append("- Generate module-level hotspot map (packages) for dashboard visualization.")
    lines.append("")

    return "\n".join(lines)


def _serialize_clustering_result(r: Any) -> Dict[str, Any]:
    """
    responsibility_clusterer_impl contains AST nodes inside MethodInfo.node.
    We drop AST to keep JSON clean.
    """
    out: Dict[str, Any] = {
        "class_name": getattr(r, "class_name", None),
        "file_path": getattr(r, "file_path", None),
        "total_methods": getattr(r, "total_methods", None),
        "unclustered_ratio": getattr(r, "unclustered_ratio", None),
        "average_cohesion": getattr(r, "average_cohesion", None),
        "silhouette_score": getattr(r, "silhouette_score", None),
        "extraction_recommended": getattr(r, "extraction_recommended", None),
        "confidence": getattr(r, "confidence", None),
        "recommendations": list(getattr(r, "recommendations", []) or []),
    }

    ev = getattr(r, "evidence", None)
    if ev is not None:
        out["evidence"] = ev.to_dict() if hasattr(ev, "to_dict") else str(ev)

    # clusters
    clusters = []
    for c in (getattr(r, "clusters", []) or []):
        clusters.append(
            {
                "cluster_id": getattr(c, "cluster_id", None),
                "suggested_name": getattr(c, "suggested_name", None),
                "cohesion_score": getattr(c, "cohesion_score", None),
                "confidence": getattr(c, "confidence", None),
                "quality": getattr(getattr(c, "quality", None), "value", getattr(c, "quality", None)),
                "methods": [
                    {"name": getattr(m, "name", None), "line_start": getattr(m, "line_start", None), "line_end": getattr(m, "line_end", None)}
                    for m in (getattr(c, "methods", []) or [])
                ],
                "shared_attributes": sorted(list(getattr(c, "shared_attributes", set()) or [])),
                "shared_dependencies": sorted(list(getattr(c, "shared_dependencies", set()) or [])),
                "dominant_responsibilities": list(getattr(c, "dominant_responsibilities", []) or []),
            }
        )
    out["clusters"] = clusters

    # unclustered methods
    out["unclustered_methods"] = [
        {"name": getattr(m, "name", None), "line_start": getattr(m, "line_start", None), "line_end": getattr(m, "line_end", None)}
        for m in (getattr(r, "unclustered_methods", []) or [])
    ]

    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _norm_file_path_for_join(project_root: Path, file_path: str) -> Path:
    """
    Convert file_path (absolute/relative) to an absolute path under project_root when possible.
    """
    fp = Path(str(file_path or ""))
    if not fp.is_absolute():
        return (project_root / fp).resolve()
    return fp


def _read_snippet(
    *,
    project_root: Path,
    file_path: str,
    line_start: int,
    line_end: int,
    context_lines: int = 4,
    max_lines: int = 60,
) -> Dict[str, Any]:
    """
    Read code snippet with context from file (safe, BOM-friendly).
    """
    try:
        abs_path = _norm_file_path_for_join(project_root, file_path)
        text = abs_path.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()
        ls = max(1, int(line_start or 1))
        le = max(ls, int(line_end or ls))

        # apply context
        before = max(1, ls - context_lines)
        after = min(len(lines), le + context_lines)

        snippet_lines = lines[before - 1 : after]
        # hard limit
        if len(snippet_lines) > max_lines:
            snippet_lines = snippet_lines[:max_lines] + ["# ... (snippet truncated)"]

        return {
            "file_path": str(file_path),
            "line_start": ls,
            "line_end": le,
            "context_before_line": before,
            "context_after_line": after,
            "text": "\n".join(snippet_lines),
        }
    except Exception as e:
        return {
            "file_path": str(file_path),
            "line_start": int(line_start or 1),
            "line_end": int(line_end or (line_start or 1)),
            "error": str(e),
        }


def _extract_instance_loc(inst: Any) -> Dict[str, Any]:
    """
    Normalize CloneInstance (object or dict) to a stable location record.
    """
    file_path = _get(inst, "file_path", None)
    line_start = _get(inst, "line_start", None)
    line_end = _get(inst, "line_end", None)

    block_info = _get(inst, "block_info", None)
    if block_info is not None and (not file_path or not line_start or not line_end):
        fr = _get(block_info, "file_reference", None)
        if fr is not None:
            file_path = file_path or _get(fr, "file_path", None)
            line_start = line_start or _get(fr, "line_start", None)
            line_end = line_end or _get(fr, "line_end", None)

    loc = {
        "file_path": str(file_path or ""),
        "line_start": int(line_start or 1),
        "line_end": int(line_end or (line_start or 1)),
        "lines_of_code": int(_get(block_info, "lines_of_code", _get(inst, "lines_of_code", 0)) or 0),
        "statement_count": int(_get(block_info, "statement_count", _get(inst, "statement_count", 0)) or 0),
        "nesting_level": int(_get(block_info, "nesting_level", _get(inst, "nesting_level", 0)) or 0),
    }
    return loc


def _clone_group_score(g: Any) -> float:
    """
    Sorting score for groups: prefer detector ranking_score if present, else size*similarity.
    """
    rs = _safe_float(_get(g, "ranking_score", 0.0), 0.0)
    if rs > 0:
        return rs
    sim = _safe_float(_get(g, "similarity_score", 0.0), 0.0)
    inst = _get(g, "instances", []) or []
    return float(sim) * float(len(inst))


def _serialize_clone_groups_summary(
    *,
    groups: List[Any],
    project_root: Path,
    max_groups: int = 500,
    preview_instances: int = 5,
) -> Dict[str, Any]:
    groups_sorted = sorted(list(groups or []), key=_clone_group_score, reverse=True)[: int(max_groups)]

    out_groups: List[Dict[str, Any]] = []
    total_instances = 0
    by_type: Dict[str, int] = {}
    by_strategy: Dict[str, int] = {}

    for g in groups_sorted:
        inst = _get(g, "instances", []) or []
        total_instances += len(inst)

        clone_type = str(_get(getattr(g, "clone_type", None), "value", _get(g, "clone_type", "unknown")) or "unknown")
        by_type[clone_type] = by_type.get(clone_type, 0) + 1

        strat = _get(getattr(g, "extraction_strategy", None), "value", _get(g, "extraction_strategy", None))
        strat_s = str(strat) if strat is not None else "none"
        by_strategy[strat_s] = by_strategy.get(strat_s, 0) + 1

        locs = [_extract_instance_loc(x) for x in inst[: int(preview_instances)]]
        uniq_files = len({l["file_path"] for l in [_extract_instance_loc(x) for x in inst] if l.get("file_path")})
        total_loc = sum(int(_extract_instance_loc(x).get("lines_of_code", 0) or 0) for x in inst)

        out_groups.append(
            {
                "group_id": str(_get(g, "group_id", "") or ""),
                "clone_type": clone_type,
                "similarity_score": _safe_float(_get(g, "similarity_score", 0.0), 0.0),
                "ranking_score": _safe_float(_get(g, "ranking_score", 0.0), 0.0),
                "detection_channels": _get(g, "detection_channels", None),
                "extraction_strategy": strat_s,
                "extraction_confidence": _safe_float(_get(g, "extraction_confidence", 0.0), 0.0),
                "instance_count": len(inst),
                "unique_files": int(uniq_files),
                "total_loc_estimate": int(total_loc),
                "instances_preview": locs,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "limits": {"max_groups": int(max_groups), "preview_instances": int(preview_instances)},
        "summary": {
            "groups_total": len(groups or []),
            "groups_emitted": len(out_groups),
            "instances_emitted_total": total_instances,
            "by_type": by_type,
            "by_extraction_strategy": by_strategy,
        },
        "groups": out_groups,
    }


def _serialize_clone_groups_full(
    *,
    groups: List[Any],
    project_root: Path,
    max_groups: int = 200,
    max_instances_per_group: int = 50,
    snippet_instances_per_group: int = 2,
    snippet_context_lines: int = 4,
    snippet_max_lines: int = 60,
) -> Dict[str, Any]:
    groups_sorted = sorted(list(groups or []), key=_clone_group_score, reverse=True)[: int(max_groups)]

    out_groups: List[Dict[str, Any]] = []
    for g in groups_sorted:
        inst = _get(g, "instances", []) or []
        inst_locs = [_extract_instance_loc(x) for x in inst[: int(max_instances_per_group)]]

        snippets: List[Dict[str, Any]] = []
        for loc in inst_locs[: int(snippet_instances_per_group)]:
            if loc.get("file_path"):
                snippets.append(
                    _read_snippet(
                        project_root=project_root,
                        file_path=str(loc["file_path"]),
                        line_start=int(loc["line_start"]),
                        line_end=int(loc["line_end"]),
                        context_lines=int(snippet_context_lines),
                        max_lines=int(snippet_max_lines),
                    )
                )

        clone_type = str(_get(getattr(g, "clone_type", None), "value", _get(g, "clone_type", "unknown")) or "unknown")
        strat = _get(getattr(g, "extraction_strategy", None), "value", _get(g, "extraction_strategy", None))
        strat_s = str(strat) if strat is not None else "none"

        out_groups.append(
            {
                "group_id": str(_get(g, "group_id", "") or ""),
                "clone_type": clone_type,
                "similarity_score": _safe_float(_get(g, "similarity_score", 0.0), 0.0),
                "ranking_score": _safe_float(_get(g, "ranking_score", 0.0), 0.0),
                "extraction_strategy": strat_s,
                "extraction_confidence": _safe_float(_get(g, "extraction_confidence", 0.0), 0.0),
                "instance_count": len(inst),
                "instances": inst_locs,
                "snippets": snippets,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "limits": {
            "max_groups": int(max_groups),
            "max_instances_per_group": int(max_instances_per_group),
            "snippet_instances_per_group": int(snippet_instances_per_group),
            "snippet_context_lines": int(snippet_context_lines),
            "snippet_max_lines": int(snippet_max_lines),
        },
        "groups": out_groups,
    }


def _write_clone_groups_csv(
    writer: "ArtifactWriter",
    *,
    rel_path: str,
    summary_payload: Dict[str, Any],
    section: str = "dedup",
    description: str = "Block clone groups (CSV summary)",
) -> None:
    """
    CSV for task-tracker / developer workflow.
    One row per group.
    """
    groups = summary_payload.get("groups") or []
    p = writer.run_dir / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "group_id",
                "clone_type",
                "similarity_score",
                "ranking_score",
                "extraction_strategy",
                "extraction_confidence",
                "instance_count",
                "unique_files",
                "total_loc_estimate",
                "top_location",
                "instances_preview",
            ]
        )
        for g in groups:
            if not isinstance(g, dict):
                continue
            prev = g.get("instances_preview") or []
            top = ""
            if prev and isinstance(prev[0], dict):
                top = f'{prev[0].get("file_path")}:{prev[0].get("line_start")}-{prev[0].get("line_end")}'

            # compact preview list
            preview_str = "; ".join(
                f'{x.get("file_path")}:{x.get("line_start")}-{x.get("line_end")}'
                for x in prev[:5]
                if isinstance(x, dict)
            )

            w.writerow(
                [
                    g.get("group_id", ""),
                    g.get("clone_type", ""),
                    g.get("similarity_score", ""),
                    g.get("ranking_score", ""),
                    g.get("extraction_strategy", ""),
                    g.get("extraction_confidence", ""),
                    g.get("instance_count", ""),
                    g.get("unique_files", ""),
                    g.get("total_loc_estimate", ""),
                    top,
                    preview_str,
                ]
            )

    writer.add_output(section=section, kind="csv", path=p, description=description)


def run_collect(args: Any) -> Dict[str, Any]:
    project_path = Path(args.project_path).resolve()
    if not project_path.exists() or not project_path.is_dir():
        raise ValueError(f"Project path must be an existing directory: {project_path}")

    out_base = Path(getattr(args, "out", "./intellirefactor_out")).resolve()
    run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_base / run_id

    target_file = Path(args.target_file).resolve() if getattr(args, "target_file", None) else None
    if target_file and not target_file.exists():
        raise ValueError(f"--target-file not found: {target_file}")

    options = {
        "sections": {
            "refactor": bool(getattr(args, "refactor", False)),
            "dedup": bool(getattr(args, "dedup", False)),
            "decompose": bool(getattr(args, "decompose", False)),
        },
        "no_index": bool(getattr(args, "no_index", False)),
        "index_full": bool(getattr(args, "index_full", False)),
        "entry_point": getattr(args, "entry_point", None),
    }

    # If user didn't specify any section flags -> run all three.
    if not options["sections"]["refactor"] and not options["sections"]["dedup"] and not options["sections"]["decompose"]:
        options["sections"] = {"refactor": True, "dedup": True, "decompose": True}

    will_run_audit = (
        options["sections"]["refactor"]
        and options["sections"]["dedup"]
        and options["sections"]["decompose"]
    )

    writer = ArtifactWriter(
        run_dir,
        tool="intellirefactor",
        tool_version="0.1.0",
        project_path=str(project_path),
        target_file=str(target_file) if target_file else None,
        run_id=run_id,
        command="collect",
        options=options,
    )

    # Load config + API facade
    config = load_config(getattr(args, "config", None))
    ir = IntelliRefactor(config)
    full_mode = bool(getattr(args, "full", False))

    # index is considered available only if user didn't disable it
    # and the DB file exists (built in this run or previously).
    index_available = False

    # Everything below should not prevent manifest/summary from being written.
    try:
        # ------------------------------------------------------------
        # INDEX (always unless --no-index)
        # ------------------------------------------------------------
        db_path = project_path / ".intellirefactor" / "index.db"
        index_built_ok = False
        index_status: Optional[Dict[str, Any]] = None
        if not options["no_index"]:
            try:
                from intellirefactor.analysis.index.builder import IndexBuilder
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with IndexBuilder(db_path) as builder:
                    build_res = builder.build_index(project_path, incremental=not options["index_full"])
                build_payload = build_res.to_dict() if hasattr(build_res, "to_dict") else _as_dict(build_res)
                index_built_ok = bool(build_payload.get("success"))
                writer.write_json("index/build.json", build_payload, section="index", description="Index build result")
                writer.write_markdown(
                    "index/build.md",
                    _md_kv(
                        "Index Build",
                        [
                            ("success", build_payload.get("success")),
                            ("files_processed", build_payload.get("files_processed")),
                            ("files_skipped", build_payload.get("files_skipped")),
                            ("symbols_found", build_payload.get("symbols_found")),
                            ("blocks_found", build_payload.get("blocks_found")),
                            ("dependencies_found", build_payload.get("dependencies_found")),
                            ("build_time_seconds", build_payload.get("build_time_seconds")),
                            ("incremental", build_payload.get("incremental")),
                        ],
                    ),
                    section="index",
                    description="Index build summary",
                )

                # Status
                try:
                    from intellirefactor.analysis.index.store import IndexStore
                    store = IndexStore(db_path)
                    index_status = store.get_statistics() if hasattr(store, "get_statistics") else {}
                    writer.write_json("index/status.json", index_status, section="index", description="Index status/statistics")
                except Exception as e:
                    writer.add_warning(f"Index status query failed: {e}")
            except Exception as e:
                writer.add_error(f"Index build failed: {e}")
        else:
            writer.add_warning("Index build skipped (--no-index). Some analyses may be incomplete.")

        # after build attempt, decide if we can use index further
        index_available = (not options["no_index"]) and db_path.exists()

        # Prepare file list for non-index scans
        analysis_settings = getattr(config, "analysis_settings", None)
        exclude_globs_user = list(getattr(analysis_settings, "exclude_patterns", []) or [])
        exclude_globs = _merge_exclude_globs(exclude_globs_user)
        files = _discover_py_files(project_path, out_base, exclude_globs)

        # ------------------------------------------------------------
        # REFACTOR
        # ------------------------------------------------------------
        if options["sections"]["refactor"]:
            # project analysis
            try:
                res = ir.analyze_project(project_path, include_metrics=True, include_opportunities=True)
                payload = {
                    "success": res.success,
                    "data": res.data,
                    "errors": res.errors,
                    "warnings": res.warnings,
                    "metadata": res.metadata,
                }
                writer.write_json("refactor/project_analysis.json", payload, section="refactor", description="Project analysis")
                writer.write_markdown(
                    "refactor/project_analysis.md",
                    _md_kv(
                        "Refactor: Project Analysis",
                        [
                            ("success", payload["success"]),
                            ("total_files", payload["data"].get("total_files")),
                            ("total_lines", payload["data"].get("total_lines")),
                            ("opportunities", len(payload["data"].get("refactoring_opportunities", []) or [])),
                        ],
                    ),
                    section="refactor",
                    description="Project analysis summary",
                )
            except Exception as e:
                writer.add_error(f"Refactor project analysis failed: {e}")

            # file analysis + expert analysis require target_file
            if target_file:
                try:
                    res = ir.analyze_file(target_file, project_root=project_path)
                    payload = {
                        "success": res.success,
                        "data": res.data,
                        "errors": res.errors,
                        "warnings": res.warnings,
                        "metadata": res.metadata,
                    }
                    writer.write_json("refactor/file_analysis.json", payload, section="refactor", description="File analysis")
                    writer.write_markdown(
                        "refactor/file_analysis.md",
                        _md_kv(
                            "Refactor: File Analysis",
                            [
                                ("success", payload["success"]),
                                ("file", str(target_file)),
                                ("issues", len(payload["data"].get("issues", []) or [])),
                                ("recommendations", len(payload["data"].get("recommendations", []) or [])),
                            ],
                        ),
                        section="refactor",
                        description="File analysis summary",
                    )
                except Exception as e:
                    writer.add_error(f"Refactor file analysis failed: {e}")

                # expert analysis (JSON+MD)
                try:
                    from intellirefactor.analysis.refactor.expert.expert_analyzer import ExpertRefactoringAnalyzer

                    expert_out = run_dir / "refactor" / "expert"
                    analyzer = ExpertRefactoringAnalyzer(
                        project_root=str(project_path),
                        target_module=str(target_file),
                        output_dir=str(expert_out),
                    )
                    expert_result = analyzer.analyze_for_expert_refactoring()

                    writer.write_json(
                        "refactor/expert/expert_result.json",
                        expert_result.to_dict() if hasattr(expert_result, "to_dict") else _as_dict(expert_result),
                        section="refactor",
                        description="Expert refactoring analysis (core)",
                    )
                    # Generate md report using analyzer if available
                    try:
                        md_path = analyzer.generate_expert_report(str(expert_out))
                        # register existing md file as output
                        writer.add_output(section="refactor", kind="markdown", path=Path(md_path), description="Expert report (generated)")
                    except Exception as e:
                        writer.add_warning(f"Expert markdown report generation failed: {e}")
                except Exception as e:
                    writer.add_error(f"Expert analysis failed: {e}")
            else:
                writer.add_warning("Refactor file/expert analysis skipped (no --target-file).")

            # Unused code detector (JSON+MD)
            try:
                if will_run_audit:
                    # Avoid running unused twice (audit includes it too).
                    # We'll export unused_result from audit into refactor/unused.* later.
                    writer.add_warning(
                        "Refactor unused analysis skipped (will be collected via audit because all 3 sections selected)."
                    )
                    raise _SkipAnalysis("unused_via_audit")

                from intellirefactor.analysis.refactor.unused_code_detector import UnusedCodeDetector
                external_index = None
                try:
                    if index_available:
                        from intellirefactor.analysis.index.store import IndexStore
                        external_index = IndexStore(db_path)
                except Exception:
                    external_index = None

                det = UnusedCodeDetector(project_path)
                unused_res = det.detect_unused_code(
                    entry_points=None,
                    include_patterns=["**/*.py"],
                    exclude_patterns=exclude_globs,
                    min_confidence=0.5,
                    external_index=external_index,
                )
                unused_payload = unused_res.to_dict() if hasattr(unused_res, "to_dict") else _as_dict(unused_res)

                writer.write_json("refactor/unused.json", unused_payload, section="refactor", description="Unused code findings")

                stats = unused_payload.get("statistics", {}) if isinstance(unused_payload, dict) else {}
                writer.write_markdown(
                    "refactor/unused.md",
                    _md_kv(
                        "Refactor: Unused Code",
                        [
                            ("total_findings", stats.get("total_findings")),
                            ("files_analyzed", stats.get("total_files_analyzed")),
                            ("high_confidence", (stats.get("findings_by_confidence", {}) or {}).get("high")),
                            ("medium_confidence", (stats.get("findings_by_confidence", {}) or {}).get("medium")),
                            ("low_confidence", (stats.get("findings_by_confidence", {}) or {}).get("low")),
                        ],
                    ),
                    section="refactor",
                    description="Unused code summary",
                )
            except _SkipAnalysis:
                pass
            except Exception as e:
                writer.add_warning(f"Unused code analysis skipped/failed: {e}")

            # visualizations (safe: writes to run_dir, does not mutate project)
            if getattr(args, "visualize", False):
                try:
                    from intellirefactor.orchestration.global_refactoring_orchestrator import GlobalRefactoringOrchestrator
                    orch = GlobalRefactoringOrchestrator(project_root=project_path, dry_run=False, config=config.refactoring_settings)
                    viz_dir = run_dir / "refactor" / "visualizations"
                    stage = orch.run_visualization_stage(viz_dir, entry_point=getattr(args, "entry_point", None))
                    writer.write_json("refactor/visualizations/result.json", _as_dict(stage), section="refactor", description="Visualization stage result")
                except Exception as e:
                    writer.add_warning(f"Visualization stage failed: {e}")

        # ------------------------------------------------------------
        # DEDUP
        # ------------------------------------------------------------
        if options["sections"]["dedup"]:
            # Semantic similarity (index-based methods)
            try:
                if options["no_index"]:
                    writer.add_warning("Dedup semantic similarity skipped (--no-index disables index usage).")
                    raise _SkipAnalysis("dedup_semantic_no_index")

                if will_run_audit:
                    # This is the most memory-expensive part on large codebases:
                    # many implementations compute all-vs-all method similarity and keep all matches in RAM.
                    # Since audit runs (all 3 sections), skip here to avoid OOM.
                    # If needed, run separately: `collect --dedup` (without audit).
                    writer.add_warning(
                        "Dedup semantic similarity skipped (audit selected; run `collect --dedup` separately if needed)."
                    )
                    raise _SkipAnalysis("dedup_semantic_via_audit")

                if not index_available:
                    raise FileNotFoundError("index.db not found; semantic similarity requires index")
                from intellirefactor.analysis.index.store import IndexStore
                from intellirefactor.analysis.dedup.semantic_similarity_matcher import SemanticSimilarityMatcher, SimilarityType

                store = IndexStore(db_path)
                get_methods = getattr(store, "get_all_deep_method_infos", None)
                if not callable(get_methods):
                    raise RuntimeError("IndexStore.get_all_deep_method_infos() not available")

                methods = get_methods()
                matcher = SemanticSimilarityMatcher()

                sim_types = {SimilarityType.STRUCTURAL, SimilarityType.FUNCTIONAL, SimilarityType.BEHAVIORAL}
                matches = matcher.find_similar_methods(methods, target_method=None, similarity_types=sim_types)

                payload = {
                    "total_methods": len(methods),
                    "total_matches": len(matches),
                    "matches": [m.to_dict() for m in matches[: int(getattr(args, "dedup_max_results", 200) or 200)]],
                    "statistics": matcher.get_similarity_statistics(matches),
                }
                writer.write_json("dedup/semantic_similarity.json", payload, section="dedup", description="Semantic similarity matches")
                writer.write_markdown(
                    "dedup/semantic_similarity.md",
                    _md_kv(
                        "Dedup: Semantic Similarity",
                        [
                            ("methods", payload["total_methods"]),
                            ("matches", payload["total_matches"]),
                        ],
                    ),
                    section="dedup",
                    description="Semantic similarity summary",
                )
            except _SkipAnalysis:
                pass
            except Exception as e:
                writer.add_warning(f"Dedup semantic similarity skipped/failed: {e}")

            # Index-based duplicates (fast; uses DB)
            try:
                if options["no_index"]:
                    writer.add_warning("Dedup index duplicates skipped (--no-index disables index usage).")
                    raise _SkipAnalysis("dedup_index_dups_no_index")

                if not index_available:
                    raise FileNotFoundError("index.db not found; index-duplicates requires index")
                from intellirefactor.analysis.index.store import IndexStore
                from intellirefactor.analysis.index.query import IndexQuery

                store = IndexStore(db_path)
                q = IndexQuery(store)

                exact_token = q.find_exact_duplicates("token_fingerprint", min_group_size=2)
                exact_ast = q.find_exact_duplicates("ast_fingerprint", min_group_size=2)
                structural = q.find_structural_duplicates(min_similarity=0.8, min_group_size=2)
                block_clones = q.find_block_clones(min_lines=3, fingerprint_type="normalized_fingerprint")

                payload = {
                    "exact_token_groups": [g.__dict__ for g in exact_token],
                    "exact_ast_groups": [g.__dict__ for g in exact_ast],
                    "structural_groups": [g.__dict__ for g in structural],
                    "block_clone_groups": [g.__dict__ for g in block_clones],
                    "summary": {
                        "exact_token_groups": len(exact_token),
                        "exact_ast_groups": len(exact_ast),
                        "structural_groups": len(structural),
                        "block_clone_groups": len(block_clones),
                    },
                }
                writer.write_json("dedup/index_duplicates.json", payload, section="dedup", description="Duplicates via IndexQuery")
                writer.write_markdown(
                    "dedup/index_duplicates.md",
                    _md_kv(
                        "Dedup: Index Duplicates (DB)",
                        [
                            ("exact_token_groups", payload["summary"]["exact_token_groups"]),
                            ("exact_ast_groups", payload["summary"]["exact_ast_groups"]),
                            ("structural_groups", payload["summary"]["structural_groups"]),
                            ("block_clone_groups", payload["summary"]["block_clone_groups"]),
                        ],
                    ),
                    section="dedup",
                    description="Index duplicates summary",
                )
            except Exception as e:
                writer.add_warning(f"Dedup index duplicates skipped/failed: {e}")

            # Block clones (extractor-based) if modules exist
            try:
                if will_run_audit:
                    # Avoid running block-clone extraction twice:
                    # audit_engine already runs duplicate analysis (BlockExtractor+BlockCloneDetector).
                    # We'll export clone_groups from audit into dedup/block_clones.* later.
                    writer.add_warning(
                        "Dedup block clones skipped (will be collected via audit because all 3 sections selected)."
                    )
                    raise _SkipAnalysis("dedup_block_clones_via_audit")

                from intellirefactor.analysis.dedup.block_extractor import BlockExtractor
                from intellirefactor.analysis.dedup.block_clone_detector import BlockCloneDetector

                extractor = BlockExtractor()
                detector = BlockCloneDetector()

                all_blocks = []
                for fp in files:
                    try:
                        src = fp.read_text(encoding="utf-8", errors="replace")
                        all_blocks.extend(extractor.extract_blocks(src, _relposix(project_path, fp)))
                        del src
                    except Exception:
                        continue

                groups = detector.detect_clones(all_blocks)
                stats = detector.get_clone_statistics(groups)

                summary = _serialize_clone_groups_summary(
                    groups=[g.to_dict() for g in groups],
                    project_root=project_path,
                    max_groups=500,
                    preview_instances=5,
                )
                payload_summary = {
                    "generated_at": summary.get("generated_at"),
                    "mode": "summary",
                    "files_scanned": len(files),
                    "blocks_extracted": len(all_blocks),
                    "statistics": stats,
                    **summary,
                }
                writer.write_json(
                    "dedup/block_clones.json",
                    payload_summary,
                    section="dedup",
                    description="Block clone groups (summary)",
                )
                _write_clone_groups_csv(
                    writer,
                    rel_path="dedup/block_clones.csv",
                    summary_payload=summary,
                    section="dedup",
                    description="Block clone groups (CSV summary)",
                )

                if full_mode:
                    payload_full = _serialize_clone_groups_full(
                        groups=[g.to_dict() for g in groups],
                        project_root=project_path,
                        max_groups=200,
                        max_instances_per_group=50,
                        snippet_instances_per_group=2,
                        snippet_context_lines=4,
                        snippet_max_lines=60,
                    )
                    payload_full["mode"] = "full"
                    writer.write_json(
                        "dedup/block_clones_full.json",
                        payload_full,
                        section="dedup",
                        description="Block clone groups (developer/full)",
                    )

                writer.write_markdown(
                    "dedup/block_clones.md",
                    _md_kv(
                        "Dedup: Block Clones",
                        [
                            ("files_scanned", payload_summary["files_scanned"]),
                            ("blocks_extracted", payload_summary["blocks_extracted"]),
                            ("clone_groups_total", (payload_summary.get("summary", {}) or {}).get("groups_total")),
                            ("clone_groups_emitted", (payload_summary.get("summary", {}) or {}).get("groups_emitted")),
                            ("total_instances", stats.get("total_instances")),
                            ("csv", "dedup/block_clones.csv"),
                            ("full_json", "dedup/block_clones_full.json" if full_mode else "(disabled, use --full)"),
                        ],
                    ),
                    section="dedup",
                    description="Block clone summary",
                )
            except _SkipAnalysis:
                pass
            except Exception as e:
                writer.add_warning(f"Dedup block clones skipped/failed: {e}")

        # ------------------------------------------------------------
        # DECOMPOSE
        # ------------------------------------------------------------
        if options["sections"]["decompose"]:
            # Smells
            try:
                from intellirefactor.analysis.decompose.architectural_smell_detector import ArchitecturalSmellDetector

                det = ArchitecturalSmellDetector()
                smells = []
                for fp in files:
                    try:
                        src = fp.read_text(encoding="utf-8", errors="replace")
                        smells.extend(det.detect_smells(src, _relposix(project_path, fp)))
                        del src
                    except Exception:
                        continue

                payload = {
                    "files_scanned": len(files),
                    "total_smells": len(smells),
                    "smells": [s.to_dict() for s in smells],
                }
                # stats
                by_type: Dict[str, int] = {}
                by_sev: Dict[str, int] = {}
                for s in smells:
                    by_type[s.smell_type.value] = by_type.get(s.smell_type.value, 0) + 1
                    by_sev[s.severity.value] = by_sev.get(s.severity.value, 0) + 1
                payload["statistics"] = {"by_type": by_type, "by_severity": by_sev}

                writer.write_json("decompose/smells.json", payload, section="decompose", description="Architectural smells")
                writer.write_markdown(
                    "decompose/smells.md",
                    _md_table(
                        "Decompose: Smells Summary",
                        ["Metric", "Value"],
                        [
                            ["files_scanned", payload["files_scanned"]],
                            ["total_smells", payload["total_smells"]],
                        ],
                    )
                    + _md_table(
                        "Smells by Severity",
                        ["Severity", "Count"],
                        [[k, v] for k, v in sorted(by_sev.items(), key=lambda kv: kv[0])],
                    )
                    + _md_table(
                        "Smells by Type",
                        ["Type", "Count"],
                        [[k, v] for k, v in sorted(by_type.items(), key=lambda kv: kv[0])],
                    ),
                    section="decompose",
                    description="Smells summary",
                )
            except Exception as e:
                writer.add_error(f"Decompose smells failed: {e}")

            # Responsibility clustering (current impl may be placeholder; still export)
            try:
                import ast
                from intellirefactor.analysis.decompose.responsibility_clusterer import ResponsibilityClusterer, ClusteringConfig, ClusteringAlgorithm

                cfg = ClusteringConfig(algorithm=ClusteringAlgorithm.HYBRID)
                clusterer = ResponsibilityClusterer(cfg)

                results = []
                for fp in files:
                    try:
                        src = fp.read_text(encoding="utf-8", errors="replace")
                        tree = ast.parse(src)
                        del src
                    except Exception:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            try:
                                r = clusterer.analyze_class(node, _relposix(project_path, fp))
                                results.append(_serialize_clustering_result(r))
                            except Exception:
                                continue

                # Detect stub implementation: if first result has confidence=0.0 and "not implemented" in recommendations
                is_stub = False
                if results:
                    first = results[0]
                    if isinstance(first, dict):
                        conf = first.get("confidence", 1.0)
                        recs = first.get("recommendations", [])
                        if conf == 0.0 and any("not implemented" in str(r).lower() for r in recs):
                            is_stub = True

                if is_stub:
                    # Stub implementation detected - emit clean "not implemented" artifact
                    payload = {
                        "implemented": False,
                        "reason": "Responsibility clustering analysis is not yet fully implemented",
                        "files_scanned": len(files),
                        "classes_analyzed": 0,
                        "results": [],
                    }
                    writer.add_warning("Responsibility clustering is experimental and not fully implemented")
                else:
                    # Real implementation - emit full results
                    payload = {
                        "implemented": True,
                        "files_scanned": len(files),
                        "classes_analyzed": len(results),
                        "results": results,
                    }

                writer.write_json("decompose/responsibility_clustering.json", payload, section="decompose", description="Responsibility clustering")
                writer.write_markdown(
                    "decompose/responsibility_clustering.md",
                    _md_kv(
                        "Decompose: Responsibility Clustering",
                        [
                            ("implemented", payload.get("implemented", True)),
                            ("files_scanned", payload["files_scanned"]),
                            ("classes_analyzed", payload["classes_analyzed"]),
                        ],
                    ),
                    section="decompose",
                    description="Clustering summary",
                )
            except Exception as e:
                writer.add_warning(f"Decompose responsibility clustering skipped/failed: {e}")

        # ------------------------------------------------------------
        # AUDIT (only when ALL 3 sections selected)
        # ------------------------------------------------------------
        if will_run_audit:
            try:
                from intellirefactor.analysis.workflows.audit_engine import AuditEngine
                from intellirefactor.analysis.workflows.spec_generator import SpecGenerator

                ae = AuditEngine(project_path)
                # Audit should be able to use index when enabled; it may reuse an existing DB.
                include_index_for_audit = (not options["no_index"])

                audit_result = ae.run_full_audit(
                    include_index=include_index_for_audit,
                    include_duplicates=True,
                    include_unused=True,
                    generate_spec=False,  # do NOT write to project root
                    incremental_index=not options["index_full"],
                    min_confidence=0.5,
                    include_patterns=["**/*.py"],
                    exclude_patterns=exclude_globs,
                )

                audit_payload = (
                    audit_result.to_dict()
                    if hasattr(audit_result, "to_dict")
                    else _as_dict(audit_result)
                )

                if isinstance(audit_payload, dict):
                    audit_payload = _canonicalize_audit_payload(project_path, audit_payload)

                writer.write_json(
                    "audit/audit.json",
                    audit_payload,
                    section="audit",
                    description="Unified audit (index + dedup + unused + smells)",
                )

                stats = (
                    (audit_payload or {}).get("statistics", {})
                    if isinstance(audit_payload, dict)
                    else {}
                )
                writer.write_markdown(
                    "audit/audit.md",
                    _md_kv(
                        "Audit Summary",
                        [
                            ("total_findings", stats.get("total_findings")),
                            ("files_analyzed", stats.get("files_analyzed")),
                            ("analysis_time_seconds", stats.get("analysis_time_seconds")),
                        ],
                    ),
                    section="audit",
                    description="Audit summary",
                )

                # Export unused_result from audit into refactor artifacts (so UI always finds it in refactor/)
                try:
                    if options["sections"]["refactor"] and isinstance(audit_payload, dict):
                        unused_payload = audit_payload.get("unused_result")
                        if isinstance(unused_payload, dict):
                            writer.write_json(
                                "refactor/unused.json",
                                unused_payload,
                                section="refactor",
                                description="Unused code findings (from audit)",
                            )

                            stats_u = unused_payload.get("statistics", {}) if isinstance(unused_payload, dict) else {}
                            writer.write_markdown(
                                "refactor/unused.md",
                                _md_kv(
                                    "Refactor: Unused Code",
                                    [
                                        ("total_findings", stats_u.get("total_findings")),
                                        ("files_analyzed", stats_u.get("total_files_analyzed")),
                                        ("high_confidence", (stats_u.get("findings_by_confidence", {}) or {}).get("high")),
                                        ("medium_confidence", (stats_u.get("findings_by_confidence", {}) or {}).get("medium")),
                                        ("low_confidence", (stats_u.get("findings_by_confidence", {}) or {}).get("low")),
                                    ],
                                ),
                                section="refactor",
                                description="Unused code summary (from audit)",
                            )
                except Exception as e:
                    writer.add_warning(f"Failed to export refactor/unused from audit: {e}")

                # Export clone_groups from audit into dedup artifacts (so UI always finds it in dedup/)
                try:
                    if options["sections"]["dedup"] and isinstance(audit_payload, dict):
                        clone_groups_payload = audit_payload.get("clone_groups")
                        if isinstance(clone_groups_payload, list):
                            summary = _serialize_clone_groups_summary(
                                groups=clone_groups_payload,
                                project_root=project_path,
                                max_groups=500,
                                preview_instances=5,
                            )
                            dedup_payload = {
                                "generated_at": summary.get("generated_at"),
                                "mode": "summary",
                                "source": "audit.clone_groups",
                                **summary,
                            }
                            writer.write_json(
                                "dedup/block_clones.json",
                                dedup_payload,
                                section="dedup",
                                description="Block clone groups (summary, exported from audit)",
                            )
                            _write_clone_groups_csv(
                                writer,
                                rel_path="dedup/block_clones.csv",
                                summary_payload=summary,
                                section="dedup",
                                description="Block clone groups (CSV summary, exported from audit)",
                            )

                            if full_mode:
                                dedup_full = {
                                    "mode": "full",
                                    "source": "audit.clone_groups",
                                    **_serialize_clone_groups_full(
                                        groups=clone_groups_payload,
                                        project_root=project_path,
                                        max_groups=200,
                                        max_instances_per_group=50,
                                        snippet_instances_per_group=2,
                                        snippet_context_lines=4,
                                        snippet_max_lines=60,
                                    ),
                                }
                                writer.write_json(
                                    "dedup/block_clones_full.json",
                                    dedup_full,
                                    section="dedup",
                                    description="Block clone groups (developer/full, exported from audit)",
                                )
                            writer.write_markdown(
                                "dedup/block_clones.md",
                                _md_kv(
                                    "Dedup: Block Clones (from audit)",
                                    [
                                        ("clone_groups_total", (dedup_payload.get("summary", {}) or {}).get("groups_total")),
                                        ("clone_groups_emitted", (dedup_payload.get("summary", {}) or {}).get("groups_emitted")),
                                        ("csv", "dedup/block_clones.csv"),
                                        ("full_json", "dedup/block_clones_full.json" if full_mode else "(disabled, use --full)"),
                                    ],
                                ),
                                section="dedup",
                                description="Block clone summary + developer/full (from audit)",
                            )
                except Exception as e:
                    writer.add_warning(f"Failed to export dedup/block_clones from audit: {e}")

                # Generate specs into run_dir (not project root)
                try:
                    sg = SpecGenerator()
                    writer.write_markdown(
                        "audit/Requirements.md",
                        sg.generate_requirements_from_audit(audit_result),
                        section="audit",
                        description="Generated requirements from audit",
                    )
                    writer.write_markdown(
                        "audit/Design.md",
                        sg.generate_design_from_audit(audit_result),
                        section="audit",
                        description="Generated design from audit",
                    )
                    writer.write_markdown(
                        "audit/Implementation.md",
                        sg.generate_implementation_from_audit(audit_result),
                        section="audit",
                        description="Generated implementation plan from audit",
                    )
                except Exception as e:
                    writer.add_warning(f"Spec generation from audit failed: {e}")
            
                # ------------------------------------------------------------
                # DASHBOARD (derived artifacts from audit)
                # ------------------------------------------------------------
                try:
                    health = _compute_health_score(audit_payload=audit_payload, index_status=index_status)
                    writer.write_json(
                        "dashboard/health_score.json",
                        health,
                        section="dashboard",
                        description="Aggregated health score (0..100)",
                    )
            
                    hotspots = _compute_hotspots(audit_payload, top_n=25)
                    writer.write_json(
                        "dashboard/hotspots.json",
                        hotspots,
                        section="dashboard",
                        description="Top hotspots by weighted finding score",
                    )
            
                    path_md = _generate_refactoring_path_md(
                        audit_payload=audit_payload,
                        hotspots_payload=hotspots,
                        run_id=run_id,
                    )
                    writer.write_markdown(
                        "dashboard/refactoring_path.md",
                        path_md,
                        section="dashboard",
                        description="Simple refactoring path (correlation-based)",
                    )

                    # Dependency hubs / keystone components (index-based)
                    try:
                        if index_available:
                            hubs = _compute_dependency_hubs(
                                db_path=db_path,
                                hotspots_payload=hotspots,
                                top_n_files=25,
                                top_n_targets=50,
                            )
                            writer.write_json(
                                "dashboard/dependency_hubs.json",
                                hubs,
                                section="dashboard",
                                description="Dependency hubs (fan-out) + keystone score",
                            )

                            graph = _compute_dependency_graph(
                                db_path=db_path,
                                hotspots_payload=hotspots,
                                hubs_payload=hubs,
                                max_external_nodes=200,
                            )
                            writer.write_json(
                                "dashboard/dependency_graph.json",
                                graph,
                                section="dashboard",
                                description="Dependency graph payload (files/modules)",
                            )
                    except Exception as e:
                        writer.add_warning(f"Dependency hubs generation failed: {e}")            
                except Exception as e:
                    writer.add_warning(f"Dashboard generation failed: {e}")
            
            except Exception as e:
                writer.add_warning(f"Audit skipped/failed: {e}")

    except Exception as e:
        # hard failure inside collect; still finalize artifacts
        writer.add_error(f"Collect failed: {e}")

    # ------------------------------------------------------------
    # LLM CONTEXT GENERATION (after all analysis is complete)
    # ------------------------------------------------------------
    try:
        from intellirefactor.refactoring.llm_context_generator import LLMContextGenerator
        
        llm_gen = LLMContextGenerator()
        llm_context_md = llm_gen.generate_llm_context_md_from_run(
            run_dir=str(run_dir),
            project_path=str(project_path),
        )
        
        # Write the LLM context to the run directory
        llm_context_path = run_dir / "llm_context.md"
        llm_context_path.write_text(llm_context_md, encoding="utf-8")
        writer.add_output(
            section="dashboard",
            kind="markdown",
            path=llm_context_path,
            description="LLM Mission Brief for refactoring (auto-generated)"
        )
    except Exception as e:
        writer.add_warning(f"LLM context generation failed: {e}")

    # finalize (always)
    success = len(writer.errors) == 0
    try:
        writer.write_summary_markdown()
    except Exception as e:
        writer.add_error(f"Failed to write summary.md: {e}")
    try:
        manifest_path = writer.write_manifest(success=success)
    except Exception as e:
        # if even manifest fails, return minimal info
        return {
            "success": False,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest": None,
            "warnings": writer.warnings,
            "errors": writer.errors + [f"Failed to write manifest.json: {e}"],
        }

    return {
        "success": success,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "warnings": writer.warnings,
        "errors": writer.errors,
    }