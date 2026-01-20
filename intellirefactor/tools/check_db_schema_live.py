from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?"
)  # very loose ISO8601 check


# ----------------------------
# reporting
# ----------------------------

@dataclass
class Finding:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["details"] is None:
            d["details"] = {}
        return d


class Report:
    def __init__(self) -> None:
        self.findings: List[Finding] = []

    def error(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(Finding("error", code, message, details or {}))

    def warn(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(Finding("warning", code, message, details or {}))

    def info(self, code: str, message: str, **details: Any) -> None:
        self.findings.append(Finding("info", code, message, details or {}))

    def has_errors(self) -> bool:
        return any(f.level == "error" for f in self.findings)

    def summary(self) -> Dict[str, Any]:
        return {
            "errors": sum(1 for f in self.findings if f.level == "error"),
            "warnings": sum(1 for f in self.findings if f.level == "warning"),
            "infos": sum(1 for f in self.findings if f.level == "info"),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "findings": [f.to_dict() for f in self.findings]}


# ----------------------------
# helpers
# ----------------------------

def _is_iso(s: Any) -> bool:
    if not isinstance(s, str) or not s:
        return False
    return bool(ISO_RE.match(s))

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

def _is_hex_len(s: Any, lengths: Sequence[int]) -> bool:
    if not isinstance(s, str):
        return False
    if len(s) not in lengths:
        return False
    return bool(HEX_RE.match(s))

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ----------------------------
# DB checks
# ----------------------------

REQUIRED_TABLES = {
    "files",
    "symbols",
    "blocks",
    "dependencies",
    "attribute_access",
    "schema_info",
}

REQUIRED_COLS = {
    "files": {"file_path", "content_hash", "lines_of_code"},
    "symbols": {"symbol_uid", "file_id", "kind", "line_start", "line_end", "is_public", "complexity_score"},
    "blocks": {"block_uid", "symbol_id", "block_type", "line_start", "line_end", "lines_of_code", "normalized_fingerprint"},
    "dependencies": {"source_symbol_id", "dependency_kind", "usage_count", "confidence", "resolution"},
}

ALLOWED_SYMBOL_KINDS = {"module", "class", "method", "function", "variable"}
ALLOWED_DEP_KINDS = {"calls", "imports", "inherits", "uses_attr", "instantiates", "unknown"}
ALLOWED_RESOLUTION = {"exact", "probable", "unknown"}

RECOMMENDED_INDEXES = {
    # minimal set that matters for performance + correctness in your queries
    "idx_dependencies_kind",
    "idx_dependencies_source",
    "idx_symbols_kind",
    "idx_symbols_qualified_name",
    "idx_blocks_normalized_fingerprint",
    "idx_files_content_hash",
    # optional (you added / want)
    "idx_symbols_complexity_score",
}

def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None

def columns_of(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]  # r[1] = name

def index_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    return [r[0] for r in cur.fetchall() if r and r[0]]

def scalar(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> Any:
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


def check_db(db_path: Path, report: Report, *, expected_schema_version: int = 3) -> None:
    if not db_path.exists():
        report.error("db.missing", f"DB file not found: {db_path}")
        return

    try:
        conn = connect(db_path)
    except Exception as e:
        report.error("db.open_failed", f"Failed to open DB: {e}", db_path=str(db_path))
        return

    with conn:
        # integrity check
        try:
            res = scalar(conn, "PRAGMA integrity_check")
            if str(res).lower() != "ok":
                report.error("db.integrity_check_failed", "PRAGMA integrity_check failed", result=res)
            else:
                report.info("db.integrity_ok", "PRAGMA integrity_check = ok")
        except Exception as e:
            report.warn("db.integrity_check_error", f"Could not run integrity_check: {e}")

        # foreign key check
        try:
            fk_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_rows:
                report.error(
                    "db.foreign_key_violations",
                    "Foreign key violations found",
                    count=len(fk_rows),
                    examples=[tuple(r) for r in fk_rows[:10]],
                )
            else:
                report.info("db.foreign_keys_ok", "No foreign key violations")
        except Exception as e:
            report.warn("db.foreign_key_check_error", f"Could not run foreign_key_check: {e}")

        # required tables
        for t in sorted(REQUIRED_TABLES):
            if not table_exists(conn, t):
                report.error("db.table_missing", f"Missing table: {t}", table=t)

        # schema version
        try:
            v = scalar(conn, "SELECT value FROM schema_info WHERE key='version'")
            v_int = _safe_int(v, -1)
            if v_int != expected_schema_version:
                report.error(
                    "db.schema_version_mismatch",
                    f"schema_info.version={v} but expected {expected_schema_version}",
                    version=v,
                    expected=expected_schema_version,
                )
            else:
                report.info("db.schema_version_ok", f"schema_info.version={v_int}")
        except Exception as e:
            report.error("db.schema_version_read_failed", f"Failed to read schema version: {e}")

        # required columns
        for table, req in REQUIRED_COLS.items():
            if not table_exists(conn, table):
                continue
            cols = set(columns_of(conn, table))
            missing = sorted(req - cols)
            if missing:
                report.error(
                    "db.columns_missing",
                    f"Table '{table}' missing columns: {missing}",
                    table=table,
                    missing=missing,
                )

        # recommended indexes
        try:
            idx = set(index_names(conn))
            missing_idx = sorted(RECOMMENDED_INDEXES - idx)
            if missing_idx:
                report.warn("db.indexes_missing", "Some recommended indexes are missing", missing=missing_idx)
            else:
                report.info("db.indexes_ok", "Recommended indexes present")
        except Exception as e:
            report.warn("db.index_list_failed", f"Could not list indexes: {e}")

        # ----------------------------
        # data sanity checks (facts)
        # ----------------------------

        # files
        try:
            total_files = scalar(conn, "SELECT COUNT(*) FROM files") or 0
            report.info("data.files.count", "Files count", count=int(total_files))
        except Exception as e:
            report.warn("data.files.count_failed", f"Could not count files: {e}")

        try:
            bad_paths = conn.execute(
                "SELECT file_path FROM files WHERE file_path IS NULL OR TRIM(file_path) = '' LIMIT 20"
            ).fetchall()
            if bad_paths:
                report.error("data.files.empty_path", "Empty file_path in files", examples=[r[0] for r in bad_paths])
        except Exception as e:
            report.warn("data.files.path_check_failed", f"files.file_path check failed: {e}")

        try:
            backslashes = conn.execute(
                r"SELECT file_path FROM files WHERE file_path LIKE '%\%' LIMIT 20"
            ).fetchall()
            if backslashes:
                report.warn(
                    "data.files.non_posix_paths",
                    "Some files.file_path contain backslashes (expected posix-style in v3)",
                    examples=[r[0] for r in backslashes],
                )
        except Exception:
            pass

        # symbols
        try:
            orphan_symbols = scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM symbols s
                LEFT JOIN files f ON s.file_id = f.file_id
                WHERE f.file_id IS NULL
                """,
            )
            if int(orphan_symbols or 0) > 0:
                report.error("data.symbols.orphan", "Orphan symbols (file missing)", count=int(orphan_symbols))
        except Exception as e:
            report.warn("data.symbols.orphan_check_failed", f"Orphan symbol check failed: {e}")

        try:
            bad_uid = conn.execute(
                "SELECT symbol_uid FROM symbols WHERE symbol_uid IS NULL OR TRIM(symbol_uid) = '' LIMIT 20"
            ).fetchall()
            if bad_uid:
                report.error("data.symbols.empty_uid", "Empty symbol_uid", examples=[r[0] for r in bad_uid])
        except Exception:
            pass

        # validate key fields for a sample of symbols (avoid huge scan)
        try:
            rows = conn.execute(
                """
                SELECT symbol_uid, kind, line_start, line_end, is_public, complexity_score
                FROM symbols
                LIMIT 5000
                """
            ).fetchall()
            for (uid, kind, ls, le, is_public, cplx) in rows:
                if not _is_hex_len(uid, (16, 32)):
                    report.warn("data.symbols.uid_format", "symbol_uid not hex length 16/32", symbol_uid=uid)
                    break
                if kind and str(kind) not in ALLOWED_SYMBOL_KINDS:
                    report.warn("data.symbols.kind_unknown", "Unexpected symbols.kind", kind=str(kind))
                    break
                if _safe_int(ls, 0) <= 0 or _safe_int(le, 0) <= 0 or _safe_int(ls, 0) > _safe_int(le, 0):
                    report.error("data.symbols.bad_line_range", "Invalid symbols line range", symbol_uid=uid, line_start=ls, line_end=le)
                    break
                if _safe_int(is_public, 0) not in (0, 1):
                    report.warn("data.symbols.bool_invalid", "symbols.is_public not 0/1", value=is_public)
                    break
                if _safe_int(cplx, -1) < 0:
                    report.warn("data.symbols.complexity_negative", "symbols.complexity_score < 0", value=cplx)
                    break
        except Exception as e:
            report.warn("data.symbols.sample_check_failed", f"Symbol sample checks failed: {e}")

        # blocks
        try:
            orphan_blocks = scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM blocks b
                LEFT JOIN symbols s ON b.symbol_id = s.symbol_id
                WHERE s.symbol_id IS NULL
                """,
            )
            if int(orphan_blocks or 0) > 0:
                report.error("data.blocks.orphan", "Orphan blocks (symbol missing)", count=int(orphan_blocks))
        except Exception as e:
            report.warn("data.blocks.orphan_check_failed", f"Orphan block check failed: {e}")

        try:
            bad = conn.execute(
                """
                SELECT block_uid, block_type, line_start, line_end, lines_of_code, statement_count, nesting_level, normalized_fingerprint
                FROM blocks
                LIMIT 5000
                """
            ).fetchall()
            for (uid, btype, ls, le, loc, stmt, nest, nfp) in bad:
                if not _is_hex_len(uid, (16, 32)):
                    report.warn("data.blocks.uid_format", "block_uid not hex length 16/32", block_uid=uid)
                    break
                if not btype or not str(btype).strip():
                    report.error("data.blocks.empty_type", "blocks.block_type empty", block_uid=uid)
                    break
                if _safe_int(ls, 0) <= 0 or _safe_int(le, 0) <= 0 or _safe_int(ls, 0) > _safe_int(le, 0):
                    report.error("data.blocks.bad_line_range", "Invalid blocks line range", block_uid=uid, line_start=ls, line_end=le)
                    break
                if _safe_int(loc, -1) < 0:
                    report.error("data.blocks.loc_negative", "blocks.lines_of_code < 0", block_uid=uid, loc=loc)
                    break
                if _safe_int(stmt, -1) < 0 or _safe_int(nest, -1) < 0:
                    report.warn("data.blocks.metrics_negative", "blocks statement_count/nesting_level < 0", block_uid=uid)
                    break
                # fingerprints vary in your codebase: allow 16/32 hex; empty is ok but warn
                if nfp and not _is_hex_len(nfp, (16, 32)):
                    report.warn("data.blocks.fp_format", "normalized_fingerprint not hex length 16/32", block_uid=uid, normalized_fingerprint=nfp)
                    break
        except Exception as e:
            report.warn("data.blocks.sample_check_failed", f"Block sample checks failed: {e}")

        # dependencies
        try:
            orphan_deps = scalar(
                conn,
                """
                SELECT COUNT(*)
                FROM dependencies d
                LEFT JOIN symbols s ON d.source_symbol_id = s.symbol_id
                WHERE s.symbol_id IS NULL
                """,
            )
            if int(orphan_deps or 0) > 0:
                report.error("data.deps.orphan_source", "Orphan dependencies (source symbol missing)", count=int(orphan_deps))
        except Exception as e:
            report.warn("data.deps.orphan_check_failed", f"Orphan dependency check failed: {e}")

        try:
            bad_rows = conn.execute(
                """
                SELECT dependency_id, dependency_kind, usage_count, confidence, resolution, target_symbol_id, target_external
                FROM dependencies
                LIMIT 8000
                """
            ).fetchall()
            for (dep_id, kind, usage_count, conf, res, target_id, target_ext) in bad_rows:
                k = str(kind or "").strip()
                if not k:
                    report.error("data.deps.empty_kind", "dependencies.dependency_kind empty", dependency_id=dep_id)
                    break
                # allow extra kinds, but warn
                if k not in ALLOWED_DEP_KINDS:
                    report.warn("data.deps.kind_unknown", "Unexpected dependency_kind", dependency_id=dep_id, dependency_kind=k)
                    break
                uc = _safe_int(usage_count, 0)
                if uc <= 0:
                    report.error("data.deps.bad_usage_count", "dependencies.usage_count must be >= 1", dependency_id=dep_id, usage_count=usage_count)
                    break
                c = _safe_float(conf, -1.0)
                if c < 0.0 or c > 1.0:
                    report.error("data.deps.bad_confidence", "dependencies.confidence must be 0..1", dependency_id=dep_id, confidence=conf)
                    break
                r = str(res or "").strip() or "probable"
                if r not in ALLOWED_RESOLUTION:
                    report.warn("data.deps.resolution_unknown", "Unexpected dependencies.resolution", dependency_id=dep_id, resolution=r)
                    break
                if target_id is None and (target_ext is None or str(target_ext).strip() == ""):
                    report.error("data.deps.no_target", "Dependency has neither target_symbol_id nor target_external", dependency_id=dep_id)
                    break
        except Exception as e:
            report.warn("data.deps.sample_check_failed", f"Dependency sample checks failed: {e}")

    conn.close()


# ----------------------------
# Dashboard checks
# ----------------------------

def check_dashboard(run_dir: Path, report: Report, *, db_path: Optional[Path] = None) -> None:
    if not run_dir.exists() or not run_dir.is_dir():
        report.error("dash.run_dir_missing", f"Run dir not found: {run_dir}")
        return

    dash_dir = run_dir / "dashboard"
    if not dash_dir.exists():
        report.warn("dash.missing", "No dashboard/ directory in run_dir", run_dir=str(run_dir))
        return

    # --- health_score.json
    hs_path = dash_dir / "health_score.json"
    if hs_path.exists():
        hs = _read_json(hs_path)
        if not isinstance(hs, dict):
            report.error("dash.health.invalid_json", "health_score.json is not valid JSON object")
        else:
            score = hs.get("score")
            if not isinstance(score, int) or score < 0 or score > 100:
                report.error("dash.health.bad_score", "health_score.score must be int 0..100", score=score)
            if not _is_iso(hs.get("computed_at")):
                report.warn("dash.health.no_computed_at", "health_score.computed_at not ISO-like", computed_at=hs.get("computed_at"))
    else:
        report.warn("dash.health.missing", "dashboard/health_score.json missing")

    # --- hotspots.json
    hot_path = dash_dir / "hotspots.json"
    hotspots = None
    if hot_path.exists():
        hotspots = _read_json(hot_path)
        if not isinstance(hotspots, dict):
            report.error("dash.hotspots.invalid_json", "hotspots.json is not valid JSON object")
            hotspots = None
        else:
            top_n = _safe_int(hotspots.get("top_n"), -1)
            hs_list = hotspots.get("hotspots")
            if top_n <= 0:
                report.warn("dash.hotspots.bad_top_n", "hotspots.top_n should be > 0", top_n=top_n)
            if not isinstance(hs_list, list):
                report.error("dash.hotspots.bad_list", "hotspots.hotspots must be a list")
            else:
                if top_n > 0 and len(hs_list) > top_n:
                    report.warn("dash.hotspots.too_many", "hotspots list longer than top_n", top_n=top_n, count=len(hs_list))
                for h in hs_list[:50]:
                    if not isinstance(h, dict) or not h.get("file_path"):
                        report.error("dash.hotspots.bad_item", "hotspot item missing file_path")
                        break
                    sc = _safe_float(h.get("score"), -1.0)
                    if sc < 0:
                        report.warn("dash.hotspots.score_negative", "hotspot.score < 0?", score=h.get("score"), file_path=h.get("file_path"))
                        break
    else:
        report.warn("dash.hotspots.missing", "dashboard/hotspots.json missing")

    # --- dependency_hubs.json
    hubs_path = dash_dir / "dependency_hubs.json"
    hubs = None
    if hubs_path.exists():
        hubs = _read_json(hubs_path)
        if not isinstance(hubs, dict):
            report.error("dash.hubs.invalid_json", "dependency_hubs.json is not valid JSON object")
            hubs = None
        else:
            if not _is_iso(hubs.get("computed_at")):
                report.warn("dash.hubs.no_computed_at", "dependency_hubs.computed_at not ISO-like", computed_at=hubs.get("computed_at"))

            totals = hubs.get("totals") or {}
            if isinstance(totals, dict):
                dep_total = totals.get("dependencies_total")
                if dep_total is None:
                    report.warn("dash.hubs.no_dep_total", "dependency_hubs.totals.dependencies_total missing")
                elif not isinstance(dep_total, int) or dep_total < 0:
                    report.error("dash.hubs.bad_dep_total", "dependencies_total must be int >= 0", dependencies_total=dep_total)

                # verify against DB count if possible
                if isinstance(dep_total, int) and db_path and db_path.exists():
                    try:
                        conn = connect(db_path)
                        real = scalar(conn, "SELECT COUNT(*) FROM dependencies") or 0
                        conn.close()
                        if int(real) != int(dep_total):
                            report.warn(
                                "dash.hubs.dep_total_mismatch",
                                "dependency_hubs totals.dependencies_total != DB COUNT(*)",
                                hubs_total=dep_total,
                                db_total=int(real),
                            )
                    except Exception as e:
                        report.warn("dash.hubs.db_check_failed", f"Could not compare dependency totals with DB: {e}")

            top_files = hubs.get("top_files_by_fanout")
            if isinstance(top_files, list) and top_files:
                # keystone formula check
                for row in top_files[:50]:
                    if not isinstance(row, dict):
                        continue
                    deps_total = _safe_float(row.get("deps_total"), -1.0)
                    hs_norm = _safe_float(row.get("hotspot_score_norm"), -1.0)
                    ks = _safe_float(row.get("keystone_score"), -1.0)

                    if deps_total < 0:
                        report.error("dash.hubs.bad_deps_total", "deps_total must be >= 0", row=row)
                        break
                    if hs_norm < 0.0 or hs_norm > 1.0:
                        report.warn("dash.hubs.bad_hotspot_norm", "hotspot_score_norm should be 0..1", hotspot_score_norm=hs_norm, file=row.get("file_path"))
                        break
                    expected = float(deps_total) * (1.0 + float(hs_norm if hs_norm >= 0 else 0.0))
                    if ks >= 0 and not _approx_equal(ks, expected, tol=1e-6):
                        report.warn(
                            "dash.hubs.keystone_mismatch",
                            "keystone_score != deps_total*(1+hotspot_score_norm) (tolerance)",
                            file=row.get("file_path"),
                            keystone_score=ks,
                            expected=expected,
                        )
                        break
            else:
                report.warn("dash.hubs.no_top_files", "dependency_hubs.top_files_by_fanout missing/empty")
    else:
        report.warn("dash.hubs.missing", "dashboard/dependency_hubs.json missing")

    # --- dependency_graph.json
    graph_path = dash_dir / "dependency_graph.json"
    if graph_path.exists():
        graph = _read_json(graph_path)
        if not isinstance(graph, dict):
            report.error("dash.graph.invalid_json", "dependency_graph.json is not valid JSON object")
        else:
            nodes = graph.get("nodes")
            edges = graph.get("edges")
            if not isinstance(nodes, list) or not isinstance(edges, list):
                report.error("dash.graph.bad_shape", "dependency_graph nodes/edges must be lists")
            else:
                node_ids = set()
                for n in nodes:
                    if isinstance(n, dict) and n.get("id"):
                        node_ids.add(str(n["id"]))
                # all edges must reference existing nodes
                bad_edges = 0
                for e in edges[:5000]:
                    if not isinstance(e, dict):
                        continue
                    s = str(e.get("source"))
                    t = str(e.get("target"))
                    if s not in node_ids or t not in node_ids:
                        bad_edges += 1
                        if bad_edges <= 5:
                            report.warn("dash.graph.edge_ref_missing_node", "Edge references missing node", edge=e)
                if bad_edges > 0:
                    report.warn("dash.graph.edges_with_missing_nodes", "Some edges reference nodes that are not present (often external nodes trimmed)", count=bad_edges)

                # summary consistency
                summary = graph.get("summary") or {}
                if isinstance(summary, dict):
                    nt = _safe_int(summary.get("nodes_total"), -1)
                    et = _safe_int(summary.get("edges_total"), -1)
                    if nt >= 0 and nt != len(nodes):
                        report.warn("dash.graph.summary_nodes_mismatch", "summary.nodes_total != len(nodes)", summary=nt, actual=len(nodes))
                    if et >= 0 and et != len(edges):
                        report.warn("dash.graph.summary_edges_mismatch", "summary.edges_total != len(edges)", summary=et, actual=len(edges))
    else:
        report.warn("dash.graph.missing", "dashboard/dependency_graph.json missing")


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Live DB + dashboard validator (IntelliRefactor)")
    ap.add_argument("--db", required=True, help="Path to index.db")
    ap.add_argument("--run-dir", default=None, help="Path to a run dir (intellirefactor_out/<run_id>) to validate dashboard")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    report = Report()

    db_path = Path(args.db).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None

    check_db(db_path, report, expected_schema_version=3)

    if run_dir:
        check_dashboard(run_dir, report, db_path=db_path)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        s = report.summary()
        print(f"Errors: {s['errors']}  Warnings: {s['warnings']}  Infos: {s['infos']}")
        for f in report.findings:
            if f.level in ("error", "warning"):
                det = f.details or {}
                det_str = ""
                if det:
                    det_str = " | " + ", ".join(f"{k}={det.get(k)!r}" for k in sorted(det.keys())[:8])
                print(f"[{f.level}] {f.code}: {f.message}{det_str}")

    return 2 if report.has_errors() else 0


if __name__ == "__main__":
    raise SystemExit(main())