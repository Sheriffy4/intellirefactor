from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict())
        except Exception:
            pass
    return str(obj)


class ArtifactWriter:
    """
    Writes analysis artifacts to a run directory and maintains manifest.json.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        tool: str = "intellirefactor",
        tool_version: str = "0.1.0",
        project_path: Optional[str] = None,
        target_file: Optional[str] = None,
        run_id: Optional[str] = None,
        command: str = "collect",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.tool = tool
        self.tool_version = tool_version
        self.project_path = project_path
        self.target_file = target_file
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.command = command
        self.options = options or {}

        self.outputs: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

        self.started_at = datetime.now().isoformat()
        self.completed_at: Optional[str] = None

    def relpath(self, p: Path) -> str:
        try:
            return p.relative_to(self.run_dir).as_posix()
        except Exception:
            # make it stable even on Windows
            return str(p).replace(os.sep, "/")

    def add_output(self, *, section: str, kind: str, path: Path, description: str = "") -> None:
        self.outputs.append(
            {
                "section": section,
                "kind": kind,
                "path": self.relpath(path),
                "description": description,
            }
        )

    def write_json(self, rel_path: str, payload: Any, *, section: str, description: str = "") -> Path:
        p = self.run_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        data = _json_safe(payload)
        _atomic_write_text(
            p,
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self.add_output(section=section, kind="json", path=p, description=description)
        return p

    def write_markdown(self, rel_path: str, md: str, *, section: str, description: str = "") -> Path:
        p = self.run_dir / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(p, md, encoding="utf-8")
        self.add_output(section=section, kind="markdown", path=p, description=description)
        return p

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def write_summary_markdown(self) -> Path:
        # summary is commonly written before manifest; still show a meaningful completion timestamp
        if self.completed_at is None:
            self.completed_at = datetime.now().isoformat()
        lines: List[str] = []
        lines.append("# IntelliRefactor Collect Summary")
        lines.append("")
        lines.append(f"- **Run ID:** `{self.run_id}`")
        if self.project_path:
            lines.append(f"- **Project:** `{self.project_path}`")
        if self.target_file:
            lines.append(f"- **Target file:** `{self.target_file}`")
        lines.append(f"- **Started:** `{self.started_at}`")
        if self.completed_at:
            lines.append(f"- **Completed:** `{self.completed_at}`")
        lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if self.errors:
            lines.append("## Errors")
            for e in self.errors:
                lines.append(f"- {e}")
            lines.append("")

        lines.append("## Outputs")
        by_section: Dict[str, List[Dict[str, Any]]] = {}
        for o in self.outputs:
            by_section.setdefault(o["section"], []).append(o)

        for section in sorted(by_section.keys()):
            lines.append(f"### {section}")
            for o in by_section[section]:
                desc = f" — {o['description']}" if o.get("description") else ""
                lines.append(f"- `{o['path']}` ({o['kind']}){desc}")
            lines.append("")

        return self.write_markdown("summary.md", "\n".join(lines), section="meta", description="Run summary")

    def write_manifest(self, *, success: bool) -> Path:
        self.completed_at = datetime.now().isoformat()
        manifest = {
            "tool": self.tool,
            "tool_version": self.tool_version,
            "command": self.command,
            "run_id": self.run_id,
            "project_path": self.project_path,
            "target_file": self.target_file,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": bool(success),
            "options": _json_safe(self.options),
            "outputs": self.outputs,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        p = self.run_dir / "manifest.json"
        _atomic_write_text(
            p,
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return p