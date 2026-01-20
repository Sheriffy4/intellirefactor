"""
LLM Context Generator for Complex Refactorings.

Generates structured context and prompts for LLM-assisted refactoring:
- Collects code snippets with surrounding context
- Summarizes evidence
- Adds constraints, success criteria, and optional knowledge insights
- Produces a structured prompt payload

Extended (2026):
- Can generate a narrow "Mission Brief" (llm_context.md) for a target hotspot file
  picked from dashboard/refactoring_path.md, using run artifacts:
    - dashboard/refactoring_path.md
    - dashboard/dependency_hubs.json
    - dedup/block_clones.json
    - refactor/unused.json
    - decompose/smells.json
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence, Iterable

logger = logging.getLogger(__name__)

# Robust imports (package execution vs. direct run/test)
try:
    from intellirefactor.analysis.foundation.models import Evidence, FileReference
    from intellirefactor.analysis.refactor.refactoring_decision_engine import (
        RefactoringDecision,
        RefactoringType,
    )
    from intellirefactor.analysis.index.neighbor_extractor import NeighborExtractor
    from intellirefactor.analysis.index.store import IndexStore
    from ..knowledge.knowledge_manager import KnowledgeManager
except Exception:  # pragma: no cover
    Evidence = Any  # type: ignore
    FileReference = Any  # type: ignore
    RefactoringDecision = Any  # type: ignore
    RefactoringType = Any  # type: ignore
    NeighborExtractor = Any  # type: ignore
    IndexStore = Any  # type: ignore
    KnowledgeManager = Any  # type: ignore


class ContextType(Enum):
    """Types of LLM context generation."""

    REFACTORING_GUIDANCE = "refactoring_guidance"
    CODE_REVIEW = "code_review"
    IMPLEMENTATION_PLAN = "implementation_plan"
    RISK_ASSESSMENT = "risk_assessment"
    TESTING_STRATEGY = "testing_strategy"


class PromptTemplate(Enum):
    """Available prompt templates."""

    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    DECOMPOSE_GOD_CLASS = "decompose_god_class"
    ELIMINATE_DUPLICATES = "eliminate_duplicates"
    REDUCE_COMPLEXITY = "reduce_complexity"
    REMOVE_UNUSED_CODE = "remove_unused_code"
    GENERAL_REFACTORING = "general_refactoring"

    # New: plan-first template (narrow mission brief style)
    REFACTORING_PLANNER = "refactoring_planner"


@dataclass(frozen=True)
class SimpleFileRef:
    """Internal lightweight file reference (avoids depending on FileReference import)."""

    file_path: str
    line_start: int
    line_end: int


@dataclass
class CodeContext:
    """Context information about code to be refactored."""

    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    surrounding_context: str = ""
    dependencies: List[str] = field(default_factory=list)
    related_methods: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # e.g. ["SM1", "CG:exact_xxx:target", "UN3"]
    test_coverage: Optional[float] = None

    def get_full_context(self) -> str:
        """Return full code context including surrounding code."""
        if self.surrounding_context:
            return f"{self.surrounding_context}\n\n# TARGET CODE:\n{self.code_snippet}"
        return self.code_snippet


@dataclass
class RefactoringContext:
    """Complete context for a refactoring operation."""

    decision: RefactoringDecision
    target_code: List[CodeContext]
    evidence_summary: str
    knowledge_insights: List[str] = field(default_factory=list)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class LLMPrompt:
    """Structured prompt for LLM interaction."""

    template_type: PromptTemplate
    context_type: ContextType
    system_prompt: str
    user_prompt: str
    context_data: Dict[str, Any]
    expected_output_format: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "template_type": self.template_type.value,
            "context_type": self.context_type.value,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context_data": self.context_data,
            "expected_output_format": self.expected_output_format,
            "confidence": self.confidence,
        }


@dataclass
class MissionBrief:
    """
    Narrow mission brief for refactoring a single target file in the context of a run.

    This is the "target bundle" that reduces noise: only relevant findings,
    curated top groups, and precise code slices.
    """

    run_id: str
    project_path: str
    target_file: str
    goal: str
    constraints: List[str]
    acceptance_criteria: List[str]

    # Navigation payload (curated)
    clone_groups: List[Dict[str, Any]] = field(default_factory=list)
    smells: List[Dict[str, Any]] = field(default_factory=list)
    unused: List[Dict[str, Any]] = field(default_factory=list)
    dependency_notes: Dict[str, Any] = field(default_factory=dict)

    # Targeted code contexts (snippets around important line ranges)
    code_contexts: List[CodeContext] = field(default_factory=list)

    # Direct neighbors for LLM navigation (top 10-30 files)
    allowed_files: List[str] = field(default_factory=list)
    neighbor_summary: Dict[str, Any] = field(default_factory=dict)
    # neighbor_summary expected keys (best-effort): total_edges, relationship_summary, neighbors


class LLMContextGenerator:
    """
    Generates rich context for LLM-assisted refactoring operations.

    Integrates with optional knowledge management to provide relevant examples,
    patterns, and best practices for complex refactoring scenarios.

    Extended:
      - generate_llm_context_md_from_run(): build narrow "Mission Brief" from run artifacts.
    """

    def __init__(self, knowledge_manager: Optional[KnowledgeManager] = None) -> None:
        self.knowledge_manager = knowledge_manager
        self.prompt_templates = self._load_prompt_templates()
        logger.info("LLMContextGenerator initialized")

    def _resolve_existing_disk_paths(self, project_path: str, file_path: str) -> List[str]:
        """
        Best-effort path resolution for helping the LLM find the real file on disk.

        Problem observed in practice:
          - analysis stores paths like: intellirefactor/analysis/...
          - in repo the package root can be nested: <repo>/intellirefactor/intellirefactor/analysis/...
          - sometimes users run commands from <repo>/intellirefactor/intellirefactor and refer to paths like analysis/...

        We return a list of existing candidate paths (absolute).
        """
        root = Path(project_path)
        p = self._norm_path(file_path)

        # Candidate relative paths
        rels: List[str] = [p]

        # alias: strip leading "intellirefactor/"
        if p.startswith("intellirefactor/"):
            rels.append(p[len("intellirefactor/") :])
        else:
            rels.append("intellirefactor/" + p)

        # common nested package root: <repo>/intellirefactor/<path_without_prefix>
        if p.startswith("intellirefactor/"):
            rels.append("intellirefactor/" + p)  # may become duplicated but harmless
        else:
            rels.append("intellirefactor/" + p)

        seen: Set[str] = set()
        out: List[str] = []
        for r in rels:
            r = self._norm_path(r)
            if r in seen:
                continue
            seen.add(r)
            cand = root / r
            if cand.exists():
                out.append(str(cand.resolve()))
        return out

    # ---------------------------------------------------------------------
    # Public API (new): run-based mission brief
    # ---------------------------------------------------------------------

    def generate_llm_context_md_from_run(
        self,
        run_dir: str,
        project_path: Optional[str] = None,
        *,
        goal: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        max_clone_groups: int = 12,
        max_smells: int = 20,
        max_unused: int = 30,
        max_code_contexts: int = 18,
        context_lines: int = 6,
        max_neighbors: int = 30,
        min_confidence: float = 0.5,
    ) -> str:
        """
        Generate a narrow Mission Brief Markdown (llm_context.md) using run artifacts.

        Expected run_dir layout (relative paths):
          - dashboard/refactoring_path.md
          - dedup/block_clones.json   (preferred over CSV for line ranges)
          - refactor/unused.json
          - decompose/smells.json
          - dashboard/dependency_hubs.json (optional)

        Args:
            run_dir: path to run artifacts directory (e.g. "20260119_101739")
            project_path: project root (if None, tries manifest.json in run_dir)
            goal: optional override goal
            constraints: optional override constraints list
            acceptance_criteria: optional override acceptance list

        Returns:
            Markdown string content suitable for writing into llm_context.md
        """
        mission = self.generate_mission_brief_from_run(
            run_dir=run_dir,
            project_path=project_path,
            goal=goal,
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            max_clone_groups=max_clone_groups,
            max_smells=max_smells,
            max_unused=max_unused,
            max_code_contexts=max_code_contexts,
            context_lines=context_lines,
            max_neighbors=max_neighbors,
            min_confidence=min_confidence,
        )
        return self._render_mission_brief_md(mission)

    def generate_mission_brief_from_run(
        self,
        run_dir: str,
        project_path: Optional[str] = None,
        *,
        goal: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        max_clone_groups: int = 12,
        max_smells: int = 20,
        max_unused: int = 30,
        max_code_contexts: int = 18,
        context_lines: int = 6,
        max_neighbors: int = 30,
        min_confidence: float = 0.5,
    ) -> MissionBrief:
        """
        Build MissionBrief from run artifacts, targeting the hotspot file in dashboard/refactoring_path.md.
        """
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")

        manifest = self._try_load_json(run_path / "manifest.json") or {}
        run_id = str(manifest.get("run_id") or run_path.name)

        resolved_project_path = (
            project_path
            or str(manifest.get("project_path") or "")
            or str(self._infer_project_path_from_run_dir(run_path) or "")
        )
        if not resolved_project_path:
            raise ValueError("project_path is required (could not infer from manifest.json)")

        # 1) Pick target file from dashboard/refactoring_path.md
        refactoring_path_md = (
            self._try_read_text(run_path / "dashboard" / "refactoring_path.md") or ""
        )
        target_file = self._pick_target_file_from_refactoring_path(refactoring_path_md)
        if not target_file:
            raise ValueError("Could not determine target file from dashboard/refactoring_path.md")

        # Normalize target_file
        target_file = self._norm_path(target_file)

        # Defaults: goal / constraints / acceptance
        goal_final = goal or (
            "Refactor the target file to reduce duplication, complexity and architectural smells, "
            "without changing behavior or breaking public APIs."
        )

        constraints_final = constraints or [
            "Preserve behavior (no functional regressions).",
            "Maintain backward compatibility for public APIs/CLI interfaces.",
            "You may split code into multiple files within the same package to reduce complexity, "
            "but keep the original import path working via re-export or thin wrappers.",
            "Prefer small incremental changes; avoid global rewrites unrelated to the target.",
        ]

        acceptance_final = acceptance_criteria or [
            "All existing tests continue to pass.",
            "CLI entrypoints still work (no breakage).",
            "Diff is minimal and scoped to the target + necessary supporting files.",
            "Duplication/complexity/smells are measurably reduced in the target area.",
        ]

        # 2) Load and filter artifacts strictly for target
        clone_groups = self._load_clone_groups_for_target(
            run_path=run_path,
            target_file=target_file,
            limit=max_clone_groups,
        )
        smells = self._load_smells_for_target(
            run_path=run_path,
            target_file=target_file,
            limit=max_smells,
        )
        unused = self._load_unused_for_target(
            run_path=run_path,
            target_file=target_file,
            limit=max_unused,
        )
        dependency_notes = self._load_dependency_notes_for_target(
            run_path=run_path,
            target_file=target_file,
        )

        # 3) Build targeted code contexts (snippets around relevant line ranges)
        code_contexts = self._build_code_contexts_for_mission(
            project_path=resolved_project_path,
            target_file=target_file,
            clone_groups=clone_groups,
            smells=smells,
            unused=unused,
            max_code_contexts=max_code_contexts,
            context_lines=context_lines,
        )

        # 4) Extract direct neighbors for LLM navigation
        allowed_files, neighbor_summary = self._extract_allowed_files_for_target(
            project_path=resolved_project_path,
            target_file=target_file,
            max_neighbors=max_neighbors,
            min_confidence=min_confidence,
        )

        # Expand allowed_files with "other side" of clone groups (needed for real dedup across files)
        allowed_files = self._augment_allowed_files_with_clone_files(
            allowed_files=allowed_files,
            clone_groups=clone_groups,
            target_file=target_file,
            extra_limit=15,
        )

        return MissionBrief(
            run_id=run_id,
            project_path=resolved_project_path,
            target_file=target_file,
            goal=goal_final,
            constraints=constraints_final,
            acceptance_criteria=acceptance_final,
            clone_groups=clone_groups,
            smells=smells,
            unused=unused,
            dependency_notes=dependency_notes,
            code_contexts=code_contexts,
            allowed_files=allowed_files,
            neighbor_summary=neighbor_summary,
        )

    # ---------------------------------------------------------------------
    # Existing API (decision-based)
    # ---------------------------------------------------------------------

    def generate_refactoring_context(
        self, decision: RefactoringDecision, project_path: str
    ) -> RefactoringContext:
        """
        Generate comprehensive context for a refactoring decision.

        Args:
            decision: refactoring decision
            project_path: filesystem path to project root

        Returns:
            RefactoringContext with all necessary information
        """
        decision_id = getattr(decision, "decision_id", "<unknown>")
        logger.info("Generating context for decision: %s", decision_id)

        target_code = self._extract_target_code(decision, project_path)
        evidence_summary = self._generate_evidence_summary(getattr(decision, "evidence", []) or [])
        knowledge_insights = self._get_knowledge_insights(decision)
        similar_cases = self._find_similar_cases(decision)
        constraints = self._generate_constraints(decision)
        success_criteria = self._generate_success_criteria(decision)

        context = RefactoringContext(
            decision=decision,
            target_code=target_code,
            evidence_summary=evidence_summary,
            knowledge_insights=knowledge_insights,
            similar_cases=similar_cases,
            constraints=constraints,
            success_criteria=success_criteria,
        )

        logger.info("Generated context with %d code contexts", len(target_code))
        return context

    def generate_llm_prompt(
        self,
        context: RefactoringContext,
        context_type: ContextType = ContextType.REFACTORING_GUIDANCE,
    ) -> LLMPrompt:
        """
        Generate structured LLM prompt from refactoring context.
        """
        template_type = self._get_template_type(getattr(context.decision, "refactoring_type", None))
        template = self.prompt_templates.get(
            template_type, self.prompt_templates[PromptTemplate.GENERAL_REFACTORING]
        )

        system_prompt = self._generate_system_prompt(template, context, context_type)
        user_prompt = self._generate_user_prompt(template, context, context_type)

        decision = context.decision
        ref_type = getattr(
            getattr(decision, "refactoring_type", None),
            "value",
            str(getattr(decision, "refactoring_type", "")),
        )
        priority = getattr(
            getattr(decision, "priority", None),
            "value",
            str(getattr(decision, "priority", "")),
        )
        evidence = getattr(decision, "evidence", []) or []
        implementation_plan = getattr(decision, "implementation_plan", []) or []
        target_files = getattr(decision, "target_files", []) or []

        context_data = {
            "decision_id": getattr(decision, "decision_id", "<unknown>"),
            "refactoring_type": ref_type,
            "target_files": target_files,
            "confidence": float(getattr(decision, "confidence", 1.0) or 1.0),
            "priority": priority,
            "evidence_count": len(evidence),
            "implementation_steps": len(implementation_plan),
        }

        output_format = self._get_expected_output_format(context_type)

        prompt = LLMPrompt(
            template_type=template_type,
            context_type=context_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_data=context_data,
            expected_output_format=output_format,
            confidence=float(getattr(decision, "confidence", 1.0) or 1.0),
        )

        logger.info("Generated LLM prompt for %s", context_type.value)
        return prompt

    # ---------------------------------------------------------------------
    # Mission Brief rendering
    # ---------------------------------------------------------------------

    def _render_mission_brief_md(self, mission: MissionBrief) -> str:
        """
        Render MissionBrief into a practical llm_context.md (plan-first contract).
        """
        lines: List[str] = []
        lines.append("# LLM Mission Brief (auto-generated)")
        lines.append("")
        lines.append(f"- **Run ID:** `{mission.run_id}`")
        lines.append(f"- **Project:** `{mission.project_path}`")
        lines.append(f"- **Target file:** `{mission.target_file}`")
        lines.append("")

        # Path resolution hints (prevents wasted LLM effort on "find the file")
        lines.append("## File Path Resolution (important)")
        lines.append("")
        lines.append(
            "The analysis uses `file_path` values that may not match your current working directory."
        )
        lines.append("Use one of the following **existing on-disk paths**:")
        resolved = self._resolve_existing_disk_paths(mission.project_path, mission.target_file)
        if resolved:
            for p in resolved[:5]:
                lines.append(f"- `{p}`")
        else:
            lines.append(
                "- (Could not resolve an existing on-disk path automatically; check your repo layout.)"
            )
        lines.append("")
        lines.append("Recommended quick import check from repo root:")
        lines.append("")
        lines.append("```bash")
        lines.append(
            "python -c \"import sys; sys.path.insert(0, '.'); import intellirefactor; print('import ok')\""
        )
        lines.append("```")
        lines.append("")

        # ------------------------------------------------------------
        # Per-step Validation Checklist (micro-loop)
        # ------------------------------------------------------------
        lines.append("## Per-step Validation Checklist (run after EACH small step)")
        lines.append("")
        lines.append("Use this micro-loop after every extraction/move/delete:")
        lines.append("")
        lines.append("1) **Import check (fast)**")
        lines.append("```bash")
        lines.append(
            "python -c \"import sys; sys.path.insert(0,'.'); import intellirefactor; print('import ok')\""
        )
        lines.append("```")
        lines.append("")
        lines.append("2) **Run ONE relevant test file (fast)**")
        lines.append("```bash")
        lines.append("python -m pytest -q tests/test_imports.py")
        lines.append("```")
        lines.append("")
        lines.append("3) **Formatter**")
        lines.append("```bash")
        lines.append("python -m black <changed_paths> --line-length 100")
        lines.append("```")
        lines.append("")
        lines.append("4) **Lint (optional but recommended)**")
        lines.append("```bash")
        lines.append("# Ruff (preferred if installed)")
        lines.append("python -m ruff check <changed_paths>")
        lines.append("")
        lines.append("# Flake8 (alternative)")
        lines.append("python -m flake8 <changed_paths>")
        lines.append("```")
        lines.append("")
        lines.append("5) **After 2–3 steps**: run full tests")
        lines.append("```bash")
        lines.append("python -m pytest -q")
        lines.append("```")
        lines.append("")

        lines.append("## Role & Output Contract")
        lines.append("")
        lines.append("You are a **senior refactoring engineer** operating in two phases:")
        lines.append("")
        lines.append("1) **Planner phase (this response):**")
        lines.append("   - Provide a **6–12 step plan** (each step ≤ 30 minutes).")
        lines.append("   - For each step specify:")
        lines.append("     - what you change (functions/classes/line ranges)")
        lines.append("     - which findings/groups you address (use IDs below when available)")
        lines.append("     - risk level and how to validate (tests/CLI/smoke checks)")
        lines.append(
            "   - Provide an **Implementation Brief**: what to extract, new module/file names, re-exports/wrappers."
        )
        lines.append(
            "   - **Do not rewrite the whole file.** If code is needed, show only short snippets (≤ 30–60 lines)."
        )
        lines.append("")
        lines.append("2) **Executor phase (only after confirmation):**")
        lines.append("   - Implement step 1–2 and provide a patch/diff.")
        lines.append("")

        lines.append("## Goal")
        lines.append("")
        lines.append(mission.goal)
        lines.append("")

        lines.append("## Constraints")
        lines.append("")
        for c in mission.constraints:
            lines.append(f"- {c}")
        # Explicit scope rule (requested)
        if mission.allowed_files:
            lines.append(
                f"- Allowed change scope: modify only `{mission.target_file}` and files listed in **Allowed Files (Direct Neighbors)**. "
                "If you need changes outside this scope, STOP and ask for confirmation."
            )
        else:
            lines.append(
                f"- Allowed change scope: modify only `{mission.target_file}`. "
                "If you need changes outside this scope, STOP and ask for confirmation."
            )
        lines.append("")

        lines.append("## Acceptance Criteria")
        lines.append("")
        for c in mission.acceptance_criteria:
            lines.append(f"- {c}")
        lines.append("")

        # Execution playbook (order + convergence)
        lines.append("## Execution Playbook (follow this order)")
        lines.append("")
        has_critical_god_class = any(
            (s.get("smell_type") == "god_class" and str(s.get("severity")).lower() == "critical")
            for s in (mission.smells or [])
        )
        has_clones = bool(mission.clone_groups)
        has_unused = bool(mission.unused)

        lines.append("1) **Baseline / Safety**")
        lines.append("   - Run minimal smoke + tests (or at least import checks).")
        lines.append(
            "   - Save a baseline output for 1 fixture (generate Requirements/Design/Implementation once) to compare later."
        )
        lines.append("")

        if has_critical_god_class:
            lines.append("2) **Decomposition first (because `god_class` is critical)**")
            lines.append(
                "   - Split responsibilities into focused modules/files inside the same package."
            )
            lines.append("   - Keep backward compatibility via re-exports / thin wrappers.")
            lines.append("")
        else:
            lines.append("2) **Refactor structure incrementally**")
            lines.append(
                "   - Extract cohesive helpers/classes only where it reduces complexity/duplication."
            )
            lines.append("")

        if has_clones:
            lines.append("3) **Dedup during extraction / decomposition**")
            lines.append("   - When moving code, do NOT copy-paste.")
            lines.append(
                "   - Prefer extracting shared helpers (e.g., small markdown builders / formatting utilities) to prevent new clone groups."
            )
            lines.append("")
        else:
            lines.append("3) **Dedup (if any duplicates appear during refactor)**")
            lines.append("   - Avoid creating new duplication while splitting files.")
            lines.append("")

        if has_unused:
            lines.append("4) **Unused cleanup (only after call graph becomes explicit)**")
            lines.append("   - Remove only high-confidence unused code first (>= 0.8).")
            lines.append("   - Medium confidence: keep, mark, or require manual review.")
            lines.append("")
        else:
            lines.append("4) **Unused cleanup (if applicable)**")
            lines.append("   - Only remove high-confidence unused code after structure stabilizes.")
            lines.append("")

        lines.append("5) **Stabilize API**")
        lines.append("   - Ensure original imports/entrypoints still work.")
        lines.append("   - Keep thin wrappers and re-exports until migration is safe.")
        lines.append("")

        lines.append("6) **Re-run IntelliRefactor and verify convergence**")
        lines.append("   - Re-run analysis and compare:")
        lines.append("     - hotspot score for target")
        lines.append("     - smells (god_class/long_method/high_complexity)")
        lines.append("     - clone groups touching the target")
        lines.append("     - unused findings in target")
        lines.append("")

        lines.append("7) **Iterate**")
        lines.append(
            "   - Pick next best improvement (highest confidence / lowest risk) and repeat."
        )
        lines.append("")

        # Budget rules to prevent LOC explosion / boilerplate bloat
        lines.append("## Budget Rules (prevent LOC explosion)")
        lines.append("")
        lines.append("- Prefer functions over new classes unless a class adds real cohesion/value.")
        lines.append(
            "- Avoid adding docstrings to every small class/function; keep docstrings at module level + key public APIs."
        )
        lines.append(
            "- Do not increase total LOC in the refactoring area by **>20%** without explicit justification (e.g., better tests/clarity)."
        )
        lines.append("- Avoid boilerplate re-export files unless required for compatibility.")
        lines.append("")

        # External tooling guidance (black/ruff/flake8/pytest/mypy)
        lines.append("## Tooling & Validation Commands (use if available)")
        lines.append("")
        lines.append("Prefer running a fast validation after each small step:")
        lines.append("")
        lines.append("### Formatting")
        lines.append("```bash")
        lines.append("# Black (if installed)")
        lines.append("python -m black <paths> --line-length 100")
        lines.append("")
        lines.append("# Ruff formatter (if using ruff)")
        lines.append("python -m ruff format <paths>")
        lines.append("```")
        lines.append("")
        lines.append("### Lint")
        lines.append("```bash")
        lines.append("# Ruff (recommended if present)")
        lines.append("python -m ruff check <paths>")
        lines.append("")
        lines.append("# Flake8 (legacy alternative)")
        lines.append("python -m flake8 <paths>")
        lines.append("```")
        lines.append("")
        lines.append("### Tests / Type checks")
        lines.append("```bash")
        lines.append("# Unit tests")
        lines.append("python -m pytest -q")
        lines.append("")
        lines.append("# Type checking (if configured)")
        lines.append("python -m mypy <paths>")
        lines.append("```")
        lines.append("")

        # Synergy rules: connect plan steps to concrete findings
        lines.append("## Synergy Rules (plan must map to findings)")
        lines.append("")
        lines.append("- Each plan step must explicitly close **at least one** of:")
        lines.append("  - a smell entry (by smell_type + symbol + lines), or")
        lines.append("  - a clone group (by group_id), or")
        lines.append("  - an unused item (by symbol + lines).")
        lines.append(
            "- Prefer steps that reduce multiple issue types at once (e.g., decomposing a god_class while deduplicating shared formatting)."
        )
        lines.append("")

        # Dependency notes
        lines.append("## Dependency Notes (awareness only)")
        lines.append("")
        if mission.dependency_notes:
            for k, v in mission.dependency_notes.items():
                lines.append(f"- **{k}:** {v}")
        else:
            lines.append("- (No dependency notes available from artifacts.)")
        lines.append("")

        # Neighbor relationship summary + top-N table
        lines.append("## Neighbor Relationship Summary (Direct Neighbors)")
        lines.append("")
        if mission.neighbor_summary:
            total_edges = mission.neighbor_summary.get("total_edges", None)
            rel_sum = mission.neighbor_summary.get("relationship_summary") or {}

            if total_edges is not None:
                lines.append(f"- **total_edges:** {total_edges}")

            if rel_sum:
                ordered = ["imports", "imported_by", "calls", "called_by"]
                for k in ordered:
                    if k in rel_sum:
                        lines.append(f"- **{k}:** {rel_sum.get(k)}")
                extras = [k for k in rel_sum.keys() if k not in set(ordered)]
                for k in sorted(extras):
                    lines.append(f"- **{k}:** {rel_sum.get(k)}")
            else:
                lines.append("- (relationship_summary is empty)")

            neighbors = mission.neighbor_summary.get("neighbors") or []
            if neighbors:

                def _edge(n: Dict[str, Any]) -> int:
                    try:
                        return int(n.get("edge_count") or 0)
                    except Exception:
                        return 0

                top_n = sorted(neighbors, key=_edge, reverse=True)[:10]
                lines.append("")
                lines.append("### Top Neighbors (by edge_count)")
                lines.append("")
                lines.append("| File | Relationship | Edges | Conf | Symbols (top) |")
                lines.append("|---|---:|---:|---:|---|")
                for n in top_n:
                    fp = n.get("file_path", "<unknown>")
                    rel = n.get("relationship_type", "unknown")
                    edges = _edge(n)
                    conf = n.get("confidence", "")
                    syms = n.get("symbols_involved") or []
                    sym_preview = ", ".join(syms[:5])
                    lines.append(f"| `{fp}` | {rel} | {edges} | {conf} | {sym_preview} |")
        else:
            lines.append("- (No neighbor_summary available)")
        lines.append("")

        # Curated findings
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
                # add target+other location (operational)
                previews = g.get("instances_preview") or []
                target_loc = ""
                other_loc = ""
                for p in previews:
                    if not isinstance(p, dict):
                        continue
                    fp = self._norm_path(p.get("file_path") or "")
                    ls = p.get("line_start")
                    le = p.get("line_end")
                    loc = f"{fp}:{ls}-{le}"
                    if fp and self._path_matches(fp, mission.target_file) and not target_loc:
                        target_loc = loc
                    elif fp and (not self._path_matches(fp, mission.target_file)) and not other_loc:
                        other_loc = loc
                loc_suffix = ""
                if target_loc or other_loc:
                    loc_suffix = f" | target={target_loc or '-'} | other={other_loc or '-'}"
                lines.append(
                    f"- `{gid}` | type={ctype} | instances={inst} | files={uniq} | strategy={strat} | score={score}{loc_suffix}"
                )
        lines.append("")

        lines.append("### Smells (top)")
        if not mission.smells:
            lines.append("- (No smells found for target in artifacts.)")
        else:
            for s in mission.smells:
                sid = s.get("sm_id") or ""
                st = s.get("smell_type", "unknown")
                sev = s.get("severity", "unknown")
                sym = s.get("symbol_name", "")
                ls = s.get("line_start", "")
                le = s.get("line_end", "")
                prefix = f"{sid} " if sid else ""
                lines.append(f"- {prefix}{st} | severity={sev} | `{sym}` | lines {ls}-{le}")
        lines.append("")

        lines.append("### Unused (top)")
        if not mission.unused:
            lines.append("- (No unused findings found for target in artifacts.)")
        else:
            for u in mission.unused:
                uid = u.get("un_id") or ""
                name = u.get("symbol_name") or u.get("name") or u.get("symbol") or "<unknown>"
                conf = u.get("confidence", "")
                ls = u.get("line_start", "")
                le = u.get("line_end", "")
                ut = u.get("unused_type", u.get("type", ""))
                prefix = f"{uid} " if uid else ""
                lines.append(
                    f"- {prefix}`{name}` | unused_type={ut} | confidence={conf} | lines {ls}-{le}"
                )
        lines.append("")

        # ------------------------------------------------------------
        # File Outline (operational replacement for giant slices)
        # ------------------------------------------------------------
        lines.append("## File Outline (for precise navigation)")
        lines.append("")
        outline_md = self._generate_file_outline_md(
            project_path=mission.project_path,
            target_file=mission.target_file,
            smells=mission.smells,
            max_items=80,
        )
        lines.extend(outline_md)
        lines.append("")

        # Navigation map
        lines.append("## Navigation Map (start here)")
        lines.append("")
        lines.append(
            "Use the following **targeted code slices** as anchors. Start refactoring from the highest impact slices."
        )
        lines.append("")

        if not mission.code_contexts:
            lines.append(
                "- (No code contexts were extracted; check file paths and line ranges in artifacts.)"
            )
        else:
            for i, ctx in enumerate(mission.code_contexts, 1):
                lines.append(
                    f"### Slice {i}: `{ctx.file_path}` lines {ctx.line_start}-{ctx.line_end}"
                )
                if ctx.tags:
                    lines.append(f"- tags: {', '.join(ctx.tags)}")
                if ctx.dependencies:
                    lines.append(
                        f"- deps: {', '.join(ctx.dependencies[:20])}"
                        + (" ..." if len(ctx.dependencies) > 20 else "")
                    )
                if ctx.related_methods:
                    lines.append(
                        f"- methods: {', '.join(ctx.related_methods[:20])}"
                        + (" ..." if len(ctx.related_methods) > 20 else "")
                    )
                lines.append("")
                lines.append("```python")
                lines.append(ctx.get_full_context())
                lines.append("```")
                lines.append("")

        # Allowed files (direct neighbors for navigation)
        if mission.allowed_files:
            lines.append("## Allowed Files (Direct Neighbors)")
            lines.append("")
            lines.append(
                "These files are directly connected to the target (imports, calls, or called by)."
            )
            lines.append(
                "When refactoring, focus on these files for understanding dependencies and impact."
            )
            lines.append("")

            # Group by relationship type if available
            if mission.neighbor_summary and "neighbors" in mission.neighbor_summary:
                neighbors = mission.neighbor_summary.get("neighbors", [])
                neighbor_paths: Set[str] = set()
                for neighbor in neighbors[:15]:  # Top 15
                    fp = neighbor.get("file_path", "")
                    if fp:
                        neighbor_paths.add(self._norm_path(fp))
                    rel_type = neighbor.get("relationship_type", "")
                    edge_count = neighbor.get("edge_count", 0)
                    symbols = neighbor.get("symbols_involved", [])
                    lines.append(f"- `{fp}` ({rel_type}, {edge_count} edges)")
                    if symbols:
                        lines.append(f"  - symbols: {', '.join(symbols[:5])}")

                # Show additional allowed files that were appended from clone groups (not necessarily neighbors)
                extra_allowed = [
                    self._norm_path(p)
                    for p in mission.allowed_files
                    if self._norm_path(p) not in neighbor_paths
                    and self._norm_path(p) != self._norm_path(mission.target_file)
                ]
                if extra_allowed:
                    lines.append("")
                    lines.append("### Additional allowed files (from clone groups)")
                    lines.append("")
                    for fp in extra_allowed[:20]:
                        lines.append(f"- `{fp}`")
            else:
                # Fallback: just list files
                for fp in mission.allowed_files[:20]:
                    lines.append(f"- `{fp}`")
            lines.append("")

        # Task request
        lines.append("## Task Request")
        lines.append("")
        lines.append("1) Provide a **6–12 step plan** (each step ≤ 30 minutes).")
        lines.append("2) For each step, specify:")
        lines.append("   - what to change (functions/classes/line ranges)")
        lines.append(
            "   - which clone groups / smells / unused items it addresses (use IDs above where possible)"
        )
        lines.append("   - risk and validation")
        lines.append(
            "   - rule: each step must close at least 1 smell OR 1 clone group OR 1 unused item"
        )
        lines.append("3) Provide an **Implementation Brief**:")
        lines.append("   - what to extract")
        lines.append("   - suggested new file/module names (if splitting)")
        lines.append(
            "   - which names to re-export/keep as thin wrappers in the original file for compatibility"
        )
        lines.append("")

        return "\n".join(lines)

    # ---------------------------------------------------------------------
    # Artifact loading (target filtering)
    # ---------------------------------------------------------------------

    def _pick_target_file_from_refactoring_path(self, refactoring_path_md: str) -> str:
        """
        Extract target file from dashboard/refactoring_path.md.

        Expected line:
          - **File:** `intellirefactor/analysis/workflows/spec_generator.py`
        """
        m = re.search(r"-\s+\*\*File:\*\*\s+`([^`]+)`", refactoring_path_md)
        if m:
            return self._norm_path(m.group(1))

        # Fallback: sometimes format may differ; try any backticked .py after "Target hotspot"
        m2 = re.search(r"Target hotspot.*?`([^`]+\.py)`", refactoring_path_md, flags=re.S)
        if m2:
            return self._norm_path(m2.group(1))

        return ""

    def _parse_instance_str(self, s: str) -> Optional[Dict[str, Any]]:
        """
        Parse "path:123-456" into dict. Uses rightmost ':' to avoid problems.
        """
        try:
            s = str(s).strip()
            if not s:
                return None
            left, rng = s.rsplit(":", 1)
            m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", rng)
            if not m:
                return None
            return {
                "file_path": self._norm_path(left),
                "line_start": int(m.group(1)),
                "line_end": int(m.group(2)),
            }
        except Exception:
            return None

    def _path_aliases(self, p: str) -> Set[str]:
        """
        Provide a small alias set to reduce mismatch between:
          - "intellirefactor/analysis/..." and "analysis/..."
        """
        p = self._norm_path(p)
        out = {p}
        if p.startswith("intellirefactor/"):
            out.add(p[len("intellirefactor/") :])
        else:
            out.add("intellirefactor/" + p)
        return out

    def _path_matches(self, candidate: str, target: str) -> bool:
        cand = self._norm_path(candidate)
        return cand in self._path_aliases(target)

    def _load_clone_groups_for_target(
        self, run_path: Path, target_file: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Prefer audit/audit.json because dedup/block_clones.json is summary with instances_preview (<=5).
        Fallback to dedup/block_clones.json and then to CSV.
        """
        target_file = self._norm_path(target_file)

        # 0) Prefer audit/audit.json (full all_instances)
        audit_path = run_path / "audit" / "audit.json"
        if audit_path.exists():
            data = self._try_load_json(audit_path) or {}
            findings = data.get("findings") or []
            dup_findings = [
                f
                for f in findings
                if str(f.get("finding_type")) == "duplicate_block"
                and self._path_matches(str(f.get("file_path") or ""), target_file)
            ]

            severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

            groups: Dict[str, Dict[str, Any]] = {}
            for f in dup_findings:
                meta = f.get("metadata") or {}
                gid = meta.get("group_id") or f.get("group_id") or "<unknown>"
                ev = f.get("evidence") or {}
                ev_meta = ev.get("metadata") or {}
                clone_type = meta.get("clone_type") or ev_meta.get("clone_type") or "unknown"
                extraction_strategy = (
                    meta.get("extraction_strategy") or ev_meta.get("extraction_strategy") or "n/a"
                )
                extraction_confidence = meta.get("extraction_confidence") or ev_meta.get(
                    "extraction_confidence"
                )
                try:
                    similarity_score = float(
                        ev_meta.get("similarity_score") or (1.0 if clone_type == "exact" else 0.0)
                    )
                except Exception:
                    similarity_score = 0.0

                all_instances_raw = ev_meta.get("all_instances") or []
                parsed_instances = [self._parse_instance_str(s) for s in all_instances_raw]
                parsed_instances = [x for x in parsed_instances if x]

                inst_count = int(
                    ev_meta.get("instance_count")
                    or len(all_instances_raw)
                    or len(parsed_instances)
                    or 0
                )
                uniq_files = (
                    len({self._norm_path(i["file_path"]) for i in parsed_instances})
                    if parsed_instances
                    else 0
                )

                sev = str(f.get("severity") or "low").lower()
                conf = float(f.get("confidence") or 0.0)

                # pick target + other instance for practical dedup
                target_inst = next(
                    (
                        i
                        for i in parsed_instances
                        if self._path_matches(i["file_path"], target_file)
                    ),
                    None,
                )
                other_inst = next(
                    (
                        i
                        for i in parsed_instances
                        if not self._path_matches(i["file_path"], target_file)
                    ),
                    None,
                )

                g = groups.get(gid)
                if not g:
                    # ranking_score: coarse but useful for ordering
                    # severity dominates, then confidence, then size (instance_count/unique_files), then similarity
                    ranking_score = (
                        severity_rank.get(sev, 1) * 10.0
                        + conf * 5.0
                        + min(inst_count, 50) * 0.1
                        + min(uniq_files, 20) * 0.05
                        + similarity_score * 1.0
                    )
                    groups[gid] = {
                        "group_id": gid,
                        "clone_type": clone_type,
                        "severity": sev,
                        "confidence": conf,
                        "instance_count": inst_count,
                        "unique_files": uniq_files,
                        "similarity_score": similarity_score,
                        "ranking_score": ranking_score,
                        "extraction_strategy": extraction_strategy,
                        "extraction_confidence": extraction_confidence,
                        "instances_preview": [x for x in [target_inst, other_inst] if x],
                    }
                else:
                    # keep max severity + max counts
                    if severity_rank.get(sev, 1) > severity_rank.get(
                        str(g.get("severity") or "low"), 1
                    ):
                        g["severity"] = sev
                    g["confidence"] = max(float(g.get("confidence") or 0.0), conf)
                    g["instance_count"] = max(int(g.get("instance_count") or 0), inst_count)
                    g["unique_files"] = max(int(g.get("unique_files") or 0), uniq_files)
                    g["similarity_score"] = max(
                        float(g.get("similarity_score") or 0.0), similarity_score
                    )
                    if g.get("extraction_strategy") in (None, "", "n/a") and extraction_strategy:
                        g["extraction_strategy"] = extraction_strategy
                    if g.get("extraction_confidence") in (None, "", 0) and extraction_confidence:
                        g["extraction_confidence"] = extraction_confidence

                    # ensure we keep target/other if absent
                    prev = g.get("instances_preview") or []
                    have_target = any(
                        self._path_matches(p.get("file_path", ""), target_file)
                        for p in prev
                        if isinstance(p, dict)
                    )
                    have_other = any(
                        (not self._path_matches(p.get("file_path", ""), target_file))
                        for p in prev
                        if isinstance(p, dict)
                    )
                    if (not have_target) and target_inst:
                        prev.insert(0, target_inst)
                    if (not have_other) and other_inst:
                        prev.append(other_inst)
                    g["instances_preview"] = prev[:5]

            def _rank(g: Dict[str, Any]) -> Tuple[float, int, int]:
                score = float(g.get("ranking_score") or 0.0)
                inst = int(g.get("instance_count") or 0)
                files = int(g.get("unique_files") or 0)
                return (score, inst, files)

            out = list(groups.values())
            out.sort(key=_rank, reverse=True)
            return out[: max(0, int(limit))]

        json_path = run_path / "dedup" / "block_clones.json"
        if json_path.exists():
            data = self._try_load_json(json_path) or {}
            groups = data.get("groups") or []
            out: List[Dict[str, Any]] = []
            for g in groups:
                previews = g.get("instances_preview") or []
                if any(self._path_matches(p.get("file_path", ""), target_file) for p in previews):
                    out.append(
                        {
                            "group_id": g.get("group_id"),
                            "clone_type": g.get("clone_type"),
                            "similarity_score": g.get("similarity_score"),
                            "ranking_score": g.get("ranking_score"),
                            "extraction_strategy": g.get("extraction_strategy"),
                            "extraction_confidence": g.get("extraction_confidence"),
                            "instance_count": g.get("instance_count"),
                            "unique_files": g.get("unique_files"),
                            "instances_preview": previews,
                        }
                    )
            out.sort(
                key=lambda x: (
                    float(x.get("ranking_score") or 0.0),
                    int(x.get("instance_count") or 0),
                    int(x.get("unique_files") or 0),
                ),
                reverse=True,
            )
            return out[: max(0, int(limit))]

        # CSV fallback
        csv_path = run_path / "dedup" / "block_clones.csv"
        if csv_path.exists():
            out2: List[Dict[str, Any]] = []
            try:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        preview = str(row.get("instances_preview") or "")
                        if target_file in self._norm_path(preview):
                            out2.append(row)
            except Exception as e:
                logger.warning("Failed reading %s: %s", csv_path, e)
                return []

            # best-effort sort
            def _to_float(v: Any) -> float:
                try:
                    return float(v)
                except Exception:
                    return 0.0

            def _to_int(v: Any) -> int:
                try:
                    return int(float(v))
                except Exception:
                    return 0

            out2.sort(
                key=lambda r: (
                    _to_float(r.get("ranking_score")),
                    _to_int(r.get("instance_count")),
                    _to_int(r.get("unique_files")),
                ),
                reverse=True,
            )
            return out2[: max(0, int(limit))]

        return []

    def _load_smells_for_target(
        self, run_path: Path, target_file: str, limit: int
    ) -> List[Dict[str, Any]]:
        target_file = self._norm_path(target_file)
        smells_path = run_path / "decompose" / "smells.json"
        data = self._try_load_json(smells_path) or {}
        smells = data.get("smells") or []
        target_smells = [
            s for s in smells if self._norm_path(str(s.get("file_path") or "")) == target_file
        ]

        severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

        def _rank(s: Dict[str, Any]) -> Tuple[int, int]:
            sev = str(s.get("severity") or "low").lower()
            # prefer earlier lines slightly (stable-ish)
            ls = int(s.get("line_start") or 10**9)
            return (severity_rank.get(sev, 1), -ls)

        target_smells.sort(key=_rank, reverse=True)
        out = target_smells[: max(0, int(limit))]
        # assign stable-ish IDs for planning references
        for i, s in enumerate(out, 1):
            s["sm_id"] = f"SM{i}"
        return out

    def _load_unused_for_target(
        self, run_path: Path, target_file: str, limit: int
    ) -> List[Dict[str, Any]]:
        target_file = self._norm_path(target_file)
        unused_path = run_path / "refactor" / "unused.json"
        data = self._try_load_json(unused_path) or {}
        findings = data.get("findings") or data.get("unused") or []
        out = [f for f in findings if self._norm_path(str(f.get("file_path") or "")) == target_file]

        # Avoid useless "module_unreachable line 1--1" slices
        out = [f for f in out if str(f.get("unused_type") or "") != "module_unreachable"]

        # sort: confidence desc, then line_start asc
        out.sort(
            key=lambda f: (
                float(f.get("confidence") or 0.0),
                -int(f.get("line_start") or 10**9),
            ),
            reverse=True,
        )
        out = out[: max(0, int(limit))]
        for i, u in enumerate(out, 1):
            u["un_id"] = f"UN{i}"
        return out

    def _load_dependency_notes_for_target(self, run_path: Path, target_file: str) -> Dict[str, Any]:
        """
        dependency_hubs.json is global/top fanout. We only extract row if target appears there.
        Otherwise return a note that detailed edges are not available here.
        """
        target_file = self._norm_path(target_file)
        hubs_path = run_path / "dashboard" / "dependency_hubs.json"
        data = self._try_load_json(hubs_path) or {}
        top_files = data.get("top_files_by_fanout") or []
        for row in top_files:
            fp = self._norm_path(str(row.get("file_path") or ""))
            if fp == target_file:
                return {
                    "fanout_deps_total": row.get("deps_total"),
                    "fanout_unique_targets": row.get("unique_targets"),
                    "fanin_total": row.get("fanin_total"),
                    "hotspot_score": row.get("hotspot_score"),
                    "keystone_score": row.get("keystone_score"),
                }
        # If not found, keep a minimal note
        return {
            "note": "Target file is not in dependency_hubs top list. Consider extracting direct edges from dependency_graph/index.db for precise neighbor list."
        }

    # ---------------------------------------------------------------------
    # Neighbor extraction for LLM navigation
    # ---------------------------------------------------------------------

    def _extract_allowed_files_for_target(
        self,
        project_path: str,
        target_file: str,
        max_neighbors: int = 30,
        min_confidence: float = 0.5,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract direct neighbors (imports, calls) for target file using index.db.

        Returns:
            (allowed_files_list, neighbor_summary_dict)
        """
        try:
            # Try to load index.db from .intellirefactor directory
            project_root = Path(project_path)
            index_db_path = project_root / ".intellirefactor" / "index.db"

            if not index_db_path.exists():
                logger.warning(
                    "Index database not found at %s, skipping neighbor extraction",
                    index_db_path,
                )
                return [], {}

            store = IndexStore(str(index_db_path))
            extractor = NeighborExtractor(store)

            # Extract neighbors (top 30 files)
            graph = extractor.extract_neighbors(
                target_file=self._norm_path(target_file),
                max_neighbors=int(max_neighbors),
                min_confidence=float(min_confidence),
            )

            allowed_files = extractor.to_allowed_files_list(graph)
            neighbor_summary = extractor.to_dict(graph)

            logger.info(
                "Extracted %d neighbors for target %s (total edges: %d)",
                len(allowed_files),
                target_file,
                graph.total_edges,
            )

            return allowed_files, neighbor_summary

        except Exception as e:
            logger.warning("Failed to extract neighbors for %s: %s", target_file, e)
            return [], {}

    # ---------------------------------------------------------------------
    # Code context extraction for mission
    # ---------------------------------------------------------------------

    def _augment_allowed_files_with_clone_files(
        self,
        *,
        allowed_files: List[str],
        clone_groups: List[Dict[str, Any]],
        target_file: str,
        extra_limit: int = 15,
    ) -> List[str]:
        """
        Extend allowed_files with non-target files appearing in clone groups (needed for cross-file dedup).
        Keeps order and uniqueness.
        """
        target_file = self._norm_path(target_file)
        seen: Set[str] = set(self._norm_path(p) for p in (allowed_files or []))
        out: List[str] = [self._norm_path(p) for p in (allowed_files or [])]

        extras: List[str] = []
        for g in clone_groups:
            previews = g.get("instances_preview") or []
            for p in previews:
                if not isinstance(p, dict):
                    continue
                fp = self._norm_path(p.get("file_path", ""))
                if not fp:
                    continue
                if self._path_matches(fp, target_file):
                    continue
                extras.append(fp)

        # unique preserve order
        uniq_extras: List[str] = []
        for fp in extras:
            if fp in seen:
                continue
            seen.add(fp)
            uniq_extras.append(fp)
            if len(uniq_extras) >= int(extra_limit):
                break

        out.extend(uniq_extras)
        return out

    def _build_code_contexts_for_mission(
        self,
        *,
        project_path: str,
        target_file: str,
        clone_groups: List[Dict[str, Any]],
        smells: List[Dict[str, Any]],
        unused: List[Dict[str, Any]],
        max_code_contexts: int,
        context_lines: int,
    ) -> List[CodeContext]:
        """
        Build targeted snippets for:
          - top smells (3-5)
          - top clone instances for target (3-5) PLUS at least one "other" instance per group
          - unused high confidence (5-10)

        Deduplicates overlapping ranges.
        """
        target_file = self._norm_path(target_file)
        ranges: List[Tuple[str, int, int, str]] = []

        # smells (top 3-5)
        for s in smells[:5]:
            ls = int(s.get("line_start") or 1)
            le = int(s.get("line_end") or ls)
            ranges.append((target_file, ls, le, f"smell:{s.get('smell_type')}"))

        # clone instances (top 3-5 groups, pick target + other)
        for g in clone_groups[:5]:
            previews = g.get("instances_preview") or []
            target_inst = None
            other_inst = None
            for p in previews:
                if not isinstance(p, dict):
                    continue
                fp = self._norm_path(str(p.get("file_path") or ""))
                if self._path_matches(fp, target_file) and target_inst is None:
                    target_inst = p
                if (not self._path_matches(fp, target_file)) and other_inst is None:
                    other_inst = p
            if target_inst:
                fp = self._norm_path(str(target_inst.get("file_path") or target_file))
                ls = int(target_inst.get("line_start") or 1)
                le = int(target_inst.get("line_end") or ls)
                ranges.append((fp, ls, le, f"clone:{g.get('group_id')}:target"))
            if other_inst:
                fp = self._norm_path(str(other_inst.get("file_path") or ""))
                ls = int(other_inst.get("line_start") or 1)
                le = int(other_inst.get("line_end") or ls)
                if fp:
                    ranges.append((fp, ls, le, f"clone:{g.get('group_id')}:other"))

        # unused (prefer high confidence first)
        for u in unused[:10]:
            ls = int(u.get("line_start") or 1)
            le = int(u.get("line_end") or ls)
            ranges.append(
                (
                    target_file,
                    ls,
                    le,
                    f"unused:{u.get('symbol_name') or u.get('name') or ''}",
                )
            )

        # normalize & dedupe
        uniq: Set[Tuple[str, int, int]] = set()
        merged: List[Tuple[str, int, int, str]] = []
        for fp, ls, le, tag in ranges:
            ls2 = max(1, ls)
            le2 = max(ls2, le)
            fp2 = self._norm_path(fp)
            key = (fp2, ls2, le2)
            if key in uniq:
                continue
            uniq.add(key)
            merged.append((fp2, ls2, le2, tag))

        # limit
        merged = merged[: max(0, int(max_code_contexts))]

        # simple cache to avoid re-reading files repeatedly
        content_cache: Dict[str, str] = {}

        contexts: List[CodeContext] = []
        for fp, ls, le, _tag in merged:
            fp = self._norm_path(fp)

            if fp not in content_cache:
                full_path = (Path(project_path) / fp) if not Path(fp).is_absolute() else Path(fp)
                if not full_path.exists():
                    # try alias (with/without leading intellirefactor/)
                    for alt in self._path_aliases(fp):
                        alt_path = Path(project_path) / alt
                        if alt_path.exists():
                            fp = alt
                            full_path = alt_path
                            break
                if not full_path.exists():
                    logger.warning("Snippet file not found on disk: %s", full_path)
                    continue
                try:
                    content_cache[fp] = full_path.read_text(encoding="utf-8-sig")
                except Exception as e:
                    logger.warning("Failed reading snippet file %s: %s", full_path, e)
                    continue

            content = content_cache.get(fp, "")
            fr = SimpleFileRef(file_path=fp, line_start=ls, line_end=le)
            snippet = self._extract_code_snippet(content, fr)
            surrounding = self._extract_surrounding_context(
                content, fr, context_lines=context_lines
            )
            contexts.append(
                CodeContext(
                    file_path=fp,
                    line_start=ls,
                    line_end=le,
                    code_snippet=snippet,
                    surrounding_context=surrounding,
                    dependencies=self._extract_dependencies(snippet),
                    related_methods=self._extract_related_methods(snippet),
                )
            )
        return contexts

    def _generate_file_outline_md(
        self,
        *,
        project_path: str,
        target_file: str,
        smells: List[Dict[str, Any]],
        max_items: int = 80,
    ) -> List[str]:
        """
        Generate an outline of classes/functions with line ranges.
        Also shows which smell IDs overlap each symbol range (SM1..).
        """
        # resolve file on disk (best-effort)
        candidates = [target_file, *list(self._path_aliases(target_file))]
        content: Optional[str] = None
        used_path: Optional[str] = None
        for rel in candidates:
            p = Path(project_path) / self._norm_path(rel)
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8-sig")
                    used_path = self._norm_path(rel)
                    break
                except Exception:
                    continue

        if not content:
            return ["- (Outline unavailable: could not read target file from disk)"]

        try:
            tree = ast.parse(content)
        except Exception as e:
            return [f"- (Outline unavailable: AST parse failed: {e})"]

        # build smell intervals with IDs
        smell_intervals: List[Tuple[int, int, str]] = []
        for s in smells or []:
            sid = s.get("sm_id") or ""
            if not sid:
                continue
            try:
                ls = int(s.get("line_start") or 1)
                le = int(s.get("line_end") or ls)
            except Exception:
                continue
            if le < ls:
                le = ls
            smell_intervals.append((ls, le, sid))

        items: List[Dict[str, Any]] = []

        def add_item(kind: str, name: str, node: ast.AST):
            ls = int(getattr(node, "lineno", 1) or 1)
            le = int(getattr(node, "end_lineno", ls) or ls)
            loc = max(1, le - ls + 1)
            overlapping = [sid for (sls, sle, sid) in smell_intervals if not (le < sls or ls > sle)]
            items.append(
                {
                    "kind": kind,
                    "name": name,
                    "line_start": ls,
                    "line_end": le,
                    "loc": loc,
                    "smells": ", ".join(overlapping[:6]) + ("..." if len(overlapping) > 6 else ""),
                }
            )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                add_item("class", node.name, node)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        add_item("method", f"{node.name}.{sub.name}", sub)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_item("function", node.name, node)

        items.sort(key=lambda x: x["line_start"])
        items = items[: max(0, int(max_items))]

        md: List[str] = []
        md.append(f"- Resolved path: `{used_path}`")
        md.append("")
        md.append("| Kind | Symbol | Lines | LOC | Smell IDs |")
        md.append("|---|---|---:|---:|---|")
        for it in items:
            md.append(
                f"| {it['kind']} | `{it['name']}` | {it['line_start']}-{it['line_end']} | {it['loc']} | {it['smells']} |"
            )
        return md

    # ---------------------------------------------------------------------
    # Helpers: I/O and normalization
    # ---------------------------------------------------------------------

    def _norm_path(self, p: str) -> str:
        return str(p).replace("\\", "/").strip()

    def _try_read_text(self, path: Path) -> Optional[str]:
        try:
            if not path.exists():
                return None
            return path.read_text(encoding="utf-8-sig")
        except Exception as e:
            logger.warning("Failed reading %s: %s", path, e)
            return None

    def _try_load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            logger.warning("Failed loading json %s: %s", path, e)
            return None

    def _infer_project_path_from_run_dir(self, run_path: Path) -> Optional[str]:
        # Best-effort: manifest.json usually has it. Otherwise None.
        _ = run_path
        return None

    # ---------------------------------------------------------------------
    # Existing internal utilities (kept, slightly hardened)
    # ---------------------------------------------------------------------

    def _iter_file_refs_from_evidence(self, evidence: Any) -> List[Dict[str, Any]]:
        """
        Normalize evidence.file_references / evidence.locations into a list of dicts:
          {"file_path": str, "line_start": int, "line_end": int}
        Supports:
          - foundation.models.Evidence objects
          - dict evidence produced by analysis.refactor.refactoring_decision_engine
        """
        if evidence is None:
            return []

        # dict evidence (current decision engine uses dicts)
        if isinstance(evidence, dict):
            refs = evidence.get("file_references") or []
            locs = evidence.get("locations") or []
            out: List[Dict[str, Any]] = []
            for r in refs:
                if isinstance(r, dict) and r.get("file_path"):
                    out.append(
                        {
                            "file_path": str(r.get("file_path")),
                            "line_start": int(r.get("line_start") or 1),
                            "line_end": int(r.get("line_end") or (r.get("line_start") or 1)),
                        }
                    )
            for l in locs:
                if isinstance(l, dict) and l.get("file_path"):
                    out.append(
                        {
                            "file_path": str(l.get("file_path")),
                            "line_start": int(l.get("line_start") or 1),
                            "line_end": int(l.get("line_end") or (l.get("line_start") or 1)),
                        }
                    )
            return out

        # object evidence
        refs = getattr(evidence, "file_references", None) or []
        out2: List[Dict[str, Any]] = []
        for fr in refs:
            fp = getattr(fr, "file_path", None)
            if not fp:
                continue
            out2.append(
                {
                    "file_path": str(fp),
                    "line_start": int(getattr(fr, "line_start", 1) or 1),
                    "line_end": int(getattr(fr, "line_end", getattr(fr, "line_start", 1)) or 1),
                }
            )
        return out2

    def _iter_locations_from_evidence(self, evidence: Any) -> List[Dict[str, Any]]:
        """
        Normalize evidence into list of dicts:
          {"file_path": str, "line_start": int, "line_end": int}

        Supports:
          - dict evidence from RefactoringDecisionEngine (_evidence_dict)
          - Evidence objects with file_references/locations
        """
        out: List[Dict[str, Any]] = []
        if evidence is None:
            return out

        # dict evidence (current decision engine uses dicts)
        if isinstance(evidence, dict):
            refs = evidence.get("file_references") or []
            locs = evidence.get("locations") or []
            for r in refs:
                if isinstance(r, dict) and r.get("file_path"):
                    out.append(
                        {
                            "file_path": str(r["file_path"]),
                            "line_start": int(r.get("line_start") or 1),
                            "line_end": int(r.get("line_end") or (r.get("line_start") or 1)),
                        }
                    )
            for l in locs:
                if isinstance(l, dict) and l.get("file_path"):
                    out.append(
                        {
                            "file_path": str(l["file_path"]),
                            "line_start": int(l.get("line_start") or 1),
                            "line_end": int(l.get("line_end") or (l.get("line_start") or 1)),
                        }
                    )
            return out

        # Evidence object
        for fr in getattr(evidence, "file_references", None) or []:
            fp = getattr(fr, "file_path", None)
            if not fp:
                continue
            out.append(
                {
                    "file_path": str(fp),
                    "line_start": int(getattr(fr, "line_start", 1) or 1),
                    "line_end": int(getattr(fr, "line_end", getattr(fr, "line_start", 1)) or 1),
                }
            )
        for loc in getattr(evidence, "locations", None) or []:
            fp = getattr(loc, "file_path", None)
            if not fp:
                continue
            out.append(
                {
                    "file_path": str(fp),
                    "line_start": int(getattr(loc, "line_start", 1) or 1),
                    "line_end": int(getattr(loc, "line_end", getattr(loc, "line_start", 1)) or 1),
                }
            )
        return out

    def _extract_target_code(
        self, decision: RefactoringDecision, project_path: str
    ) -> List[CodeContext]:
        """Extract code contexts for target files from evidence file references."""
        contexts: List[CodeContext] = []
        seen: Set[Tuple[str, int, int]] = set()

        target_files = getattr(decision, "target_files", []) or []
        evidence_list = getattr(decision, "evidence", []) or []

        for file_path in target_files:
            file_path_norm = self._norm_path(str(file_path))
            full_path = (
                (Path(project_path) / file_path_norm)
                if not Path(file_path_norm).is_absolute()
                else Path(file_path_norm)
            )
            if not full_path.exists():
                logger.warning("Target file not found: %s", file_path)
                continue

            try:
                content = full_path.read_text(encoding="utf-8-sig")
            except Exception as e:
                logger.error("Error reading %s: %s", file_path, e)
                continue

            for evidence in evidence_list:
                locs = self._iter_locations_from_evidence(evidence)
                for loc in locs:
                    if self._norm_path(str(loc.get("file_path"))) != file_path_norm:
                        continue

                    line_start = int(loc.get("line_start") or 1)
                    line_end = int(loc.get("line_end") or line_start)

                    key = (file_path_norm, line_start, line_end)
                    if key in seen:
                        continue
                    seen.add(key)

                    fr_obj = SimpleFileRef(
                        file_path=file_path_norm,
                        line_start=line_start,
                        line_end=line_end,
                    )
                    code_snippet = self._extract_code_snippet(content, fr_obj)
                    surrounding_context = self._extract_surrounding_context(content, fr_obj)

                    contexts.append(
                        CodeContext(
                            file_path=file_path_norm,
                            line_start=line_start,
                            line_end=line_end,
                            code_snippet=code_snippet,
                            surrounding_context=surrounding_context,
                            dependencies=self._extract_dependencies(code_snippet),
                            related_methods=self._extract_related_methods(code_snippet),
                        )
                    )

        return contexts

    def _extract_code_snippet(self, content: str, file_ref: Any) -> str:
        """Extract code snippet from file content using file reference line numbers."""
        lines = content.splitlines()
        start_idx = max(0, int(getattr(file_ref, "line_start", 1) or 1) - 1)
        end_idx = min(
            len(lines),
            int(getattr(file_ref, "line_end", start_idx + 1) or (start_idx + 1)),
        )
        return "\n".join(lines[start_idx:end_idx])

    def _extract_surrounding_context(
        self, content: str, file_ref: Any, context_lines: int = 5
    ) -> str:
        """Extract surrounding context around target code."""
        lines = content.splitlines()
        line_start = int(getattr(file_ref, "line_start", 1) or 1)
        line_end = int(getattr(file_ref, "line_end", line_start) or line_start)

        before_start = max(0, line_start - context_lines - 1)
        before_end = max(0, line_start - 1)
        before_lines = lines[before_start:before_end] if before_end > before_start else []

        after_start = min(len(lines), line_end)
        after_end = min(len(lines), line_end + context_lines)
        after_lines = lines[after_start:after_end] if after_end > after_start else []

        context_parts: List[str] = []
        if before_lines:
            context_parts.append("# BEFORE:\n" + "\n".join(before_lines))
        if after_lines:
            context_parts.append("# AFTER:\n" + "\n".join(after_lines))

        return "\n\n".join(context_parts)

    def _extract_dependencies(self, code_snippet: str) -> List[str]:
        """
        Extract dependencies from code snippet.

        Best-effort:
        - Try AST parsing first (cleaner call extraction)
        - Fallback to regex if snippet isn't parseable
        """
        deps: Set[str] = set()

        try:
            tree = compile(
                code_snippet,
                "<snippet>",
                "exec",
                flags=ast.PyCF_ONLY_AST,
                dont_inherit=True,
                optimize=0,
            )
            for node in ast.walk(tree):  # type: ignore[name-defined]
                if isinstance(node, ast.Import):  # type: ignore[name-defined]
                    for alias in node.names:
                        deps.add(alias.name)
                elif isinstance(node, ast.ImportFrom):  # type: ignore[name-defined]
                    mod = node.module or ""
                    for alias in node.names:
                        deps.add(f"{mod}.{alias.name}" if mod else alias.name)
                elif isinstance(node, ast.Call):  # type: ignore[name-defined]
                    fn = node.func
                    if isinstance(fn, ast.Name):  # type: ignore[name-defined]
                        deps.add(fn.id)
                    elif isinstance(fn, ast.Attribute):  # type: ignore[name-defined]
                        # capture dotted call a.b.c()
                        parts: List[str] = []
                        cur = fn
                        while isinstance(cur, ast.Attribute):  # type: ignore[name-defined]
                            parts.append(cur.attr)
                            cur = cur.value  # type: ignore[assignment]
                        if isinstance(cur, ast.Name):  # type: ignore[name-defined]
                            parts.append(cur.id)
                        deps.add(".".join(reversed(parts)))
            return sorted(deps)
        except Exception:
            # Regex fallback
            import_matches = re.findall(r"(?:from\s+(\S+)\s+)?import\s+([^#\n]+)", code_snippet)
            for module, items in import_matches:
                item = items.strip()
                deps.add(f"{module}.{item}" if module else item)

            call_matches = re.findall(r"(\w+(?:\.\w+)*)\s*\(", code_snippet)
            deps.update(call_matches)
            return sorted(deps)

    def _extract_related_methods(self, code_snippet: str) -> List[str]:
        """Extract related method names from code snippet."""
        method_matches = re.findall(r"def\s+(\w+)\s*\(", code_snippet)
        self_method_matches = re.findall(r"self\.(\w+)\s*\(", code_snippet)
        return sorted(set(method_matches + self_method_matches))

    def _generate_evidence_summary(self, evidence_list: Sequence[Evidence]) -> str:
        """Generate summary of evidence supporting the refactoring decision."""
        if not evidence_list:
            return "No specific evidence provided."

        summary_parts: List[str] = []
        for i, evidence in enumerate(evidence_list, 1):
            if isinstance(evidence, dict):
                conf = float(evidence.get("confidence") or 0.0)
                desc = str(evidence.get("description") or "<no description>")
                file_refs = evidence.get("file_references") or evidence.get("locations") or []
                code_snips = evidence.get("code_snippets") or []
            else:
                conf = float(getattr(evidence, "confidence", 0.0) or 0.0)
                desc = getattr(evidence, "description", "<no description>")
                file_refs = getattr(evidence, "file_references", []) or []
                code_snips = getattr(evidence, "code_snippets", []) or []

            confidence_desc = "high" if conf > 0.8 else "medium" if conf > 0.5 else "low"

            summary_parts.append(
                f"{i}. {desc} (confidence: {confidence_desc})\n"
                f"   - Affects {len(file_refs)} file(s)\n"
                f"   - {len(code_snips)} code example(s) available"
            )

        return "\n\n".join(summary_parts)

    def _get_knowledge_insights(self, decision: RefactoringDecision) -> List[str]:
        """Get relevant insights from knowledge manager."""
        if not self.knowledge_manager:
            return []

        try:
            ref_type = getattr(
                getattr(decision, "refactoring_type", None),
                "value",
                str(getattr(decision, "refactoring_type", "")),
            )
            target_symbols = getattr(decision, "target_symbols", []) or []
            query = f"{ref_type} {' '.join(target_symbols)}".strip()

            insights = self.knowledge_manager.query_knowledge(query)
            return [i.get("description", "") for i in (insights or [])[:3]]
        except Exception as e:
            logger.warning("Failed to get knowledge insights: %s", e)
            return []

    def _find_similar_cases(self, decision: RefactoringDecision) -> List[Dict[str, Any]]:
        _ = decision
        return []

    def _generate_constraints(self, decision: RefactoringDecision) -> List[str]:
        """Generate constraints for the refactoring."""
        constraints: List[str] = []
        feasibility = getattr(decision, "feasibility", None)

        if feasibility:
            if getattr(feasibility, "breaking_changes_risk", False):
                constraints.append("Must maintain backward compatibility")
            if getattr(feasibility, "test_coverage_risk", False):
                constraints.append("Must preserve existing test coverage")
            if getattr(feasibility, "dependency_risk", False):
                constraints.append("Must not break external dependencies")

        # Use refactoring_type.name to avoid AttributeError if enum members differ.
        rt = getattr(decision, "refactoring_type", None)
        rt_name = getattr(rt, "name", str(rt))

        type_constraints_by_name: Dict[str, List[str]] = {
            "EXTRACT_METHOD": [
                "Extracted method must have clear single responsibility",
                "Method parameters should be minimal and cohesive",
            ],
            "EXTRACT_CLASS": [
                "New class must have high cohesion",
                "Interface should be minimal and focused",
            ],
            "DECOMPOSE_GOD_CLASS": [
                "Each resulting class must have single responsibility",
                "Dependencies between new classes should be minimal",
            ],
        }

        constraints.extend(type_constraints_by_name.get(rt_name, []))
        return constraints

    def _generate_success_criteria(self, decision: RefactoringDecision) -> List[str]:
        """Generate success criteria for the refactoring."""
        criteria: List[str] = [
            "All existing tests continue to pass",
            "Code compiles without errors",
            "No regression in functionality",
        ]

        impact_assessments = getattr(decision, "impact_assessments", []) or []
        for assessment in impact_assessments:
            benefits = getattr(assessment, "quantified_benefits", []) or []
            criteria.extend(list(benefits))

        rt = getattr(decision, "refactoring_type", None)
        rt_name = getattr(rt, "name", str(rt))

        type_criteria_by_name: Dict[str, List[str]] = {
            "ELIMINATE_DUPLICATES": [
                "Code duplication is eliminated",
                "Maintainability is improved",
            ],
            "REDUCE_METHOD_COMPLEXITY": [
                "Cyclomatic complexity is reduced",
                "Method is easier to understand",
            ],
            "REMOVE_UNUSED_CODE": [
                "Unused code is completely removed",
                "No references to removed code remain",
            ],
        }

        criteria.extend(type_criteria_by_name.get(rt_name, []))
        return criteria

    def _load_prompt_templates(self) -> Dict[PromptTemplate, Dict[str, str]]:
        """Load prompt templates for different refactoring types."""
        return {
            PromptTemplate.REFACTORING_PLANNER: {
                "system": (
                    "You are a senior refactoring planner. You produce small, safe, incremental plans "
                    "and a navigation map. You avoid global rewrites and focus on the target file."
                ),
                "user_base": "Build a mission plan to refactor the target file based on the curated findings and code slices.",
                "guidance": "Provide a 6–12 step plan (each step <= 30 minutes) + implementation brief. Keep code snippets short.",
                "review": "Review the plan for feasibility and minimal-risk execution.",
                "plan": "Create a step-by-step plan with validation criteria per step.",
                "risk": "Assess risks per step and propose mitigations.",
                "testing": "Suggest a testing strategy per step (unit/integration/smoke).",
            },
            PromptTemplate.EXTRACT_METHOD: {
                "system": "You are an expert software engineer specializing in method extraction refactoring.",
                "user_base": "Please help me extract a method from the following code:",
                "guidance": "Focus on creating a cohesive method with clear responsibility.",
                "review": "Review the proposed method extraction for correctness and best practices.",
                "plan": "Create a detailed implementation plan for method extraction.",
                "risk": "Assess the risks associated with this method extraction.",
                "testing": "Suggest testing strategies for the extracted method.",
            },
            PromptTemplate.EXTRACT_CLASS: {
                "system": "You are an expert software engineer specializing in class extraction refactoring.",
                "user_base": "Please help me extract a class from the following code:",
                "guidance": "Focus on creating a cohesive class with single responsibility.",
                "review": "Review the proposed class extraction for design quality.",
                "plan": "Create a detailed implementation plan for class extraction.",
                "risk": "Assess the risks associated with this class extraction.",
                "testing": "Suggest testing strategies for the extracted class.",
            },
            PromptTemplate.DECOMPOSE_GOD_CLASS: {
                "system": "You are an expert software engineer specializing in God Class decomposition.",
                "user_base": "Please help me decompose this God Class:",
                "guidance": "Focus on identifying distinct responsibilities and creating focused classes.",
                "review": "Review the proposed decomposition strategy.",
                "plan": "Create a step-by-step decomposition plan.",
                "risk": "Assess the risks of decomposing this large class.",
                "testing": "Suggest testing strategies during decomposition.",
            },
            PromptTemplate.ELIMINATE_DUPLICATES: {
                "system": "You are an expert software engineer specializing in duplicate code elimination.",
                "user_base": "Please help me eliminate the following code duplicates:",
                "guidance": "Focus on creating reusable components that eliminate duplication.",
                "review": "Review the proposed duplicate elimination strategy.",
                "plan": "Create an implementation plan for duplicate elimination.",
                "risk": "Assess the risks of eliminating these duplicates.",
                "testing": "Suggest testing strategies for duplicate elimination.",
            },
            PromptTemplate.GENERAL_REFACTORING: {
                "system": "You are an expert software engineer specializing in code refactoring.",
                "user_base": "Please help me refactor the following code:",
                "guidance": "Focus on improving code quality while preserving functionality.",
                "review": "Review the proposed refactoring approach.",
                "plan": "Create a detailed refactoring implementation plan.",
                "risk": "Assess the risks associated with this refactoring.",
                "testing": "Suggest comprehensive testing strategies.",
            },
        }

    def _get_template_type(self, refactoring_type: Any) -> PromptTemplate:
        """
        Map refactoring type to prompt template.

        Uses refactoring_type.name (string) to avoid hard dependency on exact Enum members.
        """
        rt_name = getattr(refactoring_type, "name", str(refactoring_type))

        mapping_by_name: Dict[str, PromptTemplate] = {
            "EXTRACT_METHOD": PromptTemplate.EXTRACT_METHOD,
            "EXTRACT_CLASS": PromptTemplate.EXTRACT_CLASS,
            "DECOMPOSE_GOD_CLASS": PromptTemplate.DECOMPOSE_GOD_CLASS,
            "ELIMINATE_DUPLICATES": PromptTemplate.ELIMINATE_DUPLICATES,
            "PARAMETERIZE_DUPLICATES": PromptTemplate.ELIMINATE_DUPLICATES,
            "TEMPLATE_METHOD_PATTERN": PromptTemplate.ELIMINATE_DUPLICATES,
            "REDUCE_METHOD_COMPLEXITY": PromptTemplate.EXTRACT_METHOD,
            "REMOVE_UNUSED_CODE": PromptTemplate.GENERAL_REFACTORING,
            "IMPROVE_COHESION": PromptTemplate.EXTRACT_CLASS,
            "REDUCE_COUPLING": PromptTemplate.EXTRACT_CLASS,
        }
        return mapping_by_name.get(rt_name, PromptTemplate.GENERAL_REFACTORING)

    def _generate_system_prompt(
        self,
        template: Dict[str, str],
        context: RefactoringContext,
        context_type: ContextType,
    ) -> str:
        """Generate system prompt from template and context."""
        _ = context
        base_system = template["system"]
        additions = {
            ContextType.REFACTORING_GUIDANCE: "Provide clear, actionable refactoring guidance.",
            ContextType.CODE_REVIEW: "Focus on code quality and best practices in your review.",
            ContextType.IMPLEMENTATION_PLAN: "Create detailed, step-by-step implementation plans.",
            ContextType.RISK_ASSESSMENT: "Thoroughly assess risks and provide mitigation strategies.",
            ContextType.TESTING_STRATEGY: "Focus on comprehensive testing approaches.",
        }
        return f"{base_system} {additions.get(context_type, '')}".strip()

    def _generate_user_prompt(
        self,
        template: Dict[str, str],
        context: RefactoringContext,
        context_type: ContextType,
    ) -> str:
        """Generate user prompt from template and context."""
        template_key = {
            ContextType.REFACTORING_GUIDANCE: "guidance",
            ContextType.CODE_REVIEW: "review",
            ContextType.IMPLEMENTATION_PLAN: "plan",
            ContextType.RISK_ASSESSMENT: "risk",
            ContextType.TESTING_STRATEGY: "testing",
        }.get(context_type, "guidance")

        base_prompt = template.get(template_key, template["user_base"])

        decision = context.decision
        ref_type = getattr(
            getattr(decision, "refactoring_type", None),
            "value",
            str(getattr(decision, "refactoring_type", "")),
        )
        priority = getattr(
            getattr(decision, "priority", None),
            "value",
            str(getattr(decision, "priority", "")),
        )
        confidence = float(getattr(decision, "confidence", 1.0) or 1.0)

        context_info: List[str] = []
        context_info.append(f"Refactoring Type: {ref_type}")
        context_info.append(f"Priority: {priority}")
        context_info.append(f"Confidence: {confidence:.2f}")

        if context.evidence_summary:
            context_info.append(f"Evidence:\n{context.evidence_summary}")

        if context.target_code:
            context_info.append("Target Code:")
            for i, code_ctx in enumerate(context.target_code, 1):
                context_info.append(
                    f"File {i}: {code_ctx.file_path} (lines {code_ctx.line_start}-{code_ctx.line_end})"
                )
                context_info.append(f"```python\n{code_ctx.get_full_context()}\n```")

        if context.constraints:
            context_info.append("Constraints:\n" + "\n".join(f"- {c}" for c in context.constraints))

        if context.success_criteria:
            context_info.append(
                "Success Criteria:\n" + "\n".join(f"- {c}" for c in context.success_criteria)
            )

        if context.knowledge_insights:
            context_info.append(
                "Relevant Insights:\n" + "\n".join(f"- {i}" for i in context.knowledge_insights)
            )

        return f"{base_prompt}\n\n" + "\n\n".join(context_info)

    def _get_expected_output_format(self, context_type: ContextType) -> str:
        """Get expected output format for context type."""
        formats = {
            ContextType.REFACTORING_GUIDANCE: "Provide step-by-step guidance with code examples",
            ContextType.CODE_REVIEW: "Provide structured review with specific recommendations",
            ContextType.IMPLEMENTATION_PLAN: "Provide numbered steps with time estimates and validation criteria",
            ContextType.RISK_ASSESSMENT: "Provide risk analysis with likelihood, impact, and mitigation strategies",
            ContextType.TESTING_STRATEGY: "Provide testing approach with specific test cases and coverage requirements",
        }
        return formats.get(context_type, "Provide clear, structured response")
