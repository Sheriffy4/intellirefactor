"""
Main Decomposition Analyzer

Orchestrates the complete functional decomposition pipeline and provides:
- analysis (functional map, clusters, plans)
- report generation
- safe-first application of consolidation plans

This module includes:
- apply_safe/apply_assisted
- SAFE_EXACT_OK/FAIL evaluation per plan (logged into apply_manifest and returned)
- safe wrapper generation with decorator support for:
  @staticmethod/@classmethod/@property/@cached_property
- apply_assisted UPDATE_IMPORTS (from-import only), skipping imports that contain comments
- moved_impl (copy body to unified) only for top-level functions (not methods)
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ApplicationMode,
    CanonicalizationPlan,
    DecompositionConfig,
    ProjectFunctionalMap,
    SimilarityCluster,
    RecommendationType,
    PatchStepKind,
    FunctionalBlock,
    RiskLevel,
    EffortClass,
)
from .functional_map import FunctionalMapBuilder
from .consolidation_planner import ConsolidationPlanner
from .report_generator import DecompositionReportGenerator
from .file_operations import FileOperations
from .safe_exact_evaluator import SafeExactEvaluator
from .unified_symbol_generator import UnifiedSymbolGenerator
from . import ast_utils

logger = logging.getLogger(__name__)


class DecompositionAnalyzer:
    """
    Main analyzer for functional decomposition and consolidation.
    """

    _WRAPPER_MARKER = "[IR_DELEGATED] Auto-generated wrapper (functional decomposition)"

    _SAFE_DECORATORS_ALLOWLIST = {
        "staticmethod",
        "classmethod",
        "property",
        "cached_property",
        "functools.cached_property",
    }

    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.config = config or DecompositionConfig.default()
        self.logger = logger

        self.map_builder = FunctionalMapBuilder(self.config)

        try:
            self.planner = ConsolidationPlanner(self.config)  # type: ignore[arg-type]
        except TypeError:
            self.planner = ConsolidationPlanner()

        try:
            self.report_generator = DecompositionReportGenerator(self.config)  # type: ignore[arg-type]
        except TypeError:
            self.report_generator = DecompositionReportGenerator()

        self._functional_map: Optional[ProjectFunctionalMap] = None
        self._clusters: List[SimilarityCluster] = []
        self._plans: List[CanonicalizationPlan] = []

        # File operations with caching (extracted from god class)
        self.file_ops = FileOperations(self.config, self.logger)
        
        # SAFE_EXACT evaluator (extracted from god class)
        self.safe_exact_eval = SafeExactEvaluator(
            self.file_ops, 
            ast_utils, 
            self._SAFE_DECORATORS_ALLOWLIST
        )
        
        # Unified symbol generator (extracted from god class)
        self.unified_gen = UnifiedSymbolGenerator(
            self.file_ops,
            ast_utils,
            self._WRAPPER_MARKER
        )
        
        # Wrapper patcher (extracted from god class)
        from .wrapper_patcher import WrapperPatcher
        self.wrapper_patcher = WrapperPatcher(
            self._WRAPPER_MARKER,
            self._SAFE_DECORATORS_ALLOWLIST,
            ast_utils
        )
        
        # Import updater (extracted from god class)
        from .import_updater import ImportUpdater
        self.import_updater = ImportUpdater(self.file_ops, self.logger)
        
        # Unified alias validator (extracted from god class)
        from .validation import UnifiedAliasValidator
        self.alias_validator = UnifiedAliasValidator()
        
        # Statistics generator (extracted from god class)
        from .statistics_generator import StatisticsGenerator
        self.stats_gen = StatisticsGenerator()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def _stable_suffix(self, s: str) -> str:
        """Delegate to unified_symbol_generator.stable_suffix."""
        from .unified_symbol_generator import stable_suffix
        return stable_suffix(s)
    
    def _has_top_level_line(self, src: str, line: str) -> bool:
        """Delegate to unified_symbol_generator.has_top_level_line."""
        from .unified_symbol_generator import has_top_level_line
        return has_top_level_line(src, line)
    
    def analyze_project(
        self,
        project_root: str,
        output_dir: Optional[str] = None,
        mode: ApplicationMode = ApplicationMode.ANALYZE_ONLY,
    ) -> Dict[str, Any]:
        start_time = datetime.now()
        self.logger.info("Starting functional decomposition analysis of %s", project_root)

        try:
            # Step 1: Build functional map
            self.logger.info("Building functional map...")
            self._functional_map = self.map_builder.build_functional_map(project_root)

            if hasattr(self._functional_map, "recompute_stats"):
                self._functional_map.recompute_stats()  # type: ignore[attr-defined]

            # Step 2: Clusters
            self.logger.info("Extracting similarity clusters...")
            self._clusters = list(self._functional_map.clusters.values())

            # Step 3: Plans
            if mode != ApplicationMode.ANALYZE_ONLY or output_dir:
                self.logger.info("Generating consolidation plans...")
                plan_mode = mode if mode != ApplicationMode.ANALYZE_ONLY else ApplicationMode.PLAN_ONLY
                self._plans = self.planner.create_consolidation_plans(
                    self._clusters, self._functional_map, plan_mode
                )
            else:
                self._plans = []

            # Step 4: Reports
            report_files: Dict[str, str] = {}
            if output_dir:
                self.logger.info("Generating reports...")
                report_files = self.report_generator.generate_all_reports(
                    self._functional_map, self._clusters, self._plans, output_dir
                )

            # Step 5: Apply
            applied_changes: List[Dict[str, Any]] = []
            plan_evaluations: List[Dict[str, Any]] = []

            if mode in (ApplicationMode.APPLY_SAFE, ApplicationMode.APPLY_ASSISTED):
                if not output_dir:
                    self.logger.warning("apply_* mode requires output_dir; skipping apply.")
                else:
                    self.logger.info("Applying consolidation plans...")
                    applied_changes, plan_evaluations = self._apply_consolidation(
                        plans=self._plans,
                        functional_map=self._functional_map,
                        clusters=self._clusters,
                        project_root=project_root,
                        output_dir=output_dir,
                        mode=mode,
                    )

            duration = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "duration_seconds": duration,
                "functional_map": self._functional_map,
                "clusters": self._clusters,
                "plans": self._plans,
                "report_files": report_files,
                "statistics": self._generate_statistics(),
                "recommendations": self._generate_recommendations(),
                "applied_changes": applied_changes,
                "plan_evaluations": plan_evaluations,
            }

        except Exception as e:
            self.logger.error("Analysis failed: %s", e, exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

    
    
    def get_functional_map(self) -> Optional[ProjectFunctionalMap]:
        return self._functional_map

    def get_clusters(self) -> List[SimilarityCluster]:
        return self._clusters

    def get_plans(self) -> List[CanonicalizationPlan]:
        return self._plans

    def get_cluster_details(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        cluster = next((c for c in self._clusters if c.id == cluster_id), None)
        if not cluster or not self._functional_map:
            return None

        cluster_blocks = [
            self._functional_map.blocks[bid]
            for bid in cluster.blocks
            if bid in self._functional_map.blocks
        ]
        plan = next((p for p in self._plans if p.cluster_id == cluster_id), None)

        return {
            "cluster": cluster,
            "blocks": cluster_blocks,
            "plan": plan,
            "block_count": len(cluster_blocks),
            "total_loc": sum(b.loc for b in cluster_blocks),
            "avg_complexity": (sum(b.cyclomatic for b in cluster_blocks) / len(cluster_blocks)) if cluster_blocks else 0.0,
            "files_involved": sorted({b.file_path for b in cluster_blocks}),
        }

    def get_top_opportunities(self, limit: int = 10) -> List[Dict[str, Any]]:
        if not self._functional_map:
            return []

        opportunities: List[Dict[str, Any]] = []
        for cluster in self._clusters:
            benefit_score = self._calculate_cluster_benefit(cluster)
            details = self.get_cluster_details(cluster.id)
            if not details:
                continue

            opportunities.append(
                {
                    "cluster_id": cluster.id,
                    "category": f"{cluster.category}:{cluster.subcategory}",
                    "recommendation": cluster.recommendation.value,
                    "benefit_score": benefit_score,
                    "block_count": details["block_count"],
                    "total_loc": details["total_loc"],
                    "avg_similarity": cluster.avg_similarity,
                    "risk_level": cluster.risk_level.value,
                    "effort_class": cluster.effort_class.value,
                }
            )

        opportunities.sort(key=lambda x: x["benefit_score"], reverse=True)
        return opportunities[:limit]

    def export_results(self, output_file: str, fmt: str = "json") -> bool:
        try:
            if not self._functional_map:
                self.logger.error("No analysis results to export")
                return False

            if fmt.lower() == "json":
                self.report_generator.generate_json_report(
                    self._functional_map, self._clusters, self._plans, output_file
                )
            else:
                self.logger.error("Unsupported export format: %s", fmt)
                return False

            self.logger.info("Results exported to %s", output_file)
            return True

        except Exception as e:
            self.logger.error("Export failed: %s", e, exc_info=True)
            return False

    # ---------------------------------------------------------------------
    # APPLY ENGINE
    # ---------------------------------------------------------------------

    def _apply_consolidation(
        self,
        *,
        plans: List[CanonicalizationPlan],
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        project_root: str,
        output_dir: str,
        mode: ApplicationMode,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        project_root_path = Path(project_root).resolve()
        package_root, package_name = self._detect_package_root(project_root_path)
        self.logger.info("Detected package_root=%s package_name=%s", package_root, package_name)

        clusters_by_id = {c.id: c for c in clusters}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = out_dir / "_backups" / ts
        patch_root = out_dir / "patches" / ts
        backup_root.mkdir(parents=True, exist_ok=True)
        patch_root.mkdir(parents=True, exist_ok=True)

        evaluations: List[Dict[str, Any]] = []
        applied: List[Dict[str, Any]] = []

        # In safe mode, prefilter by low risk + small effort to reduce scanning cost
        if mode == ApplicationMode.APPLY_SAFE:
            candidate_plans = [
                p for p in plans
                if p.risk_assessment == RiskLevel.LOW and p.estimated_effort in (EffortClass.XS, EffortClass.S)
            ]
        else:
            candidate_plans = list(plans)

        for plan in candidate_plans:
            cluster = clusters_by_id.get(plan.cluster_id)

            safe_exact, reason = self._evaluate_safe_exact(plan, cluster, functional_map, package_root)
            evaluations.append(
                {
                    "cluster_id": plan.cluster_id,
                    "target_module": plan.target_module,
                    "target_symbol": plan.target_symbol,
                    "risk": getattr(plan.risk_assessment, "value", str(plan.risk_assessment)),
                    "effort": getattr(plan.estimated_effort, "value", str(plan.estimated_effort)),
                    "recommendation": getattr(cluster.recommendation, "value", "unknown") if cluster else "unknown",
                    "similarity": float(getattr(cluster, "avg_similarity", 0.0)) if cluster else 0.0,
                    "safe_exact": safe_exact,
                    "safe_exact_reason": reason,
                }
            )

            if mode == ApplicationMode.APPLY_SAFE:
                if safe_exact != "SAFE_EXACT_OK":
                    continue
                if not cluster or cluster.recommendation == RecommendationType.KEEP_SEPARATE:
                    continue

            if mode == ApplicationMode.APPLY_ASSISTED and safe_exact != "SAFE_EXACT_OK":
                if not sys.stdin.isatty():
                    continue
                if input(f"Plan {plan.cluster_id} is not SAFE_EXACT ({reason}). Apply anyway? [y/N]: ").strip().lower() != "y":
                    continue

            plan_record: Dict[str, Any] = {
                "cluster_id": plan.cluster_id,
                "target_module": plan.target_module,
                "target_symbol": plan.target_symbol,
                "mode": mode.value,
                "status": "PENDING",
                "canonical_block_id": "",
                "strategy": "",
                "safe_exact": safe_exact,
                "safe_exact_reason": reason,
                "files_modified": [],
                "files_created": [],
                "backups": [],
                "patches": [],
                "skipped_steps": [],
                "warnings": [],
                "errors": [],
            }

            created_files: List[Path] = []
            modified_files: List[Path] = []
            backups: List[Tuple[Path, Path]] = []

            try:
                if not cluster:
                    plan_record["status"] = "SKIPPED"
                    plan_record["warnings"].append("cluster not found")
                    applied.append(plan_record)
                    continue

                canonical_id = self._choose_canonical_block_id(plan, cluster, functional_map, package_root)
                plan_record["canonical_block_id"] = canonical_id or ""
                canonical_block = functional_map.blocks.get(canonical_id) if canonical_id else None
                if not canonical_block:
                    plan_record["status"] = "SKIPPED"
                    plan_record["warnings"].append("canonical block not resolved")
                    applied.append(plan_record)
                    continue

                # moved_impl only for top-level functions + only if safe_exact ok
                moved_impl_ok = (
                    safe_exact == "SAFE_EXACT_OK"
                    and not canonical_block.is_method
                    and float(getattr(cluster, "avg_similarity", 0.0)) >= float(getattr(self.config, "apply_safe_move_min_similarity", 0.995))
                    and bool(getattr(self.config, "apply_safe_move_impl", True))
                )

                # Run plan steps
                for step in plan.steps:
                    if mode == ApplicationMode.APPLY_SAFE and step.kind not in (PatchStepKind.ADD_NEW_MODULE, PatchStepKind.ADD_WRAPPER):
                        plan_record["skipped_steps"].append(
                            {"step_id": step.id, "kind": step.kind.value, "reason": "apply_safe skips this kind"}
                        )
                        continue

                    if step.kind == PatchStepKind.ADD_NEW_MODULE:
                        target_module = step.target_module or plan.target_module
                        target_symbol = step.target_symbol or plan.target_symbol
                        target_file = self._resolve_target_module_path(package_root=package_root, target_module=target_module)

                        self._ensure_package_files(target_file, package_root=package_root)

                        old_unified = self._read_text(target_file)

                        if moved_impl_ok:
                            new_unified, ok, warns = self._ensure_unified_symbol_moved_impl_top_level(
                                target_file=target_file,
                                target_symbol=target_symbol,
                                canonical_block=canonical_block,
                                package_root=package_root,
                                package_name=package_name,
                            )
                            moved_impl_ok = ok
                            plan_record["warnings"].extend([f"move_impl: {w}" for w in warns])

                            if ok and new_unified != old_unified:
                                if target_file.exists():
                                    bkp = self._backup_file(target_file, backup_root, package_root)
                                    backups.append((target_file, bkp))
                                    plan_record["backups"].append(str(bkp))

                                self._write_text(target_file, new_unified)
                                if not old_unified:
                                    created_files.append(target_file)
                                    plan_record["files_created"].append(str(target_file))
                                else:
                                    modified_files.append(target_file)
                                    plan_record["files_modified"].append(str(target_file))

                                patch_path = patch_root / f"{plan.cluster_id}_{step.id}_ADD_NEW_MODULE.patch"
                                self._write_patch(patch_path, old_unified, new_unified, str(target_file))
                                plan_record["patches"].append(str(patch_path))

                        if not moved_impl_ok:
                            new_unified = self._ensure_unified_symbol_delegating(
                                target_file=target_file,
                                target_symbol=target_symbol,
                                canonical_block=canonical_block,
                                canonical_block_id=canonical_id or "",
                                cluster_id=plan.cluster_id,
                                package_root=package_root,
                                package_name=package_name,
                            )

                            if new_unified != old_unified:
                                if target_file.exists():
                                    bkp = self._backup_file(target_file, backup_root, package_root)
                                    backups.append((target_file, bkp))
                                    plan_record["backups"].append(str(bkp))

                                self._write_text(target_file, new_unified)
                                if not old_unified:
                                    created_files.append(target_file)
                                    plan_record["files_created"].append(str(target_file))
                                else:
                                    modified_files.append(target_file)
                                    plan_record["files_modified"].append(str(target_file))

                                patch_path = patch_root / f"{plan.cluster_id}_{step.id}_ADD_NEW_MODULE.patch"
                                self._write_patch(patch_path, old_unified, new_unified, str(target_file))
                                plan_record["patches"].append(str(patch_path))

                    elif step.kind == PatchStepKind.ADD_WRAPPER:
                        if not step.source_blocks:
                            plan_record["warnings"].append(f"{step.id}: no source_blocks")
                            continue

                        block_id = step.source_blocks[0]
                        block = functional_map.blocks.get(block_id)
                        if not block:
                            plan_record["warnings"].append(f"{step.id}: missing block {block_id}")
                            continue

                        # If delegate strategy, wrapping canonical causes recursion -> skip
                        if (not moved_impl_ok) and canonical_id and block_id == canonical_id:
                            plan_record["skipped_steps"].append(
                                {"step_id": step.id, "kind": step.kind.value, "reason": "canonical skipped (delegate recursion)"}
                            )
                            continue

                        src_file = self._resolve_block_file_path(block, package_root)
                        if not src_file.exists():
                            plan_record["warnings"].append(f"{step.id}: file not found: {src_file}")
                            continue

                        old_src = self._read_text(src_file, bom=True)
                        seg = self._slice_lines(old_src, block.lineno, block.end_lineno)
                        if self._WRAPPER_MARKER in seg:
                            plan_record["skipped_steps"].append(
                                {"step_id": step.id, "kind": step.kind.value, "reason": "already wrapped (marker in callable slice)"}
                            )
                            continue

                        new_src = self._apply_wrapper_patch(
                            source_code=old_src,
                            block=block,
                            unified_module=step.target_module or plan.target_module,
                            unified_symbol=step.target_symbol or plan.target_symbol,
                            package_name=package_name,
                        )

                        if new_src != old_src:
                            bkp = self._backup_file(src_file, backup_root, package_root)
                            backups.append((src_file, bkp))
                            plan_record["backups"].append(str(bkp))

                            self._write_text(src_file, new_src)
                            modified_files.append(src_file)
                            plan_record["files_modified"].append(str(src_file))

                            patch_path = patch_root / f"{plan.cluster_id}_{step.id}_ADD_WRAPPER.patch"
                            self._write_patch(patch_path, old_src, new_src, str(src_file))
                            plan_record["patches"].append(str(patch_path))
                        else:
                            plan_record["warnings"].append(f"{step.id}: wrapper patch made no changes ({block.qualname})")

                    elif step.kind == PatchStepKind.UPDATE_IMPORTS:
                        if mode != ApplicationMode.APPLY_ASSISTED:
                            plan_record["skipped_steps"].append(
                                {"step_id": step.id, "kind": step.kind.value, "reason": "UPDATE_IMPORTS only in apply_assisted"}
                            )
                            continue

                        changed_files, warns = self._apply_update_imports_assisted(
                            step=step,
                            plan=plan,
                            functional_map=functional_map,
                            package_root=package_root,
                            package_name=package_name,
                            backup_root=backup_root,
                            patch_root=patch_root,
                            backups=backups,
                        )
                        plan_record["warnings"].extend(warns)
                        for f in changed_files:
                            plan_record["files_modified"].append(str(f))

                    else:
                        plan_record["skipped_steps"].append(
                            {"step_id": step.id, "kind": step.kind.value, "reason": "not auto-implemented"}
                        )

                plan_record["strategy"] = "moved_impl" if moved_impl_ok else "delegate"

                touched = sorted({*created_files, *modified_files}, key=lambda p: str(p))
                self._validate_files(touched)

                plan_record["status"] = "APPLIED"
                applied.append(plan_record)

            except Exception as e:
                plan_record["status"] = "FAILED"
                plan_record["errors"].append(str(e))
                self.logger.error("Apply failed for cluster %s: %s", plan.cluster_id, e, exc_info=True)
                self._rollback(backups=backups, created_files=created_files)
                applied.append(plan_record)

        manifest = {
            "timestamp": ts,
            "mode": mode.value,
            "package_root": str(package_root),
            "package_name": package_name,
            "evaluations": evaluations,
            "applied": applied,
        }
        (out_dir / f"apply_manifest_{ts}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return applied, evaluations

    # ---------------------------------------------------------------------
    # SAFE_EXACT evaluation
    # ---------------------------------------------------------------------
    def _evaluate_safe_exact(
        self,
        plan: CanonicalizationPlan,
        cluster: Optional[SimilarityCluster],
        fm: ProjectFunctionalMap,
        package_root: Path,
    ) -> Tuple[str, str]:
        """Delegate to safe_exact_eval.evaluate_safe_exact."""
        return self.safe_exact_eval.evaluate_safe_exact(plan, cluster, fm, package_root)

    # ---------------------------------------------------------------------
    # Canonical selection
    # ---------------------------------------------------------------------

    def _choose_canonical_block_id(
        self,
        plan: CanonicalizationPlan,
        cluster: SimilarityCluster,
        functional_map: ProjectFunctionalMap,
        package_root: Path,
    ) -> Optional[str]:
        # Prefer cluster-provided candidate
        if getattr(cluster, "canonical_candidate", "") and cluster.canonical_candidate in functional_map.blocks:
            return cluster.canonical_candidate

        # Prefer first ADD_NEW_MODULE source blocks if present
        for step in plan.steps:
            if step.kind == PatchStepKind.ADD_NEW_MODULE and step.source_blocks:
                blocks = [functional_map.blocks.get(bid) for bid in step.source_blocks]
                blocks = [b for b in blocks if b is not None]
                if blocks:
                    return self._choose_best_block(blocks, package_root)

        # Fallback: choose from cluster blocks
        blocks2 = [functional_map.blocks.get(bid) for bid in cluster.blocks]
        blocks2 = [b for b in blocks2 if b is not None]
        if blocks2:
            return self._choose_best_block(blocks2, package_root)

        return None

    def _choose_best_block(self, blocks: List[FunctionalBlock], package_root: Path) -> str:
        """
        Prefer:
          - not already wrapped (marker not present in that callable slice)
          - more callers
          - lower complexity
          - smaller LOC
        """
        def is_wrapped(b: FunctionalBlock) -> bool:
            try:
                p = self._resolve_block_file_path(b, package_root)
                src = self._read_text(p, bom=True)
                seg = self._slice_lines(src, b.lineno, b.end_lineno)
                return self._WRAPPER_MARKER in seg
            except Exception:
                return False

        candidates = [b for b in blocks if not is_wrapped(b)] or blocks

        best = max(
            candidates,
            key=lambda b: (
                len(getattr(b, "called_by", []) or []),
                -int(getattr(b, "cyclomatic", 0) or 0),
                -int(getattr(b, "loc", 0) or 0),
            ),
        )
        return best.id

    # ---------------------------------------------------------------------
    # Unified generation
    # ---------------------------------------------------------------------

    def _ensure_unified_symbol_moved_impl_top_level(
        self,
        *,
        target_file: Path,
        target_symbol: str,
        canonical_block: FunctionalBlock,
        package_root: Path,
        package_name: str,
    ) -> Tuple[str, bool, List[str]]:
        """Delegate to unified_gen.ensure_unified_symbol_moved_impl."""
        return self.unified_gen.ensure_unified_symbol_moved_impl(
            target_file=target_file,
            target_symbol=target_symbol,
            canonical_block=canonical_block,
            package_root=package_root,
            package_name=package_name,
        )

    def _ensure_unified_symbol_delegating(
        self,
        *,
        target_file: Path,
        target_symbol: str,
        canonical_block: FunctionalBlock,
        canonical_block_id: str,
        cluster_id: str,
        package_root: Path,
        package_name: str,
    ) -> str:
        """Delegate to unified_gen.ensure_unified_symbol_delegating."""
        return self.unified_gen.ensure_unified_symbol_delegating(
            target_file=target_file,
            target_symbol=target_symbol,
            canonical_block=canonical_block,
            canonical_block_id=canonical_block_id,
            cluster_id=cluster_id,
            package_root=package_root,
            package_name=package_name,
            module_dotted_from_filepath_fn=self._module_dotted_from_filepath,
            qualify_module_fn=self._qualify_module,
        )

    def _find_existing_unified_symbol_meta(self, src: str, symbol: str) -> Optional[Dict[str, str]]:
        """Delegate to unified_symbol_generator.find_existing_unified_symbol_meta."""
        from .unified_symbol_generator import find_existing_unified_symbol_meta
        return find_existing_unified_symbol_meta(src, symbol)
    
    def _canonical_method_call_expr(self, cls_expr: str, meth: str, dec_kind: str) -> str:
        """Delegate to unified_symbol_generator.canonical_method_call_expr."""
        from .unified_symbol_generator import canonical_method_call_expr
        return canonical_method_call_expr(cls_expr, meth, dec_kind)

    # ---------------------------------------------------------------------
    # Wrapper patch
    # ---------------------------------------------------------------------

    def _apply_wrapper_patch(
        self,
        *,
        source_code: str,
        block: FunctionalBlock,
        unified_module: str,
        unified_symbol: str,
        package_name: str,
    ) -> str:
        """Delegate to wrapper_patcher.apply_wrapper_patch."""
        return self.wrapper_patcher.apply_wrapper_patch(
            source_code=source_code,
            block=block,
            unified_module=unified_module,
            unified_symbol=unified_symbol,
            package_name=package_name,
        )

    def _build_call_arguments(self, args: ast.arguments) -> str:
        """Delegate to unified_symbol_generator.build_call_arguments."""
        from .unified_symbol_generator import build_call_arguments
        return build_call_arguments(args)

    # ---------------------------------------------------------------------
    # apply_assisted UPDATE_IMPORTS: from-import only, skip if comments
    # ---------------------------------------------------------------------

    def _apply_update_imports_assisted(
        self,
        *,
        step: Any,
        plan: CanonicalizationPlan,
        functional_map: ProjectFunctionalMap,
        package_root: Path,
        package_name: str,
        backup_root: Path,
        patch_root: Path,
        backups: List[Tuple[Path, Path]],
    ) -> Tuple[List[Path], List[str]]:
        """Delegate to import_updater.apply_update_imports_assisted."""
        return self.import_updater.apply_update_imports_assisted(
            step=step,
            plan=plan,
            functional_map=functional_map,
            package_root=package_root,
            package_name=package_name,
            backup_root=backup_root,
            patch_root=patch_root,
            backups=backups,
            module_dotted_from_target_module_fn=self._module_dotted_from_target_module,
            qualify_module_fn=self._qualify_module,
        )

    def _segment_has_comment(self, segment: str) -> bool:
        """Delegate to import_updater._segment_has_comment."""
        return self.import_updater._segment_has_comment(segment)

    # ---------------------------------------------------------------------
    # Path / files / caches / validation
    # ---------------------------------------------------------------------

    def _detect_package_root(self, project_root: Path) -> Tuple[Path, str]:
        """Delegate to file_ops.detect_package_root."""
        return self.file_ops.detect_package_root(project_root)

    def _ensure_package_files(self, target_file: Path, *, package_root: Path) -> None:
        """Delegate to file_ops.ensure_package_files."""
        self.file_ops.ensure_package_files(target_file, package_root=package_root)

    def _resolve_target_module_path(self, *, package_root: Path, target_module: str) -> Path:
        """Delegate to file_ops.resolve_target_module_path."""
        return self.file_ops.resolve_target_module_path(package_root=package_root, target_module=target_module)

    def _resolve_block_file_path(self, block: FunctionalBlock, package_root: Path) -> Path:
        """Delegate to file_ops.resolve_block_file_path."""
        return self.file_ops.resolve_block_file_path(block, package_root)

    def _backup_file(self, file_path: Path, backup_root: Path, package_root: Path) -> Path:
        """Delegate to file_ops.backup_file."""
        return self.file_ops.backup_file(file_path, backup_root, package_root)

    def _rollback(self, *, backups: List[Tuple[Path, Path]], created_files: List[Path]) -> None:
        """Delegate to file_ops.rollback."""
        self.file_ops.rollback(backups=backups, created_files=created_files)

    def _write_patch(self, patch_path: Path, old: str, new: str, file_label: str) -> None:
        """Delegate to file_ops.write_patch."""
        self.file_ops.write_patch(patch_path, old, new, file_label)

    def _validate_files(self, files: List[Path]) -> None:
        """Delegate to file_ops.validate_files."""
        self.file_ops.validate_files(files, validate_unified_aliases_fn=self._validate_unified_import_aliases)

    def _validate_unified_import_aliases(self, *, file_path: Path, code: str) -> None:
        """Delegate to alias_validator.validate_unified_import_aliases."""
        self.alias_validator.validate_unified_import_aliases(file_path=file_path, code=code)

    def _read_text(self, path: Path, bom: bool = False) -> str:
        """Delegate to file_ops.read_text."""
        return self.file_ops.read_text(path, bom)

    def _write_text(self, path: Path, content: str) -> None:
        """Delegate to file_ops.write_text."""
        self.file_ops.write_text(path, content)

    def _parse_file(self, path: Path) -> ast.Module:
        """Delegate to file_ops.parse_file."""
        return self.file_ops.parse_file(path)

    def _slice_lines(self, src: str, lineno: int, end_lineno: int) -> str:
        """Delegate to file_ops.slice_lines."""
        return self.file_ops.slice_lines(src, lineno, end_lineno)

    # ---------------------------------------------------------------------
    # AST lookup & decorators
    # ---------------------------------------------------------------------

    def _find_def_node(self, tree: ast.AST, qualname: str, lineno: Optional[int] = None) -> ast.AST:
        """Delegate to ast_utils.find_def_node."""
        return ast_utils.find_def_node(tree, qualname, lineno)

    def _decorator_name(self, d: ast.AST) -> str:
        """Delegate to ast_utils.get_decorator_name."""
        return ast_utils.get_decorator_name(d)

    def _decorator_kind(self, fn: ast.AST) -> str:
        """Delegate to ast_utils.get_decorator_kind."""
        return ast_utils.get_decorator_kind(fn)

    def _detect_async_def(self, file_path: Path, qualname: str, lineno: int) -> bool:
        """Detect if a function is async."""
        try:
            tree = self.file_ops.parse_file(file_path)
            node = ast_utils.find_def_node(tree, qualname, lineno=lineno)
            return isinstance(node, ast.AsyncFunctionDef)
        except Exception:
            return False

    def _free_names(self, fn: ast.AST) -> Set[str]:
        """Delegate to ast_utils.collect_free_names."""
        return ast_utils.collect_free_names(fn)

    # ---------------------------------------------------------------------
    # UPDATE_IMPORTS helpers
    # ---------------------------------------------------------------------

    def _iter_python_files(self, package_root: Path) -> Iterable[Path]:
        skip_dirs = {"__pycache__", ".git", ".venv", "venv", "build", "dist"}
        for p in package_root.rglob("*.py"):
            if any(part in skip_dirs for part in p.parts):
                continue
            if "unified" in p.parts:
                continue
            yield p

    def _file_module_name(self, file_path: Path, package_root: Path, package_name: str) -> str:
        rel = file_path.resolve().relative_to(package_root.resolve()).as_posix()
        if rel.endswith(".py"):
            rel = rel[:-3]
        if rel.endswith("/__init__"):
            rel = rel[: -len("/__init__")]
        rel = rel.replace("/", ".")
        return f"{package_name}.{rel}" if rel else package_name

    def _resolve_importfrom_abs_module(self, node: ast.ImportFrom, file_module: str) -> str:
        mod = node.module or ""
        level = int(node.level or 0)
        if level <= 0:
            return mod
        parts = file_module.split(".")
        base = parts[:-level] if level <= len(parts) else []
        if mod:
            base += mod.split(".")
        return ".".join([p for p in base if p])

    def _format_importfrom(self, module: Optional[str], names: List[ast.alias], level: int) -> str:
        prefix = "." * max(0, level)
        mod = module or ""
        parts = []
        for a in names:
            parts.append(f"{a.name} as {a.asname}" if a.asname else a.name)
        return f"from {prefix}{mod} import " + ", ".join(parts)

    # ---------------------------------------------------------------------
    # Module dotted helpers
    # ---------------------------------------------------------------------

    def _module_dotted_from_target_module(self, target_module: str) -> str:
        """Delegate to wrapper_patcher._module_dotted_from_target_module."""
        return self.wrapper_patcher._module_dotted_from_target_module(target_module)

    def _module_dotted_from_filepath(self, file_path: Path, package_root: Path, package_name: str) -> str:
        rel = file_path.resolve().relative_to(package_root.resolve()).as_posix()
        if rel.endswith(".py"):
            rel = rel[:-3]
        return f"{package_name}.{rel.replace('/', '.')}"

    def _qualify_module(self, mod: str, package_name: str) -> str:
        mod = (mod or "").strip()
        if not mod:
            return mod
        if mod == package_name or mod.startswith(package_name + "."):
            return mod
        return f"{package_name}.{mod}"

    # ---------------------------------------------------------------------
    # Statistics / recommendations / benefit
    # ---------------------------------------------------------------------

    def _generate_statistics(self) -> Dict[str, Any]:
        """Delegate to stats_gen.generate_statistics."""
        if not self._functional_map:
            return {}
        return self.stats_gen.generate_statistics(
            self._functional_map,
            self._clusters,
            self._plans
        )

    def _generate_recommendations(self) -> List[str]:
        """Delegate to stats_gen.generate_recommendations."""
        if not self._functional_map:
            return ["Run analysis first to get recommendations"]
        return self.stats_gen.generate_recommendations(
            self._functional_map,
            self._clusters
        )

    def _calculate_cluster_benefit(self, cluster: SimilarityCluster) -> float:
        """Delegate to stats_gen.calculate_cluster_benefit."""
        return self.stats_gen.calculate_cluster_benefit(cluster)