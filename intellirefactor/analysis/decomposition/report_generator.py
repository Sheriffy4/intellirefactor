"""
Decomposition Report Generator

Generates comprehensive reports in multiple formats (JSON, Markdown, Mermaid)
for functional decomposition analysis results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    ProjectFunctionalMap,
    SimilarityCluster,
    CanonicalizationPlan,
    FunctionalBlock,
    Capability,
    RecommendationType,
    RiskLevel,
    EffortClass,
    DecompositionConfig,
)

logger = logging.getLogger(__name__)


class DecompositionReportGenerator:
    """
    Generates comprehensive reports for functional decomposition analysis.
    """

    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.logger = logger
        self.config = config or DecompositionConfig.default()

    # --------------------------
    # Public API
    # --------------------------

    def generate_all_reports(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        plans: List[CanonicalizationPlan],
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate all report formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_files: Dict[str, str] = {}

        # JSON report
        json_file = output_path / f"functional_map_{timestamp}.json"
        self.generate_json_report(functional_map, clusters, plans, str(json_file))
        generated_files["json"] = str(json_file)

        # Compatibility without timestamp
        json_compat = output_path / "functional_map.json"
        self.generate_json_report(functional_map, clusters, plans, str(json_compat))

        # Markdown catalog
        catalog_file = output_path / f"catalog_{timestamp}.md"
        self.generate_catalog_markdown(functional_map, clusters, str(catalog_file))
        generated_files["catalog"] = str(catalog_file)

        catalog_compat = output_path / "catalog.md"
        self.generate_catalog_markdown(functional_map, clusters, str(catalog_compat))

        # Consolidation plan
        plan_file = output_path / f"consolidation_plan_{timestamp}.md"
        self.generate_plan_markdown(clusters, plans, str(plan_file))
        generated_files["plan"] = str(plan_file)

        plan_compat = output_path / "consolidation_plan.md"
        self.generate_plan_markdown(clusters, plans, str(plan_compat))

        # Mermaid diagram
        diagram_file = output_path / f"functional_graph_{timestamp}.mmd"
        self.generate_mermaid_diagram(functional_map, clusters, str(diagram_file))
        generated_files["diagram"] = str(diagram_file)

        diagram_compat = output_path / "functional_graph.mmd"
        self.generate_mermaid_diagram(functional_map, clusters, str(diagram_compat))

        # Summary report
        summary_file = output_path / f"summary_{timestamp}.md"
        self.generate_summary_report(functional_map, clusters, plans, str(summary_file))
        generated_files["summary"] = str(summary_file)

        summary_compat = output_path / "summary.md"
        self.generate_summary_report(functional_map, clusters, plans, str(summary_compat))

        self.logger.info(f"Generated {len(generated_files)} reports in {output_dir}")
        return generated_files

    def generate_json_report(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        plans: List[CanonicalizationPlan],
        output_file: str,
    ) -> None:
        """Generate machine-readable JSON report."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        report_data: Dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_root": functional_map.project_root,
                "analysis_timestamp": functional_map.timestamp,
            },
            "statistics": {
                "total_blocks": functional_map.total_blocks,
                "total_capabilities": functional_map.total_capabilities,
                "total_clusters": functional_map.total_clusters,
                "resolution_rate": functional_map.resolution_rate,
                "resolution_rate_internal": getattr(functional_map, "resolution_rate_internal", 0.0),
                "resolution_rate_actionable": getattr(functional_map, "resolution_rate_actionable", 0.0),
                "external_calls_count": getattr(functional_map, "external_calls_count", 0),
                "dynamic_attribute_calls_count": getattr(functional_map, "dynamic_attribute_calls_count", 0),
                "total_plans": len(plans),
            },
            "blocks": {bid: self._block_to_dict(b) for bid, b in functional_map.blocks.items()},
            "capabilities": {name: self._capability_to_dict(c) for name, c in functional_map.capabilities.items()},
            "clusters": [self._cluster_to_dict(c) for c in clusters],
            "consolidation_plans": [self._plan_to_dict(p) for p in plans],
            "call_graph": {
                "edges": functional_map.call_edges,
                "unresolved_calls": functional_map.unresolved_calls,
            },
        }

        with out.open("w", encoding="utf-8") as f:
            json.dump(
                report_data,
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=self._json_default,
            )

        self.logger.info(f"Generated JSON report: {output_file}")

    def generate_catalog_markdown(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        output_file: str,
    ) -> None:
        """Generate human-readable catalog in Markdown format."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Functional Decomposition Catalog",
            "",
            f"**Project:** {functional_map.project_root}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Analysis Timestamp:** {functional_map.timestamp}",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Blocks:** {functional_map.total_blocks}",
            f"- **Total Capabilities:** {functional_map.total_capabilities}",
            f"- **Total Clusters:** {functional_map.total_clusters}",
            f"- **Call Resolution Rate (Overall):** {functional_map.resolution_rate:.1%}",
            f"- **Call Resolution Rate (Internal):** {getattr(functional_map, 'resolution_rate_internal', 0.0):.1%}",
            f"- **Call Resolution Rate (Actionable):** {getattr(functional_map, 'resolution_rate_actionable', 0.0):.1%}",
            f"- **External Calls Count:** {getattr(functional_map, 'external_calls_count', 0)}",
            f"- **Dynamic Attribute Calls Count:** {getattr(functional_map, 'dynamic_attribute_calls_count', 0)}",
            "",
        ]

        lines.extend(self._generate_categories_section(functional_map))
        lines.extend(self._generate_capabilities_section(functional_map))
        lines.extend(self._generate_clusters_section(clusters, functional_map))
        lines.extend(self._generate_top_blocks_section(functional_map))

        out.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Generated catalog: {output_file}")

    def generate_plan_markdown(
        self,
        clusters: List[SimilarityCluster],
        plans: List[CanonicalizationPlan],
        output_file: str,
    ) -> None:
        """Generate consolidation plan in Markdown format."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [
            "# Consolidation Plan",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"Found **{len(clusters)}** similarity clusters and **{len(plans)}** consolidation plans.",
            "",
        ]

        lines.extend(self._generate_priority_matrix(plans))
        lines.extend(self._generate_detailed_plans(plans, clusters))

        out.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Generated consolidation plan: {output_file}")

    def generate_mermaid_diagram(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        output_file: str,
    ) -> None:
        """Generate Mermaid diagram showing functional relationships."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = ["graph TD", ""]

        cluster_colors = {
            RecommendationType.MERGE: "#ff6b6b",
            RecommendationType.EXTRACT_BASE: "#4ecdc4",
            RecommendationType.WRAP_ONLY: "#45b7d1",
            RecommendationType.KEEP_SEPARATE: "#96ceb4",
        }

        for cluster in clusters:
            color = cluster_colors.get(cluster.recommendation, "#ddd")
            safe_id = f"C_{cluster.id.replace('-', '_')}"
            label = f"{cluster.category}:{cluster.subcategory}<br/>{len(cluster.blocks)} blocks<br/>{cluster.recommendation.value}"
            # Escape quotes for Mermaid
            label = label.replace('"', '\\"')
            lines.append(f'    {safe_id}["{label}"]')
            lines.append(f"    style {safe_id} fill:{color}")

        lines.append("")
        lines.append("    %% Cluster relationships would go here")

        out.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Generated Mermaid diagram: {output_file}")

    def generate_summary_report(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        plans: List[CanonicalizationPlan],
        output_file: str,
    ) -> None:
        """Generate executive summary report."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)

        total_benefit = self._calculate_total_benefit(clusters, plans)

        internal_rate = getattr(functional_map, "resolution_rate_internal", functional_map.resolution_rate)
        actionable_rate = getattr(functional_map, "resolution_rate_actionable", internal_rate)
        external_count = getattr(functional_map, "external_calls_count", 0)
        dynamic_count = getattr(functional_map, "dynamic_attribute_calls_count", 0)

        lines: List[str] = [
            "# Functional Decomposition Summary",
            "",
            f"**Project:** {functional_map.project_root}",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Key Findings",
            "",
            f"- Analyzed **{functional_map.total_blocks}** functional blocks across the project",
            f"- Identified **{functional_map.total_capabilities}** distinct capabilities",
            f"- Found **{len(clusters)}** similarity clusters with consolidation opportunities",
            f"- Generated **{len(plans)}** actionable consolidation plans",
            f"- Call resolution rate: **{actionable_rate:.1%}** actionable, **{internal_rate:.1%}** internal, **{functional_map.resolution_rate:.1%}** overall",
            f"- External calls: **{external_count}** (libraries, builtins, etc.)",
            f"- Dynamic attribute calls: **{dynamic_count}** (method calls on unknown objects)",
            f"- Estimated benefit score: **{total_benefit:.1f}** (higher is better)",
            "",
            "## Recommendations Priority",
            "",
        ]

        high_priority_plans = [p for p in plans if p.risk_assessment == RiskLevel.LOW][:5]
        for i, plan in enumerate(high_priority_plans, 1):
            cluster = next((c for c in clusters if c.id == plan.cluster_id), None)
            if not cluster:
                continue
            lines.append(f"{i}. **{cluster.category}:{cluster.subcategory}** - {cluster.recommendation.value}")
            lines.append(f"   - {len(cluster.blocks)} blocks, {cluster.avg_similarity:.2f} similarity")
            lines.append(f"   - Risk: {plan.risk_assessment.value}, Effort: {plan.estimated_effort.value}")
            lines.append("")

        lines.extend(
            [
                "## Next Steps",
                "",
                "1. Review high-priority consolidation plans",
                "2. Start with low-risk, high-benefit opportunities",
                "3. Apply changes incrementally using patch-based approach",
                "4. Validate each step before proceeding",
                "",
                "See `consolidation_plan.md` for detailed implementation steps.",
            ]
        )

        out.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"Generated summary report: {output_file}")

    # --------------------------
    # Serialization helpers
    # --------------------------

    def _json_default(self, obj: Any) -> Any:
        """Make JSON dumping robust for enums/paths/dataclasses/sets/tuples."""
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return str(obj)

    def _block_to_dict(self, block: FunctionalBlock) -> Dict[str, Any]:
        return {
            "id": block.id,
            "module": block.module,
            "file_path": block.file_path,
            "qualname": block.qualname,
            "lineno": block.lineno,
            "end_lineno": block.end_lineno,
            "category": block.category,
            "subcategory": block.subcategory,
            "tags": block.tags,
            "signature": block.signature,
            "loc": block.loc,
            "cyclomatic": block.cyclomatic,

            # полезно для дебага резолва
            "raw_calls": list(block.raw_calls),
            "imports_used": list(block.imports_used),
            "imports_context": list(getattr(block, "imports_context", []) or []),
            "local_defs": list(getattr(block, "local_defs", []) or []),
            "local_assigned": list(getattr(block, "local_assigned", []) or []),
            "local_type_hints": dict(getattr(block, "local_type_hints", {}) or {}),

            # метрики
            "calls_count": len(block.calls),
            "called_by_count": len(block.called_by),
            "raw_calls_count": len(block.raw_calls),
            "imports_count": len(block.imports_used),
        }

    def _capability_to_dict(self, capability: Capability) -> Dict[str, Any]:
        return {
            "name": capability.name,
            "description": capability.description,
            "block_count": capability.block_count,
            "blocks": list(capability.blocks),
            "owners": list(capability.owners),
        }

    def _cluster_to_dict(self, cluster: SimilarityCluster) -> Dict[str, Any]:
        return {
            "id": cluster.id,
            "category": cluster.category,
            "subcategory": cluster.subcategory,
            "block_count": cluster.block_count,
            "blocks": list(cluster.blocks),
            "avg_similarity": cluster.avg_similarity,
            "recommendation": cluster.recommendation.value,
            "canonical_candidate": cluster.canonical_candidate,
            "proposed_target": cluster.proposed_target,
            "risk_level": cluster.risk_level.value,
            "effort_class": cluster.effort_class.value,
            "notes": list(cluster.notes),
        }

    def _plan_to_dict(self, plan: CanonicalizationPlan) -> Dict[str, Any]:
        return {
            "cluster_id": plan.cluster_id,
            "target_module": plan.target_module,
            "target_symbol": plan.target_symbol,
            "step_count": plan.step_count,
            "estimated_effort": plan.estimated_effort.value,
            "risk_assessment": plan.risk_assessment.value,
            "dependencies": list(plan.dependencies),
            "removal_criteria": plan.removal_criteria,
            "steps": [
                {
                    "id": step.id,
                    "kind": step.kind.value,
                    "description": step.description,
                    "files_touched": list(step.files_touched),
                    "preconditions": list(step.preconditions),
                    "validations": list(step.validations),
                    "target_module": step.target_module,
                    "target_symbol": step.target_symbol,
                    "source_blocks": list(step.source_blocks),
                }
                for step in plan.steps
            ],
        }

    # --------------------------
    # Markdown helpers
    # --------------------------

    def _generate_categories_section(self, functional_map: ProjectFunctionalMap) -> List[str]:
        category_counts: Dict[str, int] = {}
        for block in functional_map.blocks.values():
            key = f"{block.category}:{block.subcategory}"
            category_counts[key] = category_counts.get(key, 0) + 1

        lines = [
            "## Categories Breakdown",
            "",
            "| Category:Subcategory | Block Count |",
            "|---------------------|-------------|",
        ]
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {category} | {count} |")
        lines.append("")
        return lines

    def _generate_capabilities_section(self, functional_map: ProjectFunctionalMap) -> List[str]:
        lines = ["## Capabilities", ""]
        for capability in sorted(functional_map.capabilities.values(), key=lambda c: c.name):
            lines.extend(
                [
                    f"### {capability.name}",
                    "",
                    f"**Description:** {capability.description}",
                    f"**Blocks:** {capability.block_count}",
                    f"**Suggested Owners:** {', '.join(capability.owners)}",
                    "",
                ]
            )
        return lines

    def _generate_clusters_section(self, clusters: List[SimilarityCluster], functional_map: ProjectFunctionalMap) -> List[str]:
        lines = ["## Similarity Clusters", ""]
        for cluster in sorted(clusters, key=lambda c: c.avg_similarity, reverse=True):
            lines.extend(
                [
                    f"### {cluster.category}:{cluster.subcategory}",
                    "",
                    f"- **Blocks:** {cluster.block_count}",
                    f"- **Average Similarity:** {cluster.avg_similarity:.2f}",
                    f"- **Recommendation:** {cluster.recommendation.value}",
                    f"- **Risk Level:** {cluster.risk_level.value}",
                    f"- **Effort:** {cluster.effort_class.value}",
                    "",
                    "**Blocks in cluster:**",
                ]
            )
            for block_id in cluster.blocks:
                block = functional_map.blocks.get(block_id)
                if block:
                    lines.append(f"- `{block.qualname}` ({block.file_path}:{block.lineno})")
            lines.append("")
        return lines

    def _generate_top_blocks_section(self, functional_map: ProjectFunctionalMap) -> List[str]:
        blocks = list(functional_map.blocks.values())
        complex_blocks = sorted(blocks, key=lambda b: b.cyclomatic, reverse=True)[:10]
        popular_blocks = sorted(blocks, key=lambda b: len(b.called_by), reverse=True)[:10]

        lines = [
            "## Notable Blocks",
            "",
            "### Most Complex (by Cyclomatic Complexity)",
            "",
            "| Block | Complexity | LOC | File |",
            "|-------|------------|-----|------|",
        ]
        for block in complex_blocks:
            lines.append(f"| `{block.qualname}` | {block.cyclomatic} | {block.loc} | {block.file_path} |")

        lines.extend(
            [
                "",
                "### Most Popular (by Call Count)",
                "",
                "| Block | Callers | LOC | File |",
                "|-------|---------|-----|------|",
            ]
        )
        for block in popular_blocks:
            lines.append(f"| `{block.qualname}` | {len(block.called_by)} | {block.loc} | {block.file_path} |")

        lines.append("")
        return lines

    def _generate_priority_matrix(self, plans: List[CanonicalizationPlan]) -> List[str]:
        lines = [
            "## Priority Matrix",
            "",
            "| Risk \\ Effort | XS | S | M | L | XL |",
            "|---------------|----|----|----|----|-----|",
        ]

        matrix: Dict[tuple[RiskLevel, EffortClass], int] = {}
        for plan in plans:
            key = (plan.risk_assessment, plan.estimated_effort)
            matrix[key] = matrix.get(key, 0) + 1

        for risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
            row = [f"**{risk.value}**"]
            for effort in [EffortClass.XS, EffortClass.S, EffortClass.M, EffortClass.L, EffortClass.XL]:
                count = matrix.get((risk, effort), 0)
                row.append(str(count) if count > 0 else "-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        return lines

    def _generate_detailed_plans(self, plans: List[CanonicalizationPlan], clusters: List[SimilarityCluster]) -> List[str]:
        lines = ["## Detailed Consolidation Plans", ""]
        cluster_dict = {c.id: c for c in clusters}

        for i, plan in enumerate(plans[:10], 1):
            cluster = cluster_dict.get(plan.cluster_id)
            if not cluster:
                continue

            lines.extend(
                [
                    f"### Plan {i}: {cluster.category}:{cluster.subcategory}",
                    "",
                    f"**Target:** `{plan.target_module}::{plan.target_symbol}`",
                    f"**Risk:** {plan.risk_assessment.value} | **Effort:** {plan.estimated_effort.value}",
                    f"**Blocks:** {len(cluster.blocks)} | **Similarity:** {cluster.avg_similarity:.2f}",
                    "",
                    "**Steps:**",
                ]
            )

            for j, step in enumerate(plan.steps, 1):
                lines.append(f"{j}. **{step.kind.value}** - {step.description}")

            lines.append("")

        return lines

    def _calculate_total_benefit(self, clusters: List[SimilarityCluster], plans: List[CanonicalizationPlan]) -> float:
        benefit = 0.0
        for cluster in clusters:
            benefit += len(cluster.blocks) * 2
            benefit += cluster.avg_similarity * 10
            if cluster.recommendation == RecommendationType.MERGE:
                benefit += 5
            elif cluster.recommendation == RecommendationType.EXTRACT_BASE:
                benefit += 3
            elif cluster.recommendation == RecommendationType.WRAP_ONLY:
                benefit += 1
        return benefit