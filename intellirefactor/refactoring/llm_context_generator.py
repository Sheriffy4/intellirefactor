"""
LLM Context Generator for Complex Refactorings.

Generates structured context and prompts for LLM-assisted refactoring:
- Collects code snippets with surrounding context
- Summarizes evidence
- Adds constraints, success criteria, and optional knowledge insights
- Produces a structured prompt payload
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Robust imports (package execution vs. direct run/test)
try:
    from ..analysis.models import Evidence, FileReference
    from ..analysis.refactoring_decision_engine import RefactoringDecision, RefactoringType
    from ..knowledge.knowledge_manager import KnowledgeManager
except Exception:  # pragma: no cover
    Evidence = Any  # type: ignore
    FileReference = Any  # type: ignore
    RefactoringDecision = Any  # type: ignore
    RefactoringType = Any  # type: ignore
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


class LLMContextGenerator:
    """
    Generates rich context for LLM-assisted refactoring operations.

    Integrates with optional knowledge management to provide relevant examples,
    patterns, and best practices for complex refactoring scenarios.
    """

    def __init__(self, knowledge_manager: Optional[KnowledgeManager] = None) -> None:
        self.knowledge_manager = knowledge_manager
        self.prompt_templates = self._load_prompt_templates()
        logger.info("LLMContextGenerator initialized")

    def generate_refactoring_context(self, decision: RefactoringDecision, project_path: str) -> RefactoringContext:
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
        template = self.prompt_templates.get(template_type, self.prompt_templates[PromptTemplate.GENERAL_REFACTORING])

        system_prompt = self._generate_system_prompt(template, context, context_type)
        user_prompt = self._generate_user_prompt(template, context, context_type)

        decision = context.decision
        ref_type = getattr(getattr(decision, "refactoring_type", None), "value", str(getattr(decision, "refactoring_type", "")))
        priority = getattr(getattr(decision, "priority", None), "value", str(getattr(decision, "priority", "")))
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

    def _extract_target_code(self, decision: RefactoringDecision, project_path: str) -> List[CodeContext]:
        """Extract code contexts for target files from evidence file references."""
        contexts: List[CodeContext] = []
        seen: Set[Tuple[str, int, int]] = set()

        target_files = getattr(decision, "target_files", []) or []
        evidence_list = getattr(decision, "evidence", []) or []

        for file_path in target_files:
            full_path = Path(project_path) / file_path
            if not full_path.exists():
                logger.warning("Target file not found: %s", file_path)
                continue

            try:
                content = full_path.read_text(encoding="utf-8-sig")
            except Exception as e:
                logger.error("Error reading %s: %s", file_path, e)
                continue

            for evidence in evidence_list:
                file_refs = getattr(evidence, "file_references", []) or []
                for file_ref in file_refs:
                    if getattr(file_ref, "file_path", None) != file_path:
                        continue

                    line_start = int(getattr(file_ref, "line_start", 1) or 1)
                    line_end = int(getattr(file_ref, "line_end", line_start) or line_start)

                    key = (file_path, line_start, line_end)
                    if key in seen:
                        continue
                    seen.add(key)

                    code_snippet = self._extract_code_snippet(content, file_ref)
                    surrounding_context = self._extract_surrounding_context(content, file_ref)

                    contexts.append(
                        CodeContext(
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                            code_snippet=code_snippet,
                            surrounding_context=surrounding_context,
                            dependencies=self._extract_dependencies(code_snippet),
                            related_methods=self._extract_related_methods(code_snippet),
                        )
                    )

        return contexts

    def _extract_code_snippet(self, content: str, file_ref: FileReference) -> str:
        """Extract code snippet from file content using file reference line numbers."""
        lines = content.splitlines()
        start_idx = max(0, int(getattr(file_ref, "line_start", 1) or 1) - 1)
        end_idx = min(len(lines), int(getattr(file_ref, "line_end", start_idx + 1) or (start_idx + 1)))
        return "\n".join(lines[start_idx:end_idx])

    def _extract_surrounding_context(self, content: str, file_ref: FileReference, context_lines: int = 5) -> str:
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
            tree = compile(code_snippet, "<snippet>", "exec", dont_inherit=True, optimize=0, ast.PyCF_ONLY_AST)  # type: ignore[arg-type]
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
            conf = float(getattr(evidence, "confidence", 0.0) or 0.0)
            confidence_desc = "high" if conf > 0.8 else "medium" if conf > 0.5 else "low"

            file_refs = getattr(evidence, "file_references", []) or []
            code_snips = getattr(evidence, "code_snippets", []) or []
            desc = getattr(evidence, "description", "<no description>")

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
            ref_type = getattr(getattr(decision, "refactoring_type", None), "value", str(getattr(decision, "refactoring_type", "")))
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
            "ELIMINATE_DUPLICATES": ["Code duplication is eliminated", "Maintainability is improved"],
            "REDUCE_METHOD_COMPLEXITY": ["Cyclomatic complexity is reduced", "Method is easier to understand"],
            "REMOVE_UNUSED_CODE": ["Unused code is completely removed", "No references to removed code remain"],
        }

        criteria.extend(type_criteria_by_name.get(rt_name, []))
        return criteria

    def _load_prompt_templates(self) -> Dict[PromptTemplate, Dict[str, str]]:
        """Load prompt templates for different refactoring types."""
        return {
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

    def _generate_system_prompt(self, template: Dict[str, str], context: RefactoringContext, context_type: ContextType) -> str:
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

    def _generate_user_prompt(self, template: Dict[str, str], context: RefactoringContext, context_type: ContextType) -> str:
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
        ref_type = getattr(getattr(decision, "refactoring_type", None), "value", str(getattr(decision, "refactoring_type", "")))
        priority = getattr(getattr(decision, "priority", None), "value", str(getattr(decision, "priority", "")))
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
                context_info.append(f"File {i}: {code_ctx.file_path} (lines {code_ctx.line_start}-{code_ctx.line_end})")
                context_info.append(f"```python\n{code_ctx.get_full_context()}\n```")

        if context.constraints:
            context_info.append("Constraints:\n" + "\n".join(f"- {c}" for c in context.constraints))

        if context.success_criteria:
            context_info.append("Success Criteria:\n" + "\n".join(f"- {c}" for c in context.success_criteria))

        if context.knowledge_insights:
            context_info.append("Relevant Insights:\n" + "\n".join(f"- {i}" for i in context.knowledge_insights))

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