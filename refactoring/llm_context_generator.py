"""
LLM Context Generator for Complex Refactorings

This module generates rich context with code examples, evidence, and structured
prompts for complex refactoring operations that require LLM assistance.

Features:
- Rich context generation with code examples and evidence
- Structured prompts for different refactoring types
- Integration with existing knowledge management
- Context templates for various refactoring scenarios
- Evidence-based context with confidence scores
"""

import json
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from ..analysis.models import Evidence, FileReference
from ..analysis.refactoring_decision_engine import RefactoringDecision, RefactoringType
from ..knowledge.knowledge_manager import KnowledgeManager


logger = logging.getLogger(__name__)


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
    surrounding_context: str = ""  # Additional context around the target code
    dependencies: List[str] = field(default_factory=list)
    related_methods: List[str] = field(default_factory=list)
    test_coverage: Optional[float] = None
    
    def get_full_context(self) -> str:
        """Get full code context including surrounding code."""
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
            'template_type': self.template_type.value,
            'context_type': self.context_type.value,
            'system_prompt': self.system_prompt,
            'user_prompt': self.user_prompt,
            'context_data': self.context_data,
            'expected_output_format': self.expected_output_format,
            'confidence': self.confidence
        }


class LLMContextGenerator:
    """
    Generates rich context for LLM-assisted refactoring operations.
    
    Integrates with knowledge management to provide relevant examples,
    patterns, and best practices for complex refactoring scenarios.
    """
    
    def __init__(self, knowledge_manager: Optional[KnowledgeManager] = None):
        """Initialize context generator with optional knowledge manager."""
        self.knowledge_manager = knowledge_manager
        self.prompt_templates = self._load_prompt_templates()
        
        logger.info("LLMContextGenerator initialized")
    
    def generate_refactoring_context(self, decision: RefactoringDecision, 
                                   project_path: str) -> RefactoringContext:
        """
        Generate comprehensive context for a refactoring decision.
        
        Args:
            decision: Refactoring decision to generate context for
            project_path: Path to the project being refactored
            
        Returns:
            RefactoringContext with all necessary information
        """
        logger.info(f"Generating context for decision: {decision.decision_id}")
        
        # Extract target code contexts
        target_code = self._extract_target_code(decision, project_path)
        
        # Generate evidence summary
        evidence_summary = self._generate_evidence_summary(decision.evidence)
        
        # Get knowledge insights
        knowledge_insights = self._get_knowledge_insights(decision)
        
        # Find similar cases
        similar_cases = self._find_similar_cases(decision)
        
        # Generate constraints and success criteria
        constraints = self._generate_constraints(decision)
        success_criteria = self._generate_success_criteria(decision)
        
        context = RefactoringContext(
            decision=decision,
            target_code=target_code,
            evidence_summary=evidence_summary,
            knowledge_insights=knowledge_insights,
            similar_cases=similar_cases,
            constraints=constraints,
            success_criteria=success_criteria
        )
        
        logger.info(f"Generated context with {len(target_code)} code contexts")
        return context
    
    def generate_llm_prompt(self, context: RefactoringContext, 
                           context_type: ContextType = ContextType.REFACTORING_GUIDANCE) -> LLMPrompt:
        """
        Generate structured LLM prompt from refactoring context.
        
        Args:
            context: Refactoring context
            context_type: Type of context to generate
            
        Returns:
            LLMPrompt ready for LLM interaction
        """
        template_type = self._get_template_type(context.decision.refactoring_type)
        template = self.prompt_templates.get(template_type, self.prompt_templates[PromptTemplate.GENERAL_REFACTORING])
        
        # Generate system prompt
        system_prompt = self._generate_system_prompt(template, context, context_type)
        
        # Generate user prompt
        user_prompt = self._generate_user_prompt(template, context, context_type)
        
        # Prepare context data
        context_data = {
            'decision_id': context.decision.decision_id,
            'refactoring_type': context.decision.refactoring_type.value,
            'target_files': context.decision.target_files,
            'confidence': context.decision.confidence,
            'evidence_count': len(context.decision.evidence),
            'implementation_steps': len(context.decision.implementation_plan)
        }
        
        # Determine expected output format
        output_format = self._get_expected_output_format(context_type)
        
        prompt = LLMPrompt(
            template_type=template_type,
            context_type=context_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context_data=context_data,
            expected_output_format=output_format,
            confidence=context.decision.confidence
        )
        
        logger.info(f"Generated LLM prompt for {context_type.value}")
        return prompt
    
    def _extract_target_code(self, decision: RefactoringDecision, project_path: str) -> List[CodeContext]:
        """Extract code contexts for target files."""
        contexts = []
        
        for file_path in decision.target_files:
            try:
                full_path = Path(project_path) / file_path
                if not full_path.exists():
                    logger.warning(f"Target file not found: {file_path}")
                    continue
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract relevant code sections based on evidence
                for evidence in decision.evidence:
                    for file_ref in evidence.file_references:
                        if file_ref.file_path == file_path:
                            code_snippet = self._extract_code_snippet(content, file_ref)
                            surrounding_context = self._extract_surrounding_context(content, file_ref)
                            
                            context = CodeContext(
                                file_path=file_path,
                                line_start=file_ref.line_start,
                                line_end=file_ref.line_end,
                                code_snippet=code_snippet,
                                surrounding_context=surrounding_context,
                                dependencies=self._extract_dependencies(code_snippet),
                                related_methods=self._extract_related_methods(code_snippet)
                            )
                            contexts.append(context)
                
            except Exception as e:
                logger.error(f"Error extracting code from {file_path}: {e}")
        
        return contexts
    
    def _extract_code_snippet(self, content: str, file_ref: FileReference) -> str:
        """Extract code snippet from file content."""
        lines = content.splitlines()
        start_idx = max(0, file_ref.line_start - 1)
        end_idx = min(len(lines), file_ref.line_end)
        
        return '\n'.join(lines[start_idx:end_idx])
    
    def _extract_surrounding_context(self, content: str, file_ref: FileReference, 
                                   context_lines: int = 5) -> str:
        """Extract surrounding context around target code."""
        lines = content.splitlines()
        
        # Get context before
        before_start = max(0, file_ref.line_start - context_lines - 1)
        before_end = file_ref.line_start - 1
        before_lines = lines[before_start:before_end] if before_end > before_start else []
        
        # Get context after
        after_start = file_ref.line_end
        after_end = min(len(lines), file_ref.line_end + context_lines)
        after_lines = lines[after_start:after_end] if after_end > after_start else []
        
        context_parts = []
        if before_lines:
            context_parts.append("# BEFORE:\n" + '\n'.join(before_lines))
        if after_lines:
            context_parts.append("# AFTER:\n" + '\n'.join(after_lines))
        
        return '\n\n'.join(context_parts)
    
    def _extract_dependencies(self, code_snippet: str) -> List[str]:
        """Extract dependencies from code snippet."""
        dependencies = []
        
        # Simple regex-based extraction (could be enhanced with AST)
        import re
        
        # Import statements
        import_matches = re.findall(r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)', code_snippet)
        for module, items in import_matches:
            if module:
                dependencies.append(f"{module}.{items.strip()}")
            else:
                dependencies.append(items.strip())
        
        # Function calls
        call_matches = re.findall(r'(\w+(?:\.\w+)*)\s*\(', code_snippet)
        dependencies.extend(call_matches)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _extract_related_methods(self, code_snippet: str) -> List[str]:
        """Extract related method names from code snippet."""
        import re
        
        # Method definitions
        method_matches = re.findall(r'def\s+(\w+)\s*\(', code_snippet)
        
        # Method calls on self
        self_method_matches = re.findall(r'self\.(\w+)\s*\(', code_snippet)
        
        return list(set(method_matches + self_method_matches))
    
    def _generate_evidence_summary(self, evidence_list: List[Evidence]) -> str:
        """Generate summary of evidence supporting the refactoring decision."""
        if not evidence_list:
            return "No specific evidence provided."
        
        summary_parts = []
        
        for i, evidence in enumerate(evidence_list, 1):
            confidence_desc = "high" if evidence.confidence > 0.8 else "medium" if evidence.confidence > 0.5 else "low"
            
            summary_parts.append(
                f"{i}. {evidence.description} (confidence: {confidence_desc})\n"
                f"   - Affects {len(evidence.file_references)} file(s)\n"
                f"   - {len(evidence.code_snippets)} code example(s) available"
            )
        
        return "\n\n".join(summary_parts)
    
    def _get_knowledge_insights(self, decision: RefactoringDecision) -> List[str]:
        """Get relevant insights from knowledge manager."""
        if not self.knowledge_manager:
            return []
        
        try:
            # Query knowledge base for similar refactoring patterns
            query = f"{decision.refactoring_type.value} {' '.join(decision.target_symbols)}"
            insights = self.knowledge_manager.query_knowledge(query)
            
            return [insight.get('description', '') for insight in insights[:3]]  # Top 3 insights
        except Exception as e:
            logger.warning(f"Failed to get knowledge insights: {e}")
            return []
    
    def _find_similar_cases(self, decision: RefactoringDecision) -> List[Dict[str, Any]]:
        """Find similar refactoring cases from history."""
        # This would query decision history for similar cases
        # For now, return empty list
        return []
    
    def _generate_constraints(self, decision: RefactoringDecision) -> List[str]:
        """Generate constraints for the refactoring."""
        constraints = []
        
        if decision.feasibility:
            if decision.feasibility.breaking_changes_risk:
                constraints.append("Must maintain backward compatibility")
            
            if decision.feasibility.test_coverage_risk:
                constraints.append("Must preserve existing test coverage")
            
            if decision.feasibility.dependency_risk:
                constraints.append("Must not break external dependencies")
        
        # Add constraints based on refactoring type
        type_constraints = {
            RefactoringType.EXTRACT_METHOD: [
                "Extracted method must have clear single responsibility",
                "Method parameters should be minimal and cohesive"
            ],
            RefactoringType.EXTRACT_CLASS: [
                "New class must have high cohesion",
                "Interface should be minimal and focused"
            ],
            RefactoringType.DECOMPOSE_GOD_CLASS: [
                "Each resulting class must have single responsibility",
                "Dependencies between new classes should be minimal"
            ]
        }
        
        constraints.extend(type_constraints.get(decision.refactoring_type, []))
        
        return constraints
    
    def _generate_success_criteria(self, decision: RefactoringDecision) -> List[str]:
        """Generate success criteria for the refactoring."""
        criteria = [
            "All existing tests continue to pass",
            "Code compiles without errors",
            "No regression in functionality"
        ]
        
        # Add criteria based on impact assessments
        for assessment in decision.impact_assessments:
            if assessment.quantified_benefits:
                criteria.extend(assessment.quantified_benefits)
        
        # Add criteria based on refactoring type
        type_criteria = {
            RefactoringType.ELIMINATE_DUPLICATES: [
                "Code duplication is eliminated",
                "Maintainability is improved"
            ],
            RefactoringType.REDUCE_METHOD_COMPLEXITY: [
                "Cyclomatic complexity is reduced",
                "Method is easier to understand"
            ],
            RefactoringType.REMOVE_UNUSED_CODE: [
                "Unused code is completely removed",
                "No references to removed code remain"
            ]
        }
        
        criteria.extend(type_criteria.get(decision.refactoring_type, []))
        
        return criteria
    
    def _load_prompt_templates(self) -> Dict[PromptTemplate, Dict[str, str]]:
        """Load prompt templates for different refactoring types."""
        return {
            PromptTemplate.EXTRACT_METHOD: {
                'system': "You are an expert software engineer specializing in method extraction refactoring.",
                'user_base': "Please help me extract a method from the following code:",
                'guidance': "Focus on creating a cohesive method with clear responsibility.",
                'review': "Review the proposed method extraction for correctness and best practices.",
                'plan': "Create a detailed implementation plan for method extraction.",
                'risk': "Assess the risks associated with this method extraction.",
                'testing': "Suggest testing strategies for the extracted method."
            },
            PromptTemplate.EXTRACT_CLASS: {
                'system': "You are an expert software engineer specializing in class extraction refactoring.",
                'user_base': "Please help me extract a class from the following code:",
                'guidance': "Focus on creating a cohesive class with single responsibility.",
                'review': "Review the proposed class extraction for design quality.",
                'plan': "Create a detailed implementation plan for class extraction.",
                'risk': "Assess the risks associated with this class extraction.",
                'testing': "Suggest testing strategies for the extracted class."
            },
            PromptTemplate.DECOMPOSE_GOD_CLASS: {
                'system': "You are an expert software engineer specializing in God Class decomposition.",
                'user_base': "Please help me decompose this God Class:",
                'guidance': "Focus on identifying distinct responsibilities and creating focused classes.",
                'review': "Review the proposed decomposition strategy.",
                'plan': "Create a step-by-step decomposition plan.",
                'risk': "Assess the risks of decomposing this large class.",
                'testing': "Suggest testing strategies during decomposition."
            },
            PromptTemplate.ELIMINATE_DUPLICATES: {
                'system': "You are an expert software engineer specializing in duplicate code elimination.",
                'user_base': "Please help me eliminate the following code duplicates:",
                'guidance': "Focus on creating reusable components that eliminate duplication.",
                'review': "Review the proposed duplicate elimination strategy.",
                'plan': "Create an implementation plan for duplicate elimination.",
                'risk': "Assess the risks of eliminating these duplicates.",
                'testing': "Suggest testing strategies for duplicate elimination."
            },
            PromptTemplate.GENERAL_REFACTORING: {
                'system': "You are an expert software engineer specializing in code refactoring.",
                'user_base': "Please help me refactor the following code:",
                'guidance': "Focus on improving code quality while preserving functionality.",
                'review': "Review the proposed refactoring approach.",
                'plan': "Create a detailed refactoring implementation plan.",
                'risk': "Assess the risks associated with this refactoring.",
                'testing': "Suggest comprehensive testing strategies."
            }
        }
    
    def _get_template_type(self, refactoring_type: RefactoringType) -> PromptTemplate:
        """Map refactoring type to prompt template."""
        mapping = {
            RefactoringType.EXTRACT_METHOD: PromptTemplate.EXTRACT_METHOD,
            RefactoringType.EXTRACT_CLASS: PromptTemplate.EXTRACT_CLASS,
            RefactoringType.DECOMPOSE_GOD_CLASS: PromptTemplate.DECOMPOSE_GOD_CLASS,
            RefactoringType.ELIMINATE_DUPLICATES: PromptTemplate.ELIMINATE_DUPLICATES,
            RefactoringType.PARAMETERIZE_DUPLICATES: PromptTemplate.ELIMINATE_DUPLICATES,
            RefactoringType.TEMPLATE_METHOD_PATTERN: PromptTemplate.ELIMINATE_DUPLICATES,
            RefactoringType.REDUCE_METHOD_COMPLEXITY: PromptTemplate.EXTRACT_METHOD,
            RefactoringType.REMOVE_UNUSED_CODE: PromptTemplate.GENERAL_REFACTORING,
            RefactoringType.IMPROVE_COHESION: PromptTemplate.EXTRACT_CLASS,
            RefactoringType.REDUCE_COUPLING: PromptTemplate.EXTRACT_CLASS
        }
        return mapping.get(refactoring_type, PromptTemplate.GENERAL_REFACTORING)
    
    def _generate_system_prompt(self, template: Dict[str, str], context: RefactoringContext, 
                               context_type: ContextType) -> str:
        """Generate system prompt from template and context."""
        base_system = template['system']
        
        # Add context-specific instructions
        context_additions = {
            ContextType.REFACTORING_GUIDANCE: "Provide clear, actionable refactoring guidance.",
            ContextType.CODE_REVIEW: "Focus on code quality and best practices in your review.",
            ContextType.IMPLEMENTATION_PLAN: "Create detailed, step-by-step implementation plans.",
            ContextType.RISK_ASSESSMENT: "Thoroughly assess risks and provide mitigation strategies.",
            ContextType.TESTING_STRATEGY: "Focus on comprehensive testing approaches."
        }
        
        addition = context_additions.get(context_type, "")
        return f"{base_system} {addition}".strip()
    
    def _generate_user_prompt(self, template: Dict[str, str], context: RefactoringContext, 
                             context_type: ContextType) -> str:
        """Generate user prompt from template and context."""
        template_key = {
            ContextType.REFACTORING_GUIDANCE: 'guidance',
            ContextType.CODE_REVIEW: 'review',
            ContextType.IMPLEMENTATION_PLAN: 'plan',
            ContextType.RISK_ASSESSMENT: 'risk',
            ContextType.TESTING_STRATEGY: 'testing'
        }.get(context_type, 'guidance')
        
        base_prompt = template.get(template_key, template['user_base'])
        
        # Add context information
        context_info = []
        
        # Add decision information
        context_info.append(f"Refactoring Type: {context.decision.refactoring_type.value}")
        context_info.append(f"Priority: {context.decision.priority.value}")
        context_info.append(f"Confidence: {context.decision.confidence:.2f}")
        
        # Add evidence summary
        if context.evidence_summary:
            context_info.append(f"Evidence:\n{context.evidence_summary}")
        
        # Add target code
        if context.target_code:
            context_info.append("Target Code:")
            for i, code_ctx in enumerate(context.target_code, 1):
                context_info.append(f"File {i}: {code_ctx.file_path} (lines {code_ctx.line_start}-{code_ctx.line_end})")
                context_info.append(f"```python\n{code_ctx.get_full_context()}\n```")
        
        # Add constraints
        if context.constraints:
            context_info.append(f"Constraints:\n" + "\n".join(f"- {c}" for c in context.constraints))
        
        # Add success criteria
        if context.success_criteria:
            context_info.append(f"Success Criteria:\n" + "\n".join(f"- {c}" for c in context.success_criteria))
        
        # Add knowledge insights
        if context.knowledge_insights:
            context_info.append(f"Relevant Insights:\n" + "\n".join(f"- {i}" for i in context.knowledge_insights))
        
        return f"{base_prompt}\n\n" + "\n\n".join(context_info)
    
    def _get_expected_output_format(self, context_type: ContextType) -> str:
        """Get expected output format for context type."""
        formats = {
            ContextType.REFACTORING_GUIDANCE: "Provide step-by-step guidance with code examples",
            ContextType.CODE_REVIEW: "Provide structured review with specific recommendations",
            ContextType.IMPLEMENTATION_PLAN: "Provide numbered steps with time estimates and validation criteria",
            ContextType.RISK_ASSESSMENT: "Provide risk analysis with likelihood, impact, and mitigation strategies",
            ContextType.TESTING_STRATEGY: "Provide testing approach with specific test cases and coverage requirements"
        }
        return formats.get(context_type, "Provide clear, structured response")