"""
Pipeline выполнения анализа с единообразной обработкой ошибок.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .safe_access import SafeAnalyzerRegistry

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Приоритет выполнения анализа."""
    P0_CRITICAL = 0     # Обязательные для безопасного рефакторинга
    P1_IMPORTANT = 1    # Важные для качественного рефакторинга
    P2_ENHANCEMENT = 2  # Желательные улучшения


@dataclass
class AnalysisStep:
    """Конфигурация одного шага анализа."""
    name: str
    analyzer_key: str
    method_name: str
    result_attr: str
    failure_message: str
    priority: Priority
    requires_ast: bool = True
    log_on_fail: str = "error"  # "error", "warning", or "debug"
    
    # Дополнительные аргументы для метода
    extra_args_factory: Optional[Callable[[Dict[str, Any]], Tuple]] = None


# Конфигурация всех шагов анализа
ANALYSIS_STEPS: List[AnalysisStep] = [
    # ===== P0: Critical =====
    AnalysisStep(
        name="Call graph",
        analyzer_key="call_graph",
        method_name="analyze_internal_calls",
        result_attr="call_graph",
        failure_message="manual dependency review required",
        priority=Priority.P0_CRITICAL,
    ),
    AnalysisStep(
        name="External callers",
        analyzer_key="caller",
        method_name="find_external_callers",
        result_attr="external_callers",
        failure_message="check dependencies manually",
        priority=Priority.P0_CRITICAL,
        requires_ast=False,
        extra_args_factory=lambda ctx: (ctx['target_module_str'],),
    ),
    AnalysisStep(
        name="Behavioral contracts",
        analyzer_key="contract",
        method_name="extract_contracts_from_docstrings",
        result_attr="behavioral_contracts",
        failure_message="review method contracts manually",
        priority=Priority.P0_CRITICAL,
    ),
    AnalysisStep(
        name="Dependency interfaces",
        analyzer_key="dependency",
        method_name="extract_dependency_interfaces",
        result_attr="dependency_interfaces",
        failure_message="review imports manually",
        priority=Priority.P0_CRITICAL,
    ),
    
    # ===== P1: Important =====
    AnalysisStep(
        name="Test discovery",
        analyzer_key="test",
        method_name="find_existing_tests",
        result_attr="test_discovery",
        failure_message="verify test coverage manually",
        priority=Priority.P1_IMPORTANT,
        requires_ast=False,
    ),
    AnalysisStep(
        name="Characterization tests",
        analyzer_key="characterization",
        method_name="generate_characterization_tests",
        result_attr="characterization_tests",
        failure_message="create tests manually before refactoring",
        priority=Priority.P1_IMPORTANT,
    ),
    AnalysisStep(
        name="Duplicate fragments",
        analyzer_key="duplicate",
        method_name="find_concrete_duplicates",
        result_attr="duplicate_fragments",
        failure_message="review code manually for duplicates",
        priority=Priority.P1_IMPORTANT,
        requires_ast=False,
    ),
    
    # ===== P2: Enhancement =====
    AnalysisStep(
        name="Git history",
        analyzer_key="git",
        method_name="analyze_change_patterns",
        result_attr="git_patterns",
        failure_message="",  # Не критично
        priority=Priority.P2_ENHANCEMENT,
        requires_ast=False,
        log_on_fail="warning",
    ),
    AnalysisStep(
        name="Impact assessment",
        analyzer_key="compatibility",
        method_name="assess_breaking_change_impact",
        result_attr="impact_assessment",
        failure_message="assess breaking changes manually",
        priority=Priority.P2_ENHANCEMENT,
        requires_ast=False,
        extra_args_factory=lambda ctx: ([],),  # Пустой список изменений
    ),
    AnalysisStep(
        name="Compatibility constraints",
        analyzer_key="compatibility",
        method_name="determine_compatibility_constraints",
        result_attr="compatibility_constraints",
        failure_message="",
        priority=Priority.P2_ENHANCEMENT,
        requires_ast=False,
        log_on_fail="warning",
    ),
]


@dataclass
class StepResult:
    """Результат выполнения одного шага."""
    step: AnalysisStep
    success: bool
    value: Any = None
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": {
                "name": self.step.name,
                "analyzer_key": self.step.analyzer_key,
                "method_name": self.step.method_name,
                "result_attr": self.step.result_attr,
                "priority": int(self.step.priority),
            },
            "success": self.success,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class PipelineResult:
    """Результат выполнения pipeline."""
    results: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    step_results: List[StepResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Процент успешных шагов."""
        if not self.step_results:
            return 0.0
        successful = sum(1 for r in self.step_results if r.success)
        return successful / len(self.step_results) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results_keys": sorted(self.results.keys()),
            "failures": list(self.failures),
            "success_rate": self.success_rate,
            "step_results": [sr.to_dict() for sr in self.step_results],
        }


class AnalysisPipelineExecutor:
    """
    Исполнитель pipeline анализа.
    
    Преимущества:
    - Единообразная обработка ошибок
    - Конфигурируемые шаги
    - Отслеживание прогресса
    - Поддержка приоритетов
    """
    
    def __init__(
        self,
        analyzers: SafeAnalyzerRegistry,
        context: Dict[str, Any]
    ):
        """
        Args:
            analyzers: Реестр анализаторов
            context: Контекст с дополнительными данными (target_module_str и т.д.)
        """
        self.analyzers = analyzers
        self.context = context
    
    def execute_step(
        self,
        step: AnalysisStep,
        module_ast: Any
    ) -> StepResult:
        """Выполнить один шаг анализа."""
        analyzer = self.analyzers.get(step.analyzer_key)
        if analyzer is None:
            log_method = getattr(logger, step.log_on_fail, logger.error)
            log_method(f"{step.name} analysis failed: analyzer '{step.analyzer_key}' not found")
            return StepResult(
                step=step,
                success=False,
                error=KeyError(f"Analyzer '{step.analyzer_key}' not found")
            )
        
        method = getattr(analyzer, step.method_name, None)
        if method is None:
            log_method = getattr(logger, step.log_on_fail, logger.error)
            log_method(f"{step.name} analysis failed: method '{step.method_name}' not found")
            return StepResult(
                step=step,
                success=False,
                error=AttributeError(f"Method '{step.method_name}' not found")
            )
        
        try:
            # Подготовка аргументов
            args = []
            if step.requires_ast:
                args.append(module_ast)
            if step.extra_args_factory:
                args.extend(step.extra_args_factory(self.context))
            
            # Выполнение
            result = method(*args)
            
            logger.info(f"✓ {step.name} analysis completed")
            return StepResult(step=step, success=True, value=result)
            
        except Exception as e:
            # Логирование с нужным уровнем
            log_method = getattr(logger, step.log_on_fail, logger.error)
            log_method(f"{step.name} analysis failed: {e}")
            
            return StepResult(step=step, success=False, error=e)
    
    def execute_pipeline(
        self,
        module_ast: Any,
        steps: Optional[List[AnalysisStep]] = None,
        max_priority: Priority = Priority.P2_ENHANCEMENT,
        stop_on_critical_failure: bool = False
    ) -> PipelineResult:
        """
        Выполнить весь pipeline.
        
        Args:
            module_ast: AST модуля
            steps: Шаги для выполнения (по умолчанию ANALYSIS_STEPS)
            max_priority: Максимальный приоритет для выполнения
            stop_on_critical_failure: Остановить при ошибке P0 шага
            
        Returns:
            Результат выполнения pipeline
        """
        steps = steps or ANALYSIS_STEPS
        result = PipelineResult()
        
        # Группировка по приоритетам
        for priority in Priority:
            if priority > max_priority:
                break
            
            priority_steps = [s for s in steps if s.priority == priority]
            if not priority_steps:
                continue
            
            logger.info(f"Phase: {priority.name} ({len(priority_steps)} steps)")
            
            for step in priority_steps:
                step_result = self.execute_step(step, module_ast)
                result.step_results.append(step_result)
                
                if step_result.success:
                    result.results[step.result_attr] = step_result.value
                else:
                    if step.failure_message:
                        result.failures.append(
                            f"{step.name} analysis failed - {step.failure_message}"
                        )
                    
                    # Остановка при критической ошибке
                    if stop_on_critical_failure and priority == Priority.P0_CRITICAL:
                        logger.error(f"Critical step '{step.name}' failed, stopping pipeline")
                        return result
        
        return result


# ===== Export steps for detailed analysis =====

EXPORT_STEPS: List[AnalysisStep] = [
    AnalysisStep(
        name="Exception contracts",
        analyzer_key="exception_contract",
        method_name="analyze_exception_contracts",
        result_attr="exception_contracts",
        failure_message="",
        priority=Priority.P1_IMPORTANT,
    ),
    AnalysisStep(
        name="Data schemas",
        analyzer_key="data_schema",
        method_name="analyze_data_schemas",
        result_attr="data_schemas",
        failure_message="",
        priority=Priority.P1_IMPORTANT,
    ),
    AnalysisStep(
        name="Import dependencies",
        analyzer_key="dependency",
        method_name="analyze_import_dependencies",
        result_attr="import_dependencies_raw",
        failure_message="",
        priority=Priority.P1_IMPORTANT,
    ),
    AnalysisStep(
        name="External dependency contracts",
        analyzer_key="dependency",
        method_name="extract_external_dependency_contracts",
        result_attr="external_dependency_contracts",
        failure_message="",
        priority=Priority.P1_IMPORTANT,
        log_on_fail="warning",
    ),
    AnalysisStep(
        name="Optional dependencies",
        analyzer_key="optional_dependencies",
        method_name="analyze_optional_dependencies",
        result_attr="optional_dependencies_raw",
        failure_message="",
        priority=Priority.P2_ENHANCEMENT,
    ),
    AnalysisStep(
        name="Golden traces",
        analyzer_key="golden_traces",
        method_name="extract_golden_traces",
        result_attr="golden_traces_raw",
        failure_message="",
        priority=Priority.P2_ENHANCEMENT,
    ),
]