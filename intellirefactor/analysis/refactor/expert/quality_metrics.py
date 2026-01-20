"""
Модуль расчета метрик качества анализа.

Все функции — чистые (pure functions), не зависят от состояния.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import ExpertAnalysisResult, RiskLevel


@dataclass(frozen=True)
class QualityWeights:
    """Конфигурация весов для расчета качества."""
    # Critical (60%)
    call_graph: float = 15.0
    external_callers: float = 15.0
    behavioral_contracts: float = 15.0
    dependency_interfaces: float = 15.0
    # Important (30%)
    cohesion_matrix: float = 10.0
    test_discovery: float = 10.0
    characterization_tests: float = 10.0
    # Enhancement (10%)
    duplicate_fragments: float = 5.0
    git_patterns: float = 3.0
    impact_assessment: float = 2.0


DEFAULT_WEIGHTS = QualityWeights()


@dataclass(frozen=True)
class RiskThresholds:
    """Пороги для определения уровня риска."""
    critical: int = 6
    high: int = 4
    medium: int = 2


DEFAULT_THRESHOLDS = RiskThresholds()


def calculate_quality_score(
    result: ExpertAnalysisResult,
    weights: QualityWeights = DEFAULT_WEIGHTS,
    max_score: float = 100.0
) -> float:
    """
    Расчет общего качества анализа (0-100).
    
    Args:
        result: Результат анализа
        weights: Веса компонентов
        max_score: Максимальный балл
        
    Returns:
        Оценка качества от 0 до max_score
    """
    score = 0.0
    
    # Mapping атрибутов результата к весам
    component_weights = {
        'call_graph': weights.call_graph,
        'external_callers': weights.external_callers,
        'behavioral_contracts': weights.behavioral_contracts,
        'dependency_interfaces': weights.dependency_interfaces,
        'cohesion_matrix': weights.cohesion_matrix,
        'test_discovery': weights.test_discovery,
        'characterization_tests': weights.characterization_tests,
        'duplicate_fragments': weights.duplicate_fragments,
        'git_patterns': weights.git_patterns,
        'impact_assessment': weights.impact_assessment,
    }
    
    for attr, weight in component_weights.items():
        value = getattr(result, attr, None)
        if value is not None:
            # Дополнительная проверка для коллекций
            if hasattr(value, '__len__') and len(value) == 0:
                continue
            score += weight
    
    return min(score, max_score)


def assess_overall_risk(
    result: ExpertAnalysisResult,
    thresholds: RiskThresholds = DEFAULT_THRESHOLDS
) -> RiskLevel:
    """
    Оценка общего уровня риска рефакторинга.
    
    Args:
        result: Результат анализа
        thresholds: Пороги уровней риска
        
    Returns:
        Уровень риска
    """
    risk_factors = 0
    
    # High risk factors (+3)
    if not result.call_graph:
        risk_factors += 3
    if not result.external_callers:
        risk_factors += 3
    
    # Medium-high risk factors (+2)
    if not result.test_discovery or not getattr(result.test_discovery, 'existing_test_files', None):
        risk_factors += 2
    if not result.behavioral_contracts:
        risk_factors += 2
    
    # Medium risk factors (+1)
    usage = result.usage_analysis
    if usage and getattr(usage, 'total_callers', 0) > 10:
        risk_factors += 1
    
    duplicates = result.duplicate_fragments
    if duplicates and len(duplicates) > 5:
        risk_factors += 1
    
    # Determine level
    if risk_factors >= thresholds.critical:
        return RiskLevel.CRITICAL
    elif risk_factors >= thresholds.high:
        return RiskLevel.HIGH
    elif risk_factors >= thresholds.medium:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def generate_recommendations(result: ExpertAnalysisResult) -> List[str]:
    """
    Генерация общих рекомендаций на основе результатов анализа.
    
    Args:
        result: Результат анализа
        
    Returns:
        Список рекомендаций
    """
    recommendations = []
    
    if result.analysis_quality_score < 50:
        recommendations.append(
            "Low analysis quality - consider manual review before refactoring"
        )
    
    if result.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        recommendations.append(
            "High risk refactoring - create comprehensive tests first"
        )
    
    external = result.external_callers
    if external and len(external) > 5:
        recommendations.append(
            "Module has many external dependencies - plan migration carefully"
        )
    
    call_graph = result.call_graph
    if call_graph and getattr(call_graph, 'cycles', None):
        recommendations.append(
            "Circular dependencies detected - resolve before major refactoring"
        )
    
    return recommendations