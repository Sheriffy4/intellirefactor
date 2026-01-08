"""
Advanced Refactoring Patterns for IntelliRefactor

Расширенные паттерны рефакторинга, извлеченные из спецификации attack-registry-refactoring
и других источников для автоматического применения в IntelliRefactor.
"""

import re
import ast
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class RefactoringComplexity(Enum):
    """Уровень сложности рефакторинга."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class RefactoringPattern:
    """Паттерн рефакторинга с метаданными."""

    name: str
    description: str
    complexity: RefactoringComplexity
    confidence: float
    before_pattern: str
    after_pattern: str
    conditions: List[str]
    method_patterns: List[str] = None
    size_thresholds: Dict[str, int] = None

    def matches_class(self, class_node: ast.ClassDef, class_size: int) -> bool:
        """Проверяет, подходит ли паттерн для данного класса."""
        if self.size_thresholds:
            min_size = self.size_thresholds.get("min_lines", 0)
            max_size = self.size_thresholds.get("max_lines", float("inf"))
            if not (min_size <= class_size <= max_size):
                return False

        if self.method_patterns:
            methods = [
                n.name for n in class_node.body if isinstance(n, ast.FunctionDef)
            ]
            pattern_matches = 0
            for pattern in self.method_patterns:
                pattern_matches += sum(
                    1 for method in methods if re.match(pattern, method)
                )

            min_matches = (
                self.size_thresholds.get("min_pattern_matches", 1)
                if self.size_thresholds
                else 1
            )
            if pattern_matches < min_matches:
                return False

        return True


class AdvancedRefactoringPatterns:
    """Коллекция расширенных паттернов рефакторинга."""

    # Паттерны из attack-registry-refactoring
    MONOLITH_TO_COMPOSITION = RefactoringPattern(
        name="monolith_to_composition",
        description="Разделение монолитного класса на специализированные компоненты с композицией",
        complexity=RefactoringComplexity.ADVANCED,
        confidence=0.95,
        before_pattern="Один большой класс (>1000 строк) с множественными ответственностями",
        after_pattern="Главный класс + набор специализированных компонентов через композицию",
        conditions=[
            "Класс превышает 1000 строк кода",
            "Класс имеет множественные ответственности",
            "Можно выделить четкие функциональные области",
        ],
        size_thresholds={
            "min_lines": 1000,
            "max_component_size": 300,
            "min_methods": 20,
        },
    )

    FACTORY_EXTRACTION = RefactoringPattern(
        name="factory_extraction",
        description="Извлечение методов создания объектов в отдельную фабрику",
        complexity=RefactoringComplexity.MODERATE,
        confidence=0.90,
        before_pattern="Множество методов _create_*_handler() в основном классе",
        after_pattern="Отдельный Factory класс с методами создания",
        conditions=[
            "Есть множество методов создания с общим префиксом",
            "Методы создания логически связаны",
            "Можно выделить общий интерфейс создания",
        ],
        method_patterns=[
            r"_create_.*",
            r"_build_.*",
            r"_make_.*",
            r"create_.*",
            r"build_.*",
        ],
        size_thresholds={"min_pattern_matches": 3, "min_lines": 200},
    )

    STATE_MANAGER_EXTRACTION = RefactoringPattern(
        name="state_manager_extraction",
        description="Извлечение управления состоянием в отдельный менеджер",
        complexity=RefactoringComplexity.MODERATE,
        confidence=0.85,
        before_pattern="Состояние и логика управления смешаны в основном классе",
        after_pattern="Отдельные менеджеры для разных аспектов состояния",
        conditions=[
            "Есть сложная логика управления состоянием",
            "Состояние можно разделить на логические группы",
            "Нужна изоляция для тестирования",
        ],
        method_patterns=[
            r"manage_.*",
            r"handle_.*",
            r"_handle_.*",
            r"process_.*",
            r"control_.*",
        ],
        size_thresholds={"min_pattern_matches": 2, "min_lines": 300},
    )

    VALIDATOR_EXTRACTION = RefactoringPattern(
        name="validator_extraction",
        description="Извлечение логики валидации в отдельный компонент",
        complexity=RefactoringComplexity.SIMPLE,
        confidence=0.88,
        before_pattern="Методы валидации разбросаны по основному классу",
        after_pattern="Централизованный Validator компонент",
        conditions=[
            "Есть методы validate_* или _validate_*",
            "Валидация имеет сложную логику",
            "Нужна переиспользуемость валидации",
        ],
        method_patterns=[
            r"validate_.*",
            r"_validate_.*",
            r"check_.*",
            r"verify_.*",
            r"ensure_.*",
        ],
        size_thresholds={"min_pattern_matches": 2, "min_lines": 150},
    )

    CONFIGURATION_LAYER = RefactoringPattern(
        name="configuration_layer",
        description="Создание централизованного конфигурационного слоя",
        complexity=RefactoringComplexity.SIMPLE,
        confidence=0.82,
        before_pattern="Настройки разбросаны по коду как константы и переменные",
        after_pattern="Централизованный Config с типизированными настройками",
        conditions=[
            "Много магических констант в коде",
            "Настройки дублируются в разных местах",
            "Нужна централизованная конфигурация",
        ],
        method_patterns=[r".*config.*", r".*setting.*", r".*option.*"],
        size_thresholds={"min_lines": 100},
    )

    BACKWARD_COMPATIBILITY_DELEGATION = RefactoringPattern(
        name="backward_compatibility_delegation",
        description="Сохранение обратной совместимости через делегирование",
        complexity=RefactoringComplexity.MODERATE,
        confidence=0.92,
        before_pattern="Прямая реализация методов в монолитном классе",
        after_pattern="Методы-делегаты, которые вызывают соответствующие компоненты",
        conditions=[
            "Нужно сохранить существующий API",
            "Есть внешние зависимости от класса",
            "Рефакторинг должен быть прозрачным",
        ],
        size_thresholds={"min_lines": 500},
    )

    @classmethod
    def get_all_patterns(cls) -> List[RefactoringPattern]:
        """Возвращает все доступные паттерны."""
        return [
            cls.MONOLITH_TO_COMPOSITION,
            cls.FACTORY_EXTRACTION,
            cls.STATE_MANAGER_EXTRACTION,
            cls.VALIDATOR_EXTRACTION,
            cls.CONFIGURATION_LAYER,
            cls.BACKWARD_COMPATIBILITY_DELEGATION,
        ]

    @classmethod
    def get_applicable_patterns(
        cls, class_node: ast.ClassDef, class_size: int, min_confidence: float = 0.80
    ) -> List[RefactoringPattern]:
        """Возвращает применимые паттерны для данного класса."""
        applicable = []

        for pattern in cls.get_all_patterns():
            if pattern.confidence >= min_confidence and pattern.matches_class(
                class_node, class_size
            ):
                applicable.append(pattern)

        # Сортируем по уверенности (убывание)
        applicable.sort(key=lambda p: p.confidence, reverse=True)
        return applicable

    @classmethod
    def suggest_refactoring_strategy(
        cls, class_node: ast.ClassDef, class_size: int
    ) -> Dict[str, any]:
        """Предлагает стратегию рефакторинга для класса."""

        applicable_patterns = cls.get_applicable_patterns(class_node, class_size)

        if not applicable_patterns:
            return {
                "strategy": "no_refactoring_needed",
                "reason": "No applicable patterns found",
                "patterns": [],
            }

        # Определяем основную стратегию
        primary_pattern = applicable_patterns[0]

        if primary_pattern.name == "monolith_to_composition":
            strategy = "comprehensive_decomposition"
        elif primary_pattern.complexity == RefactoringComplexity.ADVANCED:
            strategy = "advanced_refactoring"
        elif len(applicable_patterns) > 2:
            strategy = "multi_pattern_refactoring"
        else:
            strategy = "focused_refactoring"

        return {
            "strategy": strategy,
            "primary_pattern": primary_pattern.name,
            "confidence": primary_pattern.confidence,
            "patterns": [p.name for p in applicable_patterns],
            "complexity": primary_pattern.complexity.value,
            "estimated_components": len(applicable_patterns) + 1,
        }


class ComponentExtractionRules:
    """Правила для извлечения компонентов."""

    RESPONSIBILITY_KEYWORDS = {
        "config": ["config", "setting", "option", "parameter", "preference"],
        "validation": ["validate", "check", "verify", "ensure", "confirm"],
        "factory": ["create", "build", "make", "generate", "construct"],
        "manager": ["manage", "handle", "process", "control", "coordinate"],
        "loader": ["load", "import", "discover", "fetch", "retrieve"],
        "registry": ["register", "unregister", "lookup", "find", "search"],
        "cache": ["cache", "store", "retrieve", "invalidate", "refresh"],
        "parser": ["parse", "decode", "encode", "serialize", "deserialize"],
        "formatter": ["format", "render", "display", "print", "show"],
        "analyzer": ["analyze", "examine", "inspect", "scan", "evaluate"],
    }

    @classmethod
    def categorize_methods(cls, methods: List[str]) -> Dict[str, List[str]]:
        """Категоризирует методы по ответственностям."""
        categories = {category: [] for category in cls.RESPONSIBILITY_KEYWORDS}
        categories["other"] = []

        for method in methods:
            method_lower = method.lower()
            categorized = False

            for category, keywords in cls.RESPONSIBILITY_KEYWORDS.items():
                if any(keyword in method_lower for keyword in keywords):
                    categories[category].append(method)
                    categorized = True
                    break

            if not categorized:
                categories["other"].append(method)

        # Удаляем пустые категории
        return {k: v for k, v in categories.items() if v}

    @classmethod
    def suggest_component_names(cls, category: str, methods: List[str]) -> str:
        """Предлагает имя компонента на основе категории и методов."""

        base_names = {
            "config": "Config",
            "validation": "Validator",
            "factory": "Factory",
            "manager": "Manager",
            "loader": "Loader",
            "registry": "Registry",
            "cache": "Cache",
            "parser": "Parser",
            "formatter": "Formatter",
            "analyzer": "Analyzer",
        }

        if category in base_names:
            return base_names[category]

        # Для "other" пытаемся найти общий паттерн
        if category == "other" and methods:
            # Ищем общие префиксы
            common_prefixes = []
            for method in methods:
                parts = method.split("_")
                if len(parts) > 1:
                    common_prefixes.append(parts[0])

            if common_prefixes:
                most_common = max(set(common_prefixes), key=common_prefixes.count)
                return f"{most_common.title()}Service"

        return "Service"


# Экспорт основных классов
__all__ = [
    "AdvancedRefactoringPatterns",
    "ComponentExtractionRules",
    "RefactoringPattern",
    "RefactoringComplexity",
]
