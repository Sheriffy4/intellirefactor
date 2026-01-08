"""
Manual Refactoring Analyzer

Анализирует ручной рефакторинг для извлечения паттернов и алгоритмов,
которые можно автоматизировать в IntelliRefactor.
"""

import ast
import re
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RefactoringPattern:
    """Паттерн рефакторинга, извлеченный из ручного анализа."""

    name: str
    description: str
    before_pattern: str
    after_pattern: str
    conditions: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ModuleAnalysis:
    """Результат анализа модуля."""

    imports: List[str]
    classes: List[str]
    functions: List[str]
    logging_pattern: str
    complexity_metrics: Dict[str, int]
    dependencies: Set[str]


class ManualRefactoringAnalyzer:
    """Анализирует ручной рефакторинг для извлечения автоматизируемых паттернов."""

    def __init__(self):
        self.patterns: List[RefactoringPattern] = []
        self.module_analyses: Dict[str, ModuleAnalysis] = {}

    def analyze_original_module(self, file_path: str) -> ModuleAnalysis:
        """Анализирует оригинальный модуль перед рефакторингом."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return ModuleAnalysis([], [], [], "", {}, set())

        analysis = ModuleAnalysis(
            imports=self._extract_imports(tree),
            classes=self._extract_classes(tree),
            functions=self._extract_functions(tree),
            logging_pattern=self._detect_logging_pattern(content),
            complexity_metrics=self._calculate_complexity(tree),
            dependencies=self._extract_dependencies(content),
        )

        self.module_analyses[file_path] = analysis
        return analysis

    def analyze_refactored_components(self, components_dir: str) -> Dict[str, ModuleAnalysis]:
        """Анализирует рефакторированные компоненты."""
        components_path = Path(components_dir)
        analyses = {}

        for py_file in components_path.glob("*.py"):
            if py_file.name != "__init__.py":
                analyses[str(py_file)] = self.analyze_original_module(str(py_file))

        return analyses

    def extract_refactoring_patterns(
        self, original_path: str, refactored_dir: str
    ) -> List[RefactoringPattern]:
        """Извлекает паттерны рефакторинга из сравнения оригинала и результата."""
        original_analysis = self.analyze_original_module(original_path)
        refactored_analyses = self.analyze_refactored_components(refactored_dir)

        patterns = []

        # Паттерн 1: Извлечение сервисов по семантическим группам
        patterns.append(
            self._extract_service_extraction_pattern(original_analysis, refactored_analyses)
        )

        # Паттерн 2: Исправление импортов
        patterns.append(self._extract_import_fixing_pattern(original_analysis, refactored_analyses))

        # Паттерн 3: Стандартизация логирования
        patterns.append(
            self._extract_logging_standardization_pattern(original_analysis, refactored_analyses)
        )

        # Паттерн 4: Owner Proxy pattern
        patterns.append(self._extract_owner_proxy_pattern(refactored_analyses))

        self.patterns.extend(patterns)
        return patterns

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Извлекает все импорты из AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        return imports

    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """Извлекает все классы из AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes

    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Извлекает все функции из AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions

    def _detect_logging_pattern(self, content: str) -> str:
        """Определяет паттерн логирования в модуле."""
        if re.search(r"LOG = logging\.getLogger", content):
            return "LOG"
        elif re.search(r"logger = logging\.getLogger", content):
            return "logger"
        elif re.search(r"self\.logger = logging\.getLogger", content):
            return "self.logger"
        return "none"

    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, int]:
        """Вычисляет метрики сложности."""
        metrics = {
            "lines": 0,
            "classes": 0,
            "functions": 0,
            "imports": 0,
            "complexity": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
            elif isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["imports"] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                metrics["complexity"] += 1

        return metrics

    def _extract_dependencies(self, content: str) -> Set[str]:
        """Извлекает зависимости модуля."""
        dependencies = set()

        # Ищем импорты из core.*
        core_imports = re.findall(r"from (core\.[^\s]+)", content)
        dependencies.update(core_imports)

        return dependencies

    def _extract_service_extraction_pattern(
        self, original: ModuleAnalysis, refactored: Dict[str, ModuleAnalysis]
    ) -> RefactoringPattern:
        """Извлекает паттерн выделения сервисов."""
        return RefactoringPattern(
            name="service_extraction",
            description="Extract semantic groups of methods into separate service classes",
            before_pattern="Large monolithic class with mixed responsibilities",
            after_pattern="Multiple focused service classes with single responsibility",
            conditions=[
                "Class has more than 50 methods",
                "Methods can be grouped by semantic similarity",
                "Class has multiple distinct responsibilities",
            ],
            confidence=0.9,
        )

    def _extract_import_fixing_pattern(
        self, original: ModuleAnalysis, refactored: Dict[str, ModuleAnalysis]
    ) -> RefactoringPattern:
        """Извлекает паттерн исправления импортов."""
        return RefactoringPattern(
            name="import_fixing",
            description="Automatically fix non-existent imports and add fallbacks",
            before_pattern="from core.cli.* import ...",
            after_pattern="# Removed non-existent import or added fallback",
            conditions=[
                "Import references non-existent module",
                "Module path does not exist in codebase",
            ],
            confidence=1.0,
        )

    def _extract_logging_standardization_pattern(
        self, original: ModuleAnalysis, refactored: Dict[str, ModuleAnalysis]
    ) -> RefactoringPattern:
        """Извлекает паттерн стандартизации логирования."""
        return RefactoringPattern(
            name="logging_standardization",
            description="Standardize logging variable names across codebase",
            before_pattern="LOG = logging.getLogger(...)",
            after_pattern="logger = logging.getLogger(...)",
            conditions=[
                "Codebase uses mixed logging variable names",
                "Majority of files use 'logger' instead of 'LOG'",
            ],
            confidence=0.8,
        )

    def _extract_owner_proxy_pattern(
        self, refactored: Dict[str, ModuleAnalysis]
    ) -> RefactoringPattern:
        """Извлекает паттерн Owner Proxy."""
        return RefactoringPattern(
            name="owner_proxy_pattern",
            description="Implement Owner Proxy pattern for state delegation",
            before_pattern="Direct state access and method calls",
            after_pattern="Proxy-based delegation with __slots__ optimization",
            conditions=[
                "Multiple classes need access to shared state",
                "State management can be centralized",
                "Memory optimization is desired",
            ],
            confidence=0.9,
        )

    def generate_automation_rules(self) -> Dict[str, Any]:
        """Генерирует правила автоматизации на основе извлеченных паттернов."""
        rules = {
            "import_fixes": [],
            "logging_standardization": {},
            "service_extraction_heuristics": [],
            "quality_checks": [],
        }

        for pattern in self.patterns:
            if pattern.name == "import_fixing":
                rules["import_fixes"].append(
                    {
                        "pattern": pattern.before_pattern,
                        "replacement": pattern.after_pattern,
                        "confidence": pattern.confidence,
                    }
                )
            elif pattern.name == "logging_standardization":
                rules["logging_standardization"] = {
                    "preferred_pattern": "logger = logging.getLogger(__name__)",
                    "replacements": [
                        {
                            "from": "LOG = logging.getLogger",
                            "to": "logger = logging.getLogger",
                        },
                        {"from": "LOG.", "to": "logger."},
                    ],
                }

        return rules

    def save_patterns_to_knowledge_base(self, knowledge_dir: str):
        """Сохраняет извлеченные паттерны в базу знаний."""
        knowledge_path = Path(knowledge_dir)
        knowledge_path.mkdir(exist_ok=True)

        # Сохраняем паттерны в JSON для использования в IntelliRefactor
        import json

        patterns_data = {
            "patterns": [
                {
                    "name": p.name,
                    "description": p.description,
                    "before_pattern": p.before_pattern,
                    "after_pattern": p.after_pattern,
                    "conditions": p.conditions,
                    "confidence": p.confidence,
                }
                for p in self.patterns
            ],
            "automation_rules": self.generate_automation_rules(),
        }

        with open(knowledge_path / "extracted_patterns.json", "w", encoding="utf-8") as f:
            json.dump(patterns_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.patterns)} patterns to knowledge base")


# Функция для запуска анализа
def analyze_manual_refactoring(original_file: str, refactored_dir: str, knowledge_dir: str):
    """Запускает анализ ручного рефакторинга и сохраняет результаты."""
    analyzer = ManualRefactoringAnalyzer()

    # Анализируем оригинальный файл
    logger.info(f"Analyzing original file: {original_file}")
    original_analysis = analyzer.analyze_original_module(original_file)

    # Извлекаем паттерны рефакторинга
    logger.info(f"Extracting refactoring patterns from {refactored_dir}")
    patterns = analyzer.extract_refactoring_patterns(original_file, refactored_dir)

    # Сохраняем в базу знаний
    logger.info(f"Saving patterns to knowledge base: {knowledge_dir}")
    analyzer.save_patterns_to_knowledge_base(knowledge_dir)

    return analyzer, patterns


if __name__ == "__main__":
    # Пример использования
    analyze_manual_refactoring(
        "core/cli_payload/adaptive_cli_wrapper.py",
        "core/cli_payload/components_new",
        "intellirefactor/knowledge",
    )
