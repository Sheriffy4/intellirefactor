"""
Import Fixing Patterns for IntelliRefactor

Автоматические алгоритмы для исправления импортов и логирования
на основе анализа реального рефакторинга adaptive_cli_wrapper.py
"""

import re
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ImportFix:
    """Описание исправления импорта."""

    pattern: str
    replacement: str
    reason: str


@dataclass
class LoggingFix:
    """Описание исправления логирования."""

    pattern: str
    replacement: str
    reason: str


class ImportFixingPatterns:
    """Паттерны для автоматического исправления импортов."""

    # Паттерны исправления несуществующих импортов
    IMPORT_FIXES = [
        ImportFix(
            pattern=r"from core\.cli\.error_handler import.*",
            replacement="# Removed non-existent import: core.cli.error_handler",
            reason="Module core.cli does not exist in codebase",
        ),
        ImportFix(
            pattern=r"from core\.cli\.([^\\s]+) import.*",
            replacement=r"# Removed non-existent import: core.cli.\1",
            reason="Module core.cli does not exist in codebase",
        ),
    ]

    # Паттерны исправления логирования
    LOGGING_FIXES = [
        LoggingFix(
            pattern=r"LOG = logging\.getLogger\(",
            replacement="logger = logging.getLogger(",
            reason="Standardize on 'logger' variable name instead of 'LOG'",
        ),
        LoggingFix(
            pattern=r"LOG\.([a-zA-Z_]+)\(",
            replacement=r"logger.\1(",
            reason="Use standardized logger variable",
        ),
    ]

    # Паттерны добавления fallback импортов
    FALLBACK_PATTERNS = [
        {
            "detect": r"from (core\.[^\\s]+) import ([^\\s]+)",
            "template": """# Import fallbacks for missing modules
try:
    from {module} import {imports}
    {availability_flag} = True
except ImportError:
    {availability_flag} = False
    class {main_class}:
        def __init__(self, *args, **kwargs): pass""",
        }
    ]

    @classmethod
    def fix_imports(cls, content: str) -> str:
        """Автоматически исправляет импорты в коде."""
        fixed_content = content

        for fix in cls.IMPORT_FIXES:
            fixed_content = re.sub(fix.pattern, fix.replacement, fixed_content)

        return fixed_content

    @classmethod
    def fix_logging(cls, content: str) -> str:
        """Автоматически исправляет паттерны логирования."""
        fixed_content = content

        for fix in cls.LOGGING_FIXES:
            fixed_content = re.sub(fix.pattern, fix.replacement, fixed_content)

        return fixed_content

    @classmethod
    def add_fallback_imports(cls, content: str) -> str:
        """Добавляет fallback импорты для несуществующих модулей."""
        # Этот метод будет расширен на основе дальнейшего анализа
        return content

    @classmethod
    def apply_all_fixes(cls, content: str) -> str:
        """Применяет все исправления к коду."""
        content = cls.fix_imports(content)
        content = cls.fix_logging(content)
        content = cls.add_fallback_imports(content)
        return content


class CodebaseAnalysisPatterns:
    """Паттерны для анализа кодовой базы перед рефакторингом."""

    @classmethod
    def analyze_logging_patterns(cls, codebase_files: List[str]) -> Dict[str, int]:
        """Анализирует паттерны логирования в кодовой базе."""
        patterns = {
            "LOG = logging.getLogger": 0,
            "logger = logging.getLogger": 0,
            "self.logger = logging.getLogger": 0,
        }

        for file_content in codebase_files:
            for pattern in patterns:
                patterns[pattern] += len(re.findall(pattern, file_content))

        return patterns

    @classmethod
    def find_existing_modules(cls, codebase_files: List[str]) -> List[str]:
        """Находит существующие модули в кодовой базе."""
        modules = set()

        for file_content in codebase_files:
            # Ищем импорты вида "from module import ..."
            imports = re.findall(r"from ([^\\s]+) import", file_content)
            modules.update(imports)

        return list(modules)

    @classmethod
    def recommend_logging_standard(cls, codebase_files: List[str]) -> str:
        """Рекомендует стандарт логирования на основе анализа кодовой базы."""
        patterns = cls.analyze_logging_patterns(codebase_files)

        if patterns["logger = logging.getLogger"] > patterns["LOG = logging.getLogger"]:
            return "logger"
        else:
            return "LOG"


class RefactoringQualityPatterns:
    """Паттерны для обеспечения качества рефакторинга."""

    QUALITY_CHECKS = [
        {
            "name": "consistent_logging",
            "pattern": r"(LOG|logger) = logging\.getLogger",
            "check": lambda matches: len(set(match.group(1) for match in matches)) == 1,
            "message": "Use consistent logging variable name throughout the file",
        },
        {
            "name": "no_missing_imports",
            "pattern": r"from core\.cli\.",
            "check": lambda matches: len(matches) == 0,
            "message": "Remove imports from non-existent core.cli module",
        },
    ]

    @classmethod
    def check_quality(cls, content: str) -> List[str]:
        """Проверяет качество рефакторированного кода."""
        issues = []

        for check in cls.QUALITY_CHECKS:
            matches = list(re.finditer(check["pattern"], content))
            if not check["check"](matches):
                issues.append(check["message"])

        return issues


# Экспорт основных классов для использования в IntelliRefactor
__all__ = [
    "ImportFixingPatterns",
    "CodebaseAnalysisPatterns",
    "RefactoringQualityPatterns",
    "ImportFix",
    "LoggingFix",
]
