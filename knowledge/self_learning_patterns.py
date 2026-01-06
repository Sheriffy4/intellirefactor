"""
Self-Learning Patterns for IntelliRefactor

Система самообучения, которая анализирует ручные исправления и автоматически
применяет извлеченные паттерны к новым рефакторингам.
"""

import ast
import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Паттерн, извлеченный из ручного анализа."""
    name: str
    description: str
    pattern_type: str  # 'import_fix', 'logging_fix', 'code_pattern', etc.
    before_regex: str
    after_template: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 1.0
    conditions: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CodebaseStandards:
    """Стандарты кодовой базы, извлеченные из анализа."""
    logging_variable: str = "logger"
    import_style: str = "absolute"
    error_handling_pattern: str = "try_except_fallback"
    preferred_modules: Set[str] = field(default_factory=set)
    deprecated_modules: Set[str] = field(default_factory=set)
    common_fallbacks: Dict[str, str] = field(default_factory=dict)


class SelfLearningSystem:
    """Система самообучения IntelliRefactor."""
    
    def __init__(self, knowledge_dir: str = "intellirefactor/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(exist_ok=True)
        
        self.patterns: List[LearningPattern] = []
        self.codebase_standards: CodebaseStandards = CodebaseStandards()
        self.learning_history: List[Dict[str, Any]] = []
        
        # Загружаем существующие паттерны
        self._load_existing_patterns()
    
    def analyze_manual_corrections(
        self, 
        original_file: str, 
        corrected_files: List[str],
        correction_description: str = ""
    ) -> List[LearningPattern]:
        """Анализирует ручные исправления и извлекает паттерны."""
        logger.info(f"Analyzing manual corrections: {original_file} -> {len(corrected_files)} files")
        
        # Читаем оригинальный файл
        with open(original_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Читаем исправленные файлы
        corrected_contents = {}
        for corrected_file in corrected_files:
            with open(corrected_file, 'r', encoding='utf-8') as f:
                corrected_contents[corrected_file] = f.read()
        
        # Извлекаем паттерны
        new_patterns = []
        
        # 1. Анализируем исправления импортов
        import_patterns = self._extract_import_fix_patterns(original_content, corrected_contents)
        new_patterns.extend(import_patterns)
        
        # 2. Анализируем исправления логирования
        logging_patterns = self._extract_logging_fix_patterns(original_content, corrected_contents)
        new_patterns.extend(logging_patterns)
        
        # 3. Анализируем структурные изменения
        structural_patterns = self._extract_structural_patterns(original_content, corrected_contents)
        new_patterns.extend(structural_patterns)
        
        # 4. Обновляем стандарты кодовой базы
        self._update_codebase_standards(original_content, corrected_contents)
        
        # Сохраняем новые паттерны
        for pattern in new_patterns:
            self._add_pattern(pattern)
        
        # Записываем в историю обучения
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "original_file": original_file,
            "corrected_files": corrected_files,
            "description": correction_description,
            "patterns_extracted": len(new_patterns),
            "pattern_names": [p.name for p in new_patterns]
        })
        
        # Сохраняем обновленную базу знаний
        self._save_knowledge_base()
        
        logger.info(f"Extracted {len(new_patterns)} new patterns")
        return new_patterns
    
    def _extract_import_fix_patterns(
        self, 
        original: str, 
        corrected: Dict[str, str]
    ) -> List[LearningPattern]:
        """Извлекает паттерны исправления импортов."""
        patterns = []
        
        # Ищем импорты в оригинальном файле
        original_imports = re.findall(r'from ([^\s]+) import ([^\n]+)', original)
        
        for file_path, corrected_content in corrected.items():
            # Ищем исправления импортов
            
            # Паттерн 1: Замена несуществующих импортов на fallback
            fallback_imports = re.findall(
                r'# Import fallbacks for missing modules\ntry:\n\s+from ([^\s]+) import ([^\n]+)\n\s+([A-Z_]+) = True\nexcept ImportError:\n\s+([A-Z_]+) = False',
                corrected_content,
                re.MULTILINE
            )
            
            for module, imports, flag1, flag2 in fallback_imports:
                if flag1 == flag2:  # Проверяем что флаги одинаковые
                    pattern = LearningPattern(
                        name=f"fallback_import_{module.replace('.', '_')}",
                        description=f"Add fallback import for {module}",
                        pattern_type="import_fix",
                        before_regex=f"from {re.escape(module)} import {re.escape(imports)}",
                        after_template=f"""# Import fallbacks for missing modules
try:
    from {module} import {imports}
    {flag1} = True
except ImportError:
    {flag1} = False
    # Add fallback classes/functions here""",
                        confidence=0.95,
                        conditions=[f"Module {module} does not exist in codebase"],
                        examples=[{
                            "before": f"from {module} import {imports}",
                            "after": f"try:\n    from {module} import {imports}\nexcept ImportError:\n    pass"
                        }]
                    )
                    patterns.append(pattern)
            
            # Паттерн 2: Удаление несуществующих импортов
            removed_imports = re.findall(r'# Removed non-existent import: (.+)', corrected_content)
            for removed_import in removed_imports:
                pattern = LearningPattern(
                    name=f"remove_nonexistent_import",
                    description="Remove non-existent imports",
                    pattern_type="import_fix",
                    before_regex=re.escape(removed_import.strip()),
                    after_template=f"# Removed non-existent import: {removed_import.strip()}",
                    confidence=0.90,
                    conditions=["Import references non-existent module"],
                    examples=[{
                        "before": removed_import.strip(),
                        "after": f"# Removed non-existent import: {removed_import.strip()}"
                    }]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_logging_fix_patterns(
        self, 
        original: str, 
        corrected: Dict[str, str]
    ) -> List[LearningPattern]:
        """Извлекает паттерны исправления логирования."""
        patterns = []
        
        # Анализируем изменения в логировании
        original_log_vars = re.findall(r'(LOG|logger) = logging\.getLogger', original)
        
        for file_path, corrected_content in corrected.items():
            corrected_log_vars = re.findall(r'(LOG|logger) = logging\.getLogger', corrected_content)
            
            # Если в оригинале было LOG, а в исправленном logger
            if 'LOG' in original_log_vars and 'logger' in corrected_log_vars:
                pattern = LearningPattern(
                    name="standardize_logging_variable",
                    description="Standardize logging variable from LOG to logger",
                    pattern_type="logging_fix",
                    before_regex=r"LOG = logging\.getLogger",
                    after_template="logger = logging.getLogger",
                    confidence=0.85,
                    conditions=["Codebase primarily uses 'logger' variable"],
                    examples=[{
                        "before": "LOG = logging.getLogger(__name__)",
                        "after": "logger = logging.getLogger(__name__)"
                    }]
                )
                patterns.append(pattern)
                
                # Также добавляем паттерн для замены использования LOG на logger
                pattern2 = LearningPattern(
                    name="replace_log_usage",
                    description="Replace LOG usage with logger",
                    pattern_type="logging_fix",
                    before_regex=r"LOG\.",
                    after_template="logger.",
                    confidence=0.85,
                    conditions=["LOG variable was replaced with logger"],
                    examples=[{
                        "before": "LOG.info('message')",
                        "after": "logger.info('message')"
                    }]
                )
                patterns.append(pattern2)
        
        return patterns
    
    def _extract_structural_patterns(
        self, 
        original: str, 
        corrected: Dict[str, str]
    ) -> List[LearningPattern]:
        """Извлекает структурные паттерны рефакторинга."""
        patterns = []
        
        # Анализируем структурные изменения
        try:
            original_tree = ast.parse(original)
            original_classes = [node.name for node in ast.walk(original_tree) if isinstance(node, ast.ClassDef)]
            original_functions = [node.name for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef)]
        except SyntaxError:
            logger.warning("Failed to parse original file for structural analysis")
            return patterns
        
        for file_path, corrected_content in corrected.items():
            try:
                corrected_tree = ast.parse(corrected_content)
                corrected_classes = [node.name for node in ast.walk(corrected_tree) if isinstance(node, ast.ClassDef)]
                corrected_functions = [node.name for node in ast.walk(corrected_tree) if isinstance(node, ast.FunctionDef)]
                
                # Если появились новые классы (например, OwnerProxyMixin)
                new_classes = set(corrected_classes) - set(original_classes)
                if 'OwnerProxyMixin' in new_classes:
                    pattern = LearningPattern(
                        name="add_owner_proxy_mixin",
                        description="Add OwnerProxyMixin for state delegation",
                        pattern_type="structural_pattern",
                        before_regex=r"class (\w+):",
                        after_template="class \\1(OwnerProxyMixin):",
                        confidence=0.80,
                        conditions=["Component needs access to facade state"],
                        examples=[{
                            "before": "class AnalysisService:",
                            "after": "class AnalysisService(OwnerProxyMixin):"
                        }]
                    )
                    patterns.append(pattern)
                
            except SyntaxError:
                logger.warning(f"Failed to parse corrected file {file_path} for structural analysis")
                continue
        
        return patterns
    
    def _update_codebase_standards(
        self, 
        original: str, 
        corrected: Dict[str, str]
    ) -> None:
        """Обновляет стандарты кодовой базы на основе исправлений."""
        
        # Анализируем предпочтения в логировании
        logger_count = sum(content.count('logger = logging.getLogger') for content in corrected.values())
        log_count = sum(content.count('LOG = logging.getLogger') for content in corrected.values())
        
        if logger_count > log_count:
            self.codebase_standards.logging_variable = "logger"
        else:
            self.codebase_standards.logging_variable = "LOG"
        
        # Анализируем паттерны обработки ошибок
        for content in corrected.values():
            if "try:" in content and "except ImportError:" in content:
                self.codebase_standards.error_handling_pattern = "try_except_fallback"
        
        # Анализируем устаревшие модули
        for content in corrected.values():
            removed_imports = re.findall(r'# Removed non-existent import: from ([^\s]+)', content)
            for module in removed_imports:
                self.codebase_standards.deprecated_modules.add(module)
    
    def apply_learned_patterns(self, content: str, file_path: str = "") -> str:
        """Применяет изученные паттерны к новому коду."""
        logger.info(f"Applying {len(self.patterns)} learned patterns to {file_path}")
        
        modified_content = content
        applied_patterns = []
        
        for pattern in sorted(self.patterns, key=lambda p: p.confidence, reverse=True):
            try:
                if pattern.pattern_type == "import_fix":
                    new_content = self._apply_import_pattern(modified_content, pattern)
                elif pattern.pattern_type == "logging_fix":
                    new_content = self._apply_logging_pattern(modified_content, pattern)
                elif pattern.pattern_type == "structural_pattern":
                    new_content = self._apply_structural_pattern(modified_content, pattern)
                else:
                    continue
                
                if new_content != modified_content:
                    modified_content = new_content
                    applied_patterns.append(pattern.name)
                    pattern.usage_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to apply pattern {pattern.name}: {e}")
                continue
        
        if applied_patterns:
            logger.info(f"Applied patterns: {', '.join(applied_patterns)}")
        
        return modified_content
    
    def _apply_import_pattern(self, content: str, pattern: LearningPattern) -> str:
        """Применяет паттерн исправления импортов."""
        if pattern.name.startswith("fallback_import_"):
            # Ищем импорты, которые нужно заменить на fallback
            matches = re.finditer(pattern.before_regex, content)
            for match in matches:
                # Проверяем, что это не уже исправленный импорт
                if "try:" not in content[max(0, match.start()-100):match.start()]:
                    content = content.replace(match.group(0), pattern.after_template)
        
        elif pattern.name == "remove_nonexistent_import":
            # Удаляем несуществующие импорты
            content = re.sub(pattern.before_regex, pattern.after_template, content)
        
        return content
    
    def _apply_logging_pattern(self, content: str, pattern: LearningPattern) -> str:
        """Применяет паттерн исправления логирования."""
        return re.sub(pattern.before_regex, pattern.after_template, content)
    
    def _apply_structural_pattern(self, content: str, pattern: LearningPattern) -> str:
        """Применяет структурный паттерн."""
        return re.sub(pattern.before_regex, pattern.after_template, content)
    
    def _add_pattern(self, pattern: LearningPattern) -> None:
        """Добавляет новый паттерн в базу знаний."""
        # Проверяем, не существует ли уже такой паттерн
        existing = next((p for p in self.patterns if p.name == pattern.name), None)
        if existing:
            # Обновляем существующий паттерн
            existing.confidence = max(existing.confidence, pattern.confidence)
            existing.examples.extend(pattern.examples)
        else:
            self.patterns.append(pattern)
    
    def _load_existing_patterns(self) -> None:
        """Загружает существующие паттерны из базы знаний."""
        patterns_file = self.knowledge_dir / "learned_patterns.json"
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for pattern_data in data.get("patterns", []):
                    pattern = LearningPattern(**pattern_data)
                    self.patterns.append(pattern)
                
                standards_data = data.get("codebase_standards", {})
                if standards_data:
                    self.codebase_standards = CodebaseStandards(
                        logging_variable=standards_data.get("logging_variable", "logger"),
                        import_style=standards_data.get("import_style", "absolute"),
                        error_handling_pattern=standards_data.get("error_handling_pattern", "try_except_fallback"),
                        preferred_modules=set(standards_data.get("preferred_modules", [])),
                        deprecated_modules=set(standards_data.get("deprecated_modules", [])),
                        common_fallbacks=standards_data.get("common_fallbacks", {})
                    )
                
                self.learning_history = data.get("learning_history", [])
                
                logger.info(f"Loaded {len(self.patterns)} patterns from knowledge base")
                
            except Exception as e:
                logger.warning(f"Failed to load existing patterns: {e}")
    
    def _save_knowledge_base(self) -> None:
        """Сохраняет базу знаний."""
        patterns_file = self.knowledge_dir / "learned_patterns.json"
        
        data = {
            "patterns": [
                {
                    "name": p.name,
                    "description": p.description,
                    "pattern_type": p.pattern_type,
                    "before_regex": p.before_regex,
                    "after_template": p.after_template,
                    "confidence": p.confidence,
                    "usage_count": p.usage_count,
                    "success_rate": p.success_rate,
                    "conditions": p.conditions,
                    "examples": p.examples
                }
                for p in self.patterns
            ],
            "codebase_standards": {
                "logging_variable": self.codebase_standards.logging_variable,
                "import_style": self.codebase_standards.import_style,
                "error_handling_pattern": self.codebase_standards.error_handling_pattern,
                "preferred_modules": list(self.codebase_standards.preferred_modules),
                "deprecated_modules": list(self.codebase_standards.deprecated_modules),
                "common_fallbacks": self.codebase_standards.common_fallbacks
            },
            "learning_history": self.learning_history
        }
        
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.patterns)} patterns to knowledge base")
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по паттернам."""
        pattern_types = Counter(p.pattern_type for p in self.patterns)
        
        return {
            "total_patterns": len(self.patterns),
            "pattern_types": dict(pattern_types),
            "average_confidence": sum(p.confidence for p in self.patterns) / len(self.patterns) if self.patterns else 0,
            "most_used_patterns": sorted(self.patterns, key=lambda p: p.usage_count, reverse=True)[:5],
            "learning_sessions": len(self.learning_history),
            "codebase_standards": {
                "logging_variable": self.codebase_standards.logging_variable,
                "deprecated_modules_count": len(self.codebase_standards.deprecated_modules)
            }
        }


# Глобальный экземпляр системы самообучения
_learning_system = None

def get_learning_system() -> SelfLearningSystem:
    """Возвращает глобальный экземпляр системы самообучения."""
    global _learning_system
    if _learning_system is None:
        _learning_system = SelfLearningSystem()
    return _learning_system


# Экспорт основных классов
__all__ = [
    'SelfLearningSystem',
    'LearningPattern',
    'CodebaseStandards',
    'get_learning_system'
]