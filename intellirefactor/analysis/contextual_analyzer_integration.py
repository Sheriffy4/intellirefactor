#!/usr/bin/env python3
"""
Интеграция StructuredUltimateAnalyzer с IntelliRefactor.

Этот модуль позволяет IntelliRefactor использовать богатые результаты анализа
из structured_ultimate_analyzer.py вместо примитивного анализа в auto_refactor.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextualAnalyzerIntegration:
    """Интеграция с результатами StructuredUltimateAnalyzer."""

    def __init__(self, analysis_results_dir: Optional[Path] = None):
        """
        Инициализация интеграции.
        
        Args:
            analysis_results_dir: Путь к папке с результатами анализа (например, analysis_results3)
        """
        self.analysis_results_dir = analysis_results_dir
        self._cached_analysis: Dict[str, Any] = {}
        
    def find_analysis_results_dir(self, project_path: Path) -> Optional[Path]:
        """Автоматически находит папку с результатами анализа."""
        logger.info(f"Looking for analysis results directory from project path: {project_path}")
        
        if self.analysis_results_dir and self.analysis_results_dir.exists():
            logger.info(f"Using configured analysis results directory: {self.analysis_results_dir}")
            return self.analysis_results_dir
            
        # Ищем в проекте
        for candidate in ["analysis_results3", "analysis_results", "structured_analysis"]:
            candidate_path = project_path / candidate
            logger.info(f"Checking candidate path: {candidate_path}")
            if candidate_path.exists() and candidate_path.is_dir():
                logger.info(f"Found analysis results directory: {candidate_path}")
                return candidate_path
                
        logger.warning(f"No analysis results directory found in {project_path}")
        return None
    
    def load_analysis_for_file(self, file_path: Path, project_path: Path) -> Optional[Dict[str, Any]]:
        """
        Загружает результаты анализа для конкретного файла.
        
        Args:
            file_path: Путь к анализируемому файлу
            project_path: Путь к корню проекта
            
        Returns:
            Словарь с результатами анализа или None если не найдено
        """
        cache_key = str(file_path)
        if cache_key in self._cached_analysis:
            return self._cached_analysis[cache_key]
            
        results_dir = self.find_analysis_results_dir(project_path)
        if not results_dir:
            logger.warning("Analysis results directory not found")
            return None
            
        # Загружаем различные типы анализа
        analysis_data = {}
        
        # 1. Canonical snapshot - основная структура файла
        canonical_file = self._find_latest_file(results_dir / "json", "canonical_analysis_snapshot_*.json")
        if canonical_file:
            canonical_data = self._load_json_file(canonical_file)
            if canonical_data:
                # Сравниваем пути более гибко
                canonical_path = Path(canonical_data.get("file_path", ""))
                if canonical_path.name == file_path.name and str(canonical_path).endswith(str(file_path)):
                    analysis_data["canonical"] = canonical_data
                    logger.info(f"Loaded canonical data for {file_path}")
                else:
                    logger.warning(f"Canonical data file path mismatch: expected {file_path}, got {canonical_data.get('file_path')}")
            else:
                logger.warning("Failed to load canonical data")
        else:
            logger.warning("No canonical analysis snapshot found")
                
        # 2. Target file opportunities - возможности рефакторинга для конкретного файла
        opportunities_file = self._find_latest_file(results_dir / "json", "target_file_opportunities_*.json")
        if opportunities_file:
            opportunities_data = self._load_json_file(opportunities_file)
            if opportunities_data:
                analysis_data["opportunities"] = opportunities_data
                
        # 3. Contextual refactoring decisions - умные решения по рефакторингу
        decisions_file = self._find_latest_file(results_dir / "json", "contextual_refactoring_decisions_*.json")
        if decisions_file:
            decisions_data = self._load_json_file(decisions_file)
            if decisions_data:
                analysis_data["refactoring_decisions"] = decisions_data
                
        # 4. Architectural smells - архитектурные проблемы
        smells_file = self._find_latest_file(results_dir / "json", "contextual_architectural_smells_*.json")
        if smells_file:
            smells_data = self._load_json_file(smells_file)
            if smells_data:
                analysis_data["architectural_smells"] = smells_data
                
        # 5. Duplicate blocks - дублированный код
        duplicates_file = self._find_latest_file(results_dir / "json", "contextual_duplicate_blocks_*.json")
        if duplicates_file:
            duplicates_data = self._load_json_file(duplicates_file)
            if duplicates_data:
                analysis_data["duplicates"] = duplicates_data
        
        if analysis_data:
            self._cached_analysis[cache_key] = analysis_data
            logger.info(f"Loaded contextual analysis for {file_path}: {list(analysis_data.keys())}")
            
        return analysis_data if analysis_data else None
    
    def extract_god_object_info(self, analysis_data: Dict[str, Any], class_name: str) -> Optional[Dict[str, Any]]:
        """
        Извлекает информацию о God Object из результатов анализа.
        
        Args:
            analysis_data: Результаты анализа
            class_name: Имя класса для анализа
            
        Returns:
            Информация о God Object или None
        """
        canonical = analysis_data.get("canonical", {})
        structure = canonical.get("structure", {})
        classes = structure.get("classes", [])
        
        # Находим нужный класс
        target_class = None
        for cls in classes:
            if cls.get("name") == class_name:
                target_class = cls
                break
                
        if not target_class:
            return None
            
        methods = target_class.get("methods", [])
        
        return {
            "class_name": class_name,
            "total_methods": len(methods),
            "is_god_object": target_class.get("is_god_object", len(methods) > 10),
            "methods": methods,
            "line_range": (target_class.get("line", 0), target_class.get("end_line", 0)),
            "file_stats": canonical.get("file_stats", {}),
        }
    
    def extract_refactoring_opportunities(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Извлекает возможности рефакторинга из результатов анализа.
        
        Args:
            analysis_data: Результаты анализа
            
        Returns:
            Список возможностей рефакторинга
        """
        opportunities = analysis_data.get("opportunities", [])
        
        # Преобразуем в формат понятный IntelliRefactor
        refactoring_ops = []
        
        for opp in opportunities:
            if isinstance(opp, dict):
                refactoring_ops.append({
                    "id": opp.get("id", "unknown"),
                    "type": opp.get("type", "unknown"),
                    "priority": opp.get("priority", 5),
                    "description": opp.get("description", ""),
                    "target_files": opp.get("target_files", []),
                    "estimated_impact": opp.get("estimated_impact", {}),
                    "risk_level": opp.get("risk_level", "medium"),
                    "automation_confidence": opp.get("automation_confidence", 0.5),
                })
                
        return refactoring_ops
    
    def extract_method_groups_from_context(self, analysis_data: Dict[str, Any], class_name: str) -> Dict[str, List[str]]:
        """
        Извлекает группировку методов на основе контекстного анализа.
        
        Args:
            analysis_data: Результаты анализа
            class_name: Имя класса
            
        Returns:
            Словарь групп методов
        """
        # Получаем информацию о классе
        god_object_info = self.extract_god_object_info(analysis_data, class_name)
        if not god_object_info:
            return {}
            
        methods = god_object_info["methods"]
        
        # Группируем методы по семантическим паттернам
        groups = {
            "dispatch": [],
            "resolution": [],
            "strategy": [],
            "parameter": [],
            "parsing": [],
            "logging": [],
            "validation": [],
            "network": [],
            "utility": [],
        }
        
        # Более умная группировка на основе анализа кода
        for method in methods:
            method_name = method.get("name", "")
            method_def = method.get("definition", "")
            
            # Анализируем имя метода и определение
            name_lower = method_name.lower()
            
            
            assigned = False
            
            # Dispatch группа
            if any(word in name_lower for word in ["dispatch", "route", "execute", "apply", "run", "invoke", "call", "trigger"]):
                groups["dispatch"].append(method_name)
                assigned = True
            # Resolution группа
            elif any(word in name_lower for word in ["resolve", "find", "locate", "search", "extract", "detect", "identify"]):
                groups["resolution"].append(method_name)
                assigned = True
            # Strategy группа
            elif any(word in name_lower for word in ["strategy", "combo", "combination", "sequence", "recipe", "plan"]):
                groups["strategy"].append(method_name)
                assigned = True
            # Parameter группа
            elif any(word in name_lower for word in ["parameter", "param", "normalize", "map", "filter"]):
                groups["parameter"].append(method_name)
                assigned = True
            # Parsing группа
            elif any(word in name_lower for word in ["parse", "decode", "process", "handle", "interpret"]):
                groups["parsing"].append(method_name)
                assigned = True
            # Logging группа
            elif any(word in name_lower for word in ["log", "trace", "debug", "info", "warn", "error", "record"]):
                groups["logging"].append(method_name)
                assigned = True
            # Validation группа
            elif any(word in name_lower for word in ["validate", "verify", "check", "ensure", "is_valid"]):
                groups["validation"].append(method_name)
                assigned = True
            # Network группа
            elif any(word in name_lower for word in ["request", "connect", "http", "api", "fetch", "download", "send", "receive"]):
                groups["network"].append(method_name)
                assigned = True
            
            # Если не назначен, добавляем в utility
            if not assigned:
                groups["utility"].append(method_name)
        
        # Удаляем пустые группы
        return {k: v for k, v in groups.items() if v}
    
    def _find_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Находит последний файл по паттерну."""
        if not directory.exists():
            return None
            
        files = list(directory.glob(pattern))
        if not files:
            return None
            
        # Сортируем по времени модификации
        return max(files, key=lambda f: f.stat().st_mtime)
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Загружает JSON файл."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            return None