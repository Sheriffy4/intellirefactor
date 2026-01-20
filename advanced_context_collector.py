#!/usr/bin/env python3
"""
Продвинутый сборщик контекста для рефакторинга модулей.

Дополнительные возможности:
- Интерактивный режим выбора файлов
- Предварительный просмотр контекста
- Экспорт в разные форматы
- Валидация качества контекста
- Сбор данных экспертного анализа
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import asdict
import argparse
from context_collector import ContextCollector, ContextFile


class AdvancedContextCollector(ContextCollector):
    """Продвинутый сборщик контекста с дополнительными возможностями."""
    
    def __init__(self, analysis_results_dir: str, target_module_path: str):
        super().__init__(analysis_results_dir, target_module_path)
        self.quality_metrics = {}
        
        # Добавляем паттерны для экспертного анализа
        self.file_patterns['expert'] = [
            r"expert_analysis_report_\d{8}_\d{6}\.md",
            r"expert_analysis_\d{8}_\d{6}\.json",
        ]
        
        # Добавляем паттерны для structured ultimate analyzer результатов
        self.file_patterns['structured'] = [
            r"canonical_analysis_snapshot_\d{8}_\d{6}\.json",
            r"contextual_decision_analysis_\d{8}_\d{6}\.json",
            r"contextual_refactoring_decisions_\d{8}_\d{6}\.json",
        ]
    
    def collect_context_files(self) -> List[ContextFile]:
        """Собирает все релевантные файлы контекста с улучшенным контролем размера."""
        context_files = []
        
        # ЭКСПЕРТНЫЙ АНАЛИЗ (наивысший приоритет)
        for pattern in self.file_patterns['expert']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=0,  # Наивысший приоритет
                    description=f"Expert refactoring analysis: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='expert'
                ))
        
        # Архитектурные диаграммы и документация (высокий приоритет)
        for pattern in self.file_patterns['architecture']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=1,
                    description=f"Architecture analysis: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='architecture'
                ))
        
        # Планы рефакторинга (высокий приоритет)
        for pattern in self.file_patterns['plan']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=1,
                    description=f"Refactoring plan: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='plan'
                ))
        
        # Контекст для LLM (высокий приоритет)
        for pattern in self.file_patterns['context']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=1,
                    description=f"LLM context: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='context'
                ))
        
        # Структурированный анализ (высокий приоритет, контролируемый размер)
        for pattern in self.file_patterns['structured']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                # Контролируем размер structured файлов
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 10:  # Больше 10 МБ - берем только 200 строк
                    estimated_lines = 200
                elif file_size_mb > 5:  # Больше 5 МБ - берем только 300 строк
                    estimated_lines = 300
                elif file_size_mb > 1:  # Больше 1 МБ - берем только 500 строк
                    estimated_lines = 500
                else:
                    estimated_lines = min(800, self.get_file_size_estimate(file_path))
                
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=1,
                    description=f"Structured analysis (first {estimated_lines} lines): {file_path.name}",
                    estimated_lines=estimated_lines,
                    file_type='structured'
                ))
        
        # Возможности рефакторинга (средний приоритет)
        for pattern in self.file_patterns['opportunities']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=2,
                    description=f"Refactoring opportunities: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='opportunities'
                ))
        
        # Архитектурные запахи (средний приоритет, строгое ограничение размера)
        for pattern in self.file_patterns['smells']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                # Проверяем размер файла
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 5:  # Больше 5 МБ - берем только 100 строк
                    estimated_lines = 100
                elif file_size_mb > 1:  # Больше 1 МБ - берем только 200 строк
                    estimated_lines = 200
                else:
                    estimated_lines = min(300, self.get_file_size_estimate(file_path))
                
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=2,
                    description=f"Architectural smells (first {estimated_lines} lines): {file_path.name}",
                    estimated_lines=estimated_lines,
                    file_type='smells'
                ))
        
        # Дубликаты кода (средний приоритет, строгое ограничение размера)
        for pattern in self.file_patterns['duplicates']:
            files = self.find_files_by_pattern(pattern, self.analysis_dir)
            for file_path in files:
                # Проверяем размер файла
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 5:  # Больше 5 МБ - берем только 50 строк
                    estimated_lines = 50
                elif file_size_mb > 1:  # Больше 1 МБ - берем только 100 строк
                    estimated_lines = 100
                else:
                    estimated_lines = min(200, self.get_file_size_estimate(file_path))
                
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=3,
                    description=f"Code duplicates (first {estimated_lines} lines): {file_path.name}",
                    estimated_lines=estimated_lines,
                    file_type='duplicates'
                ))
        
        # Документация (низкий приоритет)
        docs_dir = self.analysis_dir / "docs"
        for pattern in self.file_patterns['docs']:
            files = self.find_files_by_pattern(pattern, docs_dir)
            for file_path in files:
                context_files.append(ContextFile(
                    path=str(file_path),
                    priority=4,
                    description=f"Documentation: {file_path.name}",
                    estimated_lines=self.get_file_size_estimate(file_path),
                    file_type='docs'
                ))
        
        # Сортируем по приоритету и размеру
        context_files.sort(key=lambda x: (x.priority, -x.estimated_lines))
        
        return context_files
    
    def analyze_context_quality(self, selected_files: List[ContextFile]) -> Dict[str, any]:
        """Анализирует качество собранного контекста с учетом экспертного анализа."""
        metrics = {
            'total_files': len(selected_files),
            'total_lines': sum(f.estimated_lines for f in selected_files),
            'coverage_score': 0.0,
            'completeness_score': 0.0,
            'balance_score': 0.0,
            'expert_analysis_score': 0.0,
            'recommendations': []
        }
        
        # Анализ покрытия типов файлов (включая экспертный анализ)
        file_types = set(f.file_type for f in selected_files)
        expected_types = {'expert', 'architecture', 'plan', 'opportunities', 'smells'}
        coverage = len(file_types.intersection(expected_types)) / len(expected_types)
        metrics['coverage_score'] = coverage
        
        # Анализ полноты (с приоритетом экспертного анализа)
        has_expert = any(f.file_type == 'expert' for f in selected_files)
        has_architecture = any(f.file_type == 'architecture' for f in selected_files)
        has_plan = any(f.file_type == 'plan' for f in selected_files)
        has_problems = any(f.file_type in ['smells', 'opportunities'] for f in selected_files)
        
        # Экспертный анализ имеет двойной вес
        completeness_factors = [has_expert, has_expert, has_architecture, has_plan, has_problems]
        completeness = sum(completeness_factors) / len(completeness_factors)
        metrics['completeness_score'] = completeness
        
        # Оценка экспертного анализа
        if has_expert:
            expert_files = [f for f in selected_files if f.file_type == 'expert']
            has_report = any('report' in f.path for f in expert_files)
            has_json = any(f.path.endswith('.json') for f in expert_files)
            
            expert_score = 0.0
            if has_report:
                expert_score += 0.6  # Отчет важнее
            if has_json:
                expert_score += 0.4  # JSON с данными
            
            metrics['expert_analysis_score'] = expert_score
        
        # Анализ баланса (с учетом экспертного анализа)
        type_distribution = {}
        for f in selected_files:
            type_distribution[f.file_type] = type_distribution.get(f.file_type, 0) + 1
        
        # Идеальное распределение с экспертным анализом
        ideal_ratios = {
            'expert': 0.3,        # Экспертный анализ - высший приоритет
            'architecture': 0.3,  # Архитектура
            'plan': 0.2,         # Планы
            'opportunities': 0.1, # Возможности
            'smells': 0.1        # Проблемы
        }
        actual_ratios = {k: v/len(selected_files) for k, v in type_distribution.items()}
        
        balance_score = 1.0
        for file_type, ideal_ratio in ideal_ratios.items():
            actual_ratio = actual_ratios.get(file_type, 0)
            balance_score -= abs(ideal_ratio - actual_ratio) * 0.3
        
        metrics['balance_score'] = max(0.0, balance_score)
        
        # Рекомендации с учетом экспертного анализа
        if not has_expert:
            metrics['recommendations'].append("🚨 КРИТИЧНО: отсутствует экспертный анализ - запустите structured_ultimate_analyzer с --expert")
        
        if coverage < 0.8:
            metrics['recommendations'].append("Добавьте больше типов файлов для полного покрытия")
        
        if not has_architecture:
            metrics['recommendations'].append("Критично: отсутствуют архитектурные диаграммы")
        
        if not has_plan:
            metrics['recommendations'].append("Рекомендуется: добавьте план рефакторинга")
        
        if has_expert and metrics['expert_analysis_score'] < 1.0:
            if not any('report' in f.path for f in [f for f in selected_files if f.file_type == 'expert']):
                metrics['recommendations'].append("Отсутствует отчет экспертного анализа (.md файл)")
            if not any(f.path.endswith('.json') for f in [f for f in selected_files if f.file_type == 'expert']):
                metrics['recommendations'].append("Отсутствуют данные экспертного анализа (.json файл)")
        
        if metrics['total_lines'] < 1000:
            metrics['recommendations'].append("Контекст может быть недостаточным (< 1000 строк)")
        
        if metrics['total_lines'] > 3000:
            metrics['recommendations'].append("Контекст может быть избыточным (> 3000 строк)")
        
        self.quality_metrics = metrics
        return metrics
    
    def interactive_file_selection(self) -> List[ContextFile]:
        """Интерактивный выбор файлов контекста."""
        all_files = self.collect_context_files()
        
        print(f"\n📁 Найдено {len(all_files)} файлов контекста для {self.module_name}")
        print("=" * 60)
        
        selected_files = []
        total_lines = 0
        
        for i, file_info in enumerate(all_files, 1):
            file_path = Path(file_info.path)
            print(f"\n{i:2d}. {file_info.file_type.upper()}: {file_path.name}")
            print(f"    📄 ~{file_info.estimated_lines} строк | Приоритет: {file_info.priority}")
            print(f"    📝 {file_info.description}")
            
            if total_lines + file_info.estimated_lines > 2500:
                print(f"    ⚠️  Превышение лимита строк ({total_lines + file_info.estimated_lines} > 2500)")
            
            choice = input("    Включить в контекст? [y/N/q]: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice in ['y', 'yes', 'д', 'да']:
                selected_files.append(file_info)
                total_lines += file_info.estimated_lines
                print(f"    ✅ Добавлено (всего строк: {total_lines})")
            else:
                print("    ❌ Пропущено")
        
        print(f"\n📊 Итого выбрано: {len(selected_files)} файлов, ~{total_lines} строк")
        return selected_files
    
    def preview_context(self, selected_files: List[ContextFile], lines_per_file: int = 10) -> str:
        """Создает предварительный просмотр контекста."""
        preview = f"# Preview: Context for {self.module_name} Refactoring\n\n"
        
        for i, file_info in enumerate(selected_files, 1):
            file_path = Path(file_info.path)
            preview += f"## {i}. {file_path.name} ({file_info.file_type})\n"
            preview += f"**Lines**: ~{file_info.estimated_lines} | **Priority**: {file_info.priority}\n\n"
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = []
                    for line_num, line in enumerate(f, 1):
                        lines.append(line.rstrip())
                        if line_num >= lines_per_file:
                            break
                    
                    preview += "```\n"
                    preview += "\n".join(lines)
                    if file_info.estimated_lines > lines_per_file:
                        preview += f"\n... ({file_info.estimated_lines - lines_per_file} more lines)"
                    preview += "\n```\n\n"
                    
            except Exception as e:
                preview += f"*Error reading file: {e}*\n\n"
        
        return preview
    
    def export_context_json(self, selected_files: List[ContextFile], output_path: str) -> None:
        """Экспортирует контекст в JSON формат с контролем размера."""
        context_data = {
            'module_info': {
                'name': self.module_name,
                'path': str(self.target_module),
                'analysis_dir': str(self.analysis_dir)
            },
            'files': [],
            'quality_metrics': self.quality_metrics,
            'total_lines': sum(f.estimated_lines for f in selected_files)
        }
        
        for file_info in selected_files:
            file_data = asdict(file_info)
            
            # Добавляем содержимое файла с контролем размера
            try:
                with open(file_info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    if file_info.file_type in ['smells', 'duplicates']:
                        # Для проблемных файлов берем только ограниченное количество строк
                        max_lines = file_info.estimated_lines
                        content_lines = []
                        for line_num, line in enumerate(f, 1):
                            content_lines.append(line.rstrip())
                            if line_num >= max_lines:
                                break
                        file_data['content'] = '\n'.join(content_lines)
                        file_data['truncated'] = True
                        file_data['truncated_at_lines'] = max_lines
                    else:
                        # Для архитектурных файлов берем полное содержимое, но с разумным лимитом
                        content = f.read()
                        if len(content) > 100000:  # Больше 100KB
                            file_data['content'] = content[:100000] + "\n\n... [TRUNCATED: Content too large]"
                            file_data['truncated'] = True
                        else:
                            file_data['content'] = content
                            file_data['truncated'] = False
            except Exception as e:
                file_data['content'] = f"Error reading file: {e}"
                file_data['truncated'] = False
            
            context_data['files'].append(file_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
    
    def generate_quality_report(self) -> str:
        """Генерирует отчет о качестве контекста с учетом экспертного анализа."""
        if not self.quality_metrics:
            return "Quality metrics not available. Run analyze_context_quality() first."
        
        metrics = self.quality_metrics
        
        # Определяем общую оценку с учетом экспертного анализа
        base_score = (
            metrics['coverage_score'] * 0.3 +
            metrics['completeness_score'] * 0.4 +
            metrics['balance_score'] * 0.2
        )
        
        # Бонус за экспертный анализ
        expert_bonus = metrics.get('expert_analysis_score', 0) * 0.1
        overall_score = base_score + expert_bonus
        
        if overall_score >= 0.9:
            grade = "🟢 Превосходный (с экспертным анализом)"
        elif overall_score >= 0.8:
            grade = "🟢 Отличный"
        elif overall_score >= 0.6:
            grade = "🟡 Хороший"
        else:
            grade = "🔴 Требует улучшения"
        
        report = f"""# Отчет о качестве контекста

## Общая оценка: {grade} ({overall_score:.1%})

### Метрики качества
- **Покрытие типов файлов**: {metrics['coverage_score']:.1%}
- **Полнота контекста**: {metrics['completeness_score']:.1%}
- **Баланс содержимого**: {metrics['balance_score']:.1%}"""
        
        if 'expert_analysis_score' in metrics:
            report += f"\n- **Экспертный анализ**: {metrics['expert_analysis_score']:.1%}"
        
        report += f"""

### Статистика
- **Всего файлов**: {metrics['total_files']}
- **Всего строк**: {metrics['total_lines']:,}

### Рекомендации
"""
        
        if metrics['recommendations']:
            for rec in metrics['recommendations']:
                report += f"- {rec}\n"
        else:
            report += "- Контекст оптимален для рефакторинга ✅\n"
        
        return report
    
    def create_context_bundle(self, max_lines: int = 2500) -> Tuple[List[ContextFile], int]:
        """Создает оптимальный набор файлов контекста в пределах лимита строк с поддержкой экспертного анализа."""
        all_files = self.collect_context_files()
        selected_files = []
        total_lines = 0
        
        # ПРИОРИТЕТ 0: Экспертный анализ (наивысший приоритет)
        for file_info in all_files:
            if file_info.priority == 0:
                if total_lines + file_info.estimated_lines <= max_lines:
                    selected_files.append(file_info)
                    total_lines += file_info.estimated_lines
                    print(f"✅ EXPERT: Added {Path(file_info.path).name} (~{file_info.estimated_lines} lines)")
        
        # ПРИОРИТЕТ 1: Архитектура и планы (высокий приоритет)
        for file_info in all_files:
            if file_info.priority == 1:
                if total_lines + file_info.estimated_lines <= max_lines:
                    selected_files.append(file_info)
                    total_lines += file_info.estimated_lines
                    print(f"✅ HIGH: Added {Path(file_info.path).name} (~{file_info.estimated_lines} lines)")
        
        # ПРИОРИТЕТ 2+: Остальные файлы (средний и низкий приоритет)
        for file_info in all_files:
            if file_info.priority >= 2:
                if total_lines + file_info.estimated_lines <= max_lines:
                    selected_files.append(file_info)
                    total_lines += file_info.estimated_lines
                    print(f"✅ MED: Added {Path(file_info.path).name} (~{file_info.estimated_lines} lines)")
        
        return selected_files, total_lines
    
    def generate_context_summary(self, selected_files: List[ContextFile], total_lines: int) -> str:
        """Генерирует сводку контекста для LLM с поддержкой экспертного анализа."""
        summary = f"""# Advanced Context Bundle for {self.module_name} Refactoring

## Target Module
- **File**: `{self.target_module}`
- **Module**: {self.module_name}

## Context Files ({len(selected_files)} files, ~{total_lines} lines)

"""
        
        # Экспертный анализ (приоритет 0)
        expert_files = [f for f in selected_files if f.priority == 0]
        if expert_files:
            summary += "### 🎯 Expert Refactoring Analysis (CRITICAL)\n"
            for file_info in expert_files:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
                if 'expert_analysis_report' in file_info.path:
                    summary += "  - 🚨 **CONTAINS**: Call graph, circular dependencies, external usage, test coverage, characterization tests\n"
                elif 'expert_analysis' in file_info.path and file_info.path.endswith('.json'):
                    summary += "  - 📊 **CONTAINS**: Quality score, risk assessment, detailed analysis data\n"
            summary += "\n"
        
        # Высокий приоритет (архитектура и планы)
        high_priority = [f for f in selected_files if f.priority == 1]
        if high_priority:
            summary += "### 🏗️ High Priority Files (Architecture & Plans)\n"
            for file_info in high_priority:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
            summary += "\n"
        
        # Средний приоритет (анализ проблем)
        medium_priority = [f for f in selected_files if f.priority == 2]
        if medium_priority:
            summary += "### 🔍 Medium Priority Files (Problem Analysis)\n"
            for file_info in medium_priority:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
            summary += "\n"
        
        # Низкий приоритет (документация и детали)
        low_priority = [f for f in selected_files if f.priority >= 3]
        if low_priority:
            summary += "### 📚 Supporting Files (Documentation & Details)\n"
            for file_info in low_priority:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
            summary += "\n"
        
        summary += """## 🎯 Expert Analysis Highlights

"""
        
        # Добавляем ключевые находки из экспертного анализа если есть
        expert_json_files = [f for f in expert_files if f.path.endswith('.json')]
        if expert_json_files:
            try:
                expert_json_path = expert_json_files[0].path
                with open(expert_json_path, 'r', encoding='utf-8') as f:
                    expert_data = json.load(f)
                    
                summary += f"- **Quality Score**: {expert_data.get('analysis_quality_score', 'N/A')}/100\n"
                summary += f"- **Risk Level**: {expert_data.get('risk_assessment', 'N/A').upper()}\n"
                
                recommendations = expert_data.get('recommendations', [])
                if recommendations:
                    summary += "- **Key Recommendations**:\n"
                    for rec in recommendations[:3]:  # Первые 3 рекомендации
                        summary += f"  - {rec}\n"
                summary += "\n"
            except Exception as e:
                summary += f"- Expert data available but could not parse: {e}\n\n"
        
        summary += f"""## Usage Instructions

1. **First message to LLM**: Send the target module file (`{self.target_module}`)
2. **Second message to LLM**: Send this context bundle with the files listed above

## 🚨 Critical Refactoring Focus Areas (Based on Expert Analysis)

- **Circular Dependencies**: Review call graph for cycles that must be resolved first
- **External Usage Impact**: Plan changes carefully due to external dependencies  
- **Test Coverage**: Create characterization tests before refactoring
- **Code Duplication**: Significant savings potential identified
- **Architectural Smells**: Systematic issues requiring attention

## 📋 Refactoring Approach

1. **Phase 1**: Address circular dependencies (CRITICAL)
2. **Phase 2**: Create/run characterization tests for safety
3. **Phase 3**: Apply expert recommendations systematically
4. **Phase 4**: Validate with external callers
5. **Phase 5**: Optimize duplicates and smells

---
*Generated by Advanced Context Collector with Expert Analysis for {self.module_name} module*
"""
        
        return summary
    
    def create_advanced_bundle(self, output_dir: str = "advanced_context_bundle", 
                             interactive: bool = False, include_preview: bool = True, 
                             export_json: bool = False) -> str:
        """Создает продвинутый контекстный набор."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Выбор файлов
        if interactive:
            selected_files = self.interactive_file_selection()
        else:
            selected_files, _ = self.create_context_bundle()
        
        # Анализ качества
        self.analyze_context_quality(selected_files)
        
        # Сохраняем основные файлы
        total_lines = sum(f.estimated_lines for f in selected_files)
        summary = self.generate_context_summary(selected_files, total_lines)
        
        with open(output_path / "CONTEXT_SUMMARY.md", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Отчет о качестве
        quality_report = self.generate_quality_report()
        with open(output_path / "QUALITY_REPORT.md", 'w', encoding='utf-8') as f:
            f.write(quality_report)
        
        # Предварительный просмотр
        if include_preview:
            preview = self.preview_context(selected_files)
            with open(output_path / "CONTEXT_PREVIEW.md", 'w', encoding='utf-8') as f:
                f.write(preview)
        
        # JSON экспорт
        if export_json:
            self.export_context_json(selected_files, output_path / "context_data.json")
        
        # Копируем файлы контекста с жестким контролем размера
        for i, file_info in enumerate(selected_files, 1):
            source_path = Path(file_info.path)
            if source_path.exists():
                dest_name = f"{i:02d}_{source_path.name}"
                dest_path = output_path / dest_name
                
                try:
                    # Для больших JSON файлов всегда ограничиваем размер
                    if file_info.file_type in ['smells', 'duplicates']:
                        max_lines = file_info.estimated_lines
                        with open(source_path, 'r', encoding='utf-8', errors='ignore') as src:
                            lines = []
                            for line_num, line in enumerate(src, 1):
                                lines.append(line)
                                if line_num >= max_lines:
                                    break
                        
                        with open(dest_path, 'w', encoding='utf-8') as dst:
                            dst.writelines(lines)
                            if line_num >= max_lines:
                                dst.write(f"\n\n... [TRUNCATED: File too large, showing only first {max_lines} lines] ...")
                    else:
                        # Копируем файл полностью для архитектурных и планов
                        import shutil
                        shutil.copy2(source_path, dest_path)
                        
                except Exception as e:
                    print(f"Warning: Could not copy {source_path}: {e}")
        
        print(f"\n🎉 Advanced context bundle created: {output_path}")
        print(f"📊 Quality score: {self.quality_metrics.get('coverage_score', 0):.1%}")
        print(f"📁 Files: {len(selected_files)} | Lines: {total_lines:,}")
        
        return str(output_path / "CONTEXT_SUMMARY.md")


def main():
    parser = argparse.ArgumentParser(description="Advanced context collector for module refactoring")
    parser.add_argument("target_module", help="Path to the target module file")
    parser.add_argument("--analysis-dir", help="Analysis results directory")
    parser.add_argument("--output-dir", default="advanced_context_bundle", help="Output directory")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive file selection")
    parser.add_argument("--no-preview", action="store_true", help="Skip context preview generation")
    parser.add_argument("--export-json", action="store_true", help="Export context as JSON")
    parser.add_argument("--max-lines", type=int, default=2500, help="Maximum lines in context")
    
    args = parser.parse_args()
    
    # Автоматически находим папку с результатами анализа
    analysis_dir = args.analysis_dir
    if not analysis_dir:
        temp_collector = AdvancedContextCollector(".", args.target_module)
        analysis_dir = temp_collector.find_analysis_results_dir()
        if not analysis_dir:
            print("❌ Error: Could not find analysis results directory")
            return 1
        print(f"🔍 Auto-detected analysis directory: {analysis_dir}")
    
    # Создаем продвинутый сборщик
    collector = AdvancedContextCollector(analysis_dir, args.target_module)
    
    # Создаем контекстный набор
    summary_path = collector.create_advanced_bundle(
        output_dir=args.output_dir,
        interactive=args.interactive,
        include_preview=not args.no_preview,
        export_json=args.export_json
    )
    
    print(f"\n📋 Summary: {summary_path}")
    print("🚀 Ready for LLM refactoring!")
    
    return 0


if __name__ == "__main__":
    exit(main())