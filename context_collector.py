#!/usr/bin/env python3
"""
Универсальный сборщик контекста для рефакторинга модулей.

Автоматически находит и собирает релевантные файлы анализа для передачи LLM,
учитывая изменяющиеся имена папок и временные метки в именах файлов.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import argparse


@dataclass
class ContextFile:
    """Информация о файле контекста."""
    path: str
    priority: int
    description: str
    estimated_lines: int
    file_type: str  # 'architecture', 'analysis', 'plan', 'problems', 'duplicates', 'docs'


class ContextCollector:
    """Сборщик контекста для рефакторинга."""
    
    def __init__(self, analysis_results_dir: str, target_module_path: str):
        self.analysis_dir = Path(analysis_results_dir)
        self.target_module = Path(target_module_path)
        self.module_name = self.target_module.stem
        self.module_name_upper = self.module_name.upper()
        
        # Паттерны для поиска файлов
        self.file_patterns = {
            'architecture': [
                f"{self.module_name_upper}_ARCHITECTURE_DIAGRAM.md",
                f"{self.module_name_upper}_MODULE_REGISTRY.md",
                f"{self.module_name_upper}_CALL_GRAPH_DETAILED.md",
            ],
            'plan': [
                r"executable_refactoring_plan_\d{8}_\d{6}\.md",
                r"refactoring_plan_\d{8}_\d{6}\.md",
            ],
            'context': [
                f"{self.module_name_upper}_LLM_CONTEXT.md",
                f"{self.module_name_upper}_ANALYSIS_FLOWCHART.md",
            ],
            'opportunities': [
                r"target_file_opportunities_\d{8}_\d{6}\.json",
                r"refactoring_opportunities_attempt_\d+_\d{8}_\d{6}\.json",
            ],
            'smells': [
                r"contextual_architectural_smells_attempt_\d+_\d{8}_\d{6}\.json",
                r"architectural_smells_\d{8}_\d{6}\.json",
            ],
            'duplicates': [
                r"contextual_duplicate_blocks_\d{8}_\d{6}\.json",
                r"duplicate_blocks_\d{8}_\d{6}\.json",
            ],
            'docs': [
                "Requirements.md",
                "Design.md", 
                "Implementation.md",
            ]
        }
    
    def find_analysis_results_dir(self, base_dir: str = ".") -> Optional[str]:
        """Автоматически находит папку с результатами анализа."""
        base_path = Path(base_dir)
        
        # Ищем папки с паттерном analysis_results*
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("analysis_results"):
                return str(item)
        
        return None
    
    def find_files_by_pattern(self, pattern: str, directory: Path) -> List[Path]:
        """Находит файлы по регулярному выражению или точному имени."""
        found_files = []
        
        if directory.exists():
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    # Проверяем точное совпадение имени
                    if file_path.name == pattern:
                        found_files.append(file_path)
                    # Проверяем регулярное выражение
                    elif re.match(pattern, file_path.name):
                        found_files.append(file_path)
        
        return found_files
    
    def get_file_size_estimate(self, file_path: Path) -> int:
        """Оценивает количество строк в файле."""
        try:
            if file_path.suffix == '.json':
                # JSON файлы могут быть очень большими, берем примерную оценку
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 10:
                    return 5000  # Большой JSON
                elif size_mb > 1:
                    return 1000  # Средний JSON
                else:
                    return 200   # Маленький JSON
            else:
                # Для текстовых файлов считаем строки
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return sum(1 for _ in f)
        except Exception:
            return 100  # Дефолтная оценка
    
    def collect_context_files(self) -> List[ContextFile]:
        """Собирает все релевантные файлы контекста."""
        context_files = []
        
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
    
    def create_context_bundle(self, max_lines: int = 2500) -> Tuple[List[ContextFile], int]:
        """Создает оптимальный набор файлов контекста в пределах лимита строк."""
        all_files = self.collect_context_files()
        selected_files = []
        total_lines = 0
        
        # Сначала берем все файлы с приоритетом 1
        for file_info in all_files:
            if file_info.priority == 1:
                if total_lines + file_info.estimated_lines <= max_lines:
                    selected_files.append(file_info)
                    total_lines += file_info.estimated_lines
        
        # Затем добавляем файлы с более низким приоритетом
        for file_info in all_files:
            if file_info.priority > 1:
                if total_lines + file_info.estimated_lines <= max_lines:
                    selected_files.append(file_info)
                    total_lines += file_info.estimated_lines
        
        return selected_files, total_lines
    
    def generate_context_summary(self, selected_files: List[ContextFile], total_lines: int) -> str:
        """Генерирует сводку контекста для LLM."""
        summary = f"""# Context Bundle for {self.module_name} Refactoring

## Target Module
- **File**: `{self.target_module}`
- **Module**: {self.module_name}

## Context Files ({len(selected_files)} files, ~{total_lines} lines)

### High Priority Files (Architecture & Plans)
"""
        
        high_priority = [f for f in selected_files if f.priority == 1]
        for file_info in high_priority:
            summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
            summary += f"  - {file_info.description}\n"
        
        medium_priority = [f for f in selected_files if f.priority == 2]
        if medium_priority:
            summary += "\n### Medium Priority Files (Analysis)\n"
            for file_info in medium_priority:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
        
        low_priority = [f for f in selected_files if f.priority >= 3]
        if low_priority:
            summary += "\n### Supporting Files (Documentation & Details)\n"
            for file_info in low_priority:
                summary += f"- **{file_info.file_type.title()}**: `{Path(file_info.path).name}` (~{file_info.estimated_lines} lines)\n"
                summary += f"  - {file_info.description}\n"
        
        summary += f"""
## Usage Instructions

1. **First message to LLM**: Send the target module file (`{self.target_module}`)
2. **Second message to LLM**: Send this context bundle with the files listed above

## Key Refactoring Focus Areas
- God Class decomposition
- Code duplication elimination  
- Architectural smell fixes
- Method complexity reduction
- SOLID principles application

---
*Generated by ContextCollector for {self.module_name} module*
"""
        
        return summary
    
    def save_context_bundle(self, output_dir: str = "context_bundle") -> str:
        """Сохраняет контекстный набор в папку."""
        selected_files, total_lines = self.create_context_bundle()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Сохраняем сводку
        summary = self.generate_context_summary(selected_files, total_lines)
        summary_path = output_path / "CONTEXT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        # Копируем файлы контекста с контролем размера
        for i, file_info in enumerate(selected_files, 1):
            source_path = Path(file_info.path)
            if source_path.exists():
                dest_name = f"{i:02d}_{source_path.name}"
                dest_path = output_path / dest_name
                
                try:
                    # Для больших JSON файлов ограничиваем размер
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
        
        print(f"Context bundle saved to: {output_path}")
        print(f"Files included: {len(selected_files)}")
        print(f"Estimated total lines: {total_lines}")
        
        return str(summary_path)


def main():
    parser = argparse.ArgumentParser(description="Collect context for module refactoring")
    parser.add_argument("target_module", help="Path to the target module file")
    parser.add_argument("--analysis-dir", help="Analysis results directory (auto-detected if not specified)")
    parser.add_argument("--output-dir", default="context_bundle", help="Output directory for context bundle")
    parser.add_argument("--max-lines", type=int, default=2500, help="Maximum lines in context bundle")
    
    args = parser.parse_args()
    
    # Автоматически находим папку с результатами анализа если не указана
    analysis_dir = args.analysis_dir
    if not analysis_dir:
        collector_temp = ContextCollector(".", args.target_module)
        analysis_dir = collector_temp.find_analysis_results_dir()
        if not analysis_dir:
            print("Error: Could not find analysis results directory")
            print("Please specify --analysis-dir or ensure analysis_results* directory exists")
            return 1
        print(f"Auto-detected analysis directory: {analysis_dir}")
    
    # Создаем сборщик контекста
    collector = ContextCollector(analysis_dir, args.target_module)
    
    # Сохраняем контекстный набор
    summary_path = collector.save_context_bundle(args.output_dir)
    
    print(f"\nContext summary saved to: {summary_path}")
    print("\nNext steps:")
    print(f"1. Send target module to LLM: {args.target_module}")
    print(f"2. Send context bundle from: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())