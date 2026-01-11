#!/usr/bin/env python3
"""
Functional Decomposition Analyzer

Новый анализатор функциональной декомпозиции, использующий систему из ref.md.
Интегрируется с существующим GUI и предоставляет полный пайплайн:
1. Извлечение функциональных блоков
2. Категоризация и кластеризация
3. Планирование консолидации
4. Генерация отчетов
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Добавляем путь к intellirefactor
sys.path.insert(0, str(Path(__file__).parent))

try:
    from intellirefactor.analysis.decomposition import (
        DecompositionAnalyzer,
        DecompositionConfig,
        ApplicationMode,
    )
except ImportError as e:
    print(f"Ошибка импорта модулей декомпозиции: {e}")
    print("Убедитесь, что модули intellirefactor.analysis.decomposition установлены корректно")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Настройка логирования."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Главная функция анализатора."""
    parser = argparse.ArgumentParser(
        description="Анализатор функциональной декомпозиции проекта",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python functional_decomposition_analyzer.py /path/to/project /path/to/output
  python functional_decomposition_analyzer.py /path/to/project /path/to/output --mode plan_only
  python functional_decomposition_analyzer.py /path/to/project /path/to/output --verbose
        """
    )
    
    parser.add_argument(
        "project_root",
        help="Корневая директория проекта для анализа"
    )
    
    parser.add_argument(
        "output_dir",
        help="Директория для сохранения результатов анализа"
    )
    
    parser.add_argument(
        "--mode",
        choices=["analyze_only", "plan_only", "apply_safe", "apply_assisted"],
        default="analyze_only",
        help="Режим работы анализатора (по умолчанию: analyze_only)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод отладочной информации"
    )
    
    parser.add_argument(
        "--config",
        help="Путь к файлу конфигурации (JSON)"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging(args.verbose)
    
    try:
        # Проверка входных параметров
        project_path = Path(args.project_root).resolve()
        if not project_path.exists():
            logger.error(f"Проект не найден: {project_path}")
            return 1
        
        if not project_path.is_dir():
            logger.error(f"Путь к проекту должен быть директорией: {project_path}")
            return 1
        
        output_path = Path(args.output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Начинаем функциональную декомпозицию проекта: {project_path}")
        logger.info(f"Результаты будут сохранены в: {output_path}")
        logger.info(f"Режим работы: {args.mode}")
        
        # Загрузка конфигурации
        config = None
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                # Здесь можно было бы создать DecompositionConfig из JSON
                logger.info(f"Загружена конфигурация из: {config_path}")
            else:
                logger.warning(f"Файл конфигурации не найден: {config_path}")
        
        if not config:
            config = DecompositionConfig.default()
            logger.info("Используется конфигурация по умолчанию")
        
        # Создание анализатора
        analyzer = DecompositionAnalyzer(config)
        
        # Определение режима
        mode = ApplicationMode(args.mode)
        
        # Запуск анализа
        start_time = datetime.now()
        logger.info("Запуск анализа функциональной декомпозиции...")
        
        results = analyzer.analyze_project(
            project_root=str(project_path),
            output_dir=str(output_path),
            mode=mode
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Обработка результатов
        if results["success"]:
            logger.info(f"Анализ успешно завершен за {duration:.1f} секунд")
            
            # Вывод статистики
            stats = results.get("statistics", {})
            logger.info("Статистика анализа:")
            logger.info(f"  - Функциональных блоков: {stats.get('total_blocks', 0)}")
            logger.info(f"  - Возможностей (capabilities): {stats.get('total_capabilities', 0)}")
            logger.info(f"  - Кластеров похожести: {stats.get('total_clusters', 0)}")
            logger.info(f"  - Планов консолидации: {stats.get('total_plans', 0)}")
            logger.info(f"  - Процент разрешения вызовов: {stats.get('resolution_rate', 0):.1%}")
            
            # Вывод топ возможностей
            opportunities = analyzer.get_top_opportunities(5)
            if opportunities:
                logger.info("Топ-5 возможностей для консолидации:")
                for i, opp in enumerate(opportunities, 1):
                    logger.info(
                        f"  {i}. {opp['category']} - {opp['recommendation']} "
                        f"({opp['block_count']} блоков, {opp['avg_similarity']:.2f} похожесть)"
                    )
            
            # Вывод рекомендаций
            recommendations = results.get("recommendations", [])
            if recommendations:
                logger.info("Рекомендации:")
                for rec in recommendations:
                    logger.info(f"  - {rec}")
            
            # Вывод созданных файлов
            report_files = results.get("report_files", {})
            if report_files:
                logger.info("Созданные отчеты:")
                for report_type, file_path in report_files.items():
                    logger.info(f"  - {report_type}: {file_path}")
            
            # Создание итогового отчета
            create_summary_report(output_path, results, duration)
            
            logger.info("Анализ функциональной декомпозиции завершен успешно!")
            return 0
            
        else:
            error_msg = results.get("error", "Неизвестная ошибка")
            logger.error(f"Анализ завершился с ошибкой: {error_msg}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Анализ прерван пользователем")
        return 1
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        return 1


def create_summary_report(output_path: Path, results: dict, duration: float):
    """Создание итогового отчета для GUI."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = output_path / f"FUNCTIONAL_DECOMPOSITION_SUMMARY_{timestamp}.md"
        
        stats = results.get("statistics", {})
        recommendations = results.get("recommendations", [])
        
        content = f"""# Отчет по функциональной декомпозиции

**Дата анализа:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Длительность:** {duration:.1f} секунд

## Основные результаты

- **Функциональных блоков:** {stats.get('total_blocks', 0)}
- **Возможностей (capabilities):** {stats.get('total_capabilities', 0)}
- **Кластеров похожести:** {stats.get('total_clusters', 0)}
- **Планов консолидации:** {stats.get('total_plans', 0)}
- **Процент разрешения вызовов:** {stats.get('resolution_rate', 0):.1%}

## Распределение по категориям

"""
        
        # Добавляем распределение по категориям
        category_dist = stats.get('category_distribution', {})
        if category_dist:
            content += "| Категория | Количество блоков |\n"
            content += "|-----------|-------------------|\n"
            for category, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True):
                content += f"| {category} | {count} |\n"
            content += "\n"
        
        # Добавляем рекомендации по кластерам
        cluster_stats = stats.get('cluster_recommendations', {})
        if cluster_stats:
            content += "## Рекомендации по консолидации\n\n"
            content += f"- **Кандидаты на слияние:** {cluster_stats.get('merge_candidates', 0)}\n"
            content += f"- **Извлечение базы:** {cluster_stats.get('extract_base_candidates', 0)}\n"
            content += f"- **Только обертки:** {cluster_stats.get('wrap_only_candidates', 0)}\n"
            content += f"- **Оставить отдельно:** {cluster_stats.get('keep_separate', 0)}\n\n"
        
        # Добавляем распределение рисков
        risk_stats = stats.get('risk_distribution', {})
        if risk_stats:
            content += "## Распределение по рискам\n\n"
            content += f"- **Низкий риск:** {risk_stats.get('low_risk', 0)}\n"
            content += f"- **Средний риск:** {risk_stats.get('medium_risk', 0)}\n"
            content += f"- **Высокий риск:** {risk_stats.get('high_risk', 0)}\n\n"
        
        # Добавляем рекомендации
        if recommendations:
            content += "## Рекомендации\n\n"
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
            content += "\n"
        
        content += """## Следующие шаги

1. Изучите детальные отчеты в файлах `functional_map.json` и `consolidation_plan.md`
2. Начните с возможностей низкого риска и высокой выгоды
3. Используйте пошаговый подход с валидацией на каждом этапе
4. Рассмотрите создание unified модулей для часто дублируемой функциональности

## Созданные файлы

"""
        
        # Добавляем список созданных файлов
        report_files = results.get("report_files", {})
        if report_files:
            for report_type, file_path in report_files.items():
                content += f"- **{report_type}:** `{Path(file_path).name}`\n"
        
        # Записываем файл
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Создан итоговый отчет: {summary_file}")
        
    except Exception as e:
        print(f"Ошибка создания итогового отчета: {e}")


if __name__ == "__main__":
    sys.exit(main())