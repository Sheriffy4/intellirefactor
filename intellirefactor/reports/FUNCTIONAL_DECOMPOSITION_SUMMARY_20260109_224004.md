# Отчет по функциональной декомпозиции

**Дата анализа:** 2026-01-09 22:40:04
**Длительность:** 1697.0 секунд

## Основные результаты

- **Функциональных блоков:** 1968
- **Возможностей (capabilities):** 15
- **Кластеров похожести:** 14
- **Планов консолидации:** 0
- **Процент разрешения вызовов:** 12.7%

## Распределение по категориям

| Категория | Количество блоков |
|-----------|-------------------|
| parsing:regex | 1009 |
| telemetry:logging | 423 |
| persistence:generic | 165 |
| factory:creation | 93 |
| serialization:json | 80 |
| concurrency:threading | 65 |
| unknown:generic | 24 |
| orchestration:execution | 23 |
| validation:generic | 22 |
| presentation:formatting | 20 |
| parsing:generic | 17 |
| configuration:initialization | 10 |
| domain:strategy | 7 |
| configuration:settings | 4 |
| serialization:yaml | 3 |
| transformation:generic | 1 |
| caching:storage | 1 |
| domain:attack | 1 |

## Рекомендации по консолидации

- **Кандидаты на слияние:** 12
- **Извлечение базы:** 2
- **Только обертки:** 0
- **Оставить отдельно:** 0

## Распределение по рискам

- **Низкий риск:** 13
- **Средний риск:** 1
- **Высокий риск:** 0

## Рекомендации

1. Found 11 high-priority, low-risk consolidation opportunities
2. Consider refactoring 74 highly complex blocks (complexity > 15)
3. Focus on categories with most duplication: parsing:regex, telemetry:logging
4. Low call resolution rate (12.7%) - consider improving import analysis

## Следующие шаги

1. Изучите детальные отчеты в файлах `functional_map.json` и `consolidation_plan.md`
2. Начните с возможностей низкого риска и высокой выгоды
3. Используйте пошаговый подход с валидацией на каждом этапе
4. Рассмотрите создание unified модулей для часто дублируемой функциональности

## Созданные файлы

- **json:** `functional_map_20260109_224004.json`
- **catalog:** `catalog_20260109_224004.md`
- **plan:** `consolidation_plan_20260109_224004.md`
- **diagram:** `functional_graph_20260109_224004.mmd`
- **summary:** `summary_20260109_224004.md`
