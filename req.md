 “Что обязательно нужно разработчику” — для refactor / dedup / decompose

Ниже — минимальный набор, который реально позволяет выполнить работу руками, а не просто “посмотреть отчёт”.
A) Refactor (рефакторинг)

Разработчику нужно:

    Приоритизация (что трогать первым)

    список “hotspots” (файлы/символы) с баллом (severity-weighted)
    причины: сколько issues и каких типов (complexity, smells, duplicates, unused, fanout)

    Точная локация + контекст

    file_path, line_start, line_end
    symbol_name (если есть)
    краткая причина/описание
    желательно: ссылки на “related findings” (в этом же файле)

    Рекомендация и “путь изменений”

    шаги (минимум 2–5)
    risk hints (тесты, обратная совместимость, side effects)
    “что проверить после” (validation criteria)

    Данные для безопасного удаления/рефакторинга

    unused code: где определено + почему считается неиспользуемым + уровень confidence + признаки dynamic usage
    зависимости/хабы: хотя бы fanout/fanin, чтобы понимать impact

Шум, который можно убрать: подробные внутренние метаданные индекса, и слишком подробный лог о dynamic imports (вы уже перевели в DEBUG и добавили фильтр — отлично).
B) Dedup (дедупликация)

Разработчику нужно:

    Группа клонов

    group_id, clone_type, similarity_score
    extraction_strategy, extraction_confidence

    Инстансы

    для каждого: file_path, line_start, line_end (+ опционально LOC/statement_count/nesting)

    Достаточный контекст

    минимум: 1–2 representative snippets на группу (в --full)
    остальным инстансам достаточно координат

Шум: evidence per instance и большие snippets для каждого — это раздувает JSON без реальной пользы.
C) Decompose (декомпозиция)

Разработчику нужно:

    Smells (структурные проблемы)

    smell_type, severity, confidence
    file_path, symbol_name, line_start, line_end
    ключевые метрики, которые привели к smell (например complexity/len/method_count/cohesion)
    рекомендации (2–4 пункта)

    Кластеризация (если применимо)

    список кластеров: какие методы куда, cohesion per cluster
    “unclustered_ratio”
    “extraction recommended” + confidence
    в --full можно давать “extraction plan” и интерфейс компонента (public methods), но без AST-объектов

Шум: повторение одного и того же на уровне “всех методов/всех узлов AST” без ограничений.