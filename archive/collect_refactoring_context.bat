@echo off
REM Универсальный сборщик контекста для рефакторинга модулей
REM Поддерживает как базовый, так и продвинутый режимы

setlocal enabledelayedexpansion

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                Context Collector для рефакторинга            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

if "%1"=="" (
    echo ❌ Использование: collect_refactoring_context.bat ^<module_path^> [options]
    echo.
    echo 📋 Опции:
    echo    --basic          Базовый режим (быстро)
    echo    --advanced       Продвинутый режим (с анализом качества)
    echo    --interactive    Интерактивный выбор файлов
    echo    --json           Экспорт в JSON формат
    echo    --analysis-dir   Папка с результатами анализа
    echo    --output-dir     Папка для сохранения контекста
    echo.
    echo 🚀 Примеры:
    echo    collect_refactoring_context.bat core\bypass\engine\attack_dispatcher.py
    echo    collect_refactoring_context.bat module.py --advanced --json
    echo    collect_refactoring_context.bat module.py --interactive --analysis-dir analysis_results3
    echo.
    pause
    exit /b 1
)

set MODULE_PATH=%1
set MODE=basic
set INTERACTIVE=
set EXPORT_JSON=
set ANALYSIS_DIR=
set OUTPUT_DIR=
set EXTRA_ARGS=

REM Парсинг аргументов
:parse_args
shift
if "%1"=="" goto :args_done

if "%1"=="--basic" (
    set MODE=basic
    goto :parse_args
)
if "%1"=="--advanced" (
    set MODE=advanced
    goto :parse_args
)
if "%1"=="--interactive" (
    set INTERACTIVE=--interactive
    goto :parse_args
)
if "%1"=="--json" (
    set EXPORT_JSON=--export-json
    goto :parse_args
)
if "%1"=="--analysis-dir" (
    shift
    set ANALYSIS_DIR=--analysis-dir "%1"
    goto :parse_args
)
if "%1"=="--output-dir" (
    shift
    set OUTPUT_DIR=--output-dir "%1"
    goto :parse_args
)

REM Неизвестный аргумент
echo ⚠️  Неизвестный аргумент: %1
goto :parse_args

:args_done

echo 🎯 Модуль: %MODULE_PATH%
echo 🔧 Режим: %MODE%

if "%MODE%"=="basic" (
    echo.
    echo 🚀 Запуск базового сборщика контекста...
    python context_collector.py "%MODULE_PATH%" %ANALYSIS_DIR% %OUTPUT_DIR%
    set RESULT=%ERRORLEVEL%
) else (
    echo.
    echo 🚀 Запуск продвинутого сборщика контекста...
    python advanced_context_collector.py "%MODULE_PATH%" %ANALYSIS_DIR% %OUTPUT_DIR% %INTERACTIVE% %EXPORT_JSON%
    set RESULT=%ERRORLEVEL%
)

echo.
if %RESULT% EQU 0 (
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║                    ✅ УСПЕШНО ЗАВЕРШЕНО                      ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo 📁 Контекст собран и готов для передачи LLM
    echo.
    echo 📋 Следующие шаги:
    echo    1️⃣  Отправьте LLM сам модуль: %MODULE_PATH%
    echo    2️⃣  Отправьте LLM файлы из созданной папки контекста
    echo.
    echo 💡 Совет: Используйте CONTEXT_SUMMARY.md для инструкций
    
    if "%MODE%"=="advanced" (
        echo 📊 Проверьте QUALITY_REPORT.md для оценки качества контекста
    )
) else (
    echo ╔══════════════════════════════════════════════════════════════╗
    echo ║                      ❌ ОШИБКА                               ║
    echo ╚══════════════════════════════════════════════════════════════╝
    echo.
    echo 🔍 Возможные причины:
    echo    - Модуль не найден: %MODULE_PATH%
    echo    - Папка анализа не найдена (укажите --analysis-dir)
    echo    - Недостаточно прав доступа
    echo    - Python не установлен или недоступен
)

echo.
pause