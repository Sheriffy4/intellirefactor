@echo off
REM Быстрый запуск сборщика контекста для рефакторинга

if "%1"=="" (
    echo Usage: collect_context.bat ^<target_module_path^> [analysis_dir] [output_dir]
    echo.
    echo Examples:
    echo   collect_context.bat core\bypass\engine\attack_dispatcher.py
    echo   collect_context.bat core\bypass\engine\attack_dispatcher.py analysis_results3
    echo   collect_context.bat core\bypass\engine\attack_dispatcher.py analysis_results3 my_context
    exit /b 1
)

set TARGET_MODULE=%1
set ANALYSIS_DIR=%2
set OUTPUT_DIR=%3

if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=context_bundle

if "%ANALYSIS_DIR%"=="" (
    python context_collector.py "%TARGET_MODULE%" --output-dir "%OUTPUT_DIR%"
) else (
    python context_collector.py "%TARGET_MODULE%" --analysis-dir "%ANALYSIS_DIR%" --output-dir "%OUTPUT_DIR%"
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Context collection completed successfully!
    echo.
    echo 📁 Context bundle location: %OUTPUT_DIR%
    echo 📄 Summary file: %OUTPUT_DIR%\CONTEXT_SUMMARY.md
    echo.
    echo 🚀 Ready for LLM refactoring:
    echo    1. Send target module: %TARGET_MODULE%
    echo    2. Send context files from: %OUTPUT_DIR%
) else (
    echo.
    echo ❌ Context collection failed!
)

pause