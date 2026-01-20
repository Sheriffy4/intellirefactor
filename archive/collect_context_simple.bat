@echo off
chcp 65001 >nul
echo.
echo Context Collector for Module Refactoring (Fixed Version)
echo ========================================================
echo.

if "%1"=="" (
    echo Usage: collect_context_simple.bat ^<module_path^> [mode] [analysis_dir]
    echo.
    echo Modes:
    echo   basic     - Fast basic collection (recommended)
    echo   advanced  - Advanced with quality analysis
    echo.
    echo Examples:
    echo   collect_context_simple.bat core\bypass\engine\attack_dispatcher.py
    echo   collect_context_simple.bat module.py advanced
    echo   collect_context_simple.bat module.py basic analysis_results3
    echo.
    echo Note: Fixed version properly limits large JSON files to prevent bloat
    pause
    exit /b 1
)

set MODULE_PATH=%1
set MODE=%2
set ANALYSIS_DIR=%3

if "%MODE%"=="" set MODE=basic

echo Target module: %MODULE_PATH%
echo Mode: %MODE%
echo.

if "%MODE%"=="advanced" (
    echo Running advanced context collector (with size limits)...
    if "%ANALYSIS_DIR%"=="" (
        python advanced_context_collector.py "%MODULE_PATH%" --export-json
    ) else (
        python advanced_context_collector.py "%MODULE_PATH%" --analysis-dir "%ANALYSIS_DIR%" --export-json
    )
) else (
    echo Running basic context collector (with size limits)...
    if "%ANALYSIS_DIR%"=="" (
        python context_collector.py "%MODULE_PATH%"
    ) else (
        python context_collector.py "%MODULE_PATH%" --analysis-dir "%ANALYSIS_DIR%"
    )
)

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: Context collection completed!
    echo.
    echo Fixed issues:
    echo - Large JSON files are now properly truncated
    echo - Context size is kept within reasonable limits
    echo - All files are optimized for LLM consumption
    echo.
    echo Next steps:
    echo 1. Send target module to LLM: %MODULE_PATH%
    echo 2. Send context files from the created bundle directory
    echo.
) else (
    echo.
    echo ERROR: Context collection failed!
    echo Check that the module path and analysis directory are correct.
)

pause