@echo off
REM Expert Analysis Runner for Windows
REM Usage: run_expert_analysis.bat <project_path> <target_file>

if "%~2"=="" (
    echo Usage: run_expert_analysis.bat ^<project_path^> ^<target_file^>
    echo Example: run_expert_analysis.bat "C:\Intel\recon" "C:\Intel\recon\core\bypass\engine\attack_dispatcher.py"
    exit /b 1
)

echo Running Expert Analysis...
echo Project: %~1
echo Target: %~2

python run_expert_analysis.py "%~1" "%~2" --detailed

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Analysis failed. Please check the error messages above.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Analysis completed successfully!
echo Check the expert_analysis_output folder for results.
pause