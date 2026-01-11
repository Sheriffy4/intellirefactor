#!/usr/bin/env python3
"""
Скрипт для создания исполняемого файла автоматизированного анализатора

Создает standalone исполняемый файл с помощью PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_pyinstaller():
    """Проверяет наличие PyInstaller"""
    try:
        import PyInstaller

        return True
    except ImportError:
        return False


def install_pyinstaller():
    """Устанавливает PyInstaller"""
    print("📦 Установка PyInstaller...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller установлен успешно")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка установки PyInstaller")
        return False


def create_spec_file():
    """Создает spec файл для PyInstaller"""
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['automated_intellirefactor_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('intellirefactor', 'intellirefactor'),
        ('README_AUTOMATED_ANALYZER.md', '.'),
    ],
    hiddenimports=[
        'intellirefactor',
        'intellirefactor.api',
        'intellirefactor.config',
        'intellirefactor.cli',
        'intellirefactor.analysis',
        'intellirefactor.refactoring',
        'intellirefactor.documentation',
        'intellirefactor.visualization',
        'intellirefactor.knowledge',
        'intellirefactor.orchestration',
        'intellirefactor.performance',
        'intellirefactor.plugins',
        'intellirefactor.reports',
        'intellirefactor.safety',
        'intellirefactor.templates',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='IntelliRefactorAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,
)
"""

    with open("analyzer.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)

    print("📄 Spec файл создан: analyzer.spec")


def build_executable():
    """Создает исполняемый файл"""
    print("🔨 Создание исполняемого файла...")

    try:
        # Создаем spec файл
        create_spec_file()

        # Запускаем PyInstaller
        cmd = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "analyzer.spec"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Исполняемый файл создан успешно!")

            # Проверяем результат
            exe_path = Path("dist/IntelliRefactorAnalyzer.exe")
            if exe_path.exists():
                print(f"📁 Исполняемый файл: {exe_path.absolute()}")
                print(f"📏 Размер файла: {exe_path.stat().st_size / 1024 / 1024:.1f} MB")

                # Создаем папку для распространения
                dist_dir = Path("IntelliRefactorAnalyzer_Portable")
                if dist_dir.exists():
                    shutil.rmtree(dist_dir)

                dist_dir.mkdir()

                # Копируем файлы
                shutil.copy2(exe_path, dist_dir / "IntelliRefactorAnalyzer.exe")
                shutil.copy2("README_AUTOMATED_ANALYZER.md", dist_dir / "README.md")

                # Создаем bat файл для запуска
                bat_content = """@echo off
echo Автоматизированный анализатор IntelliRefactor
echo.
IntelliRefactorAnalyzer.exe %*
pause
"""
                with open(dist_dir / "run_analyzer.bat", "w", encoding="utf-8") as f:
                    f.write(bat_content)

                print(f"📦 Портативная версия создана: {dist_dir.absolute()}")
                return True
            else:
                print("❌ Исполняемый файл не найден после сборки")
                return False
        else:
            print("❌ Ошибка создания исполняемого файла:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return False


def create_installer_script():
    """Создает NSIS скрипт для Windows инсталлятора"""
    nsis_script = """!define APP_NAME "IntelliRefactor Analyzer"
!define APP_VERSION "1.0.0"
!define APP_PUBLISHER "IntelliRefactor Team"
!define APP_EXE "IntelliRefactorAnalyzer.exe"

Name "${APP_NAME}"
OutFile "IntelliRefactorAnalyzer_Setup.exe"
InstallDir "$PROGRAMFILES\\${APP_NAME}"
RequestExecutionLevel admin

Page directory
Page instfiles

Section "Install"
    SetOutPath "$INSTDIR"
    File "IntelliRefactorAnalyzer_Portable\\IntelliRefactorAnalyzer.exe"
    File "IntelliRefactorAnalyzer_Portable\\README.md"
    File "IntelliRefactorAnalyzer_Portable\\run_analyzer.bat"
    
    CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}"
    CreateShortCut "$SMPROGRAMS\\${APP_NAME}\\README.lnk" "$INSTDIR\\README.md"
    CreateShortCut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXE}"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
    WriteUninstaller "$INSTDIR\\uninstall.exe"
SectionEnd

Section "Uninstall"
    Delete "$INSTDIR\\${APP_EXE}"
    Delete "$INSTDIR\\README.md"
    Delete "$INSTDIR\\run_analyzer.bat"
    Delete "$INSTDIR\\uninstall.exe"
    RMDir "$INSTDIR"
    
    Delete "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\\${APP_NAME}\\README.lnk"
    RMDir "$SMPROGRAMS\\${APP_NAME}"
    Delete "$DESKTOP\\${APP_NAME}.lnk"
    
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"
SectionEnd
"""

    with open("installer.nsi", "w", encoding="utf-8") as f:
        f.write(nsis_script)

    print("📄 NSIS скрипт создан: installer.nsi")
    print("💡 Для создания инсталлятора запустите: makensis installer.nsi")


def main():
    """Главная функция"""
    print("🏗️ Создание исполняемого файла автоматизированного анализатора IntelliRefactor")
    print("=" * 80)

    # Проверяем наличие основных файлов
    required_files = ["automated_intellirefactor_analyzer.py", "intellirefactor"]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Отсутствуют необходимые файлы:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nУбедитесь, что все файлы находятся в текущей директории")
        return False

    # Проверяем PyInstaller
    if not check_pyinstaller():
        print("📦 PyInstaller не найден")
        if input("Установить PyInstaller? (y/N): ").lower() in ["y", "yes"]:
            if not install_pyinstaller():
                return False
        else:
            print("❌ PyInstaller необходим для создания исполняемого файла")
            return False

    # Создаем исполняемый файл
    if build_executable():
        print("\n🎉 Исполняемый файл создан успешно!")

        # Предлагаем создать инсталлятор
        if input("\nСоздать NSIS скрипт для Windows инсталлятора? (y/N): ").lower() in ["y", "yes"]:
            create_installer_script()

        print("\n📋 Инструкции по использованию:")
        print("1. Запустите IntelliRefactorAnalyzer.exe из папки IntelliRefactorAnalyzer_Portable")
        print("2. Или используйте run_analyzer.bat для удобного запуска")
        print("3. Следуйте инструкциям в README.md")

        return True
    else:
        print("\n❌ Не удалось создать исполняемый файл")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        sys.exit(1)
