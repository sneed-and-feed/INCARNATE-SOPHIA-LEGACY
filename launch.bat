@echo off
chcp 65001 >nul
TITLE SOPHIA 5.0 - INCARNATE LAUNCHER

REM Change to the script's directory
cd /d "%~dp0"

REM Clear Python cache to ensure fresh code load
echo [*] Clearing Python cache...
if exist sophia\__pycache__ rmdir /s /q sophia\__pycache__ 2>nul
if exist sophia\cortex\__pycache__ rmdir /s /q sophia\cortex\__pycache__ 2>nul
if exist sophia\core\__pycache__ rmdir /s /q sophia\core\__pycache__ 2>nul
if exist sophia\tools\__pycache__ rmdir /s /q sophia\tools\__pycache__ 2>nul
if exist tools\__pycache__ rmdir /s /q tools\__pycache__ 2>nul

REM Launch in Windows Terminal for proper color support
echo [*] Launching SOPHIA in Windows Terminal...
wt.exe -d "%CD%" python launch_sophia.py

REM If Windows Terminal not found, fall back to direct launch
if errorlevel 1 (
    echo [!] Windows Terminal not found. Using fallback...
    python launch_sophia.py
    pause
)
