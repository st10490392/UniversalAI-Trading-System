@echo off
setlocal
cd /d "%~dp0"
if not exist ".bridge-venv\Scripts\python.exe" call setup_windows.bat
if errorlevel 1 exit /b 1
if not exist ".bridge.env" call setup_windows.bat
if errorlevel 1 exit /b 1
".bridge-venv\Scripts\python.exe" paper_mt5_bridge.py
if errorlevel 1 pause
