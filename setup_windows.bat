@echo off
setlocal
cd /d "%~dp0"
if not exist ".bridge-venv\Scripts\python.exe" (
  py -3.12 -m venv ".bridge-venv" || goto :failed
)
".bridge-venv\Scripts\python.exe" -m pip install -r requirements-bridge.txt || goto :failed
if not exist ".bridge.env" (
  ".bridge-venv\Scripts\python.exe" configure_bridge.py || goto :failed
)
echo Setup complete. Start the connected MT5 terminal and double-click start_mt5_bridge.bat.
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.12 with the py launcher and check the output above.
pause
exit /b 1
