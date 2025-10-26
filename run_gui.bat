@echo off
echo Starting Track Error Detection GUI...
echo.
cd /d "%~dp0"
venv\Scripts\python.exe inference\detect_gui.py
pause

