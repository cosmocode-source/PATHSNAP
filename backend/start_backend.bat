@echo off
setlocal
echo ==============================================================================
echo Starting PathSnap Evaluation API Backend
echo ==============================================================================

cd /d "%~dp0"

REM Activate virtual environment if present
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

REM Run uvicorn server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info
pause
