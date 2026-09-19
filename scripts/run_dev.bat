@echo off
echo ============================================================
echo  SatQuery AI — Development Server Launcher
echo ============================================================
echo.

cd /d "%~dp0..\backend"

:: Create virtual environment if not exists
if not exist "venv" (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo [3/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ============================================================
echo  Starting FastAPI server (MOCK mode) on http://localhost:8000
echo  Swagger docs: http://localhost:8000/docs
echo  Press Ctrl+C to stop
echo ============================================================
echo.

set INFERENCE_MODE=MOCK
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
