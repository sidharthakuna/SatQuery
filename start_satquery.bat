@echo off
title SatQuery AI — System Launcher
echo ============================================================
echo   SatQuery AI — Launching Backend and Frontend
echo ============================================================
echo.

:: Start Backend in a separate window
echo [1/2] Launching FastAPI Backend on http://localhost:8000 ...
start "SatQuery AI — Backend Server" cmd /k "cd /d %~dp0backend && python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

:: Start Frontend in a separate window
echo [2/2] Launching Vite Frontend on http://localhost:5173 ...
start "SatQuery AI — Frontend Dev Server" cmd /k "cd /d %~dp0frontend && npm run dev -- --host 127.0.0.1 --port 5173"

:: Wait 3 seconds and open browser
timeout /t 3 /nobreak >nul
echo.
echo Opening browser to http://localhost:5173 ...
start http://localhost:5173
start http://localhost:8000/docs

echo.
echo Both servers are running!
echo Frontend: http://localhost:5173
echo Backend API Docs: http://localhost:8000/docs
echo ============================================================
