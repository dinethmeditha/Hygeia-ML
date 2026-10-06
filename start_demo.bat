@echo off
title HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo   HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo.

echo [1/3] Starting AI Server (Flask)...
start "HYGEIA Server" cmd /k "set PYTHONIOENCODING=utf-8 && python server.py"

echo Waiting for server to initialise (6 seconds)...
timeout /t 6 /nobreak >nul

echo [2/3] Opening Web Dashboard...
start http://localhost:5000

echo [3/3] Starting Patient Data Streamer...
start "HYGEIA Data Streamer" cmd /k "set PYTHONIOENCODING=utf-8 && python data_streamer.py"

echo.
echo ================================================
echo   All components are running!
echo   Dashboard: http://localhost:5000
echo   Close the server/streamer windows to stop.
echo ================================================
pause
