@echo off
title HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo   HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo.
echo Select Startup Mode:
echo [1] Demo Mode (Interactive Simulation Controller)
echo [2] CSV Streaming Mode (Stream patient_vitals.csv)
echo.
set /p mode="Enter 1 or 2: "

echo.
REM --- Clear previous demo session log (keep header only) ---
echo [0/4] Clearing previous session data...
echo Timestamp,Temp,SpO2,Pulse,Mode > "data\demo_session_log.csv"

echo [1/4] Starting AI Server (Flask)...
start "HYGEIA Server" cmd /k "set PYTHONIOENCODING=utf-8 && python server.py"

echo Waiting for server to initialise (7 seconds)...
timeout /t 7 /nobreak >nul

REM --- Clear server buffer so chart starts fresh ---
echo [2/4] Resetting server data buffer...
python -c "import requests; requests.post('http://localhost:5000/api/clear')" 2>nul

echo [3/4] Opening Web Dashboard...
start http://localhost:5000

if "%mode%"=="2" (
    echo [4/4] Starting Patient Data Streamer...
    start "HYGEIA Data Streamer" cmd /k "set PYTHONIOENCODING=utf-8 && python data_streamer.py"
    echo.
    echo ================================================
    echo   All components are running! (CSV Streaming)
    echo   Dashboard: http://localhost:5000
    echo   Close the server/streamer windows to stop.
    echo ================================================
) else (
    echo [4/4] Data Streamer Skipped. Demo mode is active.
    echo.
    echo ================================================
    echo   All components are running! (Interactive Demo)
    echo   Dashboard: http://localhost:5000
    echo   Use the Simulation Controller buttons in the UI!
    echo   Close the server window to stop.
    echo ================================================
)

pause
