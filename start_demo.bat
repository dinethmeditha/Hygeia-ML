@echo off
title HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo   HYGEIA AI - Patient Health Monitoring System
echo ================================================
echo.
echo Select Startup Mode:
echo [1] Demo Mode (No hardware required - auto-generates vitals)
echo [2] Hardware Mode (Wait for real-time data from HYGEIA Glove)
echo.
set /p mode="Enter 1 or 2: "

echo.
REM --- Clear previous demo session log (keep header only) ---
echo [1/3] Clearing previous session data...
echo Timestamp,Temp,SpO2,Pulse,SimulationMode > "data\demo_session_log.csv"

echo [2/3] Starting AI Server (Flask)...
if "%mode%"=="2" (
    start "HYGEIA Server" cmd /k "set PYTHONIOENCODING=utf-8 && python server.py --hardware"
) else (
    start "HYGEIA Server" cmd /k "set PYTHONIOENCODING=utf-8 && python server.py"
)

echo Waiting for server to initialise...
echo (This may take up to 20 seconds while TensorFlow loads)

:wait_loop
python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

REM --- Clear server buffer so chart starts fresh ---
python -c "import urllib.request; req = urllib.request.Request('http://localhost:5000/api/clear', method='POST'); urllib.request.urlopen(req)" >nul 2>&1

echo [3/3] Opening Web Dashboard...
start "" "http://localhost:5000"

if "%mode%"=="2" (
    echo.
    echo ================================================
    echo   Server is running in HARDWARE MODE.
    echo   Dashboard: http://localhost:5000
    echo   Please power on your ESP32 HYGEIA Glove.
    echo   Close the server window to stop.
    echo ================================================
) else (
    echo.
    echo ================================================
    echo   Server is running in DEMO MODE.
    echo   Dashboard: http://localhost:5000
    echo   Use the Simulation Controller in the UI to change vitals!
    echo   Close the server window to stop.
    echo ================================================
)

pause
