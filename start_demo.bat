@echo off
title HYGEIA AI - Patient Health Monitoring System
color 0A
echo.
echo  =====================================================
echo    HYGEIA AI - Patient Health Monitoring System
echo  =====================================================
echo.
echo  Select Startup Mode:
echo  [1]  Demo Mode      - No hardware needed, auto-generates vitals
echo  [2]  Hardware Mode  - Live data from HYGEIA Glove over WiFi
echo.
set /p mode="  Enter 1 or 2: "

echo.

REM --- Clear previous demo session log (keep header only) ---
echo  [1/4] Clearing previous session data...
echo Timestamp,Temp,SpO2,Pulse,SimulationMode > "data\demo_session_log.csv"

REM ─────────────────────────────────────────────────────────────
REM  HARDWARE MODE – instructions shown BEFORE server starts
REM ─────────────────────────────────────────────────────────────
if "%mode%"=="2" (
    echo.
    echo  =====================================================
    echo    HARDWARE MODE – Follow these steps:
    echo  =====================================================
    echo.
    echo  STEP 1  Power on your ESP32 HYGEIA Glove.
    echo          The OLED will show:
    echo            WiFi: HYGEIA-Monitor
    echo            Pass: hygeia123
    echo            Open: http://192.168.4.1
    echo.
    echo  STEP 2  On THIS laptop, go to:
    echo            Settings ^> WiFi ^> Connect to "HYGEIA-Monitor"
    echo            Password: hygeia123
    echo.
    echo  STEP 3  Press any key here once your laptop is
    echo          connected to the HYGEIA-Monitor hotspot.
    echo.
    pause
    echo.
)

REM --- Start Flask server ---
echo  [2/4] Starting AI Server (Flask)...
if "%mode%"=="2" (
    start "HYGEIA Server" cmd /k "color 0A && set PYTHONIOENCODING=utf-8 && python server.py --hardware"
) else (
    start "HYGEIA Server" cmd /k "color 0A && set PYTHONIOENCODING=utf-8 && python server.py"
)

echo  [3/4] Waiting for server to initialise...
echo        (May take up to 20 seconds while TensorFlow loads)

:wait_loop
python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

REM --- Clear server buffer so chart starts fresh ---
python -c "import urllib.request; req = urllib.request.Request('http://localhost:5000/api/clear', method='POST'); urllib.request.urlopen(req)" >nul 2>&1

echo  [4/4] Opening Web Dashboard...
start "" "http://localhost:5000"

REM ─────────────────────────────────────────────────────────────
REM  HARDWARE MODE – launch the ESP32 bridge in a new window
REM ─────────────────────────────────────────────────────────────
if "%mode%"=="2" (
    start "HYGEIA Bridge" cmd /k "color 0B && set PYTHONIOENCODING=utf-8 && python hardware\esp32_bridge.py"
    echo.
    echo  =====================================================
    echo    HARDWARE MODE is running!
    echo.
    echo    Dashboard   : http://localhost:5000
    echo    ESP32 UI    : http://192.168.4.1
    echo.
    echo    Two windows are open:
    echo      "HYGEIA Server"  - Flask AI backend
    echo      "HYGEIA Bridge"  - Forwards glove data to server
    echo.
    echo    Place your finger on the MAX30102 sensor.
    echo    Data appears on the dashboard within ~15 seconds.
    echo.
    echo    Close BOTH windows to stop the system.
    echo  =====================================================
) else (
    echo.
    echo  =====================================================
    echo    DEMO MODE is running!
    echo.
    echo    Dashboard   : http://localhost:5000
    echo    Use the Simulation Controller in the UI to
    echo    change vitals (Normal / Critical SpO2, etc.)
    echo.
    echo    Close the "HYGEIA Server" window to stop.
    echo  =====================================================
)

echo.
pause
