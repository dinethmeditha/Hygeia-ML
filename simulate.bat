@echo off
echo Starting HYGEIA AI Health Monitoring System Simulator...

echo [1/3] Starting the Flask Web Server...
start cmd /k "python server.py"

echo Waiting 5 seconds for the server to start...
timeout /t 5 /nobreak >nul

echo [2/3] Opening Web Dashboard in Default Browser...
start http://localhost:5000

echo [3/3] Starting the ESP32 Simulator script (streaming data from CSV)...
start cmd /k "python simulator.py"

echo All components started! Close this window when you're done.
pause
