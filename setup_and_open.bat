@echo off
echo ==========================================
echo HYGEIA ML - Setup and Open Project
echo ==========================================
echo.
echo Installing required dependencies...
pip install -r requirements.txt
echo.
echo Setup complete. Starting the Flask server...
echo The web dashboard will open in your default browser.
echo.

:: Open the browser in a new window/tab
start http://localhost:5000

:: Run the server
python server.py

pause
