@echo off
title Credit Card Fraud Detection
echo ========================================================
echo   Credit Card Fraud Detection - Random Forest
echo ========================================================
echo Starting server at http://localhost:5500/
echo Opening web browser...
start "" http://localhost:5500/
echo.
echo Press Ctrl+C in this terminal to stop the server.
echo ========================================================
python -m http.server 5500
