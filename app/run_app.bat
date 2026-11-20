@echo off
rem Double-click to start the lane-detection demo; it opens in your browser at http://localhost:8501.
rem Needs: VPN/network access to the cluster, SSH key login, and app\settings.json (copy settings.example.json).
cd /d "%~dp0.."
set "ST=streamlit"
if exist "%USERPROFILE%\.venvs\lane_app\Scripts\streamlit.exe" set "ST=%USERPROFILE%\.venvs\lane_app\Scripts\streamlit.exe"
if not exist "app\settings.json" (
  echo app\settings.json is missing: copy app\settings.example.json to app\settings.json and fill in your login.
  pause
  exit /b 1
)
"%ST%" run app\streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
if errorlevel 1 pause
