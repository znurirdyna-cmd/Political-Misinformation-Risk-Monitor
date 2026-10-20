@echo off
REM Launch the Political Misinformation Risk Monitoring System (Anaconda base env).
set "PY=C:\Users\znuri\anaconda3\python.exe"
if not exist "%PY%" (
  echo Anaconda python not found at %PY%
  echo Install/update the path in run.bat or use:  streamlit run app.py
  pause
  exit /b 1
)
echo Starting Political Misinformation Risk Monitoring on http://localhost:8501 ...
"%PY%" -m streamlit run app.py