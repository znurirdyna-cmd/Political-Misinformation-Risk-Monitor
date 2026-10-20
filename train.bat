@echo off
REM Train the default model using the Anaconda base environment.
set "PY=C:\Users\znuri\anaconda3\python.exe"
if not exist "%PY%" (
  echo Anaconda python not found at %PY%
  pause
  exit /b 1
)
"%PY%" scripts\train_model.py %*