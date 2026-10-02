@echo off
setlocal
REM  Launches turing_live.exe with the DLL paths it needs on PATH.
REM  Usage: run_live.bat          -> opens camera 0
REM         run_live.bat 1        -> opens camera 1
REM         run_live.bat - 1920 1080  -> reads raw BGR frames from stdin

if "%HIP_PATH%"=="" for /d %%D in ("C:\Program Files\AMD\ROCm\*") do set "HIP_PATH=%%D"

if "%OPENCV_DIR%"=="" (
    if exist "C:\opencv\build" (set "OPENCV_DIR=C:\opencv\build") else if exist "%USERPROFILE%\opencv\build" (set "OPENCV_DIR=%USERPROFILE%\opencv\build")
)

set "PATH=%HIP_PATH%\bin;%OPENCV_DIR%\x64\vc16\bin;%PATH%"

if not exist "%~dp0turing_live.exe" (
    echo [ERROR] turing_live.exe not found. Run build_win.bat first.
    exit /b 1
)

"%~dp0turing_live.exe" %*
