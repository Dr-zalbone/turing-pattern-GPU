@echo off
setlocal
REM  Launches turing.exe with the HIP DLL paths it needs on PATH.
REM  Usage:
REM    run_turing.bat <input> <output> <width> <height> [batch_size]   video
REM    run_turing.bat <input> <output>                                  photo
REM  e.g.
REM    run_turing.bat input.mp4 output.mp4 1920 1080 64
REM    run_turing.bat photo.jpg output.png

if "%HIP_PATH%"=="" for /d %%D in ("C:\Program Files\AMD\ROCm\*") do set "HIP_PATH=%%D"
set "PATH=%HIP_PATH%\bin;%PATH%"

if not exist "%~dp0turing.exe" (
    echo [ERROR] turing.exe not found. Run build_win.bat first.
    exit /b 1
)

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo [WARNING] ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg
)

"%~dp0turing.exe" %*
exit /b %ERRORLEVEL%
