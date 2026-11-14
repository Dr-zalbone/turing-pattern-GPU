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

REM  A terminal opened before ffmpeg was installed (e.g. by setup_win.ps1) still has the old
REM  PATH, so also look in winget's install folders (user and machine scope).
where ffprobe >nul 2>nul
if errorlevel 1 call :find_ffmpeg
where ffprobe >nul 2>nul
if errorlevel 1 (
    echo [WARNING] ffmpeg/ffprobe not found. Install with: winget install --id Gyan.FFmpeg -e --source winget
    echo           then open a new terminal.
)

"%~dp0turing.exe" %*
exit /b %ERRORLEVEL%

:find_ffmpeg
for %%R in ("%LOCALAPPDATA%\Microsoft\WinGet" "%ProgramFiles%\WinGet") do (
    if exist "%%~R\Links\ffprobe.exe" (
        set "PATH=%%~R\Links;%PATH%"
        exit /b 0
    )
    for /d %%P in ("%%~R\Packages\Gyan.FFmpeg*") do (
        for /d %%V in ("%%~P\*") do (
            if exist "%%~V\bin\ffprobe.exe" (
                set "PATH=%%~V\bin;%PATH%"
                exit /b 0
            )
        )
    )
)
exit /b 0
