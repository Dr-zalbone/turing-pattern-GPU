@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  build_win.bat - native Windows build of turing_live.exe
REM  Requirements: AMD HIP SDK, VS 2022/2026 (C++ workload), OpenCV
REM ============================================================

REM --- 1. Locate Visual Studio (needed for MSVC headers + link libs) ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Please install Visual Studio with "Desktop development with C++".
    exit /b 1
)
for /f "usebackq tokens=*" %%V in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%V"
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
    echo [ERROR] vcvars64.bat not found under "%VSDIR%".
    exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
echo [INFO] Using Visual Studio: %VSDIR%

REM --- 2. Locate AMD HIP SDK (hipcc) ---
where hipcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    if not "%HIP_PATH%"=="" if exist "%HIP_PATH%\bin\hipcc.bat" set "PATH=%HIP_PATH%\bin;!PATH!"
)
where hipcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    for /d %%D in ("C:\Program Files\AMD\ROCm\*") do (
        if exist "%%D\bin\hipcc.bat" (
            set "HIP_PATH=%%D"
            set "PATH=%%D\bin;!PATH!"
        )
    )
)
where hipcc >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'hipcc' not found. Install the AMD HIP SDK or set HIP_PATH.
    exit /b 1
)
echo [INFO] Using HIP SDK: %HIP_PATH%

REM --- 3. Locate OpenCV ---
if "%OPENCV_DIR%"=="" (
    if exist "C:\opencv\build" (set "OPENCV_DIR=C:\opencv\build") else if exist "%USERPROFILE%\opencv\build" (set "OPENCV_DIR=%USERPROFILE%\opencv\build")
)
if "%OPENCV_DIR%"=="" (
    echo [ERROR] Set OPENCV_DIR to your OpenCV build folder ^(e.g. C:\opencv\build^).
    exit /b 1
)
echo [INFO] Using OPENCV_DIR=%OPENCV_DIR%

set "OPENCV_LIB="
for /d %%V in ("%OPENCV_DIR%\x64\vc*") do (
    for %%F in ("%%V\lib\opencv_world*.lib") do (
        if exist "%%F" (
            set "LIBNAME=%%~nF"
            if not "!LIBNAME:~-1!"=="d" (
                set "OPENCV_LIB=%%F"
                set "OPENCV_LIB_DIR=%%~dpF"
                set "OPENCV_LIB_NAME=%%~nF"
            )
        )
    )
)
if "%OPENCV_LIB%"=="" (
    echo [ERROR] Could not find opencv_world*.lib under %OPENCV_DIR%\x64\
    exit /b 1
)
REM strip trailing backslash (it would escape the closing quote on the -L arg)
if "!OPENCV_LIB_DIR:~-1!"=="\" set "OPENCV_LIB_DIR=!OPENCV_LIB_DIR:~0,-1!"
echo [INFO] Using library: %OPENCV_LIB%

REM --- 4. Compile ---
REM  -Ihip_shim          : fixes HIP-wrapper vs MSVC STL 14.51 (VS 2026) cmath clash
REM  --offload-arch       : gfx1101 = Radeon RX 7700 XT (RDNA3). Change if needed.
if "%HIP_GPU_ARCH%"=="" set "HIP_GPU_ARCH=gfx1101"

echo [INFO] Compiling turing_live.cpp for %HIP_GPU_ARCH% ...
hipcc -O3 -ffast-math -std=c++17 -Wno-nan-infinity-disabled ^
    -I"%~dp0hip_shim" --offload-arch=%HIP_GPU_ARCH% ^
    "%~dp0turing_live.cpp" -o "%~dp0turing_live.exe" ^
    -I"%OPENCV_DIR%\include" -L"%OPENCV_LIB_DIR%" -l%OPENCV_LIB_NAME%

if %ERRORLEVEL% neq 0 (
    echo [FAILED] Compilation error.
    exit /b 1
)
echo [SUCCESS] Built turing_live.exe
echo.
echo Run it with: run_live.bat
exit /b 0
