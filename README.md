# A GPU port of a Turing pattern generator #

This is a vibe-coded mess of half-assed attempts at porting [Patrick Gillespie's script](https://github.com/patorjk/video-to-turing-pattern/) to my AMD GPU.

It uses HIP/ROCm because I'm too poor for an NVIDIA GPU, and the codebase is really janky but just because this is on the GPU it's still a bajillion times faster than the original script. This project is public domain.

The live camera app runs natively on **Windows 11**, so the webcam is read straight through DirectShow. The video file processor still targets Linux/WSL2 with ROCm.

# Installation & Use

## Hardware & OS Requirements

* **GPU:** AMD Radeon RX 6000/7000 Series (or ROCm-compatible AMD GPU)
* **OS (live camera):** Windows 11 with the AMD HIP SDK
* **OS (video files):** Ubuntu 22.04 LTS (Native or WSL2 under Windows 11)
* **Camera:** USB Webcam supporting MJPEG 1080p stream

## Dependencies

### Windows (live camera)

Install these three things:

1. **[AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)** — provides `hipcc`.
   > [!NOTE]
   > The installer does not always add `hipcc` to `PATH`. The build script detects it automatically from `HIP_PATH` / `C:\Program Files\AMD\ROCm\`.
2. **Visual Studio** with the *Desktop development with C++* workload (needed for the MSVC headers and linker).
3. **OpenCV for Windows** — the prebuilt binary zip/installer. If it is somewhere other than `C:\opencv\build`, set `OPENCV_DIR`.

Verify:

```powershell
& "C:\Program Files\AMD\ROCm\<version>\bin\hipcc.exe" --version
```

### Linux / WSL2 (video file processor)

Install the ROCm SDK, C++ compiler, OpenCV development libraries, and Linux video utilities:

```bash
sudo apt update
sudo apt install -y build-essential libopencv-dev v4l-utils pkg-config rocm-hip-sdk

```

Verify GPU detection and the compiler installation:

```bash
hipcc --version
rocminfo

```
If you are having trouble, there is an official guide on how to install ROCm [here.](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
> [!WARNING]
> Though, note that the guide wasn't enough for me, and you will probably have a very specific and stupid problem with your installation that will make the setup process a living hell and drain hours of your precious and finite life.

## Compilation

### Live camera application (Windows)

One command, from the repo folder in any terminal:

```powershell
.\build_win.bat
```

The script auto-detects Visual Studio (`vswhere`), the HIP SDK, and OpenCV, then produces `turing_live.exe`. Override the GPU target with `HIP_GPU_ARCH` if needed:

```powershell
$env:HIP_GPU_ARCH = "gfx1100"   # default is gfx1101 (RX 7700 XT)
.\build_win.bat
```

> [!IMPORTANT]
> `hip_shim/` is not optional. The HIP SDK wrapper force-includes `<cmath>` before HIP's device-math forward declarations, and Microsoft's STL (VS 2026 / MSVC 14.51) now declares `constexpr isgreater/isless/...` there. Clang marks those `__host__ __device__`, which collides with HIP's `__device__` overloads and fails the build. The shim in `hip_shim/` includes the forward declarations first and then chains to the real SDK header. If you upgrade the HIP SDK and this header changes, the shim may need the same treatment again.

Manual build, if you would rather not use the script:

```powershell
# from "x64 Native Tools Command Prompt for VS" (or after calling vcvars64.bat)
hipcc -O3 -ffast-math -std=c++17 -Ihip_shim --offload-arch=gfx1101 `
    turing_live.cpp -o turing_live.exe `
    -I"C:\opencv\build\include" -L"C:\opencv\build\x64\vc16\lib" -lopencv_world500
```

### Video file processor (Linux / WSL2)

Build the batch file processor:

```bash
hipcc -O3 -ffast-math turing.cpp -o turing

```
---

## Execution

### 1. Live Camera Stream (`turing_live`) — Windows

Run the application:

```powershell
.\run_live.bat
```

The launcher puts the HIP and OpenCV DLL directories on `PATH` for you, so you don't have to. To pick a different camera index or feed frames from a pipe:

```powershell
.\run_live.bat 1                 # second camera
.\run_live.bat - 1920 1080       # raw BGR24 frames from stdin (FFmpeg pipe)
```

On startup it prints what the camera actually negotiated — confirm it says `1920x1080 @ 30 FPS, FOURCC: MJPG`. If it says `YUY2` instead, you are on uncompressed video and the camera will crawl at ~5 FPS.

> [!NOTE]
> The property order inside `turing_live.cpp` is deliberate: MJPG is applied **last**, after width, height and FPS. OpenCV's DirectShow backend resets its remembered FOURCC after each successful setup, so any `set()` that comes after the FOURCC request silently reverts the stream to RGB24/YUY2 — which is what used to cap ingest at 5 FPS. Reordering it is what got the pipeline to a stable 30 FPS.

Press `ESC` inside the render window to close.

Console output shows per-stage FPS so you can see where a bottleneck is:

```
Stream connected: 1920x1080 @ 30 FPS, FOURCC: MJPG
Starting GUI Pipeline. Press ESC in the window to exit.
[Camera Ingest]: 30.0 FPS
[GPU Pipeline]:  30.0 FPS
[GUI Display]:   37.5 FPS
```

### 2. File Video Processor (`turing`)

Process video files directly with VRAM batching:

```bash
./turing <input_path> <output_path> <width> <height> [batch_size]

```

Example:

```bash
./turing input.mp4 output.mp4 1920 1080 64

```
