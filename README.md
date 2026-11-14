# A GPU port of a Turing pattern generator #

This is a vibe-coded mess of half-assed attempts at porting [Patrick Gillespie's script](https://github.com/patorjk/video-to-turing-pattern/) to my AMD GPU.

It uses HIP/ROCm because I'm too poor for an NVIDIA GPU, and the codebase is really janky but just because this is on the GPU it's still a bajillion times faster than the original script. This project is public domain.

The live camera app, the video file processor and single-image (photo) processing all run natively on **Windows 11** (the webcam is read straight through DirectShow). On Linux they run natively or in Docker with ROCm.

# Installation & Use

## Hardware & OS Requirements

* **GPU:** AMD Radeon RX 6000/7000 Series (or ROCm-compatible AMD GPU)
* **OS (live camera):** Windows 11 with the AMD HIP SDK
* **OS (video files):** Ubuntu 22.04 LTS (Native or WSL2 under Windows 11)
* **Camera:** USB Webcam supporting MJPEG 1080p stream

## Linux

### Quick start (Docker, Linux)

The image bundles the ROCm toolchain, ffmpeg, OpenCV (live target) and the compiled binaries, so the host only needs the GPU driver and Docker.

#### Requirements

* **Docker** with BuildKit.
* **AMD RDNA2/RDNA3 GPU** (RX 6000/7000 Series) and the **host ROCm kernel driver** (`amdgpu-dkms`), so that `/dev/kfd` and `/dev/dri` exist. Check with `ls /dev/kfd /dev/dri`.
* No ROCm SDK, OpenCV or ffmpeg install is needed on the host. The image has them.

> [!NOTE]
> Docker Desktop on Windows/macOS (and WSL2, which exposes the GPU as `/dev/dxg` rather than `/dev/kfd`) can build the image and run the self-test, but cannot pass an AMD GPU through to ROCm, so GPU processing needs a native Linux host. Without a GPU the photo/video checks are skipped with a warning. On Windows, use the native build in the [Windows section](#windows).

#### Build

```bash
docker build -t turing-gpu .                      # headless photo/video (default target)
docker build --target live -t turing-gpu:live .   # live camera (adds OpenCV and V4L2 tools)
```

Build arguments:

* `ROCM_VERSION`: ROCm base image tag (`rocm/dev-ubuntu-22.04:<ROCM_VERSION>`). Default `6.2.4`.
* `HIP_GPU_ARCHS`: space-separated `--offload-arch` list. Default `"gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102"` (RDNA2 and RDNA3), so one image works on most cards. Narrow it for a faster build of a single target:

```bash
docker build --build-arg HIP_GPU_ARCHS=gfx1101 -t turing-gpu .
```

#### Run a photo or video

The current directory is mounted at `/data`, which is the container's working directory, so file paths are relative to it. The first argument after the image name is the command to run, so pass `turing` first.

Photo:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$PWD":/data turing-gpu turing photo.jpg output.png
```

Video:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$PWD":/data turing-gpu turing input.mp4 output.mp4
```

The optional width/height, `--batch` and `--no-audio` arguments go after the file names. See *Execution* under [Linux / WSL2](#linux--wsl2) for what they do.

> [!NOTE]
> `turing-gpu photo.jpg output.png` will not work. The entrypoint runs its arguments as the command (`exec "$@"`), so `turing` has to come first.

#### docker compose

The `turing` service in `docker-compose.yml` mounts the project folder at `/data` and passes the same devices and groups:

```bash
docker compose run --rm turing turing input.mp4 output.mp4
```

To build it for a single target: `HIP_GPU_ARCHS=gfx1101 docker compose build turing`.

#### Live camera (Linux, untested)

```bash
xhost +local:docker
docker compose --profile live run --rm live
```

This needs an X11 display and `/dev/video0`.

> [!WARNING]
> The Linux live camera path is untested by the author. The original live pipeline was verified on Windows only.

#### Self-test

The container runs a self-test on every start, before the requested command:

1. **toolchain**: `turing` is on `PATH` and its shared libraries resolve, `ffmpeg` and `ffprobe` are present, and ffmpeg has the `libx264` encoder.
2. **cli**: bad invocations are rejected with a usage or extension message.
3. **gpu**: AMD GPUs visible to `rocminfo`.
4. **photo** and **video** (only when a GPU is visible): end-to-end renders checked with ffprobe. Photo: output size matches the input, output is grayscale, has contrast, and JPEG output works. Video: size matches, size auto-detected when width/height are omitted, frame rate is preserved exactly at `24/1` and `30000/1001`, frame count is right, source audio is copied, `--no-audio` produces no audio stream, and `--batch` and auto batch both work.

Quiet output prints one line per section and a summary:

```
toolchain  <n> passed, <n> failed, <n> skipped
cli        <n> passed, <n> failed, <n> skipped
gpu        <n> passed, <n> failed, <n> skipped
...
turing self-test: <n> passed, <n> failed, <n> skipped (gpu: <names>)
```

Run it on its own with `-v` to also print every PASS and SKIP line:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined turing-gpu selftest -v
```

Environment variables:

* `TURING_SELFTEST=0`: skip the automatic self-test at startup.
* `TURING_SELFTEST_VERBOSE=1`: same as `-v`.
* `TURING_REQUIRE_GPU=1`: fail instead of skip when no GPU is visible. Use this in CI on GPU runners.

If any check fails, the self-test exits nonzero and the container stops before the command runs.

### Manual setup (without Docker)

#### Linux / WSL2

##### Dependencies

Install the ROCm SDK, C++ compiler, OpenCV development libraries, and Linux video utilities:

```bash
sudo apt update
sudo apt install -y build-essential libopencv-dev v4l-utils pkg-config rocm-hip-sdk

```

ffmpeg must be installed too, since the photo and video modes call `ffmpeg` and `ffprobe`:

```bash
sudo apt install -y ffmpeg
```

Verify GPU detection and the compiler installation:

```bash
hipcc --version
rocminfo

```

If you are having trouble, there is an official guide on how to install ROCm [here.](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

> [!WARNING]
> Though, note that the guide wasn't enough for me, and you will probably have a very specific and stupid problem with your installation that will make the setup process a living hell and drain hours of your precious and finite life.

##### Compilation

Build the batch file processor:

```bash
hipcc -O3 -ffast-math turing.cpp -o turing

```

I recommend passing the offload target of your GPU as well. Find it with `rocminfo` (the `gfx...` name), for example `gfx1101` for an RX 7700 XT:

```bash
hipcc -O3 -ffast-math --offload-arch=<your gfx> turing.cpp -o turing
```

Photo mode is compiled into the same `turing.cpp`, so the commands above already include it.

##### Execution

Process video files directly:

```bash
./turing <input_path> <output_path> [width height] [--batch N] [--no-audio]

```

Examples:

```bash
./turing input.mp4 output.mp4
./turing input.mp4 output.mp4 1280 720 --batch 16

```

The **width and height are optional**. If you leave them out, the output keeps the source size. The size is read with `ffprobe` and corrected for rotation metadata, so a portrait phone video gives the rotated size. An odd auto-detected size is rounded down to even, because libx264 `yuv420p` needs even values. Pass `width height` to rescale: the source is scaled to that size, and it doesn't have to match the input. Explicitly given video sizes must be even, and odd values are an error. The program exits non-zero if ffmpeg fails to decode or encode, so it is safe to use in scripts.

**Batch size** is the number of frames sent to the GPU at once. It is chosen automatically. The program estimates GPU memory per frame (about 10 bytes per pixel on the GPU and 2 bytes per pixel of pinned host memory), uses at most 25% of the free GPU memory reported by HIP, caps pinned host memory at 256 MiB, and caps the result at 64 frames and at the video's frame count. It prints the choice, for example `Batch size: 64 (auto, 12126 MiB GPU memory free)`. Photos always use 1.

Use `--batch N` (or `-b N`, or `--batch=N`; N from 1 to 1024) to override the automatic value:

* Lower it if you run out of GPU memory, or if you share the GPU with a display or other GPU work.
* Raise it only for small frames.

The old fifth positional argument (`turing in.mp4 out.mp4 W H 64`) still works but prints a deprecation note. Use `--batch 64` instead. Giving both a fifth positional value and `--batch` is an error.

The output keeps the **source frame rate**, probed with `ffprobe`, so `24`, `30`, `60` and `29.97` (`30000/1001`) all come out right instead of being forced to 30. It also **copies the source audio** across untouched. Pass `--no-audio` (or `-an`) to write a silent video:

```bash
./turing input.mp4 output.mp4 --no-audio
```

The flags can go anywhere on the command line, and if the source has no audio stream the output simply has none either.

Process a single image the same way as a video file, except **the width and height are read from the image itself**, so you normally don't pass them:

```bash
./turing photo.jpg output.png
./turing photo.jpg output.png 1600 1200      # optional explicit override
```

* **Input** is whatever the *input* extension says it is: `jpg · jpeg · png · bmp · webp · tif · tiff · pgm · ppm`. That is how the binary chooses between photo mode and video mode, so the video invocation above is untouched. Point it at a video with no dimensions and you still get exactly the same usage message.
* **Output** format comes from the *output* extension: `.png` stays grayscale, `.jpg` is JPEG-encoded.
* Everything else matches video mode: sigma 1.0 / 1.8, sharpen 5.0, 100 iterations, grayscale end to end. Photo mode is video mode with batch size 1 and a single still frame pushed through the same `turing_core.h` kernels.

## Windows

The live camera application and the Windows photo/video processor run natively on Windows 11 with the AMD HIP SDK. Docker cannot give the AMD GPU to ROCm on Windows, so the native build is the GPU path here.

> [!NOTE]
> Run it on the physical machine with the AMD GPU. Virtual machines (VirtualBox, Hyper-V, VMware) don't expose the GPU to HIP. In a VM the setup and build still work, but `turing.exe` fails with `HIP Error: no ROCm-capable device is detected`.
>
> If the setup installed anything, open a new terminal before running `run_turing.bat` or `run_live.bat`, so the updated PATH is picked up.

### Quick start (setup script)

The setup script installs the Windows toolchain and builds the apps. Run it from the repo folder in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_win.ps1
```

It does the following. Anything already installed is skipped, so it is safe to rerun:

1. Asks for administrator rights.
2. Installs the Visual Studio 2022 Build Tools with the C++ workload via winget.
3. Installs ffmpeg (`ffmpeg` and `ffprobe`) via winget (`Gyan.FFmpeg`).
4. Installs the AMD HIP SDK silently from `-HipInstaller`. The HIP SDK has no winget package. If it is missing and no installer is given, the script opens [AMD's HIP SDK download page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) and exits. Download the installer, then rerun with `-HipInstaller`.
5. Downloads OpenCV 4.10.0 to `C:\opencv` and sets `OPENCV_DIR` (live camera only).
6. Detects the GPU architecture.
7. Runs `build_win.bat`.

Typical first run:

1. Download the AMD HIP SDK installer from the [HIP SDK page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html). The filename changes between versions, so use the name of the file you actually downloaded.
2. Run the script with the path to that installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_win.ps1 -HipInstaller "$HOME\Downloads\<installer>.exe"
```

Parameters:

* `-Target all|turing|live`: what to build. Default `all`. `turing` builds only the photo/video app and skips OpenCV. `live` builds only the camera app.
* `-GpuArch <gfxNNNN>`: GPU target. Default: auto-detected with `hipInfo`, then from the GPU name, then all RDNA2/RDNA3 targets. For several targets, pass a space-separated list in quotes, for example `-GpuArch "gfx1030 gfx1100"`. For the target names, see the table under [Compilation](#compilation-1).
* `-HipInstaller <path>`: the downloaded AMD HIP SDK installer. Installed silently if the HIP SDK is missing.
* `-OpenCvDir <dir>`: folder to download OpenCV into and set `OPENCV_DIR` from. Default `C:\opencv`.
* `-SkipInstall`: only rebuild. Nothing is installed.
* `-Yes`: don't ask for confirmation.

### Run

Run these from the repo folder:

```powershell
.\run_turing.bat input.mp4 output.mp4
.\run_turing.bat input.mp4 output.mp4 1280 720 --batch 16
.\run_turing.bat photo.jpg output.png
.\run_live.bat           # camera 0
.\run_live.bat 1         # camera 1
```

The `run_*.bat` launchers put the HIP and OpenCV DLL directories on `PATH` for you. The input and output rules for `run_turing.bat` are the same as on Linux (see *Process a single image* under [Linux / WSL2](#linux--wsl2)). The video options (optional width/height, `--batch`, `--no-audio`) are the same too, see *Execution* under [Linux / WSL2](#linux--wsl2). For what the camera app prints on startup and how to check the stream format, see **Execution** under *Manual setup* below.

### Manual setup

#### Dependencies

The setup script above installs these. To do it by hand, install these three things:

1. **[AMD HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)**: provides `hipcc`.
   > [!NOTE]
   > The installer does not always add `hipcc` to `PATH`. The build script detects it automatically from `HIP_PATH` / `C:\Program Files\AMD\ROCm\`.
2. **Visual Studio** with the *Desktop development with C++* workload (needed for the MSVC headers and linker).
3. **OpenCV for Windows**: the prebuilt binary zip/installer. If it is somewhere other than `C:\opencv\build`, set `OPENCV_DIR`.

Verify:

```powershell
& "C:\Program Files\AMD\ROCm\<version>\bin\hipcc.exe" --version
```

#### Compilation

`build_win.bat` builds both apps by default. `.\build_win.bat live` builds only the camera app, and `.\build_win.bat turing` builds only the photo/video app.

Pick the GPU target (`HIP_GPU_ARCH`) for your card:

| GPU | Target |
| --- | --- |
| RX 7900 series | `gfx1100` |
| RX 7800 / 7700 series | `gfx1101` |
| RX 7600 series | `gfx1102` |
| RX 6800 / 6900 / 6950 | `gfx1030` |
| RX 6700 / 6750 | `gfx1031` |
| RX 6600 / 6650 / 6500 / 6400 | `gfx1032` |

##### Live camera application

One command, from the repo folder in any terminal:

```powershell
.\build_win.bat
```

The script auto-detects Visual Studio (`vswhere`), the HIP SDK, and OpenCV, then produces `turing_live.exe`. Override the GPU target with `HIP_GPU_ARCH` if needed. It also accepts a space-separated list for several targets:

```powershell
$env:HIP_GPU_ARCH = "gfx1100"   # default is gfx1101 (RX 7700 XT)
.\build_win.bat
```

```powershell
$env:HIP_GPU_ARCH = "gfx1030 gfx1100"   # several targets in one build
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

##### Photo processor

`build_win.bat` also builds the photo/video binary natively as `turing.exe`:

```powershell
.\build_win.bat            # both targets
.\build_win.bat turing     # only this one (no OpenCV required for it)
```

Either way it only needs `ffmpeg` and `ffprobe` on `PATH`:

```powershell
winget install Gyan.FFmpeg
```

#### Execution

##### 1. Live Camera Stream (`turing_live`)

Run the application:

```powershell
.\run_live.bat
```

The launcher puts the HIP and OpenCV DLL directories on `PATH` for you, so you don't have to. To pick a different camera index or feed frames from a pipe:

```powershell
.\run_live.bat 1                 # second camera
.\run_live.bat - 1920 1080       # raw BGR24 frames from stdin (FFmpeg pipe)
```

On startup it prints what the camera actually negotiated. Confirm it says `1920x1080 @ 30 FPS, FOURCC: MJPG`. If it says `YUY2` instead, you are on uncompressed video and the camera will crawl at ~5 FPS.

> [!NOTE]
> The property order inside `turing_live.cpp` is deliberate: MJPG is applied **last**, after width, height and FPS. OpenCV's DirectShow backend resets its remembered FOURCC after each successful setup, so any `set()` that comes after the FOURCC request silently reverts the stream to RGB24/YUY2, which is what used to cap ingest at 5 FPS. Reordering it is what got the pipeline to a stable 30 FPS.

Press `ESC` inside the render window to close.

Console output shows per-stage FPS so you can see where a bottleneck is:

```
Stream connected: 1920x1080 @ 30 FPS, FOURCC: MJPG
Starting GUI Pipeline. Press ESC in the window to exit.
[Camera Ingest]: 30.0 FPS
[GPU Pipeline]:  30.0 FPS
[GUI Display]:   37.5 FPS
```

##### 2. Photo Processor (`turing`)

```powershell
.\run_turing.bat photo.jpg output.png
```

`run_turing.bat` puts the HIP DLLs on `PATH` for you. The input and output rules are the same as on Linux (see *Process a single image* under [Linux / WSL2](#linux--wsl2)).

---

## Design notes

* Two binaries (`turing`, `turing_live`) share the header-only `turing_core.h`, so each builds as a single hipcc translation unit. The separable convolution takes 24 vs 74 MACs/pixel/iteration. Frames are batched via `blockIdx.z`.
* Each iteration is two LDS-tiled kernels (sharpen, then diffuse), each doing its horizontal and vertical pass in one launch with a clamped halo, so the results match the four-pass version (same edge handling) at about 16 instead of 36 bytes of memory traffic per pixel per iteration. A single fully fused kernel per iteration was tried and was slower: the halo recompute costs more than the saved traffic.
* Grayscale end to end. ffmpeg/ffprobe pipes handle decode and encode. Photo vs video is chosen by the input extension. The frame rate is kept as an exact rational from ffprobe, and source audio is copied by default.
* Iterations: 100 offline, 90 live. The live pipeline has 3 threads (ingest, GPU, display). DirectShow MJPG must be set last, or the stream falls back to YUY2 at ~5 FPS.
* `hip_shim/` is a Windows-only workaround for the HIP wrapper vs MSVC STL 14.51 clash. It is not used in the Linux/Docker build.
* Docker: the ROCm base image is pinned so the toolchain is reproducible. The binaries are fat builds for several RDNA2/3 targets, so one image works on most cards. GPU access is through `/dev/kfd` + `/dev/dri` device passthrough. The self-test runs on startup, so a broken GPU/driver setup fails fast instead of mid-render.
