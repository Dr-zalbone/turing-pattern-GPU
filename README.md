# A GPU port of a Turing pattern generator #

This is a vibe-coded mess of half-assed attempts at porting [Patrick Gillespie's script](https://github.com/patorjk/video-to-turing-pattern/) to my AMD GPU.

It uses HIP/ROCm because I'm too poor for an NVIDIA GPU, and the codebase is really janky but just because this is on the GPU it's still a bajillion times faster than the original script. This project is public domain.

The live camera app runs natively on **Windows 11**, so the webcam is read straight through DirectShow. The video file processor still targets Linux/WSL2 with ROCm. Single-image (photo) processing shares the video binary and also runs natively on Windows.

# Installation & Use

## Quick start (Docker, Linux)

The image bundles the ROCm toolchain, ffmpeg, OpenCV (live target) and the compiled binaries, so the host only needs the GPU driver and Docker.

### Requirements

* **Docker** with BuildKit.
* **AMD RDNA2/RDNA3 GPU** (RX 6000/7000 Series) and the **host ROCm kernel driver** (`amdgpu-dkms`), so that `/dev/kfd` and `/dev/dri` exist. Check with `ls /dev/kfd /dev/dri`.
* No ROCm SDK, OpenCV or ffmpeg install is needed on the host. The image has them.

> [!NOTE]
> Docker Desktop on Windows/macOS (and WSL2, which exposes the GPU as `/dev/dxg` rather than `/dev/kfd`) can build the image and run the self-test, but cannot pass an AMD GPU through to ROCm, so GPU processing needs a native Linux host. Without a GPU the photo/video checks are skipped with a warning. On Windows, use the native build below.

### Build

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

### Run a photo or video

The current directory is mounted at `/data`, which is the container's working directory, so file paths are relative to it. The first argument after the image name is the command to run, so pass `turing` first.

Photo:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$PWD":/data turing-gpu turing photo.jpg output.png
```

Video:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --security-opt seccomp=unconfined -v "$PWD":/data turing-gpu turing input.mp4 output.mp4 1920 1080 64
```

> [!NOTE]
> `turing-gpu photo.jpg output.png` will not work. The entrypoint runs its arguments as the command (`exec "$@"`), so `turing` has to come first.

### docker compose

The `turing` service in `docker-compose.yml` mounts the project folder at `/data` and passes the same devices and groups:

```bash
docker compose run --rm turing turing input.mp4 output.mp4 1920 1080 64
```

To build it for a single target: `HIP_GPU_ARCHS=gfx1101 docker compose build turing`.

### Live camera (Linux, untested)

```bash
xhost +local:docker
docker compose --profile live run --rm live
```

This needs an X11 display and `/dev/video0`.

> [!WARNING]
> The Linux live camera path is untested by the author. The original live pipeline was verified on Windows only.

### Self-test

The container runs a self-test on every start, before the requested command:

1. **toolchain**: `turing` is on `PATH` and its shared libraries resolve, `ffmpeg` and `ffprobe` are present, and ffmpeg has the `libx264` encoder.
2. **cli**: bad invocations are rejected with a usage or extension message.
3. **gpu**: AMD GPUs visible to `rocminfo`.
4. **photo** and **video** (only when a GPU is visible): end-to-end renders checked with ffprobe. Photo: output size matches the input, output is grayscale, has contrast, and JPEG output works. Video: size matches, frame rate is preserved exactly at `24/1` and `30000/1001`, frame count is right, source audio is copied, and `--no-audio` produces no audio stream.

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

## Hardware & OS Requirements

* **GPU:** AMD Radeon RX 6000/7000 Series (or ROCm-compatible AMD GPU)
* **OS (live camera):** Windows 11 with the AMD HIP SDK
* **OS (video files):** Ubuntu 22.04 LTS (Native or WSL2 under Windows 11)
* **Camera:** USB Webcam supporting MJPEG 1080p stream

## Manual setup (without Docker)

### Linux / WSL2

#### Dependencies

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

#### Compilation

Build the batch file processor:

```bash
hipcc -O3 -ffast-math turing.cpp -o turing

```

I recommend passing the offload target of your GPU as well. Find it with `rocminfo` (the `gfx...` name), for example `gfx1101` for an RX 7700 XT:

```bash
hipcc -O3 -ffast-math --offload-arch=<your gfx> turing.cpp -o turing
```

Photo mode is compiled into the same `turing.cpp`, so the commands above already include it.

#### Execution

Process video files directly with VRAM batching:

```bash
./turing <input_path> <output_path> <width> <height> [batch_size]

```

Example:

```bash
./turing input.mp4 output.mp4 1920 1080 64

```

The output keeps the **source frame rate**, probed with `ffprobe`, so `24`, `30`, `60` and `29.97` (`30000/1001`) all come out right instead of being forced to 30. It also **copies the source audio** across untouched. Pass `--no-audio` (or `-an`) to write a silent video:

```bash
./turing input.mp4 output.mp4 1920 1080 64 --no-audio
```

The flag can go anywhere on the command line, and if the source has no audio stream the output simply has none either.

Process a single image the same way as a video file, except **the width and height are read from the image itself**, so you normally don't pass them:

```bash
./turing photo.jpg output.png
./turing photo.jpg output.png 1600 1200      # optional explicit override
```

* **Input** is whatever the *input* extension says it is: `jpg · jpeg · png · bmp · webp · tif · tiff · pgm · ppm`. That is how the binary chooses between photo mode and video mode, so the video invocation above is untouched. Point it at a video with no dimensions and you still get exactly the same usage message.
* **Output** format comes from the *output* extension: `.png` stays grayscale, `.jpg` is JPEG-encoded.
* Everything else matches video mode: sigma 1.0 / 1.8, sharpen 5.0, 100 iterations, grayscale end to end. Photo mode is video mode with `batch_size = 1` and a single still frame pushed through the same `turing_core.h` kernels.

### Windows native (live camera)

The live camera application and the Windows photo path run natively on Windows 11. The Windows-native instructions are below.

#### Dependencies

Install these three things:

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

##### Live camera application

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

`run_turing.bat` puts the HIP DLLs on `PATH` for you. The input and output rules are the same as on Linux (see **Photo processor** under *Linux / WSL2* above).

---

## Design notes

* Two binaries (`turing`, `turing_live`) share the header-only `turing_core.h`, so each builds as a single hipcc translation unit. The separable convolution takes 24 vs 74 MACs/pixel/iteration. Frames are batched via `blockIdx.z`.
* Grayscale end to end. ffmpeg/ffprobe pipes handle decode and encode. Photo vs video is chosen by the input extension. The frame rate is kept as an exact rational from ffprobe, and source audio is copied by default.
* Iterations: 100 offline, 90 live. The live pipeline has 3 threads (ingest, GPU, display). DirectShow MJPG must be set last, or the stream falls back to YUY2 at ~5 FPS.
* `hip_shim/` is a Windows-only workaround for the HIP wrapper vs MSVC STL 14.51 clash. It is not used in the Linux/Docker build.
* Docker: the ROCm base image is pinned so the toolchain is reproducible. The binaries are fat builds for several RDNA2/3 targets, so one image works on most cards. GPU access is through `/dev/kfd` + `/dev/dri` device passthrough. The self-test runs on startup, so a broken GPU/driver setup fails fast instead of mid-render.
