# A Turing pattern generator using repeated sharpen and blur effects #

This is a vibe-coded mess of half-assed attempts at "porting" and "optimizing" a cool script I saw from Patrick Gillespie [here.](https://github.com/patorjk/video-to-turing-pattern/)

It uses ROCm because I'm too poor for an NVIDIA GPU, and this project is dead on arrival because I'm done with this BS.

It's still much, much faster than the script from him, but it's much jankier and with much more room for improvement than his version. The simple fact that this is a successful port to GPU makes up for the poor, vibe-coded codebase.

Also, the Python one isn't really complete.

This is built for Linux (I'm using WSL), and Feel free to use this as a reference on how not to make a port; this is public domain.

# Installation & Use

## Hardware & OS Requirements

* **GPU:** AMD Radeon RX 6000/7000 Series (or ROCm-compatible AMD GPU)
* **OS:** Ubuntu 22.04 LTS (Native or WSL2 under Windows 11)
* **Camera:** USB Webcam supporting MJPEG 1080p stream

## Dependencies

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

Build the live camera application:

```bash
hipcc -O3 -ffast-math turing_live.cpp -o turing_live `pkg-config --cflags --libs opencv4`

```

Build the batch file processor:

```bash
hipcc -O3 -ffast-math turing.cpp -o turing

```
---

## Execution

### 1. Live Camera Stream (`turing_live`)

If using WSL2, attach the camera from Windows PowerShell (Administrator):

```powershell
usbipd list
usbipd attach --wsl --busid <BUSID>

```

Lock hardware parameters to 1080p @ 30 FPS MJPEG to prevent USB bandwidth throttling:

```bash
v4l2-ctl --set-fmt-video=width=1920,height=1080,pixelformat=MJPG --set-parm=30 -d /dev/video0

```

Run the application:

```bash
./turing_live

```

* Press `ESC` inside the render window to close.

### 2. File Video Processor (`turing`)

Process video files directly with VRAM batching:

```bash
./turing <input_path> <output_path> <width> <height> [batch_size]

```

Example:

```bash
./turing input.mp4 output.mp4 1920 1080 64

```
