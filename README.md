# A Turing pattern generator using repeated sharpen and blur effects #

This is a vibe-coded mess of half-assed attempts at "porting" and "optimizing" a cool script I saw from Patrick Gillespie [here.](https://github.com/patorjk/video-to-turing-pattern/)

It uses ROCm because I'm too poor for an NVIDIA GPU, and this project is dead on arrival because I'm done with this BS.

It's still much, much faster than the script from him, but it's much jankier and with much more room for improvement than his version. The simple fact that this is a successful port to GPU makes up for the poor, vibe-coded codebase.

Also, the Python one isn't really complete.

This is built for Linux (I'm using WSL), and an official guide on how to install ROCm is [here.](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)

Though, note that the guide wasn't enough for me, and you will probably have a very specific and stupid problem with your installation that will make the setup process a living hell and drain hours of your precious and finite life.

Feel free to use this as a reference on how not to make a port; this is public domain.
