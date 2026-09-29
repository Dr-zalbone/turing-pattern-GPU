# A turing pattern generator using repeated sharpen and blur effects #
this is a vibe coded mess of half assed attepmts at "porting" and """"optimizing"""" a cool script i saw from Patrick Gillespie [here](https://github.com/pmneila/jsexp)
it uses ROCm because im too poor for an nvidia gpu and this project is dead on arrival because im done with this bs.
it still much *much* faster than the script from him, but it's much jankier and with much more room for improvment than his version. the simple fact this is a successful port to gpu makes up for the poor vibe coded codebase.
also, the python one isn't really complete.

this is built for linux(im using WSL), and an offical guide on how to install ROCm is [here](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
though, note that the guide wasn't enough for me and you will probably have a very specific and stupid problem with your installation that will make the setup process a living hell, and drain hours of your precios and finite life.
feel free to use this as a refrence on how not to make a port, this is public domain.
