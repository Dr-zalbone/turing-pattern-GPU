# syntax=docker/dockerfile:1
#
# turing-pattern-gpu: HIP/ROCm build of the Turing pattern renderer.
#
# Stages:
#   base    Shared parent: ROCm dev image + ffmpeg, turing sources copied in.
#   live    Adds OpenCV + V4L2 tools; builds turing_live and turing.
#           Build with:  docker build --target live -t turing-gpu:live .
#   turing  Headless batch renderer. This is the LAST stage, so it is the
#           default target:  docker build -t turing-gpu:latest .
#
# Build args:
#   ROCM_VERSION   ROCm base image tag (default 6.2.4).
#   HIP_GPU_ARCHS  Space-separated --offload-arch list. Each stage re-declares it.
#
# hip_shim/ is a Windows/MSVC-only workaround and is not used in this image.

ARG ROCM_VERSION=6.2.4
FROM rocm/dev-ubuntu-22.04:${ROCM_VERSION} AS base

LABEL org.opencontainers.image.title="turing-pattern-gpu" \
      org.opencontainers.image.description="Turing pattern reaction-diffusion renderer for AMD ROCm GPUs (HIP)" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ARG HIP_GPU_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102"
ENV PATH=/opt/rocm/bin:/opt/turing/bin:$PATH

WORKDIR /opt/turing/src
COPY turing.cpp turing_core.h ./

# ---------------------------------------------------------------------------
FROM base AS live
ARG HIP_GPU_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libopencv-dev pkg-config v4l-utils \
 && rm -rf /var/lib/apt/lists/*

COPY turing_live.cpp ./
RUN mkdir -p /opt/turing/bin \
 && hipcc -O3 -ffast-math -std=c++17 -Wno-unused-result \
      $(for a in $HIP_GPU_ARCHS; do printf -- '--offload-arch=%s ' "$a"; done) \
      turing_live.cpp -o /opt/turing/bin/turing_live \
      $(pkg-config --cflags --libs opencv4) \
 && hipcc -O3 -ffast-math -std=c++17 -Wno-unused-result \
      $(for a in $HIP_GPU_ARCHS; do printf -- '--offload-arch=%s ' "$a"; done) \
      turing.cpp -o /opt/turing/bin/turing

COPY docker/selftest.sh /opt/turing/selftest.sh
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh \
 && sed -i 's/\r$//' /opt/turing/selftest.sh \
 && chmod +x /opt/turing/selftest.sh

WORKDIR /data
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["turing_live"]

# ---------------------------------------------------------------------------
# Default target. Keep this stage last.
FROM base AS turing
ARG HIP_GPU_ARCHS="gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102"

RUN mkdir -p /opt/turing/bin \
 && hipcc -O3 -ffast-math -std=c++17 -Wno-unused-result \
      $(for a in $HIP_GPU_ARCHS; do printf -- '--offload-arch=%s ' "$a"; done) \
      turing.cpp -o /opt/turing/bin/turing

COPY docker/selftest.sh /opt/turing/selftest.sh
COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
 && chmod +x /usr/local/bin/docker-entrypoint.sh \
 && sed -i 's/\r$//' /opt/turing/selftest.sh \
 && chmod +x /opt/turing/selftest.sh

WORKDIR /data
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["turing"]
