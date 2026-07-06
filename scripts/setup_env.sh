#!/usr/bin/env bash
# Installs the CUDA Toolkit (nvcc, nsys, ncu) needed to build and profile
# stages 1-3. Does NOT require an NVIDIA GPU to be present/accessible -
# nvcc/ptxas/nsys/ncu install and the compile-only workflow all work
# without a live device; only *running* the produced binaries and
# *capturing* nsys/ncu profiles needs real hardware.
#
# Tested on Ubuntu (apt-based). On the reference environment this pulled
# CUDA 12.4 (nvidia-cuda-toolkit package) plus Nsight Systems/Compute.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "error: this script only supports apt-based distros (Ubuntu/Debian)." >&2
  echo "On other distros, install nvcc + nsys + ncu manually (see" >&2
  echo "https://developer.nvidia.com/cuda-downloads) and re-run this project's" >&2
  echo "cmake configure step." >&2
  exit 1
fi

echo "== Updating apt package lists =="
sudo apt-get update

echo "== Installing nvidia-cuda-toolkit (nvcc, nsys, ncu, cupti, ...) =="
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
  apt-get install -y nvidia-cuda-toolkit

hash -r
echo
echo "== Verifying installation =="
nvcc --version
command -v nsys >/dev/null && echo "nsys: $(command -v nsys)" || echo "WARNING: nsys not found"
command -v ncu  >/dev/null && echo "ncu:  $(command -v ncu)"  || echo "WARNING: ncu not found"

echo
echo "Done. Next: cmake -B build -DCU_EPILOGUE_CUDA_ARCH=70 && cmake --build build -j\$(nproc)"
echo "(set -DCU_EPILOGUE_CUDA_ARCH to match your actual GPU's SM version if not sm_70/Volta)"
