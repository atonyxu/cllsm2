#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p build

gcc -static -O3 -DFP_TYPE=float -DUSE_PTHREAD \
  -Iinclude \
  -o build/cllsm2_extract \
  cllsm2_extract.c \
  include/libllsm2/container.c \
  include/libllsm2/frame.c \
  include/libllsm2/dsputils.c \
  include/libllsm2/llsmutils.c \
  include/libllsm2/layer0.c \
  include/libllsm2/layer1.c \
  include/libllsm2/coder.c \
  include/ciglet/ciglet.c \
  include/ciglet/external/fftsg_h.c \
  include/ciglet/external/fast_median.c \
  -lm -lpthread -fopenmp
chmod 777 build/cllsm2_extract
