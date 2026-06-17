# CUDA 加速说明

## 加速点

### 1. FFT 加速 (cdft / rdft → cuFFT)

`cllsm2_extract_cuda.cu:75-122` 用 cuFFT 替换了 Ooura FFT (`fftsg_h.c`) 的三个核心函数：

| 函数 | 替换方式 | 调用场景 |
|------|----------|----------|
| `cdft` | `cufftExecC2C` | ciglet.c 中的互相关 (`cig_xcorr`)、卷积 (`cig_conv`)、FFT 滤波 |
| `rdft` | `cufftExecR2C` / `cufftExecC2R` | 频谱分析 (`llsm_compute_spectrogram`)、噪声 PSD 估计 |

这些 FFT 调用散布在整个信号处理流水线中——每帧的加窗、滤波、谱估计都依赖它们。

**未加速**: `ddct` (离散余弦变换) 保留 CPU 实现，仅在 coder.c 中编码/解码时各调用 2 次，非性能热点。

### 2. 谐波分析加速 (ha_cuda)

`cllsm2_extract_cuda.cu:244-347` 的 `ha_cuda` 通过 `#define llsm_harmonic_analysis ha_cuda` 替换 CPU 版 `llsm_harmonic_analysis`：

- **方法 0 (HMPP, 默认)**: 使用 cuFFT 批处理 (`cufftPlan1d` batch=nv) 在 GPU 上完成所有帧的 FFT，再一次性回传 CPU 做峰值拾取。每帧的加窗通过自定义 GPU kernel `extract_window_kernel` 完成，仅一次 H2D/D2H 传输。
- **方法 1 (CZT)**: 使用自定义 CUDA kernel `czt_frame_kernel`，**每个谐波分配一个 GPU 线程**并行计算 CZT。

### 3. 已实施的 P0/P1 优化

| 优先级 | 优化 | 说明 |
|--------|------|------|
| **P0** | 批处理 FFT | `extract_window_kernel` 一次性提取所有帧并加窗 → 一次 `cudaMemcpy H2D` → `cufftExecR2C` 批处理 → 一次 `cudaMemcpy D2H`，消除逐帧 H2D/D2H 开销 |
| **P1** | GPU 端加窗 | 帧提取 + Blackman 加窗在 `extract_window_kernel` 中并行完成，无需 CPU 逐帧 `fetch_frame` + `blackman` + 逐点乘 |
| **P1** | 内存池复用 | `gpu_czt` 改用 `gpialloc` 统一内存池，消除每帧 `cudaMalloc`/`cudaFree` 开销 |

GPU 端维护了 cuFFT plan 缓存和 GPU 内存池，避免重复申请资源。

### 4. 数据流程

```
音频 + F0 → llsm_analyze → llsm_analyze_harmonics → ha_cuda (GPU)
                              └→ llsm_analyze_noise → cuFFT FFT (分散调用)
```

`llsm_analyze` 的整体调用栈：
1. `llsm_analyze_harmonics` → `ha_cuda` — **批处理 GPU 加速**
2. `llsm_analyze_noise` → 内部调用 `llsm_compute_spectrogram` → `rdft` → cuFFT — GPU 加速（但目前 rdft 仍为逐帧调用）

## 理论加速估算

### FFT 层面
- Ooura FFT 是手写优化的 C 分治 FFT，对典型的 512~4096 点规模，CPU 上约 0.5~5μs
- cuFFT 利用 GPU 数千核心并行计算，同规模下 **5~20x 加速**
- 一条流水线中 FFT 调用次数 = O(帧数 × 每帧 FFT 次数)，数百到数千次

### 谐波分析层面
- **方法 0 (HMPP)**: 批处理 FFT + GPU 加窗，一次 H2D/D2H 传输 N 帧数据，消除 N-1 次传输和 N-1 次 CPU 加窗
- **方法 1 (CZT)**: 每帧 nhar 个谐波，CPU 为 O(nhar × win) 串行，GPU 为 O(win) 并行（nhar 个线程同时计算）→ 理论加速约 **nhar × win / (win + nhar)** 倍

### 整体预期
| 阶段 | CPU 耗时占比 | 预期加速 | 说明 |
|------|-------------|----------|------|
| FFT 运算 | ~30% | 5-20x | cuFFT 替代 Ooura |
| 谐波分析 (HMPP) | ~40% | 10-50x | 批处理 + GPU 加窗 |
| 其他 (I/O, 峰值拾取, DCT) | ~30% | 1x | 留在 CPU |
| **整体** | **100%** | **10-50x** | 取决于帧数、F0 密度 |

### 场景分析

| 场景 | 帧数 (hop=128) | PCIe 传输次数 (优化前) | PCIe 传输次数 (优化后) | 传输开销节省 |
|------|---------------|----------------------|----------------------|-------------|
| 1 条 8s 人声 | ~2756 帧 | ~2756 次 H2D + ~2756 次 D2H | **1 次 H2D + 1 次 D2H** | **~99.96%** |
| 1000 条 8s 人声 | ~2756 帧 × 1000 | ~275 万次 | ~1000 次 (每条 1 次) | **~99.96%** |
| 1 条 4min 人声 | ~82687 帧 | ~82687 次 | **1 次** | **~99.999%** |
| 30 条 4min 人声 | ~82687 帧 × 30 | ~248 万次 | ~30 次 | **~99.999%** |

> 注意：实际加速受 GPU 型号和 PCIe 带宽影响。批处理优化对长音频效果最为显著，因为总线传输时间被充分摊销。短音频（<2 秒）仍能从单次批处理中受益。
