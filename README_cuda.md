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

`cllsm2_extract_cuda.cu:224-313` 的 `ha_cuda` 通过 `#define llsm_harmonic_analysis ha_cuda` 替换 CPU 版 `llsm_harmonic_analysis`：

- **方法 0 (HMPP, 默认)**: 使用 cuFFT `cufftExecR2C` 在 GPU 上完成整帧 FFT，再回传 CPU 做峰值拾取。每帧的加窗也在 GPU 端完成。
- **方法 1 (CZT)**: 使用自定义 CUDA kernel `czt_frame_kernel` (`cllsm2_extract_cuda.cu:152-167`)，**每个谐波分配一个 GPU 线程**并行计算 CZT，取代 CPU 上逐谐波串行计算。

GPU 端维护了 cuFFT plan 缓存（`cllsm2_extract_cuda.cu:30-47`）和 GPU 内存池（`cllsm2_extract_cuda.cu:50-67`），避免重复申请资源。

### 3. 数据流程

```
音频 + F0 → llsm_analyze → llsm_analyze_harmonics → ha_cuda (GPU)
                              └→ llsm_analyze_noise → 含 FFT 操作 (GPU)
```

`llsm_analyze` 的整体调用栈：
1. `llsm_analyze_harmonics` → `ha_cuda` — GPU 加速
2. `llsm_analyze_noise` → 内部调用 `llsm_compute_spectrogram` → `rdft` → cuFFT — GPU 加速

## 理论加速估算

### FFT 层面
- Ooura FFT 是手写优化的 C 分治 FFT，对典型的 512~4096 点规模，CPU 上约 0.5~5μs
- cuFFT 利用 GPU 数千核心并行计算，同规模下 **5~20x 加速**
- 一条流水线中 FFT 调用次数 = O(帧数 × 每帧 FFT 次数)，数百到数千次

### 谐波分析层面
- **方法 0 (HMPP)**: 全帧 FFT 移至 GPU，每帧节省一次 O(n log n) FFT + O(n) 的 CPU 时间
- **方法 1 (CZT)**: 每帧 nhar 个谐波，CPU 为 O(nhar × win) 串行，GPU 为 O(win) 并行（nhar 个线程同时计算）→ 理论加速约 **nhar / warp_size 倍**（nhar 通常 10~50）

### 整体预期
| 阶段 | CPU 耗时占比 | 预期加速 | 说明 |
|------|-------------|----------|------|
| FFT 运算 | ~30% | 5-20x | cuFFT 替代 Ooura |
| 谐波分析 (CZT) | ~40% | 10-50x | GPU 并行 + cuFFT |
| 其他 (I/O, 峰值拾取, DCT) | ~30% | 1x | 留在 CPU |
| **整体** | **100%** | **5-50x** | 取决于帧数、F0 密度 |

保守估计：对 10 秒 44.1kHz 音频（~670 帧，hop=128），CUDA 加速预期可达到 **10-30x** 实时处理（即从 ~5x 实时提升到 ~50x 实时以上）。

> 注意：实际加速受 GPU 型号、PCIe 传输延迟（每帧需 H2D/D2H 拷贝）和帧数影响。短音频（<2 秒）因传输开销占比高，加速比偏低；长音频（>10 秒）可充分摊销开销，接近理论峰值。
