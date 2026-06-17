# cllsm2_extract 使用指南

## 概述

`cllsm2_extract` 从语音音频中提取 LLSM2（Low Level Speech Model v2）参数。输出为 72 维参数向量（含 60 阶频谱 + 5 阶非周期性 + 3 个基础维度），每帧一行 CSV。

提供两个版本：
- `cllsm2_extract.exe` — CPU（MSVC `/O2 /MT`）
- `cllsm2_extract_cuda.exe` — CUDA 加速（需要 NVIDIA GPU + cuFFT/cuBLAS DLL）

## 编译

### CPU 版

```cmd
cl /O2 /MT /DFP_TYPE=float /Iinclude /Febuild\cllsm2_extract.exe ^
  cllsm2_extract.c ^
  include\libllsm2\container.c ^
  include\libllsm2\frame.c ^
  include\libllsm2\dsputils.c ^
  include\libllsm2\llsmutils.c ^
  include\libllsm2\layer0.c ^
  include\libllsm2\layer1.c ^
  include\libllsm2\coder.c ^
  include\ciglet\ciglet.c ^
  include\ciglet\external\fftsg_h.c ^
  include\ciglet\external\fast_median.c
```

### CUDA 版

```cmd
nvcc -O3 -allow-unsupported-compiler -DFP_TYPE=float ^
  -gencode arch=compute_75,code=sm_75 ^
  -gencode arch=compute_86,code=sm_86 ^
  -gencode arch=compute_89,code=sm_89 ^
  -gencode arch=compute_120,code=sm_120 ^
  -I include ^
  -o build\cllsm2_extract_cuda.exe ^
  cllsm2_extract_cuda.cu ^
  -cudart static ^
  -lcufft -lcublas -lcublasLt
```

## 输入格式

### 音频文件 (`*.f32`)

纯 float32 裸采样，无头信息。采样率由命令行指定（不嵌入文件）。
- 单声道
- 范围归一化到 [-1, 1]
- 可用 Python 生成：`data.astype('float32').tofile('audio.f32')`

### F0 文件 (`*.csv`)

每行一个浮点数，表示对应帧的基频（Hz）。无声段为 0、负值或 NaN。可包含逗号（仅第一列被读取）。

可用 Python 生成：`numpy.savetxt('f0.csv', f0, fmt='%.1f')`

## 命令行参数

```
cllsm2_extract.exe audio.f32 f0.csv out72.csv sample_rate hop_samples nfft maxnhar maxnhar_e npsd nchannel chanfreq0 chanfreq1 chanfreq2 order_spec order_bap rel_winsize lip_radius f0_refine hm_method frames
```

| # | 参数 | 类型 | 默认值 | 说明 |
|---|------|------|--------|------|
| 1 | `audio.f32` | 输入路径 | — | float32 裸音频 |
| 2 | `f0.csv` | 输入路径 | — | F0 曲线（Hz），每行一帧 |
| 3 | `out72.csv` | 输出路径 | — | 72 维参数 CSV |
| 4 | `sample_rate` | int | 44100 | 音频采样率 (Hz) |
| 5 | `hop_samples` | int | 128 | 帧移（采样点） |
| 6 | `nfft` | int | 2048 | FFT 大小，影响层 1 转换 |
| 7 | `maxnhar` | int | 100 | 最大谐波数 |
| 8 | `maxnhar_e` | int | 5 | 噪声包络最大谐波数 |
| 9 | `npsd` | int | 128 | 噪声 PSD 向量大小 |
| 10 | `nchannel` | int | 4 | 噪声建模通道数 |
| 11 | `chanfreq0` | float | 2000 | 通道 1 频率 (Hz) |
| 12 | `chanfreq1` | float | 4000 | 通道 2 频率 (Hz) |
| 13 | `chanfreq2` | float | 8000 | 通道 3 频率 (Hz) |
| 14 | `order_spec` | int | 60 | 频谱编码阶数（输出维度的一部分） |
| 15 | `order_bap` | int | 5 | 非周期性编码阶数 |
| 16 | `rel_winsize` | float | 4.0 | 窗口大小与基频周期的比值 |
| 17 | `lip_radius` | float | 1.5 | 唇辐射半径 (cm) |
| 18 | `f0_refine` | int | 1 | F0 精炼（0=关，1=开） |
| 19 | `hm_method` | int | 1 | 谐波分析方法（0=峰值拾取 HMPP，1=CZT） |
| 20 | `frames` | int | 0 | 处理帧数上限（0=全部） |

## 典型用法

### 基础用法

```powershell
# 一条 8s 音频，标准参数
cllsm2_extract.exe audio.f32 f0.csv out72.csv 44100 128 2048 100 5 128 4 2000 4000 8000 60 5 4.0 1.5 1 1 0
```

### 配合 WORLD（D4C+Harvest）提取 F0

```powershell
python -c @"
import pyworld
import numpy as np
import soundfile as sf

x, sr = sf.read('speech.wav')
x = x.astype(np.float64)
f0, t = pyworld.harvest(x, sr)
f0 = f0.astype(np.float32)
x_f32 = x.astype(np.float32)

x_f32.tofile('speech.f32')
np.savetxt('f0.csv', f0, fmt='%.1f')
print(f'audio: {len(x_f32)} samples, F0: {len(f0)} frames')
"@

cllsm2_extract.exe speech.f32 f0.csv params.csv 44100 128 2048 100 5 128 4 2000 4000 8000 60 5 4.0 1.5 1 1 0
```

### 批处理（PowerShell）

```powershell
$files = Get-ChildItem "*.wav"
foreach ($f in $files) {
  $base = $f.BaseName
  # 用 Python 生成 f32 + F0（见上）
  python extract_f0.py $f.Name $base.f32 $base.f0.csv
  # 提取 LLSM 参数
  .\build\cllsm2_extract.exe $base.f32 $base.f0.csv $base.params.csv `
    44100 128 2048 100 5 128 4 2000 4000 8000 60 5 4.0 1.5 1 1 0
}
```

## 输出格式

输出 CSV 每行对应一帧，含 `order_spec + order_bap + 3` 列：

| 列 | 含义 | 说明 |
|----|------|------|
| 0 | F0 | 基频 (Hz) |
| 1 | 频谱编码 | `order_spec` 维 MCEP-like 参数 |
| 2 | 非周期性编码 | `order_bap` 维 BAP 参数 |
| 3 | 其他 | 编码的辅助参数 |

默认 `order_spec=60, order_bap=5` → 每行 68 列 + 3 基础维度 = 71？实际上输出为 `order_spec + order_bap + 3` 列。可通过修改 `coder.c` 或调整参数改变维度。

## 参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `hop_samples` | 128 (sr=44100) | 对应 ~2.9ms 帧移，约 2.9ms。更大的 hop（如 256）减少帧数但降低时间分辨率 |
| `nfft` | 2048 | 足够涵盖基频低至 ~50Hz。F0 极低时需增大 |
| `maxnhar` | 100 | 对于 F0 > 100Hz 的语音足够。F0 较低时可能需要 200+ |
| `rel_winsize` | 4.0 | 窗口 = 4 个基频周期。增大提高频率分辨率，降低时间分辨率 |
| `hm_method` | 1 (CZT) | CZT 精度更高但计算量大。方法 0 (HMPP) 在 CUDA 版中有批处理加速 |

## CUDA 使用注意

- CUDA 版启动时检测 GPU，无 GPU 则退出（exit code 1）
- 运行时需要 `cufft64_11.dll`、`cublas64_12.dll`、`cublasLt64_12.dll` 在 `PATH` 或同目录下
- 使用方法与 CPU 版完全相同（参数一致）
- `hm_method=1`（CZT）使用 `czt_frame_kernel` GPU 内核
- `hm_method=0`（HMPP）使用批处理 cuFFT，对多帧场景更高效
- 推荐在批量场景下使用 HMPP（method=0）以充分发挥批处理优势
