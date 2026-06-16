# benchmark.ps1 — Compare CPU vs CUDA cllsm2_extract performance
param(
  [string]$AudioF32 = "test_audio.f32",
  [string]$F0Csv    = "test_f0.csv",
  [string]$OutCpu   = "out_cpu.csv",
  [string]$OutCuda  = "out_cuda.csv",
  [int]$Runs        = 3
)

# --- Generate test data if not present ---
if (!(Test-Path $AudioF32) -or !(Test-Path $F0Csv)) {
  Write-Host "Generating test data..."
  python -c @"
import numpy as np
import wave, struct

# Try to use the first available wav file
import glob, os
wavs = glob.glob('include/libllsm2/test/*.wav') + glob.glob('*.wav')
if wavs:
    wf = wave.open(wavs[0], 'rb')
    sr = wf.getframerate()
    nf = wf.getnframes()
    raw = wf.readframes(nf)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # stereo -> mono
    if wf.getnchannels() > 1:
        data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
    wf.close()
    print(f'Using {os.path.basename(wavs[0])}: {sr}Hz, {len(data)} samples ({len(data)/sr:.1f}s)')
else:
    # Synthetic: chirp + noise
    sr = 44100
    dur = 10.0
    t = np.linspace(0, dur, int(sr*dur))
    f0_t = 200 + 100 * t/dur
    data = np.sin(2*np.pi*np.cumsum(f0_t)/sr) + 0.1*np.random.randn(len(t))
    print(f'Using synthetic signal: {sr}Hz, {len(data)} samples')

data.tofile('$AudioF32')
print(f'Wrote {len(data)} float32 samples to $AudioF32')

# Simple F0 estimation via autocorrelation
hop = int(0.005 * sr)
nfrm = len(data) // hop
f0 = np.zeros(nfrm)
for i in range(nfrm):
    start = i * hop
    end = min(start + int(0.04*sr), len(data))
    seg = data[start:end]
    if len(seg) < sr//500:
        continue
    seg = seg * np.hanning(len(seg))
    ac = np.correlate(seg, seg, mode='same')
    mid = len(ac)//2
    # search in 50-500Hz range
    lo = int(sr/500)
    hi = int(sr/50)
    peak = lo + np.argmax(ac[mid+lo:mid+hi])
    if peak > lo and ac[mid+peak] > 0:
        f0[i] = sr / peak
    else:
        f0[i] = 0
# median filter
f0 = np.array([np.median(f0[max(0,i-2):min(len(f0),i+3)]) for i in range(len(f0))])
np.savetxt('$F0Csv', f0, fmt='%.1f')
print(f'Wrote {nfrm} F0 frames to $F0Csv')
"@
  if ($LASTEXITCODE -ne 0) { Write-Error "Test data generation failed"; exit 1 }
}

# --- Common args ---
$sr = 44100
$hop = 128
$nfft = 2048
$mxh = 100
$mxhe = 5
$npsd = 128
$nch = 4
$cf0 = 2000
$cf1 = 4000
$cf2 = 8000
$os = 60
$ob = 5
$rw = 4.0
$lr = 1.5
$frf = 0
$hm = 1
$frm = 0
$args = "$AudioF32 $F0Csv $OutCpu $sr $hop $nfft $mxh $mxhe $npsd $nch $cf0 $cf1 $cf2 $os $ob $rw $lr $frf $hm $frm"

# --- CPU benchmark ---
$cpuExe = ".\build\cllsm2_extract.exe"
if (!(Test-Path $cpuExe)) { Write-Error "CPU exe not found at $cpuExe"; exit 1 }

Write-Host "`n=== CPU benchmark ==="
$cpuTimes = @()
for ($i = 0; $i -lt $Runs; $i++) {
  Remove-Item -Force $OutCpu -ErrorAction SilentlyContinue
  $t = Measure-Command { & $cpuExe $args.Split(' ') | Out-Null }
  $ms = $t.TotalMilliseconds
  $cpuTimes += $ms
  Write-Host "  Run $($i+1): $([math]::Round($ms, 1)) ms"
}
$cpuAvg = ($cpuTimes | Measure-Object -Average).Average
Write-Host "  Average: $([math]::Round($cpuAvg, 1)) ms"

# --- CUDA benchmark ---
$cudaExe = ".\build\cllsm2_extract_cuda.exe"
if (Test-Path $cudaExe) {
  Write-Host "`n=== CUDA benchmark ==="
  $cudaTimes = @()
  for ($i = 0; $i -lt $Runs; $i++) {
    Remove-Item -Force $OutCuda -ErrorAction SilentlyContinue
    $cudaArgs = $args -replace [regex]::Escape($OutCpu), $OutCuda
    $t = Measure-Command { & $cudaExe $cudaArgs.Split(' ') | Out-Null }
    $ms = $t.TotalMilliseconds
    $cudaTimes += $ms
    Write-Host "  Run $($i+1): $([math]::Round($ms, 1)) ms"
  }
  $cudaAvg = ($cudaTimes | Measure-Object -Average).Average
  $speedup = $cpuAvg / $cudaAvg
  Write-Host "  Average: $([math]::Round($cudaAvg, 1)) ms"
  Write-Host "`n*** Speedup: $([math]::Round($speedup, 2))x ***"
} else {
  Write-Host "`nCUDA exe not found. Build it with GitHub Actions first:"
  Write-Host "  1. Push to 'cuda' branch"
  Write-Host "  2. gh run download <run-id> --name cllsm2_extract_cuda-win-x64 --dir build"
  Write-Host "  or: gh run download <run-id> && copy <dir>\build\cllsm2_extract_cuda.exe .\build\"
}
