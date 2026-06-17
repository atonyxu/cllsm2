# benchmark.ps1 — Compare CPU vs CUDA cllsm2_extract performance
param(
  [string]$AudioF32 = "test_audio.f32",
  [string]$F0Csv    = "test_f0.csv",
  [string]$OutCpu   = "out_cpu.csv",
  [string]$OutCuda  = "out_cuda.csv",
  [int]$Runs        = 3
)

$ErrorActionPreference = "Stop"

# --- Ensure numpy is available ---
python -c "import numpy" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "numpy not found, installing..."
  pip install numpy 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install numpy"; exit 1 }
}

# --- Generate test data if not present ---
if (!(Test-Path $AudioF32) -or !(Test-Path $F0Csv)) {
  Write-Host "Generating test data..."
  python -c @"
import numpy as np
import wave, struct, glob, os

wavs = glob.glob('include/libllsm2/test/*.wav') + glob.glob('*.wav')
if wavs:
    wf = wave.open(wavs[0], 'rb')
    sr = wf.getframerate()
    nf = wf.getnframes()
    raw = wf.readframes(nf)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if wf.getnchannels() > 1:
        data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
    wf.close()
    print(f'Using {os.path.basename(wavs[0])}: {sr}Hz, {len(data)} samples ({len(data)/sr:.1f}s)')
else:
    sr = 44100
    dur = 10.0
    t = np.linspace(0, dur, int(sr*dur))
    f0_t = 200 + 100 * t/dur
    data = np.sin(2*np.pi*np.cumsum(f0_t)/sr) + 0.1*np.random.randn(len(t))
    print(f'Using synthetic signal: {sr}Hz, {len(data)} samples')

data.tofile('$AudioF32')
print(f'Wrote {len(data)} float32 samples to $AudioF32')

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
    lo = int(sr/500)
    hi = int(sr/50)
    peak = lo + np.argmax(ac[mid+lo:mid+hi])
    if peak > lo and ac[mid+peak] > 0:
        f0[i] = sr / peak
    else:
        f0[i] = 0
f0 = np.array([np.median(f0[max(0,i-2):min(len(f0),i+3)]) for i in range(len(f0))])
np.savetxt('$F0Csv', f0, fmt='%.1f')
print(f'Wrote {nfrm} F0 frames to $F0Csv')
"@
  if ($LASTEXITCODE -ne 0) { Write-Error "Test data generation failed"; exit 1 }
}

# --- Common args ---
$sr = 44100; $hop = 128; $nfft = 2048
$mxh = 100; $mxhe = 5; $npsd = 128; $nch = 4
$cf0 = 2000; $cf1 = 4000; $cf2 = 8000
$os = 60; $ob = 5; $rw = 4.0; $lr = 1.5
$frf = 0; $hm = 1; $frm = 0
$argStr = "$AudioF32 $F0Csv $OutCpu $sr $hop $nfft $mxh $mxhe $npsd $nch $cf0 $cf1 $cf2 $os $ob $rw $lr $frf $hm $frm"
$argList = $argStr.Split(' ')

# --- Helper: run binary N times ---
function Run-Benchmark($exe, $argList0, $outFile, $runs, $label) {
  if (!(Test-Path $exe)) { Write-Host "$label exe not found at $exe"; return $null }
  Write-Host "`n=== $label benchmark ==="
  # Verify the binary works before benchmarking
  Remove-Item -Force $outFile -ErrorAction SilentlyContinue
  & $exe $argList0 2>&1 | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "  SKIPPED ($exe failed with exit $LASTEXITCODE - requires GPU)"; Remove-Item -Force $outFile -ErrorAction SilentlyContinue; return $null }
  Remove-Item -Force $outFile -ErrorAction SilentlyContinue
  $times = @()
  for ($i = 0; $i -lt $runs; $i++) {
    $t = Measure-Command { & $exe $argList0 2>&1 | Out-Null }
    $ms = $t.TotalMilliseconds
    $times += $ms
    Write-Host "  Run $($i+1): $([math]::Round($ms, 1)) ms"
    Remove-Item -Force $outFile -ErrorAction SilentlyContinue
  }
  $avg = ($times | Measure-Object -Average).Average
  Write-Host "  Average: $([math]::Round($avg, 1)) ms"
  return @{ times = $times; avg = $avg }
}

# --- CPU benchmark ---
$cpuResult = Run-Benchmark -exe ".\build\cllsm2_extract.exe" -argList0 $argList -outFile $OutCpu -runs $Runs -label "CPU"

# --- CUDA benchmark ---
$cudaResult = $null
$cudaExe = ".\build\cllsm2_extract_cuda.exe"
if (Test-Path $cudaExe) {
  # Check for required DLLs
  $missing = @()
  foreach ($dll in @("cufft64_11.dll","cublas64_12.dll","cublasLt64_12.dll")) {
    $found = [System.IO.File]::Exists("$PSScriptRoot\$dll") -or
             [System.IO.File]::Exists("$PSScriptRoot\build\$dll") -or
             (Get-Command $dll -ErrorAction SilentlyContinue) -ne $null
    if (-not $found) { $missing += $dll }
  }
  if ($missing.Count -gt 0) {
    Write-Host "`n=== CUDA benchmark (SKIPPED) ==="
    Write-Host "  Missing DLLs: $($missing -join ', ')"
    Write-Host "  CUDA binary requires these DLLs and an NVIDIA GPU to run."
  } else {
    $cudaArgStr = $argStr -replace [regex]::Escape($OutCpu), $OutCuda
    $cudaResult = Run-Benchmark -exe $cudaExe -argList0 $cudaArgStr.Split(' ') -outFile $OutCuda -runs $Runs -label "CUDA"
  }
} else {
  Write-Host "`nCUDA exe not found at $cudaExe"
}

# --- Summary ---
Write-Host "`n$('='*50)"
Write-Host "  SUMMARY"
Write-Host "$('='*50)"
if ($cpuResult) { Write-Host "  CPU:  $([math]::Round($cpuResult.avg, 1)) ms avg over $Runs runs" }
if ($cudaResult) {
  Write-Host "  CUDA: $([math]::Round($cudaResult.avg, 1)) ms avg over $Runs runs"
  $speedup = $cpuResult.avg / $cudaResult.avg
  Write-Host "  Speedup: $([math]::Round($speedup, 2))x"
}
Write-Host "$('='*50)"
