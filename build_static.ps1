$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $ScriptDir "build"
$null = New-Item -ItemType Directory -Path $BuildDir -Force

$gcc = Get-Command gcc -ErrorAction SilentlyContinue
if (-not $gcc) {
    throw "gcc (MinGW) was not found on PATH. Install MinGW-w64 and try again."
}

$src = @(
    "cllsm2_extract.c"
    "include/libllsm2/container.c"
    "include/libllsm2/frame.c"
    "include/libllsm2/dsputils.c"
    "include/libllsm2/llsmutils.c"
    "include/libllsm2/layer0.c"
    "include/libllsm2/layer1.c"
    "include/libllsm2/coder.c"
    "include/ciglet/ciglet.c"
    "include/ciglet/external/fftsg_h.c"
    "include/ciglet/external/fast_median.c"
)

$srcPaths = $src | ForEach-Object { Join-Path $ScriptDir $_ }

$outPath = Join-Path $BuildDir "cllsm2_extract.exe"

Write-Host "Compiling cllsm2_extract.exe (fully static, no DLL deps) ..."
& gcc -static -O3 -DFP_TYPE=float -I (Join-Path $ScriptDir "include") -o $outPath $srcPaths -lm

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful: $outPath"
} else {
    throw "Build failed with exit code $LASTEXITCODE"
}
