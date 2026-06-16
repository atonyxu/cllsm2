$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..")
Set-Location $RepoRoot

$zig = Get-Command zig -ErrorAction SilentlyContinue
if (-not $zig) {
    throw "zig was not found on PATH. Install Zig or put zig.exe on PATH, then rerun this script."
}

& zig cc `
    -target x86_64-linux-gnu `
    -O3 `
    -DFP_TYPE=float `
    -DUSE_PTHREAD `
    -I. `
    -Isvtrain_next/tools/cllsm2/include `
    -Iciglet-master `
    -Iciglet-master/external `
    -Ilibllsm2-master `
    -o svtrain_next/tools/cllsm2/cllsm2_extract `
    svtrain_next/tools/cllsm2/cllsm2_extract.c `
    libllsm2-master/container.c `
    libllsm2-master/frame.c `
    libllsm2-master/dsputils.c `
    libllsm2-master/llsmutils.c `
    libllsm2-master/layer0.c `
    libllsm2-master/layer1.c `
    libllsm2-master/coder.c `
    ciglet-master/ciglet.c `
    ciglet-master/external/fftsg_h.c `
    ciglet-master/external/fast_median.c `
    -lm
