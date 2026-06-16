# 在 PowerShell 中运行：
$file = "D:\project\svtrain_next\tools\cllsm2\cllsm2_extract.exe"
$bytes = [System.IO.File]::ReadAllBytes($file)
if ($bytes[1] -eq 0x4D -and $bytes[2] -eq 0x5A) {
    # 是 PE 文件
    $machine = [BitConverter]::ToUInt16($bytes, 4)
    if ($machine -eq 0x8664) { "64-bit" }
    elseif ($machine -eq 0x014C) { "32-bit" }
    else { "Unknown" }
}