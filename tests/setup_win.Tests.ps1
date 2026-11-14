# Plain-PowerShell unit tests for the pure helpers in setup_win.ps1 (no Pester needed).
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File tests\setup_win.Tests.ps1

. (Join-Path $PSScriptRoot '..\setup_win.ps1')

$script:Passed = 0
$script:Failed = 0

function Assert-Equal {
    param($Expected, $Actual, [string]$Name)
    $exp = @($Expected)
    $act = @($Actual)
    $ok = ($exp.Count -eq $act.Count)
    if ($ok) {
        for ($i = 0; $i -lt $exp.Count; $i++) {
            if ("$($exp[$i])" -cne "$($act[$i])") { $ok = $false }
        }
    }
    if ($ok) {
        $script:Passed++
        Write-Host "PASS $Name"
    } else {
        $script:Failed++
        Write-Host "FAIL $Name"
        Write-Host "     expected: [$($exp -join ', ')] (count $($exp.Count))"
        Write-Host "     actual:   [$($act -join ', ')] (count $($act.Count))"
    }
}

# ---- ConvertFrom-HipInfo

$hipLine = 'gcnArchName:                      {0}:sramecc-:xnack-'

Assert-Equal -Expected @('gfx1101') -Name 'ConvertFrom-HipInfo: single GPU' -Actual (
    ConvertFrom-HipInfo -Lines @(
        'Device 0:',
        ($hipLine -f 'gfx1101')
    )
)

Assert-Equal -Expected @('gfx1101') -Name 'ConvertFrom-HipInfo: two GPUs, same arch -> deduped' -Actual (
    ConvertFrom-HipInfo -Lines @(
        ($hipLine -f 'gfx1101'),
        ($hipLine -f 'gfx1101')
    )
)

Assert-Equal -Expected @('gfx1030', 'gfx1100') -Name 'ConvertFrom-HipInfo: two GPUs, different archs' -Actual (
    ConvertFrom-HipInfo -Lines @(
        ($hipLine -f 'gfx1030'),
        ($hipLine -f 'gfx1100')
    )
)

Assert-Equal -Expected @() -Name 'ConvertFrom-HipInfo: no match -> empty' -Actual (
    ConvertFrom-HipInfo -Lines @(
        'hipInfo: no devices',
        'name: something else'
    )
)

Assert-Equal -Expected @('gfx1036') -Name 'ConvertFrom-HipInfo: iGPU gfx1036 line is kept' -Actual (
    ConvertFrom-HipInfo -Lines @(
        ($hipLine -f 'gfx1036')
    )
)

Assert-Equal -Expected @('gfx1036', 'gfx1100') -Name 'ConvertFrom-HipInfo: iGPU plus dGPU both kept, in order' -Actual (
    ConvertFrom-HipInfo -Lines @(
        ($hipLine -f 'gfx1036'),
        ($hipLine -f 'gfx1100')
    )
)

# ---- Get-ArchFromGpuName

$gpuCases = @(
    @{ Name = 'AMD Radeon RX 7900 XTX';   Expected = 'gfx1100' },
    @{ Name = 'AMD Radeon RX 7900 XT';    Expected = 'gfx1100' },
    @{ Name = 'AMD Radeon RX 7800 XT';    Expected = 'gfx1101' },
    @{ Name = 'AMD Radeon RX 7700 XT';    Expected = 'gfx1101' },
    @{ Name = 'AMD Radeon RX 7600';       Expected = 'gfx1102' },
    @{ Name = 'AMD Radeon RX 6950 XT';    Expected = 'gfx1030' },
    @{ Name = 'AMD Radeon RX 6900 XT';    Expected = 'gfx1030' },
    @{ Name = 'AMD Radeon RX 6800 XT';    Expected = 'gfx1030' },
    @{ Name = 'AMD Radeon RX 6750 XT';    Expected = 'gfx1031' },
    @{ Name = 'AMD Radeon RX 6700 XT';    Expected = 'gfx1031' },
    @{ Name = 'AMD Radeon RX 6650 XT';    Expected = 'gfx1032' },
    @{ Name = 'AMD Radeon RX 6600';       Expected = 'gfx1032' },
    @{ Name = 'AMD Radeon RX 6500 XT';    Expected = 'gfx1032' },
    @{ Name = 'AMD Radeon RX 6400';       Expected = 'gfx1032' },
    @{ Name = 'NVIDIA GeForce RTX 4090';  Expected = $null },
    @{ Name = 'AMD Radeon(TM) Graphics';  Expected = $null }
)

foreach ($case in $gpuCases) {
    $label = if ($null -eq $case.Expected) { 'unknown' } else { $case.Expected }
    Assert-Equal -Expected $case.Expected -Actual (Get-ArchFromGpuName $case.Name) -Name "Get-ArchFromGpuName: '$($case.Name)' -> $label"
}

# ---- Regression: arch lists must stay flat string arrays.
# Assert-Equal compares stringified elements, so it cannot see a nested array.
# A nested array once reached build_win.bat as HIP_GPU_ARCH=System.String[].

function Assert-FlatStrings {
    param($Value, [int]$Count, [string]$Name)
    $arr = @($Value)
    $ok = ($arr.Count -eq $Count)
    foreach ($item in $arr) { if (-not ($item -is [string])) { $ok = $false } }
    if ($ok) { $script:Passed++; Write-Host "PASS $Name" }
    else {
        $script:Failed++
        Write-Host "FAIL $Name"
        Write-Host "     count $($arr.Count), types: $(($arr | ForEach-Object { $_.GetType().Name }) -join ', ')"
    }
}

$twoGpus = @(($hipLine -f 'gfx1030'), ($hipLine -f 'gfx1100'))
$flat = @(ConvertFrom-HipInfo $twoGpus | ForEach-Object { $_ })
Assert-FlatStrings -Value $flat -Count 2 -Name 'ConvertFrom-HipInfo result flattens to 2 strings'
Assert-Equal -Expected 'gfx1030 gfx1100' -Actual ($flat -join ' ') -Name 'ConvertFrom-HipInfo result joins to HIP_GPU_ARCH value'

$GpuArch = 'gfx1030  gfx1100'
$fromParam = @(Step-GpuArch ([pscustomobject]@{ HipBin = $null }) | ForEach-Object { $_ })
Assert-FlatStrings -Value $fromParam -Count 2 -Name 'Step-GpuArch -GpuArch list flattens to 2 strings'
Assert-Equal -Expected 'gfx1030 gfx1100' -Actual ($fromParam -join ' ') -Name 'Step-GpuArch -GpuArch joins to HIP_GPU_ARCH value'
$GpuArch = $null

# ---- Summary

Write-Host ''
Write-Host "Passed: $script:Passed  Failed: $script:Failed"
if ($script:Failed -gt 0) { exit 1 }
exit 0
