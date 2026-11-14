<#
.SYNOPSIS
  One-shot Windows setup for turing-pattern-GPU: installs the dependencies, then runs build_win.bat.

.DESCRIPTION
  Runs on Windows PowerShell 5.1. Every step detects first and only installs what is missing.
  Steps: winget, Visual Studio C++ tools, ffmpeg, AMD HIP SDK, OpenCV (live target only),
  GPU architecture detection, then build_win.bat.

.PARAMETER Target
  all (default), turing (video/photo only, no OpenCV) or live (camera app only).

.PARAMETER GpuArch
  Override the GPU arch(s), e.g. "gfx1101" or "gfx1030 gfx1100". Default: auto-detect.

.PARAMETER HipInstaller
  Path to an already-downloaded AMD HIP SDK installer (.exe). It is run silently with -install.

.PARAMETER OpenCvDir
  OpenCV location (the folder that contains build\). Default: C:\opencv.

.PARAMETER SkipInstall
  Only build. Nothing is installed.

.PARAMETER Yes
  Do not prompt before installing.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File setup_win.ps1
.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File setup_win.ps1 -Target turing -HipInstaller C:\Downloads\hip-sdk.exe -Yes
#>
param(
    [ValidateSet('all', 'turing', 'live')]
    [string]$Target = 'all',
    [string]$GpuArch,
    [string]$HipInstaller,
    [string]$OpenCvDir = 'C:\opencv',
    [switch]$SkipInstall,
    [switch]$Yes,
    # Internal: set when the script relaunches itself as administrator, so the window waits before closing.
    [switch]$Elevated
)

$script:OpenCvUrl = 'https://github.com/opencv/opencv/releases/download/4.10.0/opencv-4.10.0-windows.exe'
$script:HipDownloadPage = 'https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html'
$script:DefaultArchs = @('gfx1030', 'gfx1031', 'gfx1032', 'gfx1100', 'gfx1101', 'gfx1102')

function Write-Tag {
    param(
        [ValidateSet('OK', 'INSTALL', 'SKIP', 'ERROR', 'INFO', 'WARN')]
        [string]$Tag,
        [string]$Message
    )
    $color = @{ OK = 'Green'; INSTALL = 'Cyan'; SKIP = 'DarkGray'; ERROR = 'Red'; INFO = 'Gray'; WARN = 'Yellow' }[$Tag]
    Write-Host ('[{0}] {1}' -f $Tag, $Message) -ForegroundColor $color
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Confirm-Step {
    param([string]$Question)
    if ($Yes) { return }
    $answer = Read-Host "$Question [y/N]"
    if ($answer -notmatch '^(y|yes)$') { throw 'Cancelled by user.' }
}

function Invoke-Winget {
    param([string[]]$Arguments)
    # Pin the community 'winget' source. Without it winget also queries 'msstore', which fails on
    # fresh installs (e.g. 0x8a15005e "server certificate did not match") and aborts with
    # "Please specify one of them using the --source option".
    # Out-Host keeps winget's output on screen and out of this function's return value.
    & winget @Arguments --source winget | Out-Host
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Tag WARN ('winget exited with code {0} (0x{1:X8}); checking the result.' -f $code, $code)
    }
    return $code
}

# ---------------------------------------------------------------- detection

function Get-VsInstallPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }
    $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    if ($found) { return ([string]$found).Trim() }
    return $null
}

function Find-HipBin {
    $candidates = @()
    if ($env:HIP_PATH) { $candidates += (Join-Path $env:HIP_PATH 'bin') }
    $rocm = 'C:\Program Files\AMD\ROCm'
    if (Test-Path -LiteralPath $rocm) {
        Get-ChildItem -LiteralPath $rocm -Directory | Sort-Object Name -Descending | ForEach-Object {
            $candidates += (Join-Path $_.FullName 'bin')
        }
    }
    foreach ($dir in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $dir 'hipcc.bat')) { return $dir }
    }
    return $null
}

function Test-OpenCvPresent {
    param([string]$Dir)
    $libDir = Join-Path $Dir 'build\x64\vc16\lib'
    $lib = Get-ChildItem -Path (Join-Path $libDir 'opencv_world*.lib') -ErrorAction SilentlyContinue | Select-Object -First 1
    return [bool]$lib
}

function Get-SetupState {
    $needOpenCv = ($Target -ne 'turing')
    return [pscustomobject]@{
        VsPath        = Get-VsInstallPath
        Ffmpeg        = Test-Command 'ffmpeg'
        HipBin        = Find-HipBin
        NeedOpenCv    = $needOpenCv
        OpenCvPresent = (-not $needOpenCv) -or (Test-OpenCvPresent $OpenCvDir)
    }
}

function Get-MissingItems {
    param([object]$State)
    $missing = @()
    if (-not $State.VsPath) { $missing += 'Visual Studio C++ tools' }
    if (-not $State.Ffmpeg) { $missing += 'ffmpeg' }
    if (-not $State.HipBin) { $missing += 'AMD HIP SDK' }
    if ($State.NeedOpenCv -and -not $State.OpenCvPresent) { $missing += 'OpenCV' }
    return ,$missing
}

function Test-NeedsAdmin {
    param([object]$State)
    if (-not $State.VsPath) { return $true }
    if ($State.NeedOpenCv -and -not $State.OpenCvPresent) { return $true }
    if ((-not $State.HipBin) -and $HipInstaller) { return $true }
    return $false
}

# ---------------------------------------------------------------- GPU arch helpers (pure, unit tested)

function ConvertFrom-HipInfo {
    param([string[]]$Lines)
    $result = New-Object System.Collections.Generic.List[string]
    foreach ($line in $Lines) {
        if ($line -match '^\s*gcnArchName\s*:\s*(\S+)') {
            $arch = $Matches[1].Split(':')[0]
            if ($arch -and -not $result.Contains($arch)) { $result.Add($arch) }
        }
    }
    return ,$result.ToArray()
}

function Get-ArchFromGpuName {
    param([string]$Name)
    if ($Name -match 'RX\s*7900\b') { return 'gfx1100' }
    if ($Name -match 'RX\s*7[78]00\b') { return 'gfx1101' }
    if ($Name -match 'RX\s*7600\b') { return 'gfx1102' }
    if ($Name -match 'RX\s*(6950|6900|6800)\b') { return 'gfx1030' }
    if ($Name -match 'RX\s*(6750|6700)\b') { return 'gfx1031' }
    if ($Name -match 'RX\s*(6650|6600|6500|6400)\b') { return 'gfx1032' }
    return $null
}

# ---------------------------------------------------------------- steps

function Invoke-Elevated {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $PSCommandPath))
    $argList += @('-Target', $Target)
    if ($GpuArch) { $argList += @('-GpuArch', ('"{0}"' -f $GpuArch)) }
    if ($HipInstaller) {
        $argList += @('-HipInstaller', ('"{0}"' -f (Resolve-Path -LiteralPath $HipInstaller).ProviderPath))
    }
    $argList += @('-OpenCvDir', ('"{0}"' -f $OpenCvDir))
    if ($Yes) { $argList += '-Yes' }
    $argList += '-Elevated'
    Write-Tag INFO 'Relaunching this script as administrator (approve the UAC prompt)...'
    try {
        $proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $argList -Verb RunAs -Wait -PassThru
    } catch {
        throw "Elevation was cancelled or failed: $($_.Exception.Message)"
    }
    return $proc.ExitCode
}

function Step-Winget {
    param([object]$State)
    if (Test-Command 'winget') { Write-Tag OK 'winget present'; return }
    if ($SkipInstall -or ((Get-MissingItems $State).Count -eq 0)) {
        Write-Tag SKIP 'winget not needed (nothing to install)'
        return
    }
    throw "winget not found. Install 'App Installer' from the Microsoft Store, then rerun this script."
}

function Step-VisualStudio {
    param([object]$State)
    if ($State.VsPath) { Write-Tag OK "Visual Studio C++ tools: $($State.VsPath)"; return }
    if ($SkipInstall) { Write-Tag SKIP 'Visual Studio C++ tools missing; install skipped (-SkipInstall)'; return }
    Confirm-Step 'Install Visual Studio 2022 Build Tools with the C++ workload (large download)?'
    Write-Tag INSTALL 'Visual Studio 2022 Build Tools with the C++ workload (winget)...'
    $code = Invoke-Winget @(
        'install', '--id', 'Microsoft.VisualStudio.2022.BuildTools', '-e',
        '--accept-source-agreements', '--accept-package-agreements',
        '--override', '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    )
    if (-not (Get-VsInstallPath)) {
        if ($code -ne 0) { throw ('winget could not install Visual Studio Build Tools (exit code 0x{0:X8}). See the winget output above.' -f $code) }
        throw 'Visual Studio C++ tools still not detected after install (a reboot may be required).'
    }
    Write-Tag OK 'Visual Studio C++ tools installed'
}

function Step-Ffmpeg {
    param([object]$State)
    if ($State.Ffmpeg) { Write-Tag OK 'ffmpeg present'; return }
    if ($SkipInstall) { Write-Tag SKIP 'ffmpeg missing; install skipped (-SkipInstall)'; return }
    Confirm-Step 'Install ffmpeg (Gyan.FFmpeg) with winget?'
    Write-Tag INSTALL 'ffmpeg (Gyan.FFmpeg) via winget...'
    $null = Invoke-Winget @('install', '--id', 'Gyan.FFmpeg', '-e', '--accept-source-agreements', '--accept-package-agreements')
    Update-SessionPath
    if (Test-Command 'ffmpeg') { Write-Tag OK 'ffmpeg installed' }
    else { Write-Tag WARN 'ffmpeg is not on PATH yet; open a new shell before running run_turing.bat.' }
}

function Step-HipSdk {
    param([object]$State)
    if ($State.HipBin) { Write-Tag OK "AMD HIP SDK: $($State.HipBin)"; return }
    if ($SkipInstall) { Write-Tag SKIP 'AMD HIP SDK missing; install skipped (-SkipInstall)'; return }
    if (-not $HipInstaller) {
        Write-Host 'The AMD HIP SDK has no winget package. Download the Windows HIP SDK installer from:'
        Write-Host "  $($script:HipDownloadPage)"
        try { Start-Process $script:HipDownloadPage } catch { }
        throw 'AMD HIP SDK not found. Download it, then rerun: setup_win.ps1 -HipInstaller <path-to-installer.exe>'
    }
    $installer = (Resolve-Path -LiteralPath $HipInstaller -ErrorAction Stop).ProviderPath
    Confirm-Step "Run the AMD HIP SDK installer silently ($installer -install)?"
    Write-Tag INSTALL "AMD HIP SDK: running $installer -install (this can take a while)..."
    $proc = Start-Process -FilePath $installer -ArgumentList '-install' -Wait -PassThru
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "HIP SDK installer exited with code $($proc.ExitCode)."
    }
    if (-not (Find-HipBin)) { throw 'HIP SDK still not detected after install (a reboot may be required).' }
    Write-Tag OK 'AMD HIP SDK installed'
}

function Step-OpenCv {
    param([object]$State)
    if (-not $State.NeedOpenCv) { Write-Tag SKIP 'OpenCV not needed for -Target turing'; return }
    $buildDir = Join-Path $OpenCvDir 'build'
    if ($State.OpenCvPresent) {
        Write-Tag OK "OpenCV found under $OpenCvDir"
    } elseif ($SkipInstall) {
        Write-Tag SKIP "OpenCV not found under $OpenCvDir; install skipped (-SkipInstall)"
        return
    } else {
        Confirm-Step "Download OpenCV 4.10.0 and extract it to $OpenCvDir (large download)?"
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $ProgressPreference = 'SilentlyContinue'
        $exe = Join-Path $env:TEMP 'opencv-4.10.0-windows.exe'
        Write-Tag INSTALL "OpenCV 4.10.0: downloading to $exe ..."
        Invoke-WebRequest -Uri $script:OpenCvUrl -OutFile $exe -UseBasicParsing

        $full = [IO.Path]::GetFullPath($OpenCvDir).TrimEnd('\')
        $parent = Split-Path -Parent $full
        $leaf = Split-Path -Leaf $full
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
        # "C:\" would leave a trailing backslash inside the quoted -o argument, so use "C:\." for drive roots
        $outArg = $parent
        if ($outArg -match '^[A-Za-z]:\\$') { $outArg = $outArg + '.' }
        Write-Tag INSTALL "OpenCV 4.10.0: extracting to $parent ..."
        $proc = Start-Process -FilePath $exe -ArgumentList ('-o"{0}" -y' -f $outArg) -Wait -PassThru
        if ($proc.ExitCode -ne 0) { throw "OpenCV extractor exited with code $($proc.ExitCode)." }

        $extracted = Join-Path $parent 'opencv'
        if (($leaf -ne 'opencv') -and (Test-Path -LiteralPath $extracted)) {
            Rename-Item -LiteralPath $extracted -NewName $leaf
        }
        if (-not (Test-OpenCvPresent $OpenCvDir)) { throw "OpenCV not found under $OpenCvDir after extraction." }
        Write-Tag OK "OpenCV installed to $OpenCvDir"
    }

    $env:OPENCV_DIR = $buildDir
    if (-not $SkipInstall) {
        $current = [Environment]::GetEnvironmentVariable('OPENCV_DIR', 'User')
        if ($current -ne $buildDir) {
            [Environment]::SetEnvironmentVariable('OPENCV_DIR', $buildDir, 'User')
            Write-Tag INFO "Saved OPENCV_DIR=$buildDir for the current user"
        }
    }
}

function Step-GpuArch {
    param([object]$State)
    if ($GpuArch) {
        $archs = @($GpuArch -split '\s+' | Where-Object { $_ })
        Write-Tag OK "GPU arch from -GpuArch: $($archs -join ' ')"
        return ,$archs
    }
    if ($State.HipBin) {
        $hipInfo = Join-Path $State.HipBin 'hipInfo.exe'
        if (Test-Path -LiteralPath $hipInfo) {
            # hipInfo may write to stderr; with ErrorActionPreference=Stop, PS 5.1 would turn that into a terminating error.
            $lines = @()
            try {
                $ErrorActionPreference = 'Continue'
                $lines = @(& $hipInfo 2>&1 | ForEach-Object { "$_" })
            } catch { $lines = @() }
            $ErrorActionPreference = 'Stop'
            $archs = @(ConvertFrom-HipInfo $lines | ForEach-Object { $_ })
            if ($archs.Count -gt 0) {
                Write-Tag OK "GPU arch from hipInfo: $($archs -join ' ')"
                return ,$archs
            }
        }
    }
    $names = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
    $mapped = @($names | ForEach-Object { Get-ArchFromGpuName $_ } | Where-Object { $_ } | Select-Object -Unique)
    if ($mapped.Count -gt 0) {
        Write-Tag OK "GPU arch from video controller names: $($mapped -join ' ')"
        return ,$mapped
    }
    Write-Tag WARN "Could not detect the GPU arch. Using the full list: $($script:DefaultArchs -join ' ')"
    Write-Tag WARN 'Pass -GpuArch <arch> to build for one arch only.'
    return ,$script:DefaultArchs
}

function Invoke-Build {
    param([string[]]$Archs)
    $env:HIP_GPU_ARCH = ($Archs -join ' ')
    Write-Tag INFO "Building target=$Target with HIP_GPU_ARCH=$env:HIP_GPU_ARCH ..."
    # Full path to build_win.bat, quoted with the cmd /c "" ... "" pattern so paths with spaces work.
    # Start-Process is used because cmd.exe does not search the current folder for the bat here.
    $bat = Join-Path $PSScriptRoot 'build_win.bat'
    $cmdLine = '/c ""' + $bat + '"'
    if ($Target -ne 'all') { $cmdLine += ' ' + $Target }
    $cmdLine += '"'
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdLine -NoNewWindow -Wait -PassThru
    return $proc.ExitCode
}
function Write-NextSteps {
    Write-Host ''
    Write-Host 'Next steps:'
    if ($Target -ne 'live') {
        Write-Host '  run_turing.bat input.mp4 output.mp4 1920 1080 64'
        Write-Host '  run_turing.bat photo.jpg output.png'
    }
    if ($Target -ne 'turing') {
        Write-Host '  run_live.bat'
    }
}

function Wait-IfElevated {
    if ($Elevated) { Read-Host 'Press Enter to close this window' | Out-Null }
}

function Main {
    $ErrorActionPreference = 'Stop'
    try {
        Write-Host '== turing-pattern-GPU: Windows setup =='
        Write-Tag INFO "Target=$Target  SkipInstall=$SkipInstall"
        $state = Get-SetupState

        if ((-not $SkipInstall) -and (Test-NeedsAdmin $state) -and (-not (Test-IsAdmin))) {
            exit (Invoke-Elevated)
        }

        Step-Winget $state
        Step-VisualStudio $state
        Step-Ffmpeg $state
        Step-HipSdk $state
        Step-OpenCv $state
        $archs = @(Step-GpuArch $state | ForEach-Object { $_ })
        Write-Host ''
        $code = Invoke-Build $archs
        if ($code -ne 0) {
            Write-Tag ERROR "build_win.bat failed with exit code $code"
            Wait-IfElevated
            exit $code
        }
        Write-Tag OK 'Build succeeded'
        Write-NextSteps
        Wait-IfElevated
        exit 0
    } catch {
        Write-Tag ERROR $_.Exception.Message
        Wait-IfElevated
        exit 1
    }
}

if ($MyInvocation.InvocationName -ne '.') { Main }
