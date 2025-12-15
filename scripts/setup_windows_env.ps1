<#
.SYNOPSIS
  Setup script for Windows: installs Miniforge (if needed), Visual C++ redistributable,
  creates a conda env with a specified Python version, installs TensorFlow and requirements,
  and runs a TensorFlow import test.

USAGE
  Open PowerShell as Administrator (required to install VC redistributable).
  ./scripts/setup_windows_env.ps1 -EnvName trading -PythonVersion 3.11
#>

param(
  [string]$EnvName = 'trading',
  [string]$PythonVersion = '3.11'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Log { param($m) Write-Host "[setup] $m" }

function Ensure-Miniforge {
  $condaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
  if (-Not (Test-Path $condaExe)) {
    Write-Log "Miniforge not found; downloading installer..."
    $installer = Join-Path $PWD 'Miniforge3-Windows-x86_64.exe'
    $url = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    Write-Log "Running Miniforge installer (silent)..."
    Start-Process -FilePath $installer -ArgumentList '/InstallationType=JustMe','/AddToPath=1','/RegisterPython=0','/S' -Wait -NoNewWindow
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
    Write-Log "Miniforge installed to $env:USERPROFILE\miniforge3"
  }
  else { Write-Log "Miniforge already installed" }
}

function Ensure-VCRedist {
  # Checks for Visual C++ 2015-2022 x64 redistributable by checking registry key
  $key = 'HKLM:\SOFTWARE\Classes\Installer\Dependencies\{e2803110-78b3-4664-a479-3611a381656a}'
  if (-Not (Test-Path $key)) {
    Write-Log "Visual C++ Redistributable not detected; downloading and installing..."
    $vcurl = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
    $vcinstaller = Join-Path $PWD 'vc_redist.x64.exe'
    Invoke-WebRequest -Uri $vcurl -OutFile $vcinstaller -UseBasicParsing
    Start-Process -FilePath $vcinstaller -ArgumentList '/install','/quiet','/norestart' -Wait -NoNewWindow
    Remove-Item $vcinstaller -Force -ErrorAction SilentlyContinue
    Write-Log "Visual C++ redistributable installed"
  } else {
    Write-Log "Visual C++ redistributable appears to be installed"
  }
}

function Ensure-CondaEnv {
  $conda = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
  & $conda info --envs | Out-Null
  $envs = & $conda info --envs 2>$null
  if ($envs -match "^$EnvName\s") {
    Write-Log "Conda env '$EnvName' already exists"
  } else {
    Write-Log "Creating conda env '$EnvName' with python=$PythonVersion..."
    & $conda create -n $EnvName python=$PythonVersion -c conda-forge -y
  }
}

function Install-Packages {
  $conda = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
  Write-Log "Upgrading pip and installing tensorflow + requirements via pip in '$EnvName'"
  & $conda run -n $EnvName python -m pip install --upgrade pip setuptools wheel
  # prefer pip install for TensorFlow to avoid solver conflicts
  & $conda run -n $EnvName python -m pip install --upgrade tensorflow
  if (Test-Path '..\requirements.txt') {
    & $conda run -n $EnvName python -m pip install -r ..\requirements.txt
  } else {
    & $conda run -n $EnvName python -m pip install -r requirements.txt
  }
}

function Run-TF-Test {
  $conda = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
  Write-Log "Running TensorFlow import test inside env '$EnvName'"
  & $conda run -n $EnvName python scripts/tf_import_test.py
}

Write-Log "Starting Windows environment setup"
Ensure-Miniforge
Ensure-VCRedist
Ensure-CondaEnv
Install-Packages
Run-TF-Test

Write-Log "Windows setup complete. Activate with: conda activate $EnvName"
Write-Log "If TF import still fails, consider checking GPU drivers or trying a CPU-only TF build."
