$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "RBCCPS portable annotator setup"
Write-Host "Bundle root: $Root"

if (-not (Test-Path -LiteralPath $Python)) {
  Write-Host "Creating local Python environment..."
  py -3 -m venv $Venv
}

Write-Host "Installing/checking Python dependencies..."
& $Python -m pip install --upgrade pip
$Editable = "${Root}[annotator,measurement,dev]"
& $Python -m pip install -e $Editable

try {
  ffmpeg -version | Out-Null
} catch {
  throw "ffmpeg was not found on PATH. Install ffmpeg, then rerun Setup-And-Launch.bat."
}

$InputRaw = Join-Path $Root "input_raw"
if (-not (Test-Path -LiteralPath $InputRaw)) {
  New-Item -ItemType Directory -Path $InputRaw | Out-Null
}

$Existing = Get-ChildItem -LiteralPath (Join-Path $Root "workspaces") -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Force = $false
if ($Existing) {
  $answer = Read-Host "Existing workspace found: $($Existing.Name). Type R to resume or B to rebuild"
  if ($answer -match '^[Bb]') { $Force = $true }
}

$Args = @(
  "-m", "rbccps_annotator", "bundle-launch",
  "--bundle-root", $Root,
  "--input-raw", $InputRaw,
  "--host", "127.0.0.1",
  "--port", "8789"
)
if ($Force) { $Args += "--force" }

Write-Host "Preparing sampled workspace and launching annotator..."
& $Python @Args
