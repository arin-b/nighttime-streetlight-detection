$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
  throw "Local venv not found. Run Setup-And-Launch.bat first."
}
$Workspace = Get-ChildItem -LiteralPath (Join-Path $Root "workspaces") -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Workspace) {
  throw "No workspace found under workspaces/."
}
$Out = Join-Path $Root ("exports\" + $Workspace.Name)
New-Item -ItemType Directory -Path $Out -Force | Out-Null
& $Python -m rbccps_annotator export-yolo --workspace $Workspace.FullName --output (Join-Path $Out "yolo") --split-dirs
& $Python -m rbccps_annotator export-measurement --workspace $Workspace.FullName --output (Join-Path $Out "measurement")
Write-Host "Exports written to $Out"
