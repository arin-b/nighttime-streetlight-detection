$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$OutRoot = Join-Path $Repo "exports\portable_annotator_bundle"
$Stage = Join-Path $OutRoot "rbccps_portable_annotator"
$Zip = Join-Path $OutRoot "rbccps_portable_annotator.zip"

if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Path $Stage | Out-Null
New-Item -ItemType Directory -Path $OutRoot -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $Repo "pyproject.toml") -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Repo "README.md") -Destination $Stage -ErrorAction SilentlyContinue
Copy-Item -LiteralPath (Join-Path $Repo "src") -Destination $Stage -Recurse
Get-ChildItem -LiteralPath (Join-Path $Stage "src") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Copy-Item -LiteralPath (Join-Path $Repo "scripts\annotator_bundle\Setup-And-Launch.bat") -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Repo "scripts\annotator_bundle\Setup-And-Launch.ps1") -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Repo "scripts\annotator_bundle\Export-Now.bat") -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Repo "scripts\annotator_bundle\Export-Now.ps1") -Destination $Stage
Copy-Item -LiteralPath (Join-Path $Repo "scripts\annotator_bundle\README_START_HERE.md") -Destination $Stage

foreach ($dir in @("input_raw", "tutorial_examples", "workspaces", "exports", "logs", "models\detector", "models\annotator\prompt_segmenter")) {
  New-Item -ItemType Directory -Path (Join-Path $Stage $dir) -Force | Out-Null
}

"Drop raw phone images/videos here." | Set-Content -LiteralPath (Join-Path $Stage "input_raw\README.txt") -Encoding UTF8
"Place tutorial image+JSON examples here." | Set-Content -LiteralPath (Join-Path $Stage "tutorial_examples\README.txt") -Encoding UTF8
"Generated workspaces will be written here." | Set-Content -LiteralPath (Join-Path $Stage "workspaces\README.txt") -Encoding UTF8
"Generated YOLO and measurement exports will be written here." | Set-Content -LiteralPath (Join-Path $Stage "exports\README.txt") -Encoding UTF8
"Launcher and ffmpeg logs will be written here." | Set-Content -LiteralPath (Join-Path $Stage "logs\README.txt") -Encoding UTF8
"Optional detector weights go here as best.pt." | Set-Content -LiteralPath (Join-Path $Stage "models\detector\README.txt") -Encoding UTF8

$Sam2 = Join-Path $Repo "models\annotator\prompt_segmenter\sam2.1_hiera_tiny.pt"
if (Test-Path -LiteralPath $Sam2) {
  Copy-Item -LiteralPath $Sam2 -Destination (Join-Path $Stage "models\annotator\prompt_segmenter\sam2.1_hiera_tiny.pt")
}

if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -Force
$SizeMb = [math]::Round((Get-Item -LiteralPath $Zip).Length / 1MB, 2)
Write-Host "Bundle written: $Zip"
Write-Host "Zip size MB: $SizeMb"
