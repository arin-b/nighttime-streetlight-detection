$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$derivedRoot = Join-Path $repoRoot 'datasets\derived\annotation_automation'
$calibrationManifest = Join-Path $derivedRoot 'reviews\calibration_subset_manifest.csv'
$annotationManifest = Join-Path $derivedRoot 'manifests\annotation_metadata_seed.csv'
$batchRoot = Join-Path $derivedRoot 'reviews\batches'
$calibrationRoot = Join-Path $batchRoot 'calibration'
$tempRoot = Join-Path $batchRoot 'calibration_boxed_tmp'
$zipPath = Join-Path $batchRoot 'calibration_boxed.zip'

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Remove-TreeIfExists {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Get-BoxColor {
    param([string]$Stratum)
    switch ($Stratum) {
        'multi_box' { return [System.Drawing.Color]::Lime }
        'single_small' { return [System.Drawing.Color]::DeepSkyBlue }
        'single_large' { return [System.Drawing.Color]::Orange }
        'negative_only' { return [System.Drawing.Color]::Gold }
        default { return [System.Drawing.Color]::Red }
    }
}

$annotationRows = Import-Csv -LiteralPath $annotationManifest
$annotationsByPath = @{}
foreach ($row in $annotationRows) {
    $key = $row.image_path
    if (-not $annotationsByPath.ContainsKey($key)) {
        $annotationsByPath[$key] = New-Object System.Collections.ArrayList
    }
    [void]$annotationsByPath[$key].Add($row)
}

$calibrationRows = Import-Csv -LiteralPath $calibrationManifest

Remove-TreeIfExists -Path $tempRoot
Ensure-Dir -Path $tempRoot

$font = New-Object System.Drawing.Font('Arial', 18, [System.Drawing.FontStyle]::Bold)
$smallFont = New-Object System.Drawing.Font('Arial', 12, [System.Drawing.FontStyle]::Bold)

$written = 0
foreach ($row in $calibrationRows) {
    $sourcePath = $row.image_path
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Source image not found: $sourcePath"
    }

    $targetDir = Join-Path $tempRoot $row.stratum
    Ensure-Dir -Path $targetDir
    $targetName = '{0}__{1}' -f $row.calibration_id, [System.IO.Path]::GetFileName($sourcePath)
    $targetPath = Join-Path $targetDir $targetName

    $bitmap = [System.Drawing.Bitmap]::FromFile($sourcePath)
    try {
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
            $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

            $labelText = '{0} | {1}' -f $row.calibration_id, $row.stratum
            $labelBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(220, 0, 0, 0))
            $textBrush = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
            $labelSize = $graphics.MeasureString($labelText, $font)
            $graphics.FillRectangle($labelBrush, 8, 8, [math]::Ceiling($labelSize.Width) + 20, [math]::Ceiling($labelSize.Height) + 12)
            $graphics.DrawString($labelText, $font, $textBrush, 18, 14)

            $boxColor = Get-BoxColor -Stratum $row.stratum
            $pen = New-Object System.Drawing.Pen($boxColor, 3)
            $boxTextBrush = [System.Drawing.SolidBrush]::new($boxColor)

            if ($annotationsByPath.ContainsKey($sourcePath)) {
                $boxIndex = 1
                foreach ($ann in $annotationsByPath[$sourcePath]) {
                    $x = [float]$ann.bbox_x
                    $y = [float]$ann.bbox_y
                    $w = [float]$ann.bbox_w
                    $h = [float]$ann.bbox_h
                    $graphics.DrawRectangle($pen, $x, $y, $w, $h)
                    $boxLabel = 'streetlight #{0}' -f $boxIndex
                    $graphics.DrawString($boxLabel, $smallFont, $boxTextBrush, $x, [Math]::Max(0, $y - 18))
                    $boxIndex += 1
                }
            }
            else {
                $note = 'negative candidate: no seed boxes'
                $graphics.FillRectangle($labelBrush, 8, $bitmap.Height - 42, 280, 28)
                $graphics.DrawString($note, $smallFont, $textBrush, 14, $bitmap.Height - 38)
            }
        }
        finally {
            $graphics.Dispose()
        }

        $jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' } | Select-Object -First 1
        $encoderParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
        $qualityEncoder = [System.Drawing.Imaging.Encoder]::Quality
        $encoderParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter($qualityEncoder, [long]92)
        $bitmap.Save($targetPath, $jpegCodec, $encoderParams)
        $written += 1
    }
    finally {
        $bitmap.Dispose()
    }
}

Remove-TreeIfExists -Path $calibrationRoot
Move-Item -LiteralPath $tempRoot -Destination $calibrationRoot

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $calibrationRoot -DestinationPath $zipPath -CompressionLevel Optimal

$font.Dispose()
$smallFont.Dispose()

Write-Output ("Rendered calibration overlays: {0}" -f $written)
Write-Output ("Calibration folder: {0}" -f $calibrationRoot)
Write-Output ("Zip archive: {0}" -f $zipPath)
