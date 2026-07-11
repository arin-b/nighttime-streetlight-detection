$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$manifestPath = Join-Path $repoRoot 'datasets\derived\annotation_automation\reviews\calibration_subset_manifest.csv'
$overlayRoot = Join-Path $repoRoot 'datasets\derived\annotation_automation\reviews\batches\calibration'
$outRoot = Join-Path $repoRoot 'datasets\derived\annotation_automation\reviews\contact_sheets'

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

Ensure-Dir $outRoot

$rows = Import-Csv -LiteralPath $manifestPath | Where-Object { $_.review_status -eq 'pending_manual_lock' }
$rows = $rows | Sort-Object stratum, calibration_id

$groups = $rows | Group-Object stratum

$thumbW = 420
$thumbH = 240
$labelH = 26
$cols = 3
$rowsPerPage = 4
$pageW = $cols * $thumbW
$pageH = $rowsPerPage * ($thumbH + $labelH)

$font = New-Object System.Drawing.Font('Arial', 12, [System.Drawing.FontStyle]::Bold)
$smallFont = New-Object System.Drawing.Font('Arial', 10)
$whiteBrush = [System.Drawing.Brushes]::White
$blackBrush = [System.Drawing.Brushes]::Black
$labelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(230, 20, 20, 20))

foreach ($group in $groups) {
    $items = @($group.Group)
    $pages = [Math]::Ceiling($items.Count / ($cols * $rowsPerPage))
    for ($pageIndex = 0; $pageIndex -lt $pages; $pageIndex++) {
        $bitmap = New-Object System.Drawing.Bitmap($pageW, $pageH)
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            try {
                $graphics.Clear([System.Drawing.Color]::Black)
                $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality

                for ($slot = 0; $slot -lt ($cols * $rowsPerPage); $slot++) {
                    $itemIndex = ($pageIndex * $cols * $rowsPerPage) + $slot
                    if ($itemIndex -ge $items.Count) { break }
                    $item = $items[$itemIndex]
                    $col = $slot % $cols
                    $row = [Math]::Floor($slot / $cols)
                    $x = $col * $thumbW
                    $y = $row * ($thumbH + $labelH)

                    $overlayPath = Join-Path (Join-Path $overlayRoot $item.stratum) ("{0}__{1}" -f $item.calibration_id, [System.IO.Path]::GetFileName($item.image_path))
                    if (-not (Test-Path -LiteralPath $overlayPath)) { continue }

                    $img = [System.Drawing.Image]::FromFile($overlayPath)
                    try {
                        $graphics.DrawImage($img, (New-Object System.Drawing.Rectangle($x, $y, $thumbW, $thumbH)))
                    }
                    finally {
                        $img.Dispose()
                    }

                    $graphics.FillRectangle($labelBrush, $x, $y + $thumbH, $thumbW, $labelH)
                    $label = "{0} | {1}" -f $item.calibration_id, $item.frame_id
                    $graphics.DrawString($label, $font, $whiteBrush, $x + 6, $y + $thumbH + 2)
                }
            }
            finally {
                $graphics.Dispose()
            }

            $outPath = Join-Path $outRoot ("{0}_page_{1:00}.jpg" -f $group.Name, ($pageIndex + 1))
            $jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' } | Select-Object -First 1
            $encoderParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
            $qualityEncoder = [System.Drawing.Imaging.Encoder]::Quality
            $encoderParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter($qualityEncoder, [long]90)
            $bitmap.Save($outPath, $jpegCodec, $encoderParams)
        }
        finally {
            $bitmap.Dispose()
        }
    }
}

$font.Dispose()
$smallFont.Dispose()
$labelBrush.Dispose()

Get-ChildItem -LiteralPath $outRoot -Filter *.jpg | Select-Object FullName, Length
