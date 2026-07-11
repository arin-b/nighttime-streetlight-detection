$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$manifestPath = Join-Path $repoRoot 'datasets\derived\annotation_automation\reviews\hard_negative_review_manifest.csv'
$outRoot = Join-Path $repoRoot 'datasets\derived\annotation_automation\reviews\hard_negative_contact_sheets'

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

Ensure-Dir $outRoot

$rows = Import-Csv -LiteralPath $manifestPath | Sort-Object review_label, source_pool, review_candidate_id
$groups = $rows | Group-Object review_label

$thumbW = 420
$thumbH = 240
$labelH = 44
$cols = 3
$rowsPerPage = 4
$pageW = $cols * $thumbW
$pageH = $rowsPerPage * ($thumbH + $labelH)

$font = New-Object System.Drawing.Font('Arial', 11, [System.Drawing.FontStyle]::Bold)
$smallFont = New-Object System.Drawing.Font('Arial', 9)
$whiteBrush = [System.Drawing.Brushes]::White
$labelBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(235, 20, 20, 20))

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

                    if (-not (Test-Path -LiteralPath $item.image_path)) { continue }
                    $img = [System.Drawing.Image]::FromFile($item.image_path)
                    try {
                        $graphics.DrawImage($img, (New-Object System.Drawing.Rectangle($x, $y, $thumbW, $thumbH)))
                    }
                    finally {
                        $img.Dispose()
                    }

                    $graphics.FillRectangle($labelBrush, $x, $y + $thumbH, $thumbW, $labelH)
                    $label1 = "{0} | {1}" -f $item.review_candidate_id, $item.review_label
                    $label2 = "{0} | {1}" -f $item.source_pool, $item.frame_id
                    $graphics.DrawString($label1, $font, $whiteBrush, $x + 6, $y + $thumbH + 2)
                    $graphics.DrawString($label2, $smallFont, $whiteBrush, $x + 6, $y + $thumbH + 22)
                }
            }
            finally {
                $graphics.Dispose()
            }

            $outPath = Join-Path $outRoot ("{0}_page_{1:00}.jpg" -f $group.Name, ($pageIndex + 1))
            $jpegCodec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' } | Select-Object -First 1
            $encoderParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
            $qualityEncoder = [System.Drawing.Imaging.Encoder]::Quality
            $encoderParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter($qualityEncoder, [long]88)
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
