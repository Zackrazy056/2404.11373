param(
    [string]$Device = "cpu",
    [string]$RunTag = "",
    [string]$ObservationTimingMode = "",
    [double]$OverrideSampleRateHz = 0,
    [double]$OverrideDurationS = 0,
    [double]$BandpassMinHz = 0,
    [double]$BandpassMaxHz = 0
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Get-Date -Format "yyyyMMdd-HHmmss"
}

$OutDir = Join-Path "reports/posteriors/fig1_paper_precision" ("kerr220_" + $RunTag)
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$StatusPath = Join-Path $OutDir "kerr220_status.json"
$HeartbeatLog = Join-Path $OutDir "kerr220_heartbeat.log"
$CaseLog = Join-Path $OutDir "kerr220.log"
$PlotLog = Join-Path $OutDir "kerr220_plot.log"
$FigPath = Join-Path "reports/figures" ("fig1_kerr220_paper_grade_" + $RunTag + ".png")
$FigSummary = Join-Path "reports/figures" ("fig1_kerr220_paper_grade_" + $RunTag + ".json")

Write-Output ("Run directory: {0}" -f (Resolve-Path $OutDir))
Write-Output ("Status file: {0}" -f (Join-Path (Resolve-Path ".") $StatusPath))
Write-Output ("Heartbeat log: {0}" -f (Join-Path (Resolve-Path ".") $HeartbeatLog))
Write-Output ("Case log: {0}" -f (Join-Path (Resolve-Path ".") $CaseLog))

$TrainArgs = @(
    "python",
    "-u",
    "scripts/08_run_fig1_paper_precision.py",
    "--case",
    "kerr220",
    "--device",
    $Device,
    "--output-dir",
    $OutDir,
    "--status-path",
    $StatusPath,
    "--heartbeat-log",
    $HeartbeatLog,
    "--heartbeat-interval-seconds",
    "60.0"
)
if (-not [string]::IsNullOrWhiteSpace($ObservationTimingMode)) {
    $TrainArgs += @("--observation-timing-mode", $ObservationTimingMode)
}
if ($OverrideSampleRateHz -gt 0) {
    $TrainArgs += @("--override-sample-rate-hz", $OverrideSampleRateHz.ToString())
}
if ($OverrideDurationS -gt 0) {
    $TrainArgs += @("--override-duration-s", $OverrideDurationS.ToString())
}
if ($BandpassMinHz -gt 0) {
    $TrainArgs += @("--bandpass-min-hz", $BandpassMinHz.ToString())
}
if ($BandpassMaxHz -gt 0) {
    $TrainArgs += @("--bandpass-max-hz", $BandpassMaxHz.ToString())
}
$TrainCmd = ($TrainArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
$TrainCmd = $TrainCmd + ' 2>&1'
cmd /c $TrainCmd | Tee-Object -FilePath $CaseLog -Append

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$PlotCmd = 'python scripts/15_plot_fig1_case_paper_grade.py --case kerr220 --run-dir "{0}" --output-figure "{1}" --output-summary "{2}" 2>&1' -f $OutDir, $FigPath, $FigSummary
cmd /c $PlotCmd | Tee-Object -FilePath $PlotLog -Append

exit $LASTEXITCODE
