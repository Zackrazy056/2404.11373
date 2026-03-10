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

if ([string]::IsNullOrWhiteSpace($RunTag)) {
    $RunTag = Get-Date -Format "yyyyMMdd-HHmmss"
}

$Case = "kerr220"
$OutDir = Join-Path "reports/posteriors/fig1_paper_precision" ("{0}_{1}" -f $Case, $RunTag)
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$MetaPath = Join-Path $OutDir "run_meta.json"
$StatusPath = Join-Path $OutDir "kerr220_status.json"
$HeartbeatLog = Join-Path $OutDir "kerr220_heartbeat.log"
$CaseLog = Join-Path $OutDir "kerr220.log"

$Args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/17_run_kerr220_worker.ps1",
    "-Device", $Device,
    "-RunTag", $RunTag
)
if (-not [string]::IsNullOrWhiteSpace($ObservationTimingMode)) {
    $Args += @("-ObservationTimingMode", $ObservationTimingMode)
}
if ($OverrideSampleRateHz -gt 0) {
    $Args += @("-OverrideSampleRateHz", $OverrideSampleRateHz)
}
if ($OverrideDurationS -gt 0) {
    $Args += @("-OverrideDurationS", $OverrideDurationS)
}
if ($BandpassMinHz -gt 0) {
    $Args += @("-BandpassMinHz", $BandpassMinHz)
}
if ($BandpassMaxHz -gt 0) {
    $Args += @("-BandpassMaxHz", $BandpassMaxHz)
}

$Proc = Start-Process `
    -FilePath "powershell" `
    -ArgumentList $Args `
    -WorkingDirectory (Resolve-Path ".") `
    -PassThru

$Meta = [ordered]@{
    case = $Case
    device = $Device
    pid = $Proc.Id
    run_tag = $RunTag
    out_dir = $OutDir
    status_path = $StatusPath
    heartbeat_log = $HeartbeatLog
    case_log = $CaseLog
    observation_timing_mode = $ObservationTimingMode
    override_sample_rate_hz = $(if ($OverrideSampleRateHz -gt 0) { $OverrideSampleRateHz } else { $null })
    override_duration_s = $(if ($OverrideDurationS -gt 0) { $OverrideDurationS } else { $null })
    bandpass_min_hz = $(if ($BandpassMinHz -gt 0) { $BandpassMinHz } else { $null })
    bandpass_max_hz = $(if ($BandpassMaxHz -gt 0) { $BandpassMaxHz } else { $null })
    started_at = (Get-Date).ToString("s")
}
($Meta | ConvertTo-Json -Depth 4) | Set-Content -Encoding UTF8 $MetaPath

Write-Output ("started pid={0}" -f $Proc.Id)
Write-Output ("out_dir={0}" -f $OutDir)
Write-Output ("status={0}" -f $StatusPath)
Write-Output ("heartbeat={0}" -f $HeartbeatLog)
Write-Output ("case_log={0}" -f $CaseLog)
Write-Output ("meta={0}" -f $MetaPath)
