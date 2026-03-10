param(
    [string]$Device = "cpu",
    [string]$Cases = "kerr220,kerr221,kerr330",
    [switch]$WithOverlay,
    [switch]$WithArtifactAudit
)

$ErrorActionPreference = "Stop"
$RunTag = Get-Date -Format "yyyyMMdd-HHmmss"
$RootOut = Join-Path "reports/posteriors/fig1_paper_precision" ("run_" + $RunTag)
New-Item -ItemType Directory -Path $RootOut -Force | Out-Null

$StdoutLog = Join-Path $RootOut "driver_stdout.log"
$StderrLog = Join-Path $RootOut "driver_stderr.log"
$MetaPath = Join-Path $RootOut "run_meta.json"

$Args = @(
    "-u",
    "scripts/12_run_fig1_paper_repro.py",
    "--cases", $Cases,
    "--device", $Device,
    "--run-tag", $RunTag
)
if ($WithOverlay) {
    $Args += "--with-overlay"
}
if ($WithArtifactAudit) {
    $Args += "--with-artifact-audit"
}

$Proc = Start-Process `
    -FilePath "python" `
    -ArgumentList $Args `
    -WorkingDirectory (Resolve-Path ".") `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

$Meta = [ordered]@{
    pid = $Proc.Id
    run_tag = $RunTag
    cases = $Cases
    device = $Device
    with_overlay = [bool]$WithOverlay
    with_artifact_audit = [bool]$WithArtifactAudit
    out_dir = $RootOut
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    started_at = (Get-Date).ToString("s")
}
($Meta | ConvertTo-Json -Depth 4) | Set-Content -Encoding UTF8 $MetaPath

Write-Output ("started pid={0}" -f $Proc.Id)
Write-Output ("out_dir={0}" -f $RootOut)
Write-Output ("stdout={0}" -f $StdoutLog)
Write-Output ("stderr={0}" -f $StderrLog)
Write-Output ("meta={0}" -f $MetaPath)
