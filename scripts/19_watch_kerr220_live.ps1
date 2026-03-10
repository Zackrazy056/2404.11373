param(
    [string]$RunDir = "",
    [int]$RefreshSeconds = 2,
    [int]$TailLines = 12,
    [switch]$StopWhenFinished
)

$ErrorActionPreference = "Stop"

function Resolve-LatestRunDir {
    param([string]$Root)
    $dirs = Get-ChildItem -Path $Root -Directory -Filter "kerr220_*" | Sort-Object LastWriteTime -Descending
    if (-not $dirs -or $dirs.Count -eq 0) {
        throw "No kerr220_* run directory found under $Root"
    }
    return $dirs[0].FullName
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    try {
        return Get-Content -Path $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Tail-File {
    param(
        [string]$Path,
        [int]$Lines
    )
    if (-not (Test-Path $Path)) {
        return @("(missing) $Path")
    }
    $content = Get-Content -Path $Path -Tail $Lines -ErrorAction SilentlyContinue
    if ($null -eq $content -or $content.Count -eq 0) {
        return @("(empty) $Path")
    }
    return $content
}

$root = Join-Path (Resolve-Path ".") "reports/posteriors/fig1_paper_precision"
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $runPath = Resolve-LatestRunDir -Root $root
} else {
    $runPath = if ([System.IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path (Resolve-Path ".") $RunDir }
}

$statusPath = Join-Path $runPath "kerr220_status.json"
$heartbeatPath = Join-Path $runPath "kerr220_heartbeat.log"
$caseLogPath = Join-Path $runPath "kerr220.log"

Write-Host ("Watching run: {0}" -f $runPath)
Write-Host ("Status: {0}" -f $statusPath)
Write-Host ("Heartbeat: {0}" -f $heartbeatPath)
Write-Host ("Case log: {0}" -f $caseLogPath)
Write-Host "Press Ctrl+C to stop."

while ($true) {
    Clear-Host
    $status = Read-JsonFile -Path $statusPath

    Write-Host ("Run:       {0}" -f $runPath)
    Write-Host ("Refreshed: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    if ($null -ne $status) {
        Write-Host ("State:     {0}" -f $status.state)
        Write-Host ("Event:     {0}" -f $status.event)
        Write-Host ("Round:     {0}" -f $status.round_index)
        Write-Host ("Phase:     {0}" -f $status.phase)
        Write-Host ("n_sim:     {0}/{1}" -f $status.num_simulations, $status.requested_num_simulations)
        Write-Host ("Elapsed:   {0:N1}s" -f ([double]$status.phase_elapsed_seconds))
        if ($null -ne $status.truncated_prior_volume) {
            Write-Host ("Volume:    {0}" -f ([double]$status.truncated_prior_volume).ToString("F6"))
        }
        if ($null -ne $status.probe_acceptance_rate) {
            Write-Host ("Probe acc: {0}" -f ([double]$status.probe_acceptance_rate).ToString("F6"))
        }
        if ($null -ne $status.epochs_trained_this_round) {
            Write-Host ("Epochs:    {0}" -f $status.epochs_trained_this_round)
        }
        if ($null -ne $status.simulation_budget_adjusted) {
            Write-Host ("Budget adj:{0}" -f ($(if ($status.simulation_budget_adjusted) { " yes" } else { " no" })))
        }
        if ($null -ne $status.truncation_relaxed) {
            Write-Host ("Trunc rel: {0}" -f ($(if ($status.truncation_relaxed) { "yes" } else { "no" })))
        }
        if ($null -ne $status.error) {
            Write-Host ("Error:     {0}" -f $status.error) -ForegroundColor Red
        }
    } else {
        Write-Host "State:     waiting for status file..."
    }

    Write-Host ""
    Write-Host "Heartbeat tail"
    Write-Host "--------------"
    Tail-File -Path $heartbeatPath -Lines $TailLines | ForEach-Object { Write-Host $_ }

    Write-Host ""
    Write-Host "Case log tail"
    Write-Host "-------------"
    Tail-File -Path $caseLogPath -Lines $TailLines | ForEach-Object { Write-Host $_ }

    if ($StopWhenFinished -and $null -ne $status) {
        if (@("completed", "failed") -contains [string]$status.state) {
            break
        }
    }

    Start-Sleep -Seconds $RefreshSeconds
}
