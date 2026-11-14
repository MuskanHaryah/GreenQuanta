# Guide 2 step 2.5: end-to-end smoke of the backend (Windows PowerShell 5.1), from the repo root:
#     powershell -NoProfile -ExecutionPolicy Bypass -File backend\smoke.ps1
# Starts python -m backend, GETs every endpoint for every size (every method's placement, every allowed GAS seed),
# checks HTTP 200 and that the body parses as JSON, then stops the server. Exit 1 on any failure.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv-gq\Scripts\python.exe"
$base = "http://127.0.0.1:5000/api"
$seeds = (& $python -B -c "import sys; sys.path.insert(0, 'optimizer'); from diagnostics.s1_end_to_end import RUN_CONFIG; print(*RUN_CONFIG['pipe_gas_seeds'])").Split(" ")
$server = Start-Process -FilePath $python -ArgumentList "-B", "-m", "backend" -WorkingDirectory $repo -PassThru `
    -WindowStyle Hidden -RedirectStandardError (Join-Path $env:TEMP "gq_backend_smoke.log")
$script:failed = 0
$script:lines = New-Object System.Collections.Generic.List[string]

function Get-Json($path) {
    $started = Get-Date
    try {
        $r = Invoke-WebRequest -UseBasicParsing "$base$path"
        $body = $r.Content | ConvertFrom-Json
        $ms = [math]::Round(((Get-Date) - $started).TotalMilliseconds)
        $script:lines.Add(("PASS {0} {1} {2:N0} bytes {3} ms" -f $r.StatusCode, $path, $r.RawContentLength, $ms))
        return , $body
    } catch {
        $script:failed++
        $script:lines.Add("FAIL $path : $($_.Exception.Message)")
        return $null
    }
}

try {
    for ($i = 0; $i -lt 120; $i++) {
        try { Invoke-WebRequest -UseBasicParsing "$base/health" | Out-Null; break } catch { Start-Sleep -Milliseconds 500 }
    }
    $null = Get-Json "/health"
    foreach ($s in (Get-Json "/sizes")) {
        $id = $s.id
        $null = Get-Json "/results/$id"
        foreach ($row in (Get-Json "/results/$id/compare")) { $null = Get-Json "/results/$id/placement/$($row.method_id)" }
        foreach ($seed in $seeds) { $null = Get-Json "/gas-trace/${id}?seed=$seed" }
        $null = Get-Json "/claims?size=$id"
    }
} finally {
    & taskkill /F /T /PID $server.Id | Out-Null
}
$script:lines.Add("server stopped: " + ($null -eq (Get-Process -Id $server.Id -ErrorAction SilentlyContinue)))
$script:lines.Add("SMOKE " + $(if ($script:failed -eq 0) { "PASS" } else { "FAIL ($($script:failed) failed)" }))
$script:lines
exit [int]($script:failed -ne 0)
