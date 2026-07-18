param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "FewShotIndex", "ValueRetrieval", "FewShotPreparation", "SchemaLinking",
        "SqlGeneration", "SqlRevision", "SqlSelection", "Evaluation", "Export",
        "Controller"
    )]
    [string]$Stage,
    [string]$ConfigPath = "config/local/config-bird-dev.toml",
    [string]$ExperimentName = "qwen3-coder-plus-bird-dev-full",
    [string]$OptSQLRoot = $env:OPTSQL_ROOT
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$resolvedConfig = Resolve-Path $ConfigPath -ErrorAction Stop
$env:PYTHONUTF8 = "1"
$env:CONFIG_PATH = $resolvedConfig.Path
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $root "cache/huggingface" }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $root "cache/uv" }

$apiStages = @(
    "FewShotIndex", "ValueRetrieval", "FewShotPreparation", "SchemaLinking",
    "SqlGeneration", "SqlRevision", "SqlSelection"
)
if ($Stage -in $apiStages) {
    if (-not $env:DS_API_KEY) { throw "DS_API_KEY is not set." }
    if (-not $env:DS_BASE_URL) { throw "DS_BASE_URL is not set." }
}

$venvPython = Join-Path $root ".venv/Scripts/python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$runRoot = Join-Path $root "workspace/runs/$ExperimentName"
$snapshot = Join-Path $runRoot "sql_selection.snapshot"
$outputDir = Join-Path $runRoot "optsql_batch"
$logRoot = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$logPath = Join-Path $logRoot ("{0}.log" -f $Stage)

function Invoke-StageCommand {
    param([string[]]$Arguments)
    $started = Get-Date
    "[$($started.ToString('s'))] Starting $Stage" | Tee-Object -FilePath $logPath -Append
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python @Arguments 2>> "$logPath.stderr.log" |
        Tee-Object -FilePath $logPath -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($exitCode -ne 0) {
        throw "$Stage failed with exit code $exitCode. See $logPath"
    }
    $hours = [math]::Round(((Get-Date) - $started).TotalHours, 2)
    "[$((Get-Date).ToString('s'))] Completed $Stage in $hours hours" |
        Tee-Object -FilePath $logPath -Append
}

switch ($Stage) {
    "FewShotIndex" {
        Invoke-StageCommand @("runner/build_few_shot_index.py", "--config", $env:CONFIG_PATH)
    }
    "ValueRetrieval" {
        Invoke-StageCommand @("runner/run_value_retrieval.py")
    }
    "FewShotPreparation" {
        Invoke-StageCommand @("runner/run_few_shot_preparation.py", "--config", $env:CONFIG_PATH)
    }
    "SchemaLinking" {
        Invoke-StageCommand @("runner/run_schema_linking.py")
    }
    "SqlGeneration" {
        Invoke-StageCommand @("runner/run_sql_generation.py")
    }
    "SqlRevision" {
        Invoke-StageCommand @("runner/run_sql_revision.py")
    }
    "SqlSelection" {
        Invoke-StageCommand @("runner/run_sql_selection.py")
    }
    "Evaluation" {
        Invoke-StageCommand @("runner/evaluation.py")
    }
    "Export" {
        if (-not (Test-Path $snapshot)) { throw "Missing snapshot: $snapshot" }
        Invoke-StageCommand @(
            "runner/convert_snapshot_to_sql.py", "--snapshot_path", $snapshot,
            "--output", (Join-Path $runRoot "predictions.json")
        )
    }
    "Controller" {
        if (-not (Test-Path $snapshot)) { throw "Missing snapshot: $snapshot" }
        if (-not $OptSQLRoot -or -not (Test-Path $OptSQLRoot)) {
            throw "Set OPTSQL_ROOT to a valid OptSQL controller checkout."
        }
        Invoke-StageCommand @(
            "integration/run_optsql_bridge_batch.py", "--snapshot", $snapshot,
            "--optsql-root", $OptSQLRoot, "--output-dir", $outputDir
        )
    }
}
