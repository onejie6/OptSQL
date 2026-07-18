param(
    [string]$ConfigPath = "config/local/config-bird-dev.toml",
    [string]$ExperimentName = "qwen3-coder-plus-bird-dev-full",
    [string]$OptSQLRoot = $env:OPTSQL_ROOT
)

$ErrorActionPreference = "Stop"
$stages = @(
    "FewShotIndex", "ValueRetrieval", "FewShotPreparation", "SchemaLinking",
    "SqlGeneration", "SqlRevision", "SqlSelection", "Evaluation", "Export",
    "Controller"
)

foreach ($stage in $stages) {
    & "$PSScriptRoot/run_full_stage.ps1" `
        -Stage $stage `
        -ConfigPath $ConfigPath `
        -ExperimentName $ExperimentName `
        -OptSQLRoot $OptSQLRoot
}
