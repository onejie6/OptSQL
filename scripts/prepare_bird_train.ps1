$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$birdRoot = Join-Path $root "data\bird"
$trainZip = Join-Path $birdRoot "train.zip"
$python = Join-Path $root ".venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $birdRoot | Out-Null
if (-not (Test-Path $trainZip)) {
    curl.exe -L --fail --retry 8 --retry-delay 5 -C - -o $trainZip `
        "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
    if ($LASTEXITCODE -ne 0) { throw "BIRD train download failed." }
}

$trainRoot = Join-Path $birdRoot "train"
$databaseRoot = Join-Path $trainRoot "train_databases"
if (-not (Test-Path $databaseRoot)) {
    & $python -c "import zipfile; zipfile.ZipFile(r'$trainZip').extractall(r'$birdRoot')"
    if ($LASTEXITCODE -ne 0) { throw "BIRD train.zip extraction failed." }
}

$nestedDatabaseZip = Join-Path $trainRoot "train_databases.zip"
if ((Test-Path $nestedDatabaseZip) -and -not (Test-Path $databaseRoot)) {
    & $python -c "import zipfile; zipfile.ZipFile(r'$nestedDatabaseZip').extractall(r'$trainRoot')"
    if ($LASTEXITCODE -ne 0) { throw "BIRD nested database extraction failed." }
}

$filteredJsonl = Join-Path $trainRoot "train_filtered.jsonl"
$columnMeaning = Join-Path $trainRoot "train_column_meaning_filtered.json"
if (-not (Test-Path $filteredJsonl)) {
    curl.exe -L --fail --retry 8 -o $filteredJsonl `
        "https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/data/train-00000-of-00001.jsonl"
}
if (-not (Test-Path $columnMeaning)) {
    curl.exe -L --fail --retry 8 -o $columnMeaning `
        "https://huggingface.co/datasets/birdsql/bird23-train-filtered/resolve/main/train_column_meaning.json"
}

$conversion = @'
import json
from pathlib import Path

root = Path(r"__TRAIN_ROOT__")
records = []
for question_id, line in enumerate((root / "train_filtered.jsonl").read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue
    record = json.loads(line)
    record.setdefault("question_id", question_id)
    record.setdefault("difficulty", "")
    records.append(record)
(root / "train.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
with (root / "train_gold.sql").open("w", encoding="utf-8") as handle:
    for record in records:
        handle.write(f"{record['SQL']}\t{record['db_id']}\n")
print(f"Prepared {len(records)} filtered BIRD train records.")
'@.Replace("__TRAIN_ROOT__", $trainRoot.Replace("\", "\\"))
$conversion | & $python -
if ($LASTEXITCODE -ne 0) { throw "BIRD train label conversion failed." }

Write-Host "BIRD train is ready at $trainRoot"
