[CmdletBinding()]
param(
    [string]$PythonPath = "D:\Anaconda\envs\pytorch\python.exe",
    [switch]$IncludeEvaluation
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ReproDirectory = Join-Path $RepositoryRoot ".repro"

if (-not (Test-Path $PythonPath)) {
    throw "Python interpreter not found: $PythonPath"
}
$Python = $PythonPath

& $Python -m pip install --upgrade pip wheel
& $Python -m pip install -e . --no-deps --no-build-isolation

# Reuse the selected Conda environment's existing PyTorch installation.
$RuntimeDependencies = @(
    "pyyaml", "pydantic", "numpy", "pandas", "scikit-learn", "tqdm",
    "openai", "transformers", "datasets", "litellm", "fastapi", "shortuuid", "uvicorn"
)
& $Python -m pip install @RuntimeDependencies

if ($IncludeEvaluation) {
    & $Python -m pip install matplotlib pandarallel sglang tiktoken
}

New-Item -ItemType Directory -Force -Path $ReproDirectory | Out-Null
& $Python -m pip freeze | Set-Content -Encoding utf8 (Join-Path $ReproDirectory "requirements.freeze.txt")
& $Python --version
Write-Host "Installed RouteLLM into $Python and wrote .repro\\requirements.freeze.txt"
