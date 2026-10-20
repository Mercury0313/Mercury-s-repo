$ErrorActionPreference = "Stop"
$Python = "D:\Anaconda\envs\pytorch\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python interpreter not found: $Python"
}

# RouteLLM creates an OpenAI client during router-module import. The random
# router does not send requests, so this placeholder is never transmitted.
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = "sk-local-smoke-test"
}
$env:LITELLM_LOCAL_MODEL_COST_MAP = "true"

& $Python (Join-Path $PSScriptRoot "smoke_test.py")
if ($LASTEXITCODE -ne 0) {
    throw "Offline smoke test failed."
}
