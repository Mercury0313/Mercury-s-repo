[CmdletBinding()]
param(
    [string]$PublishRepository = "D:\router\routeLLM-publish-snapshot",
    [string]$Message = "Sync RouteLLM from local repository"
)

$ErrorActionPreference = "Stop"
$SourceRepository = Split-Path -Parent $PSScriptRoot
$SourceRepository = [System.IO.Path]::GetFullPath($SourceRepository)
$PublishRepository = [System.IO.Path]::GetFullPath($PublishRepository)
$TargetDirectory = Join-Path $PublishRepository "routeLLM"

if (-not (Test-Path (Join-Path $PublishRepository ".git"))) {
    throw "Publish repository not found: $PublishRepository"
}
if ((git -C $SourceRepository status --porcelain)) {
    throw "Source repository has uncommitted changes. Commit them before publishing."
}
if ((git -C $PublishRepository branch --show-current) -ne "main") {
    throw "Publish repository must be on branch 'main'."
}
if ((git -C $PublishRepository status --porcelain)) {
    throw "Publish repository has uncommitted changes. Resolve them before publishing."
}

$TemporaryRoot = Join-Path $env:TEMP ("routellm-publish-" + [guid]::NewGuid())
$ArchivePath = Join-Path $TemporaryRoot "routellm.zip"
$ExportDirectory = Join-Path $TemporaryRoot "export"

try {
    New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null
    git -C $SourceRepository archive --format=zip --output=$ArchivePath HEAD
    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExportDirectory

    New-Item -ItemType Directory -Force -Path $TargetDirectory | Out-Null
    # /MIR intentionally makes only the target routeLLM directory match HEAD.
    # The repository root and its other projects are never mirrored or deleted.
    robocopy $ExportDirectory $TargetDirectory /MIR /XD .git __pycache__ | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE"
    }

    git -C $PublishRepository add -- routeLLM
    git -C $PublishRepository diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "No committed RouteLLM changes to publish."
        exit 0
    }

    git -C $PublishRepository commit -m $Message
    git -C $PublishRepository push origin main
    Write-Host "Published RouteLLM to origin/main:routeLLM"
}
finally {
    if (Test-Path $TemporaryRoot) {
        Remove-Item -LiteralPath $TemporaryRoot -Recurse -Force
    }
}
