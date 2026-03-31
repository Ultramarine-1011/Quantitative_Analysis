#Requires -Version 5.1
param(
    [string]$RepoName = "Quantitative_Analysis",
    [switch]$Private,
    [string]$GhExe = "C:\Program Files\GitHub CLI\gh.exe"
)

$ErrorActionPreference = "Stop"
$rootPath = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not (Test-Path -LiteralPath $GhExe)) {
    Write-Error "GitHub CLI not found: $GhExe. Install it or pass -GhExe."
}

Push-Location $rootPath
try {
    & $GhExe auth status
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Run once: & `"$GhExe`" auth login   OR set GH_TOKEN / GITHUB_TOKEN, then re-run this script."
    }

    $visFlag = if ($Private) { "--private" } else { "--public" }
    git remote get-url origin 2>$null
    $noOrigin = $LASTEXITCODE -ne 0
    if ($noOrigin) {
        Write-Host "Creating GitHub repo and pushing..."
        $desc = "Quantitative analysis / Streamlit"
        & $GhExe repo create $RepoName $visFlag "--source=$rootPath" "--remote=origin" "--push" "--description" $desc
    }
    else {
        Write-Host "Remote origin exists; pushing main..."
        git push -u origin main
    }
}
finally {
    Pop-Location
}
