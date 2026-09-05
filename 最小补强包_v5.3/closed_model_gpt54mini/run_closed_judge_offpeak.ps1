param(
    [ValidateRange(1, 10)]
    [int]$Concurrent = 1,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$packageRoot = Split-Path -Parent $PSScriptRoot
$codeRoot = Split-Path -Parent $packageRoot
$projectRoot = Split-Path -Parent $codeRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$pipelineRoot = Join-Path $codeRoot 'LADDER_v2_Code'
$rawDir = Join-Path $PSScriptRoot 'raw'
$judgeDir = Join-Path $PSScriptRoot 'disclosure_gpt54mini_judge'
$statsDir = Join-Path $PSScriptRoot 'statistics_gpt54mini_judge'
$shapeRawName = 'E1-gpt-5.4-mini-baseline-30s-20q-20260822_145031.json'
$ladderRawName = 'A3-gpt-5.4-mini-baseline-30s-20q-20260822_153650.json'
$shapeJudged = Join-Path $judgeDir ($shapeRawName -replace '\.json$', '_disclosure.json')
$ladderJudged = Join-Path $judgeDir ($ladderRawName -replace '\.json$', '_disclosure.json')

foreach ($requiredPath in @(
    $pythonExe,
    (Join-Path $pipelineRoot 'pipeline\run_disclosure_eval.py'),
    (Join-Path $packageRoot 'scripts\analyze_shape_vs_ladder.py'),
    (Join-Path $rawDir $shapeRawName),
    (Join-Path $rawDir $ladderRawName)
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path is missing: $requiredPath"
    }
}

if ((Test-Path -LiteralPath $shapeJudged) -and (Test-Path -LiteralPath $ladderJudged) -and -not $Force) {
    Write-Host 'Existing Judge outputs found; skipping paid API calls.'
}
else {
    if (((Test-Path -LiteralPath $shapeJudged) -or (Test-Path -LiteralPath $ladderJudged)) -and -not $Force) {
        throw 'Only one Judge output exists. Audit the partial state, then use -Force only if a paid rerun is intended.'
    }
    Push-Location $pipelineRoot
    try {
        & $pythonExe 'pipeline\run_disclosure_eval.py' `
            --input $rawDir `
            --output $judgeDir `
            --model 'gpt-5.4-mini' `
            --provider 'apinebula' `
            --concurrent $Concurrent
        if ($LASTEXITCODE -ne 0) {
            throw "GPT-5.4-mini Judge exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

foreach ($judgedPath in @($shapeJudged, $ladderJudged)) {
    if (-not (Test-Path -LiteralPath $judgedPath)) {
        throw "Expected Judge output is missing: $judgedPath"
    }
}

& $pythonExe (Join-Path $packageRoot 'scripts\analyze_shape_vs_ladder.py') `
    --shape $shapeJudged `
    --ladder $ladderJudged `
    --output $statsDir `
    --bootstrap 5000 `
    --seed 42
if ($LASTEXITCODE -ne 0) {
    throw "Paired statistics exited with code $LASTEXITCODE"
}

Write-Host "Completed. Final statistics: $statsDir"
