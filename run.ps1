param([Parameter(Position=0)][string]$Command = "help")
$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
$VENV = Join-Path $ROOT "venv\Scripts\Activate.ps1"
$AI_GATEWAY = Join-Path $ROOT "services\ai-gateway"
$API_CORE = Join-Path $ROOT "services\api-core"
$DOCKER_DIR = Join-Path $ROOT "infra\docker"
$OLLAMA_MODEL = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "llama3.2:3b" }

function Ensure-Venv {
    if (Test-Path $VENV) {
        if (-not $env:VIRTUAL_ENV) {
            . $VENV
        }
    }
}

function Show-Help {
    Write-Host ""
    Write-Host "LAMOUREUX AI PLATFORM - Commandes" -ForegroundColor Cyan
    Write-Host "  .\run.ps1 up            Demarre la stack" -ForegroundColor White
    Write-Host "  .\run.ps1 down          Arrete la stack" -ForegroundColor White
    Write-Host "  .\run.ps1 restart       Redemarre" -ForegroundColor White
    Write-Host "  .\run.ps1 logs          Logs stack" -ForegroundColor White
    Write-Host "  .\run.ps1 ps            Etat conteneurs" -ForegroundColor White
    Write-Host "  .\run.ps1 pull-model    Pull Ollama model" -ForegroundColor White
    Write-Host "  .\run.ps1 warmup        Precharge le modele Ollama" -ForegroundColor White
    Write-Host "  .\run.ps1 test          Tous les tests" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 test-gateway  Tests ai-gateway" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 test-api-core Tests api-core" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 test-root     Tests racine" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 smoke         Test end-to-end" -ForegroundColor Green
    Write-Host "  .\run.ps1 health        Statut gateway" -ForegroundColor Green
    Write-Host "  .\run.ps1 policy        Matrice privacy" -ForegroundColor Green
    Write-Host "  .\run.ps1 clean         Nettoyer caches" -ForegroundColor Magenta
    Write-Host ""
}

function Run-In($Path, $Script) {
    Push-Location $Path
    try { & $Script } finally { Pop-Location }
}

function Run-Tests($Path, $Label) {
    Write-Host ""
    Write-Host "=== Tests $Label ===" -ForegroundColor Cyan
    Push-Location $Path
    try {
        & pytest -q 2>&1 | Out-Host
        $script:lastExit = $LASTEXITCODE
    } finally { Pop-Location }
}

function Invoke-API {
    Write-Host "Demarrage de l'API knowledge (port 8002)..." -ForegroundColor Cyan
    Push-Location "$ROOT\services\knowledge"
    try {
        uvicorn app.main:app --host 0.0.0.0 --port 8002
    } finally {
        Pop-Location
    }
}

switch ($Command.ToLower()) {
    "help" { Show-Help }
    "up" { Write-Host "Demarrage..." -ForegroundColor Cyan; Run-In $DOCKER_DIR { docker compose up -d }; Write-Host "OK" -ForegroundColor Green }
    "down" { Write-Host "Arret..." -ForegroundColor Cyan; Run-In $DOCKER_DIR { docker compose down }; Write-Host "OK" -ForegroundColor Green }
    "restart" { Run-In $DOCKER_DIR { docker compose down }; Run-In $DOCKER_DIR { docker compose up -d } }
    "logs" { Run-In $DOCKER_DIR { docker compose logs -f } }
    "ps" { Run-In $DOCKER_DIR { docker compose ps } }
    "pull-model" { docker exec lamoureux-ollama ollama pull $OLLAMA_MODEL }
    "api" { Invoke-API }
    "warmup" {
        Write-Host "Warmup Ollama ($OLLAMA_MODEL)..." -ForegroundColor Cyan
        $body = "{`"model`":`"$OLLAMA_MODEL`",`"prompt`":`"hi`",`"stream`":false}"
        $null = Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method POST -ContentType "application/json" -Body $body
        Write-Host "Modele precharge" -ForegroundColor Green
    }
    "test" {
        Ensure-Venv
        Run-Tests $AI_GATEWAY "ai-gateway"; $c1 = $script:lastExit
        Run-Tests $API_CORE "api-core";   $c2 = $script:lastExit
        Run-Tests $ROOT "racine";         $c3 = $script:lastExit
        Write-Host ""
        if ($c1 -eq 0 -and $c2 -eq 0 -and $c3 -eq 0) {
            Write-Host "Toutes les suites passent" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "Echecs: gateway=$c1 api-core=$c2 racine=$c3" -ForegroundColor Red
            exit 1
        }
    }
    "test-gateway"  { Ensure-Venv; Run-Tests $AI_GATEWAY "ai-gateway"; exit $script:lastExit }
    "test-api-core" { Ensure-Venv; Run-Tests $API_CORE "api-core"; exit $script:lastExit }
    "test-root"     { Ensure-Venv; Run-Tests $ROOT "racine"; exit $script:lastExit }
    "smoke" {
        $body = '{"prompt":"Say hello in one word.","classification":"PUBLIC","tenant_id":"smoke","target":"LOCAL"}'
        $r = Invoke-RestMethod -Uri http://localhost:8001/v1/inference -Method POST -ContentType "application/json" -Body $body
        Write-Host "Content : $($r.content)"
        Write-Host "Model   : $($r.model)"
        Write-Host "Latency : $($r.latency_ms) ms"
    }
    "health" { Invoke-RestMethod -Uri http://localhost:8001/health | ConvertTo-Json }
    "policy" { Invoke-RestMethod -Uri http://localhost:8001/v1/policy | ConvertTo-Json }
    "clean" {
        Get-ChildItem -Path $ROOT -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $ROOT -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path $ROOT -Recurse -Directory -Filter ".pytest_tmp" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Clean OK" -ForegroundColor Green
    }
    "clean-docker" { docker builder prune -f; docker system prune -f }
    default { Write-Host "Inconnu: $Command" -ForegroundColor Red; Show-Help }
}


# Tests knowledge (48)
cd C:\Users\marc\source\projet_trading\services\knowledge
pytest -q

# Puis chaque autre service
cd ..\memory          ; pytest -q
cd ..\orchestrator    ; pytest -q
cd ..\model-registry  ; pytest -q
cd ..\evaluation      ; pytest -q
cd ..\self-improvement; pytest -q
