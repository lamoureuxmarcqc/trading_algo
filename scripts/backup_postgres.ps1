[CmdletBinding()]
param(
    [string]$BackupDirectory = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $BackupDirectory -PathType Container)) {
    throw "Le dossier de backup est introuvable : $BackupDirectory"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker doit etre disponible dans le PATH."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupFile = Join-Path -Path $BackupDirectory -ChildPath "backup_$timestamp.sql"

& docker exec lamoureux-postgres pg_dump -U lamoureux lamoureux > $backupFile
if ($LASTEXITCODE -ne 0) {
    throw "pg_dump a echoue avec le code de sortie $LASTEXITCODE."
}

if (-not (Test-Path -LiteralPath $backupFile -PathType Leaf)) {
    throw "Le fichier de backup n'a pas ete cree : $backupFile"
}

$backup = Get-Item -LiteralPath $backupFile
if ($backup.Length -le 0) {
    throw "Le fichier de backup est vide : $backupFile"
}

Write-Host ("Backup cree : {0} ({1:N2} MB)" -f $backup.FullName, ($backup.Length / 1MB))

$cutoff = (Get-Date).AddDays(-7)
Get-ChildItem -LiteralPath $BackupDirectory -Filter "backup_*.sql" -File |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Force
        Write-Host "Backup expire supprime : $($_.Name)"
    }
