param(
  [string]$ProjectRoot = (Resolve-Path ".").Path
)

$stateDir = Join-Path $ProjectRoot "state"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$files = @(
  "symbolic_memory.msgpack",
  "symbolic_memory.msgpack.bak",
  "game_state.json"
)

foreach ($f in $files) {
  $src = Join-Path $ProjectRoot $f
  $dst = Join-Path $stateDir $f
  if (Test-Path $src) {
    Move-Item -Force $src $dst
    Write-Host "Moved $f -> state/"
  }
}

Write-Host "Done. You can now commit the state/ folder to snapshot your current run."
