param(
  [string]$ProjectRoot = (Resolve-Path ".").Path
)

$stateDir = Join-Path $ProjectRoot "state"
$targets = @(
  (Join-Path $stateDir "symbolic_memory.msgpack"),
  (Join-Path $stateDir "symbolic_memory.msgpack.bak"),
  (Join-Path $stateDir "game_state.json")
)

foreach ($t in $targets) {
  if (Test-Path $t) {
    Remove-Item -Force $t
    Write-Host "Deleted $t"
  }
}

Write-Host "State reset complete."
