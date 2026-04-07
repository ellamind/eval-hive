#!/bin/bash
set -euo pipefail

RUN_DIR=${1:?Usage: $0 <run_dir> <config>}
CONFIG=${2:?Usage: $0 <run_dir> <config>}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

CYCLE=0
while true; do
    CYCLE=$((CYCLE + 1))
    log "=== Cycle $CYCLE start ==="
    log "Collecting results..."
    pixi run eval-hive collect "$RUN_DIR" --upload
    log "Compacting..."
    pixi run eval-hive compact "$RUN_DIR"
    log "Creating run..."
    pixi run eval-hive create-run --config "$CONFIG" --output "$RUN_DIR" --force
    log "Submitting jobs..."
    pixi run eval-hive submit "$RUN_DIR"
    log "=== Cycle $CYCLE done, sleeping 1h ==="
    sleep 3600
done
