import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from loguru import logger

from eval_hive.config import EhConfig, load_config


# ── Manifest ──────────────────────────────────────────────────


def _sanitize(s: str) -> str:
    """Replace characters not safe for SLURM job names."""
    return re.sub(r'[^a-zA-Z0-9_-]', '-', s)


def manifest_key(model_key: str, label: str) -> str:
    """Build stable manifest key from model_key and label.

    Uses '--' as separator so single dashes within names never collide.
    """
    return f"{_sanitize(model_key)}--{_sanitize(label)}"


def build_manifest(config: EhConfig) -> dict[str, dict]:
    """Build manifest mapping stable keys to model entries.

    Returns a dict keyed by manifest_key(model_key, label):
    {"Model--main": {"model_key": "...", "label": "...", "model_path": "..."}, ...}
    """
    manifest = {}
    for model_key, entry in config.models.items():
        for label, path in entry.resolve_model_paths():
            key = manifest_key(model_key, label)
            if key in manifest:
                raise ValueError(
                    f"Duplicate manifest key '{key}' from "
                    f"model_key='{model_key}', label='{label}'. "
                    f"Ensure model_key + label combinations are unique."
                )
            manifest[key] = {
                "model_key": model_key,
                "label": label,
                "model_path": str(path),
            }
    return manifest


# ── Template helpers ──────────────────────────────────────────


def build_additional_sbatch_lines(config: EhConfig) -> str:
    lines = ""
    if config.additional_sbatch_args:
        for key, value in config.additional_sbatch_args.items():
            if not key.startswith("--"):
                key = f"--{key}"
            lines += f"#SBATCH {key}={value}\n"
    return lines


def build_env_exports(config: EhConfig) -> str:
    lines = ""
    if config.env_vars:
        for key, value in config.env_vars.items():
            lines += f'export {key}="{value}"\n'
    return lines


def build_env_activation_block(config: EhConfig) -> str:
    if config.env_activation_command:
        return (
            f'log "INFO" "Activating environment with custom command"\n'
            f'{config.env_activation_command}'
        )
    return (
        f'log "INFO" "Activating pixi environment: {config.pixi_env}"\n'
        f'eval "$(pixi shell-hook --manifest-path {config.pixi_manifest} '
        f'-e {config.pixi_env} --no-install)"'
    )


def build_model_args_string(config: EhConfig) -> str:
    """Join eval.model_args as key=value,key=value.

    Values like ${EH_MODEL_PATH} and ${EH_PORT} are kept as literal text.
    Bash expands them at runtime since they appear inside double quotes.

    Injects eval.timeout and eval.max_retries unless already set in model_args.
    """
    args = dict(config.eval.model_args)
    args.setdefault("timeout", config.eval.timeout)
    args.setdefault("max_retries", config.eval.max_retries)
    return ",".join(f"{k}={v}" for k, v in args.items())


def build_lm_eval_extra_args(config: EhConfig) -> str:
    """Convert eval.lm_eval_args dict to --key value CLI flags.

    Boolean true → bare flag (--log_samples).
    Other values → --key value.
    Automatically adds --cache_requests true when request_cache_dir is set.
    """
    parts = []
    for key, value in config.eval.lm_eval_args.items():
        if value is True:
            parts.append(f"--{key}")
        elif value is not False and value is not None:
            parts.append(f"--{key} {value}")

    # Auto-enable request caching when a cache dir is configured
    if config.request_cache_dir and "cache_requests" not in config.eval.lm_eval_args:
        parts.append("--cache_requests true")

    return " ".join(parts)


def build_suites_bash_array(config: EhConfig) -> str:
    items = " ".join(f'"{s}"' for s in config.eval.suites_and_tasks)
    return f"({items})"


def clean_server_command(cmd: str) -> str:
    """Remove YAML folded block scalar artifacts (backslash + whitespace)."""
    cmd = re.sub(r'\\\s+', ' ', cmd)
    cmd = ' '.join(cmd.split())
    return cmd


# ── SBATCH template ──────────────────────────────────────────

SBATCH_TEMPLATE = r"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --qos={qos}
#SBATCH --nodes={total_nodes_per_job}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gres={gres_per_node}
#SBATCH --time={time_limit}
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output={log_dir}/%x-%j.log
#SBATCH --error={log_dir}/%x-%j.log
{additional_sbatch_lines}
# ═══════════════════════════════════════════════════════════════
# eval-hive SLURM job script (auto-generated — do not edit)
# ═══════════════════════════════════════════════════════════════

echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${{SLURM_JOB_NAME}}"
echo "SLURM_JOB_ID: ${{SLURM_JOB_ID}}"
echo "EH_TASK_KEY: ${{EH_TASK_KEY}}"
echo "SLURM_JOB_NUM_NODES: ${{SLURM_JOB_NUM_NODES}}"
echo "SLURM_JOB_NODELIST: ${{SLURM_JOB_NODELIST}}"
echo "SLURM_JOB_PARTITION: ${{SLURM_JOB_PARTITION}}"
echo "SLURM_JOB_ACCOUNT: ${{SLURM_JOB_ACCOUNT}}"
echo "============================="

set -e

# ── Paths ──────────────────────────────────────────────────────
RUN_DIR="{run_dir}"
MANIFEST="$RUN_DIR/eh_manifest.json"
COMPLETED_FILE="{progress_dir}/jobs_completed.log"
FAILED_FILE="{progress_dir}/jobs_failed.log"
TASK_KEY="${{EH_TASK_KEY}}"

if [ -z "$TASK_KEY" ]; then
    echo "FATAL: EH_TASK_KEY is not set. This script must be launched via eval-hive submit."
    exit 1
fi

# ── Logging ────────────────────────────────────────────────────
log() {{
    local level="$1"
    local message="$2"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message"
}}

log_failed() {{
    local reason="$1"
    local info="$2"
    local ts=$(date '+%Y-%m-%d %H:%M:%S')
    local job="${{SLURM_JOB_ID}}"
    echo "${{TASK_KEY}} ${{ts}} ${{reason}} ${{job}} ${{info}}" >> "${{FAILED_FILE}}"
    log "ERROR" "Marked task ${{TASK_KEY}} as failed: ${{reason}} ${{info}}"
}}

# ── Check if already completed ─────────────────────────────────
if [ -f "$COMPLETED_FILE" ]; then
    if grep -qFx "${{TASK_KEY}}" "$COMPLETED_FILE"; then
        log "INFO" "Task ${{TASK_KEY}} is already completed. Exiting."
        exit 0
    fi
fi

# ── Read manifest entry ───────────────────────────────────────
EH_MODEL_KEY=$(jq -r ".[\"${{TASK_KEY}}\"].model_key" "$MANIFEST")
EH_LABEL=$(jq -r ".[\"${{TASK_KEY}}\"].label" "$MANIFEST")
EH_MODEL_PATH=$(jq -r ".[\"${{TASK_KEY}}\"].model_path" "$MANIFEST")
export EH_MODEL_PATH

if [ "$EH_MODEL_KEY" = "null" ] || [ -z "$EH_MODEL_KEY" ]; then
    log "ERROR" "No manifest entry for task key '${{TASK_KEY}}'"
    log_failed "invalid_task_key" "No manifest entry found"
    exit 1
fi

log "INFO" "Model key:  $EH_MODEL_KEY"
log "INFO" "Label:      $EH_LABEL"
log "INFO" "Model path: $EH_MODEL_PATH"

# ── Dynamic port ──────────────────────────────────────────────
EH_PORT=$((30000 + RANDOM % 30000))
export EH_PORT
log "INFO" "Dynamic port: EH_PORT=$EH_PORT"

# ── Activate environment ──────────────────────────────────────
{env_activation_block}
log "INFO" "python path: $(which python)"

# ── Environment exports ───────────────────────────────────────
{env_exports}

# ── Request cache ─────────────────────────────────────────────
{request_cache_block}

# ── Output paths ──────────────────────────────────────────────
EH_OUTPUT_BASE="{output_path}/${{EH_MODEL_KEY}}/${{EH_LABEL}}"

# ── Suites to evaluate ────────────────────────────────────────
EH_SUITES={suites_bash_array}

# ── Signal handling ───────────────────────────────────────────
# shutdown_servers() is defined by the server lifecycle block below.

cleanup() {{
    local signal=$1
    log "INFO" "Received signal $signal, initiating cleanup..."

    # Shutdown inference infrastructure (defined by lifecycle block)
    if type shutdown_servers &>/dev/null; then
        shutdown_servers
    fi

    # Resubmit on time limit, mark failed otherwise
    local sbatch_cmd=(sbatch --export=ALL,EH_TASK_KEY="${{TASK_KEY}}" --job-name="{job_name}-${{TASK_KEY}}" "$RUN_DIR/eh_job.slurm")
    if [ "$signal" = "SIGUSR1" ]; then
        log "INFO" "Resubmitting task ${{TASK_KEY}} due to time limit..."
        if sbatch_output="$("${{sbatch_cmd[@]}}" 2>&1)"; then
            new_job_id=$(echo "$sbatch_output" | grep -o '[0-9]*')
            log "INFO" "Resubmitted as job ${{new_job_id}}"
        else
            log "ERROR" "Resubmission failed: $sbatch_output"
            log_failed "resubmission_failed" "$sbatch_output"
        fi
    elif [ "$signal" = "SIGTERM" ]; then
        log_failed "manual_cancellation" "Job was manually cancelled"
    else
        log_failed "unexpected_signal" "Signal: $signal"
    fi

    exit 0
}}

trap 'cleanup SIGUSR1' SIGUSR1
trap 'cleanup SIGTERM' SIGTERM

# ── Server lifecycle ──────────────────────────────────────────
{server_lifecycle_block}

# ── Evaluation ────────────────────────────────────────────────
{eval_loop_block}

# ── Shutdown inference server ─────────────────────────────────
{server_shutdown_block}

# ── Record result ─────────────────────────────────────────────
if [ $EVAL_FAILURES -eq 0 ]; then
    log "INFO" "All evaluations completed successfully"
    echo "${{TASK_KEY}}" >> "$COMPLETED_FILE"
else
    log "ERROR" "$EVAL_FAILURES evaluation(s) failed"
    log_failed "eval_failures" "$EVAL_FAILURES evaluation(s) failed"
fi

exit $EVAL_FAILURES
"""

# ── Server lifecycle sub-templates ────────────────────────────

SERVER_LIFECYCLE_BLOCK = r"""
INFERENCE_SERVER_LOG="{log_dir}/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}-inference-server.log"
SERVER_COMMAND="{inference_server_command}"
SERVER_PID=""
log "INFO" "Starting inference server"
log "INFO" "Command: ${{SERVER_COMMAND}}"
# Unset conflicting SLURM memory variables (they are mutually exclusive and
# srun inherits all of them from the job environment, causing a fatal error).
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

setsid srun --cpu-bind=none \
    --output="$INFERENCE_SERVER_LOG" --error="$INFERENCE_SERVER_LOG" \
    --export=ALL,NO_COLOR=1,VLLM_LOGGING_NO_COLOR=1 \
    bash -c "${{SERVER_COMMAND}}" &
SERVER_PID=$!

sleep 5

# Check if srun started successfully
if ! kill -0 $SERVER_PID 2>/dev/null; then
    wait $SERVER_PID
    SERVER_EXIT=$?
    log "ERROR" "Inference server failed to start (exit code $SERVER_EXIT)"
    log_failed "server_startup_failed" "exit code $SERVER_EXIT"
    exit 1
fi

# Health check
wait_for_server() {{
    local max_attempts=$(({health_check_max_wait_minutes} * 60 / {health_check_interval_seconds}))
    log "INFO" "Max health check wait: {health_check_max_wait_minutes} minutes"
    local attempt=0
    local health_url="http://localhost:${{EH_PORT}}/health"

    while [ $attempt -lt $max_attempts ]; do
        log "INFO" "Health check $((attempt + 1))/${{max_attempts}} for $health_url"
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR" "Server process died during health checks"
            wait $SERVER_PID 2>/dev/null
            log_failed "server_died_during_healthcheck" "PID $SERVER_PID terminated"
            return 1
        fi
        if curl -s --connect-timeout 5 --max-time 10 "$health_url" >/dev/null 2>&1; then
            log "INFO" "Inference server is healthy!"
            return 0
        fi
        log "INFO" "Not ready, waiting {health_check_interval_seconds}s..."
        sleep {health_check_interval_seconds}
        attempt=$((attempt + 1))
    done
    log "ERROR" "Server did not become healthy within timeout"
    return 1
}}

wait_for_server
if [ $? -ne 0 ]; then
    log "ERROR" "Health check failed. Exiting."
    log_failed "health_check_failed" "Server not healthy within timeout"
    exit 1
fi

shutdown_servers() {{
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        log "INFO" "Sending SIGINT to inference server (PID: $SERVER_PID)"
        kill -INT -$SERVER_PID 2>/dev/null || kill -INT $SERVER_PID 2>/dev/null
        sleep 0.1
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill -INT -$SERVER_PID 2>/dev/null || kill -INT $SERVER_PID 2>/dev/null
        fi
        local wait_count=0
        while [ $wait_count -lt 30 ] && kill -0 $SERVER_PID 2>/dev/null; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        if kill -0 $SERVER_PID 2>/dev/null; then
            log "WARN" "Force killing inference server"
            kill -KILL -$SERVER_PID 2>/dev/null || kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
}}
"""

SERVER_LIFECYCLE_BLOCK_SERVERLESS = r"""
log "INFO" "Serverless mode — no inference server to start"
shutdown_servers() { :; }
"""

SERVER_SHUTDOWN_BLOCK = r"""
log "INFO" "Shutting down inference server..."
shutdown_servers
log "INFO" "Inference server shutdown complete"
"""

# ── Multi-server lifecycle sub-templates ─────────────────────

SERVER_LIFECYCLE_BLOCK_MULTI = r"""
INFERENCE_SERVER_LOG_DIR="{log_dir}"
NUM_SERVERS={num_inference_servers}
NODES_PER_SERVER={num_nodes_per_inference_server}

# Unset conflicting SLURM memory variables
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# ── Resolve node list ──────────────────────────────────────────
ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
log "INFO" "All nodes: ${{ALL_NODES[*]}} (total: ${{#ALL_NODES[@]}})"

if [ "${{#ALL_NODES[@]}}" -ne $((NUM_SERVERS * NODES_PER_SERVER)) ]; then
    log "ERROR" "Expected $((NUM_SERVERS * NODES_PER_SERVER)) nodes, got ${{#ALL_NODES[@]}}"
    log_failed "node_count_mismatch" "expected=$((NUM_SERVERS * NODES_PER_SERVER)) got=${{#ALL_NODES[@]}}"
    exit 1
fi

# ── Port setup ─────────────────────────────────────────────────
# LB listens on EH_PORT (what lm-eval connects to).
# Backends use EH_BACKEND_PORT to avoid a conflict on the head node.
# We temporarily set EH_PORT=EH_BACKEND_PORT so the server command's
# ${{EH_PORT}} placeholder expands to the backend port.
EH_LB_PORT=$EH_PORT
EH_BACKEND_PORT=$((EH_PORT + 1))
export EH_BACKEND_PORT
log "INFO" "Load balancer port: $EH_LB_PORT, backend port: $EH_BACKEND_PORT"

EH_PORT=$EH_BACKEND_PORT
export EH_PORT
SERVER_COMMAND="{inference_server_command}"

# Restore EH_PORT to LB port (lm-eval connects here)
EH_PORT=$EH_LB_PORT
export EH_PORT

# ── Start inference servers ────────────────────────────────────
# Unset conflicting SLURM variables (they are mutually exclusive and
# srun inherits all of them from the job environment, causing fatal errors).
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

declare -a SERVER_PIDS
declare -a SERVER_HEAD_NODES
BACKEND_LIST=""
LB_PID=""

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    START_IDX=$((i * NODES_PER_SERVER))
    SERVER_NODES=()
    for j in $(seq 0 $((NODES_PER_SERVER - 1))); do
        SERVER_NODES+=("${{ALL_NODES[$((START_IDX + j))]}}")
    done
    NODELIST=$(IFS=,; echo "${{SERVER_NODES[*]}}")
    HEAD_NODE="${{SERVER_NODES[0]}}"

    SERVER_LOG="$INFERENCE_SERVER_LOG_DIR/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}-inference-server-${{i}}.log"

    log "INFO" "Starting server $i on nodes: $NODELIST (head: $HEAD_NODE)"

    setsid srun \
        --cpu-bind=none \
        --nodes=$NODES_PER_SERVER \
        --nodelist="$NODELIST" \
        --output="$SERVER_LOG" --error="$SERVER_LOG" \
        --export=ALL,NO_COLOR=1,VLLM_LOGGING_NO_COLOR=1 \
        bash -c "$SERVER_COMMAND" &

    SERVER_PIDS[$i]=$!
    SERVER_HEAD_NODES[$i]="$HEAD_NODE"

    if [ -n "$BACKEND_LIST" ]; then
        BACKEND_LIST="$BACKEND_LIST,"
    fi
    BACKEND_LIST="${{BACKEND_LIST}}${{HEAD_NODE}}:${{EH_BACKEND_PORT}}"
done

log "INFO" "All server PIDs: ${{SERVER_PIDS[*]}}"
log "INFO" "Backend list: $BACKEND_LIST"

sleep 5

# ── Verify all srun processes started ──────────────────────────
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    if ! kill -0 ${{SERVER_PIDS[$i]}} 2>/dev/null; then
        wait ${{SERVER_PIDS[$i]}}
        EXIT_CODE=$?
        log "ERROR" "Server $i failed to start (exit code $EXIT_CODE)"
        log_failed "server_startup_failed" "server=$i exit_code=$EXIT_CODE"
        for j in $(seq 0 $((NUM_SERVERS - 1))); do
            kill -KILL -${{SERVER_PIDS[$j]}} 2>/dev/null || true
        done
        exit 1
    fi
done

# ── Health check all backends ──────────────────────────────────
wait_for_all_servers() {{
    local max_attempts=$(({health_check_max_wait_minutes} * 60 / {health_check_interval_seconds}))
    log "INFO" "Health checking $NUM_SERVERS backends (max wait: {health_check_max_wait_minutes} min)"

    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        local head="${{SERVER_HEAD_NODES[$i]}}"
        local health_url="http://${{head}}:${{EH_BACKEND_PORT}}/health"
        local attempt=0

        log "INFO" "Waiting for server $i at $health_url"
        while [ $attempt -lt $max_attempts ]; do
            if ! kill -0 ${{SERVER_PIDS[$i]}} 2>/dev/null; then
                log "ERROR" "Server $i (PID ${{SERVER_PIDS[$i]}}) died during health check"
                log_failed "server_died_during_healthcheck" "server=$i"
                return 1
            fi
            if curl -s --connect-timeout 5 --max-time 10 "$health_url" >/dev/null 2>&1; then
                log "INFO" "Server $i is healthy!"
                break
            fi
            log "INFO" "Server $i not ready, waiting {health_check_interval_seconds}s... ($((attempt + 1))/$max_attempts)"
            sleep {health_check_interval_seconds}
            attempt=$((attempt + 1))
        done

        if [ $attempt -ge $max_attempts ]; then
            log "ERROR" "Server $i did not become healthy within timeout"
            return 1
        fi
    done
    log "INFO" "All $NUM_SERVERS backends are healthy"
    return 0
}}

wait_for_all_servers
if [ $? -ne 0 ]; then
    log "ERROR" "Backend health check failed. Exiting."
    log_failed "health_check_failed" "Not all backends healthy within timeout"
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        kill -KILL -${{SERVER_PIDS[$i]}} 2>/dev/null || true
    done
    exit 1
fi

# ── Start load balancer ────────────────────────────────────────
LB_LOG="$INFERENCE_SERVER_LOG_DIR/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}-load-balancer.log"
log "INFO" "Starting load balancer on :$EH_PORT -> $BACKEND_LIST"

python -m eval_hive.load_balancer \
    --listen-port "$EH_PORT" \
    --backends "$BACKEND_LIST" \
    >"$LB_LOG" 2>&1 &
LB_PID=$!

sleep 2

LB_HEALTH_URL="http://localhost:${{EH_PORT}}/health"
LB_ATTEMPTS=0
while [ $LB_ATTEMPTS -lt 15 ]; do
    if curl -s --connect-timeout 2 --max-time 5 "$LB_HEALTH_URL" >/dev/null 2>&1; then
        log "INFO" "Load balancer is healthy at $LB_HEALTH_URL"
        break
    fi
    if ! kill -0 $LB_PID 2>/dev/null; then
        log "ERROR" "Load balancer process died"
        log_failed "lb_startup_failed" "LB process exited"
        for i in $(seq 0 $((NUM_SERVERS - 1))); do
            kill -KILL -${{SERVER_PIDS[$i]}} 2>/dev/null || true
        done
        exit 1
    fi
    sleep 1
    LB_ATTEMPTS=$((LB_ATTEMPTS + 1))
done

if [ $LB_ATTEMPTS -ge 15 ]; then
    log "ERROR" "Load balancer did not become healthy"
    log_failed "lb_health_failed" "LB not healthy after 15 attempts"
    kill -KILL $LB_PID 2>/dev/null || true
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        kill -KILL -${{SERVER_PIDS[$i]}} 2>/dev/null || true
    done
    exit 1
fi

log "INFO" "Multi-server infrastructure ready: LB :$EH_PORT -> $NUM_SERVERS backends"

# ── shutdown_servers for multi-server ──────────────────────────
shutdown_servers() {{
    # Stop load balancer first
    if [ -n "$LB_PID" ] && kill -0 $LB_PID 2>/dev/null; then
        log "INFO" "Stopping load balancer (PID: $LB_PID)"
        kill -TERM $LB_PID 2>/dev/null || true
        local wc=0
        while [ $wc -lt 10 ] && kill -0 $LB_PID 2>/dev/null; do
            sleep 1
            wc=$((wc + 1))
        done
        if kill -0 $LB_PID 2>/dev/null; then
            kill -KILL $LB_PID 2>/dev/null || true
        fi
    fi
    # Send SIGINT to all inference servers
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        local pid=${{SERVER_PIDS[$i]:-}}
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            log "INFO" "Stopping server $i (PID: $pid)"
            kill -INT -$pid 2>/dev/null || kill -INT $pid 2>/dev/null
        fi
    done
    # Wait for servers to exit
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        local pid=${{SERVER_PIDS[$i]:-}}
        if [ -n "$pid" ]; then
            local wc=0
            while [ $wc -lt 30 ] && kill -0 $pid 2>/dev/null; do
                sleep 1
                wc=$((wc + 1))
            done
            if kill -0 $pid 2>/dev/null; then
                log "WARN" "Force killing server $i"
                kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null
            fi
        fi
    done
}}
"""

SERVER_SHUTDOWN_BLOCK_MULTI = r"""
log "INFO" "Shutting down multi-server infrastructure..."
shutdown_servers
log "INFO" "Multi-server shutdown complete"
"""

# ── Evaluation loop sub-templates ────────────────────────────

EVAL_LOOP_SEQUENTIAL = r"""
EVAL_FAILURES=0

for SUITE in "${{EH_SUITES[@]}}"; do
    SUITE_OUTPUT_DIR="$EH_OUTPUT_BASE/$SUITE"

    # Runtime dedup: skip if results already exist
    if ls "$SUITE_OUTPUT_DIR"/results_*.json 1>/dev/null 2>&1; then
        log "INFO" "[$SUITE] Results already exist — skipping"
        continue
    fi

    log "INFO" "[$SUITE] Starting evaluation"
    mkdir -p "$SUITE_OUTPUT_DIR"

    # --output_path with .json suffix prevents lm_eval from creating model subdirectories
    lm_eval run \
        {lm_eval_extra_args} \
        --model_args "{model_args_string}" \
        --tasks "$SUITE" \
        {include_path_arg} \
        --output_path "$SUITE_OUTPUT_DIR/results.json" \
        >"$SUITE_OUTPUT_DIR/lm_eval.log" 2>&1

    LM_EVAL_EXIT=$?
    if [ $LM_EVAL_EXIT -eq 0 ]; then
        log "INFO" "[$SUITE] Completed successfully"
    else
        log "ERROR" "[$SUITE] Failed with exit code $LM_EVAL_EXIT"
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
done
"""

EVAL_LOOP_PARALLEL = r"""
# ── Build deduplicated flat task list ────────────────────────
TASK_MAP_FILE="$RUN_DIR/eh_task_map.json"
PARALLEL_TASKS={parallel_tasks}

# Collect unique leaf tasks across all suites, interleaved (round-robin)
# so that tasks from different suites are mixed rather than sequential.
declare -A SEEN_TASKS
declare -a SUITE_TASKS
NUM_SUITES=${{#EH_SUITES[@]}}
MAX_LEN=0

# Read per-suite task lists into arrays
for (( s=0; s < NUM_SUITES; s++ )); do
    SUITE="${{EH_SUITES[${{s}}]}}"
    declare -a "SUITE_${{s}}=()"
    while IFS= read -r TASK; do
        eval "SUITE_${{s}}+=(\"$TASK\")"
    done < <(jq -r ".[\"$SUITE\"][]" "$TASK_MAP_FILE")
    eval "LEN=\${{#SUITE_${{s}}[@]}}"
    (( LEN > MAX_LEN )) && MAX_LEN=$LEN
done

# Interleave: pick one task from each suite in turn
ALL_TASKS=()
for (( i=0; i < MAX_LEN; i++ )); do
    for (( s=0; s < NUM_SUITES; s++ )); do
        eval "TASK=\${{SUITE_${{s}}[${{i}}]:-}}"
        [ -z "$TASK" ] && continue
        if [ -z "${{SEEN_TASKS[$TASK]+x}}" ]; then
            ALL_TASKS+=("$TASK")
            SEEN_TASKS[$TASK]=1
        fi
    done
done

log "INFO" "Total unique tasks: ${{#ALL_TASKS[@]}} (from ${{#EH_SUITES[@]}} suite(s), parallel=$PARALLEL_TASKS)"

# ── Run lm-eval per task with worker pool ────────────────────
EVAL_FAILURES=0

run_single_task() {{
    local TASK="$1"
    local TASK_OUTPUT_DIR="$EH_OUTPUT_BASE/$TASK"

    # Per-task dedup (works across job restarts since output is flat by task name)
    if ls "$TASK_OUTPUT_DIR"/results_*.json 1>/dev/null 2>&1; then
        log "INFO" "[$TASK] Results already exist — skipping"
        return 0
    fi

    mkdir -p "$TASK_OUTPUT_DIR"
    log "INFO" "[$TASK] Starting evaluation"

    lm_eval run \
        {lm_eval_extra_args} \
        --model_args "{model_args_string}" \
        --tasks "$TASK" \
        {include_path_arg} \
        --output_path "$TASK_OUTPUT_DIR/results.json" \
        >"$TASK_OUTPUT_DIR/lm_eval.log" 2>&1

    local LM_EVAL_EXIT=$?
    if [ $LM_EVAL_EXIT -eq 0 ]; then
        log "INFO" "[$TASK] Completed successfully"
    else
        log "ERROR" "[$TASK] Failed with exit code $LM_EVAL_EXIT"
    fi
    return $LM_EVAL_EXIT
}}

# Worker pool with manual PID tracking.
# (Cannot use $(jobs -rp | wc -l) — runs in a subshell that can't see parent's jobs.)
declare -a TASK_PIDS=()

for TASK in "${{ALL_TASKS[@]}}"; do
    # If at capacity, wait for a slot to free up
    while (( ${{#TASK_PIDS[@]}} >= PARALLEL_TASKS )); do
        STILL_RUNNING=()
        for pid in "${{TASK_PIDS[@]}}"; do
            if kill -0 "$pid" 2>/dev/null; then
                STILL_RUNNING+=("$pid")
            else
                wait "$pid" || EVAL_FAILURES=$((EVAL_FAILURES + 1))
            fi
        done
        TASK_PIDS=("${{STILL_RUNNING[@]}}")
        if (( ${{#TASK_PIDS[@]}} >= PARALLEL_TASKS )); then
            sleep 1
        fi
    done
    run_single_task "$TASK" &
    TASK_PIDS+=($!)
done

# Wait for all remaining tasks
for pid in "${{TASK_PIDS[@]}}"; do
    wait "$pid" || EVAL_FAILURES=$((EVAL_FAILURES + 1))
done
"""


# ── Main ──────────────────────────────────────────────────────


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the create-run command."""
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the eval-hive YAML configuration file",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Directory to write the run artifacts (manifest, SLURM script, config copy)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--update", action="store_true",
        help="Update an existing run (model/checkpoint and suite changes allowed)",
    )
    group.add_argument(
        "--force", action="store_true",
        help="Overwrite existing run artifacts (manifest, SLURM script, config) without wiping progress/logs",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the create-run command with parsed arguments."""
    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file '{config_path}' does not exist.")
        return 1

    # Load and validate config
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return 1

    # Build manifest
    manifest = build_manifest(config)
    if not manifest:
        logger.error("No models/checkpoints resolved. Check model paths in config.")
        return 1

    # Handle existing run directory
    run_dir = Path(args.output)
    if run_dir.exists():
        if args.force:
            logger.warning(f"Force overwriting run artifacts in: {run_dir}")
        elif args.update:
            # Check that only models and suites_and_tasks changed
            existing_config_path = run_dir / "eh_config.yaml"
            if existing_config_path.exists():
                existing = load_config(existing_config_path)
                old = existing.model_dump(exclude={"models"}, mode="json")
                new = config.model_dump(exclude={"models"}, mode="json")
                # Allow suites_and_tasks to change — staleness is handled at runtime
                old.get("eval", {}).pop("suites_and_tasks", None)
                new.get("eval", {}).pop("suites_and_tasks", None)
                if old != new:
                    logger.error(
                        "Config changes beyond models/suites detected. "
                        "Use --force to overwrite completely."
                    )
                    return 1
                # Log suite changes
                old_suites = set(existing.eval.suites_and_tasks)
                new_suites = set(config.eval.suites_and_tasks)
                if old_suites != new_suites:
                    added_suites = sorted(new_suites - old_suites)
                    removed_suites = sorted(old_suites - new_suites)
                    if added_suites:
                        logger.info(f"Adding suite(s): {', '.join(added_suites)}")
                    if removed_suites:
                        logger.info(f"Removing suite(s): {', '.join(removed_suites)}")
                    logger.info("Stale results will be re-evaluated at runtime.")
            # Check if manifest actually changed
            existing_manifest_path = run_dir / "eh_manifest.json"
            manifest_changed = True
            if existing_manifest_path.exists():
                existing_manifest = json.loads(existing_manifest_path.read_text())
                manifest_changed = existing_manifest != manifest
                # Log manifest changes
                old_keys = set(existing_manifest.keys())
                new_keys = set(manifest.keys())
                added = new_keys - old_keys
                removed = old_keys - new_keys
                if added:
                    logger.info(f"Adding {len(added)} task(s):")
                    for key in sorted(added):
                        logger.info(f"  + {key}")
                if removed:
                    logger.info(f"Removing {len(removed)} task(s):")
                    for key in sorted(removed):
                        logger.info(f"  - {key}")
            logger.info(f"Updating existing run: {run_dir}")
        else:
            logger.error(
                f"Run directory '{run_dir}' already exists. "
                f"Use --update to add checkpoints or --force to overwrite."
            )
            return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_dir}")

    progress_dir = run_dir / "progress"
    progress_dir.mkdir(exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Ensure output_path exists
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Write manifest
    manifest_path = run_dir / "eh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Manifest written: {manifest_path} ({len(manifest)} entries)")

    # Copy config for reproducibility
    config_copy = run_dir / "eh_config.yaml"
    shutil.copy2(config_path, config_copy)
    logger.info(f"Config copied to: {config_copy}")

    # Clean server command
    server_cmd = None
    if config.inference_server_command:
        server_cmd = clean_server_command(config.inference_server_command)

    total_nodes = config.num_inference_servers * config.num_nodes_per_inference_server

    # Request cache block
    if config.request_cache_dir:
        cache_tarball = config.request_cache_dir / "cache.tar.gz"
        request_cache_block = (
            f'EH_CACHE_TARBALL="{cache_tarball}"\n'
            f'if [ -f "$EH_CACHE_TARBALL" ]; then\n'
            f'    EH_LOCAL_CACHE="${{TMPDIR:-/tmp}}/lm_eval_cache"\n'
            f'    mkdir -p "$EH_LOCAL_CACHE"\n'
            f'    log "INFO" "Extracting request cache tarball to $EH_LOCAL_CACHE"\n'
            f'    tar -xzf "$EH_CACHE_TARBALL" -C "$EH_LOCAL_CACHE"\n'
            f'    CACHE_COUNT=$(ls "$EH_LOCAL_CACHE"/*.pickle 2>/dev/null | wc -l)\n'
            f'    log "INFO" "Extracted $CACHE_COUNT cache file(s) to local storage"\n'
            f'    export LM_HARNESS_CACHE_PATH="$EH_LOCAL_CACHE"\n'
            f'else\n'
            f'    log "WARN" "Cache tarball not found at $EH_CACHE_TARBALL — using shared filesystem"\n'
            f'    export LM_HARNESS_CACHE_PATH="{config.request_cache_dir}"\n'
            f'fi\n'
            f'log "INFO" "LM_HARNESS_CACHE_PATH=$LM_HARNESS_CACHE_PATH"'
        )
    else:
        request_cache_block = 'log "INFO" "No request cache directory configured — using lm-eval default"'

    # include_path arg
    include_path_arg = ""
    if config.eval.eval_suite_path:
        include_path_arg = f'--include_path "{config.eval.eval_suite_path}"'

    # Server blocks
    if server_cmd:
        if config.num_inference_servers > 1:
            server_lifecycle_block = SERVER_LIFECYCLE_BLOCK_MULTI.format(
                log_dir=str(log_dir),
                inference_server_command=server_cmd,
                num_inference_servers=config.num_inference_servers,
                num_nodes_per_inference_server=config.num_nodes_per_inference_server,
                health_check_max_wait_minutes=config.health_check_max_wait_minutes,
                health_check_interval_seconds=config.health_check_interval_seconds,
            )
            server_shutdown_block = SERVER_SHUTDOWN_BLOCK_MULTI
        else:
            server_lifecycle_block = SERVER_LIFECYCLE_BLOCK.format(
                log_dir=str(log_dir),
                inference_server_command=server_cmd,
                health_check_max_wait_minutes=config.health_check_max_wait_minutes,
                health_check_interval_seconds=config.health_check_interval_seconds,
            )
            server_shutdown_block = SERVER_SHUTDOWN_BLOCK
    else:
        server_lifecycle_block = SERVER_LIFECYCLE_BLOCK_SERVERLESS
        server_shutdown_block = ""

    # Eval loop block: parallel (per-task) or sequential (per-suite)
    eval_format_vars = {
        "lm_eval_extra_args": build_lm_eval_extra_args(config),
        "model_args_string": build_model_args_string(config),
        "include_path_arg": include_path_arg,
    }
    if config.parallel_tasks > 1:
        # Resolve suite/group names to leaf tasks and write task map into run directory
        from lm_eval.tasks import TaskManager
        from eval_hive.prepare import resolve_task_names

        logger.info("Resolving task map for parallel execution...")
        tm = TaskManager(include_path=config.eval.eval_suite_path)
        task_map = {}
        for suite_or_task in config.eval.suites_and_tasks:
            task_map[suite_or_task] = resolve_task_names(tm, [suite_or_task])

        task_map_dst = run_dir / "eh_task_map.json"
        task_map_dst.write_text(json.dumps(task_map, indent=2))
        logger.info(f"Task map written to: {task_map_dst}")

        eval_loop_block = EVAL_LOOP_PARALLEL.format(
            parallel_tasks=config.parallel_tasks,
            **eval_format_vars,
        )
    else:
        eval_loop_block = EVAL_LOOP_SEQUENTIAL.format(**eval_format_vars)

    # Note: eval_loop_block is inserted as a VALUE to SBATCH_TEMPLATE.format(),
    # so its braces are NOT interpreted as format placeholders. No re-escaping needed.

    # Format main template
    template_vars = {
        "job_name": config.job_name,
        "partition": config.partition,
        "account": config.account,
        "qos": config.qos,
        "total_nodes_per_job": total_nodes,
        "cpus_per_node": config.cpus_per_node,
        "gres_per_node": config.gres_per_node,
        "time_limit": config.time_limit,
        "log_dir": str(log_dir),
        "additional_sbatch_lines": build_additional_sbatch_lines(config),
        "run_dir": str(run_dir),
        "progress_dir": str(progress_dir),
        "env_activation_block": build_env_activation_block(config),
        "env_exports": build_env_exports(config),
        "request_cache_block": request_cache_block,
        "output_path": str(config.output_path),
        "suites_bash_array": build_suites_bash_array(config),
        "model_args_string": build_model_args_string(config),
        "lm_eval_extra_args": build_lm_eval_extra_args(config),
        "include_path_arg": include_path_arg,
        "eval_loop_block": eval_loop_block,
        "server_lifecycle_block": server_lifecycle_block,
        "server_shutdown_block": server_shutdown_block,
    }

    sbatch_script = SBATCH_TEMPLATE.format(**template_vars)

    # Write SLURM script
    job_script = run_dir / "eh_job.slurm"
    job_script.write_text(sbatch_script)
    logger.info(f"SLURM job script generated: {job_script}")

    logger.info(f"Run created at: {run_dir}")
    return 0


def main() -> int:
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(
        description="Create an eval-hive run directory with manifest and SLURM script"
    )
    add_arguments(parser)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
