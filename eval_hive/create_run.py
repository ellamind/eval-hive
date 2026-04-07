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
    for config_key, entry in config.models.items():
        effective_model_key = entry.model_key or config_key
        for label, step, path in entry.resolve_model_paths():
            key = manifest_key(effective_model_key, label)
            if key in manifest:
                raise ValueError(
                    f"Duplicate manifest key '{key}' from "
                    f"model_key='{effective_model_key}', label='{label}'. "
                    f"Ensure model_key + label combinations are unique."
                )
            if isinstance(entry.tokens_trained, list):
                # Per-step tokens_trained list — look up by step index
                step_idx = entry.steps.index(step) if step is not None and entry.steps else None
                tokens_trained = entry.tokens_trained[step_idx] if step_idx is not None else None
            else:
                tokens_trained = entry.tokens_trained
            if tokens_trained is None and entry.train_batch_size is not None:
                if step is not None:
                    tokens_trained = entry.train_batch_size * step

            effective_step = step + entry.step_offset if step is not None else None
            manifest[key] = {
                "model_key": effective_model_key,
                "label": label,
                "step": effective_step,
                "model_path": str(path),
                "display_name": entry.display_name,
                "train_batch_size": entry.train_batch_size,
                "tokens_trained": tokens_trained,
            }
    return manifest


# ── HF sync ──────────────────────────────────────────────────


def sync_hf_markers(
    run_dir: Path,
    manifest: dict[str, dict],
    task_map: dict[str, list[str]],
    suites: list[str],
    hf_repo: str,
    progress_dir: Path,
) -> dict[str, list[str]]:
    """Write a skip list for leaf tasks already covered in HuggingFace.

    Produces ``eh_hf_covered.json`` in *run_dir* mapping manifest keys to
    their covered leaf task names.  The SLURM job script reads this file
    at startup and skips those tasks (no per-task marker files needed).

    Manifest keys where *all* leaf tasks are covered are also appended to
    ``jobs_completed.log`` so that ``submit`` skips them entirely.
    """
    from eval_hive.results.hf import download_hf_parquet

    logger.info("Syncing HF coverage from {}...", hf_repo)

    df = download_hf_parquet(hf_repo)
    if df is None or len(df) == 0:
        logger.info("No HF data found, skipping HF sync")
        return {}

    # Collect all expected leaf tasks across suites
    all_expected_tasks: set[str] = set()
    for suite in suites:
        all_expected_tasks.update(task_map.get(suite, []))

    if not all_expected_tasks:
        return {}

    # Build lookup of (model, step, task) tuples present in HF
    hf_tuples: set[tuple[str, int | None, str]] = set()
    for row in df.select("model", "step", "task").unique().iter_rows(named=True):
        step = None if row["step"] is None else int(row["step"])
        hf_tuples.add((row["model"], step, row["task"]))

    covered: dict[str, list[str]] = {}
    keys_completed: list[str] = []

    for mkey, entry in manifest.items():
        model_key = entry["model_key"]
        label = entry["label"]
        step = entry.get("step")

        covered_tasks = sorted(
            task for task in all_expected_tasks
            if (model_key, step, task) in hf_tuples
        )

        if covered_tasks:
            covered[mkey] = covered_tasks

        if len(covered_tasks) == len(all_expected_tasks):
            keys_completed.append(mkey)

    # Write single coverage file (job script reads this)
    hf_covered_path = run_dir / "eh_hf_covered.json"
    hf_covered_path.write_text(json.dumps(covered, indent=2))

    # Write fully-covered keys to completed log
    if keys_completed:
        completed_file = progress_dir / "jobs_completed.log"
        existing: set[str] = set()
        if completed_file.exists():
            existing = {
                line.strip()
                for line in completed_file.read_text().splitlines()
                if line.strip()
            }
        new_keys = [k for k in keys_completed if k not in existing]
        if new_keys:
            with open(completed_file, "a") as f:
                for key in new_keys:
                    f.write(f"{key}\n")

    total_tasks = sum(len(v) for v in covered.values())
    logger.info(
        "HF sync: {}/{} tasks covered, {}/{} jobs fully covered",
        total_tasks, len(manifest) * len(all_expected_tasks),
        len(keys_completed), len(manifest),
    )

    return covered


def _mark_locally_complete(
    output_path: Path,
    manifest: dict[str, dict],
    task_map: dict[str, list[str]],
    suites: list[str],
    hf_covered: dict[str, list[str]] | None,
    progress_dir: Path,
) -> None:
    """Append manifest keys that are fully covered (disk + HF) to jobs_completed.log."""
    all_tasks: set[str] = set()
    for s in suites:
        all_tasks.update(task_map.get(s, []))
    if not all_tasks:
        return

    completed_file = progress_dir / "jobs_completed.log"
    existing: set[str] = set()
    if completed_file.exists():
        existing = {
            line.strip()
            for line in completed_file.read_text().splitlines()
            if line.strip()
        }

    _cache: dict[tuple[str, str], set[str]] = {}
    new_keys: list[str] = []

    for mkey, entry in manifest.items():
        if mkey in existing:
            continue
        mk, lbl = entry["model_key"], entry["label"]
        cache_key = (mk, lbl)
        if cache_key not in _cache:
            base = output_path / mk / lbl
            _cache[cache_key] = _collect_completed_tasks(base)
        disk = _cache[cache_key]

        hf_tasks = set(hf_covered.get(mkey, [])) if hf_covered else set()
        covered = sum(1 for t in all_tasks if t in disk or t in hf_tasks)
        if covered == len(all_tasks):
            new_keys.append(mkey)

    if new_keys:
        with open(completed_file, "a") as f:
            for key in new_keys:
                f.write(f"{key}\n")
        logger.info("Marked {} job(s) as completed from local results", len(new_keys))


def _collect_completed_tasks(base: Path) -> set[str]:
    """Scan all result files under *base* to find completed task names.

    Handles both per-task directories (old format) and batch directories
    (new format) by parsing the ``results`` key from each result file.
    """
    completed: set[str] = set()
    if not base.is_dir():
        return completed
    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue
        for rf in subdir.glob("results_*.json"):
            try:
                data = json.loads(rf.read_text())
                completed.update(data.get("results", {}).keys())
            except (json.JSONDecodeError, OSError):
                continue
    return completed


def count_task_coverage(
    output_path: Path,
    manifest: dict[str, dict],
    all_tasks: set[str],
    hf_covered: dict[str, list[str]] | None = None,
) -> tuple[int, int, int, int, int]:
    """Count leaf-task coverage across all manifest keys.

    Returns (total, on_disk, on_hf_only, jobs_covered, jobs_remaining).
    """
    total = len(manifest) * len(all_tasks)
    n_disk = 0
    n_hf_only = 0
    jobs_covered = 0

    # Cache completed tasks per (model_key, label) to avoid re-parsing
    # the same result files for manifest entries that share a directory.
    _cache: dict[tuple[str, str], set[str]] = {}

    for mkey, entry in manifest.items():
        mk, lbl = entry["model_key"], entry["label"]
        cache_key = (mk, lbl)
        if cache_key not in _cache:
            base = output_path / mk / lbl
            _cache[cache_key] = _collect_completed_tasks(base)
        completed = _cache[cache_key]

        hf_tasks = set(hf_covered.get(mkey, [])) if hf_covered else set()
        key_covered = 0

        for task in all_tasks:
            if task in completed:
                n_disk += 1
                key_covered += 1
            elif task in hf_tasks:
                n_hf_only += 1
                key_covered += 1

        if key_covered == len(all_tasks):
            jobs_covered += 1

    jobs_remaining = len(manifest) - jobs_covered
    return total, n_disk, n_hf_only, jobs_covered, jobs_remaining


def display_run_summary(
    output_path: Path,
    manifest: dict[str, dict],
    task_map: dict[str, list[str]],
    suites: list[str],
    hf_covered: dict[str, list[str]] | None = None,
) -> None:
    """Print a summary table of job and task coverage."""
    from tabulate import tabulate

    all_tasks: set[str] = set()
    for s in suites:
        all_tasks.update(task_map.get(s, []))

    total, n_disk, n_hf_only, jobs_covered, jobs_remaining = count_task_coverage(
        output_path, manifest, all_tasks, hf_covered,
    )
    n_remaining = total - n_disk - n_hf_only

    rows = [
        ["Total", f"{len(manifest)} jobs, {len(all_tasks)} tasks per job"],
        ["On disk", f"{n_disk} tasks"],
    ]
    if hf_covered:
        rows.append(["On HF", f"{n_hf_only} tasks"])
    rows.append(["To evaluate", f"{jobs_remaining} jobs, {n_remaining} tasks"])

    print(tabulate(rows, tablefmt="rounded_outline"))


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


def build_env_exports_inline(config: EhConfig) -> str:
    """Build env exports as a single-line bash snippet for use inside bash -c.

    Re-exporting env_vars inside srun steps allows variables like
    ${EH_SERVER_ID} (set per srun step) to be expanded correctly.
    """
    if not config.env_vars:
        return ""
    parts = []
    for key, value in config.env_vars.items():
        parts.append(f'export {key}="{value}";')
    return " ".join(parts) + " "


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

# ── Dynamic port (bind-tested to avoid "address already in use") ─
for _try in $(seq 1 20); do
    EH_PORT=$((30000 + RANDOM % 30000))
    if python3 -c "import socket; s=socket.socket(); s.bind(('', $EH_PORT)); s.close()" 2>/dev/null; then
        break
    fi
    log "WARN" "Port $EH_PORT already in use, retrying ($_try/20)"
    [ "$_try" -eq 20 ] && {{ log "ERROR" "No free port after 20 attempts"; exit 1; }}
done
export EH_PORT
log "INFO" "Dynamic port: EH_PORT=$EH_PORT"

# ── Activate environment ──────────────────────────────────────
{env_activation_block}
log "INFO" "python path: $(which python)"

# ── Environment exports ───────────────────────────────────────
{env_exports}

# ── Unset proxy vars (all traffic is cluster-internal) ────────
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ── Request cache ─────────────────────────────────────────────
{request_cache_block}

# ── Output paths ──────────────────────────────────────────────
EH_OUTPUT_BASE="{output_path}/${{EH_MODEL_KEY}}/${{EH_LABEL}}"

# ── Suites to evaluate ────────────────────────────────────────
EH_SUITES={suites_bash_array}

# ── HF-covered tasks (skip list from create-run) ─────────────
declare -A EH_HF_COVERED
HF_COVERED_FILE="$RUN_DIR/eh_hf_covered.json"
if [ -f "$HF_COVERED_FILE" ]; then
    while IFS= read -r task; do
        EH_HF_COVERED["$task"]=1
    done < <(jq -r ".[\"${{TASK_KEY}}\"][]? // empty" "$HF_COVERED_FILE")
    if [ ${{#EH_HF_COVERED[@]}} -gt 0 ]; then
        log "INFO" "HF coverage: ${{#EH_HF_COVERED[@]}} task(s) will be skipped"
    fi
fi

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

setsid --wait srun --cpu-bind=none \
    --output="$INFERENCE_SERVER_LOG" --error="$INFERENCE_SERVER_LOG" \
    --export=ALL,NO_COLOR=1 \
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
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR" "Server process died during health checks"
            wait $SERVER_PID 2>/dev/null
            log_failed "server_died_during_healthcheck" "PID $SERVER_PID terminated"
            return 1
        fi
        local hc_status
        hc_status="$(curl --noproxy '*' -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 --max-time 10 "$health_url")" || true
        if [ "$hc_status" = "200" ]; then
            log "INFO" "Inference server is healthy!"
            return 0
        fi
        log "INFO" "Not ready (HTTP $hc_status), waiting {health_check_interval_seconds}s... ($((attempt + 1))/${{max_attempts}})"
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
        while [ $wait_count -lt 15 ] && kill -0 $SERVER_PID 2>/dev/null; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        if kill -0 $SERVER_PID 2>/dev/null; then
            log "WARN" "Force killing inference server"
            kill -KILL -$SERVER_PID 2>/dev/null || kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
    # Kill any remaining vllm/singularity processes that survived signal-based shutdown
    # (e.g. container processes started via setsid srun that ignore parent signals)
    pkill -KILL -f "vllm serve.*${{EH_PORT}}" 2>/dev/null || true
    pkill -KILL -f "singularity exec.*${{EH_PORT}}" 2>/dev/null || true
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

    setsid --wait srun \
        --cpu-bind=none \
        --nodes=$NODES_PER_SERVER \
        --nodelist="$NODELIST" \
        --output="$SERVER_LOG" --error="$SERVER_LOG" \
        --export=ALL,NO_COLOR=1,EH_SERVER_ID=$i \
        bash -c "{env_re_exports}$SERVER_COMMAND" &

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
            local hc_status
            hc_status="$(curl --noproxy '*' -s -o /dev/null -w '%{{http_code}}' --connect-timeout 5 --max-time 10 "$health_url")" || true
            if [ "$hc_status" = "200" ]; then
                log "INFO" "Server $i is healthy!"
                break
            fi
            log "INFO" "Server $i not ready (HTTP $hc_status), waiting {health_check_interval_seconds}s... ($((attempt + 1))/$max_attempts)"
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
    if [ "$(curl --noproxy '*' -s -o /dev/null -w '%{{http_code}}' --connect-timeout 2 --max-time 5 "$LB_HEALTH_URL")" = "200" ]; then
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
        while [ $wc -lt 5 ] && kill -0 $LB_PID 2>/dev/null; do
            sleep 1
            wc=$((wc + 1))
        done
        if kill -0 $LB_PID 2>/dev/null; then
            kill -KILL $LB_PID 2>/dev/null || true
        fi
    fi
    # Send SIGINT to all inference servers at once
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        local pid=${{SERVER_PIDS[$i]:-}}
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            log "INFO" "Stopping server $i (PID: $pid)"
            kill -INT -$pid 2>/dev/null || kill -INT $pid 2>/dev/null
        fi
    done
    # Wait for all servers in parallel (single countdown)
    local wc=0
    while [ $wc -lt 15 ]; do
        local still_alive=0
        for i in $(seq 0 $((NUM_SERVERS - 1))); do
            local pid=${{SERVER_PIDS[$i]:-}}
            [ -n "$pid" ] && kill -0 $pid 2>/dev/null && still_alive=1
        done
        [ $still_alive -eq 0 ] && break
        sleep 1
        wc=$((wc + 1))
    done
    # Force kill any survivors
    for i in $(seq 0 $((NUM_SERVERS - 1))); do
        local pid=${{SERVER_PIDS[$i]:-}}
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            log "WARN" "Force killing server $i"
            kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null
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

    # Runtime dedup: skip if results already exist or covered by HF
    if ls "$SUITE_OUTPUT_DIR"/results_*.json 1>/dev/null 2>&1 || [ -n "${{EH_HF_COVERED[$SUITE]+x}}" ]; then
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
TASK_BATCH_SIZE={task_batch_size}

# Collect unique leaf tasks across all suites (task map is pre-sorted by
# request type so vLLM sees homogeneous workloads).
declare -A SEEN_TASKS
ALL_TASKS=()
for SUITE in "${{EH_SUITES[@]}}"; do
    while IFS= read -r TASK; do
        if [ -z "${{SEEN_TASKS[$TASK]+x}}" ]; then
            ALL_TASKS+=("$TASK")
            SEEN_TASKS[$TASK]=1
        fi
    done < <(jq -r ".[\"$SUITE\"][]" "$TASK_MAP_FILE")
done

log "INFO" "Total unique tasks: ${{#ALL_TASKS[@]}} (from ${{#EH_SUITES[@]}} suite(s), parallel=$PARALLEL_TASKS, batch_size=$TASK_BATCH_SIZE)"

# ── Filter out already-completed tasks ───────────────────────
# Scan existing result files (handles both per-task and batch directories)
declare -A DONE_TASKS
for result_file in "$EH_OUTPUT_BASE"/*/results_*.json; do
    [ -f "$result_file" ] || continue
    while IFS= read -r task; do
        DONE_TASKS["$task"]=1
    done < <(jq -r '.results | keys[]' "$result_file")
done

REMAINING=()
for TASK in "${{ALL_TASKS[@]}}"; do
    [ -n "${{DONE_TASKS[$TASK]+x}}" ] && continue
    [ -n "${{EH_HF_COVERED[$TASK]+x}}" ] && continue
    REMAINING+=("$TASK")
done

SKIPPED=$(( ${{#ALL_TASKS[@]}} - ${{#REMAINING[@]}} ))
if [ $SKIPPED -gt 0 ]; then
    log "INFO" "Skipping $SKIPPED already-completed task(s), ${{#REMAINING[@]}} remaining"
fi

if [ ${{#REMAINING[@]}} -eq 0 ]; then
    log "INFO" "All tasks already completed — nothing to evaluate"
fi

# ── Compute effective batch size ─────────────────────────────
# Don't under-parallelize: if we have fewer tasks than parallel_tasks,
# use batch_size=1 so each gets its own slot.
EFF_BATCH_SIZE=$TASK_BATCH_SIZE
if (( ${{#REMAINING[@]}} <= PARALLEL_TASKS )); then
    EFF_BATCH_SIZE=1
elif (( TASK_BATCH_SIZE > (${{#REMAINING[@]}} + PARALLEL_TASKS - 1) / PARALLEL_TASKS )); then
    # Also cap batch size so we fill all parallel slots
    EFF_BATCH_SIZE=$(( (${{#REMAINING[@]}} + PARALLEL_TASKS - 1) / PARALLEL_TASKS ))
fi

# ── Build batches ────────────────────────────────────────────
declare -a BATCHES=()
BATCH_IDX=0
for (( i=0; i < ${{#REMAINING[@]}}; i+=EFF_BATCH_SIZE )); do
    BATCH_CSV=""
    for (( j=i; j < i+EFF_BATCH_SIZE && j < ${{#REMAINING[@]}}; j++ )); do
        [ -n "$BATCH_CSV" ] && BATCH_CSV+=","
        BATCH_CSV+="${{REMAINING[$j]}}"
    done
    BATCHES+=("$BATCH_CSV")
done

log "INFO" "Created ${{#BATCHES[@]}} batch(es) of up to $EFF_BATCH_SIZE task(s)"

# ── Run batched lm-eval with worker pool ─────────────────────
EVAL_FAILURES=0
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=$(( (PARALLEL_TASKS + 1) / 2 + 1 ))

run_task_batch() {{
    local BATCH_IDX="$1"
    local TASKS_CSV="$2"
    local BATCH_DIR="$EH_OUTPUT_BASE/batch_${{SLURM_JOB_ID}}_$(printf '%03d' $BATCH_IDX)"

    mkdir -p "$BATCH_DIR"
    log "INFO" "[batch $BATCH_IDX] Starting evaluation: $TASKS_CSV"

    lm_eval run \
        {lm_eval_extra_args} \
        --model_args "{model_args_string}" \
        --tasks "$TASKS_CSV" \
        {include_path_arg} \
        --output_path "$BATCH_DIR/results.json" \
        >"$BATCH_DIR/lm_eval.log" 2>&1

    local LM_EVAL_EXIT=$?
    if [ $LM_EVAL_EXIT -eq 0 ]; then
        log "INFO" "[batch $BATCH_IDX] Completed successfully"
    else
        log "ERROR" "[batch $BATCH_IDX] Failed with exit code $LM_EVAL_EXIT"
    fi
    return $LM_EVAL_EXIT
}}

# Worker pool with manual PID tracking.
declare -a BATCH_PIDS=()
declare -A PID_TO_BATCH=()

for (( b=0; b < ${{#BATCHES[@]}}; b++ )); do
    # If at capacity, wait for a slot to free up
    while (( ${{#BATCH_PIDS[@]}} >= PARALLEL_TASKS )); do
        STILL_RUNNING=()
        for pid in "${{BATCH_PIDS[@]}}"; do
            if kill -0 "$pid" 2>/dev/null; then
                STILL_RUNNING+=("$pid")
            else
                if wait "$pid"; then
                    CONSECUTIVE_FAILURES=0
                else
                    EVAL_FAILURES=$((EVAL_FAILURES + 1))
                    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                    log "ERROR" "[batch ${{PID_TO_BATCH[$pid]}}] Failed (pid $pid)"
                fi
                unset PID_TO_BATCH[$pid]
            fi
        done
        BATCH_PIDS=("${{STILL_RUNNING[@]}}")
        if (( ${{#BATCH_PIDS[@]}} >= PARALLEL_TASKS )); then
            sleep 1
        fi
    done
    if (( CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
        log "ERROR" "Aborting: $CONSECUTIVE_FAILURES consecutive batch failures (server likely dead)"
        break
    fi
    run_task_batch "$b" "${{BATCHES[$b]}}" &
    BATCH_PIDS+=($!)
    PID_TO_BATCH[$!]=$b
done

# Wait for all remaining batches (kill survivors if threshold reached)
for pid in "${{BATCH_PIDS[@]}}"; do
    if (( CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
        log "ERROR" "Killing remaining batches (${{#BATCH_PIDS[@]}} left) — server likely dead"
        for kpid in "${{BATCH_PIDS[@]}}"; do
            kill "$kpid" 2>/dev/null
        done
        wait
        break
    fi
    if wait "$pid"; then
        CONSECUTIVE_FAILURES=0
    else
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        log "ERROR" "[batch ${{PID_TO_BATCH[$pid]}}] Failed (pid $pid)"
    fi
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
            # Clear stale completion tracking so HF sync rebuilds it cleanly.
            # Keep jobs_failed.log — it's diagnostic history, not used for scheduling.
            completed_file = run_dir / "progress" / "jobs_completed.log"
            if completed_file.exists():
                completed_file.unlink()
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
            f'    EH_LOCAL_CACHE="${{TMPDIR:-/tmp}}/lm_eval_cache_${{SLURM_JOB_ID}}"\n'
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
                env_re_exports=build_env_exports_inline(config),
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

    # Resolve suite/group names to leaf tasks and write task map into run directory.
    # Always generated (used by HF sync, status, and parallel eval mode).
    from lm_eval.tasks import TaskManager
    from eval_hive.prepare import resolve_task_names

    logger.info("Resolving task map...")
    tm = TaskManager(include_path=config.eval.eval_suite_path)

    # Sort tasks by request type so vLLM sees homogeneous workloads:
    # slow generation types first (they take longest), then fast loglikelihood types.
    # Within each type: alphabetical by name. Unknown suffixes sort alphabetically after known ones.
    _SUFFIX_ORDER = {"_code": 0, "_cot": 1, "_gen": 2, "_mc": 3, "_rc": 4}

    def _task_sort_key(name: str) -> tuple[int, str, str]:
        for suffix, order in _SUFFIX_ORDER.items():
            if name.endswith(suffix):
                return (order, suffix, name)
        # Unknown suffixes: sort after all known types, alphabetically by suffix then name
        parts = name.rsplit("_", 1)
        unknown_suffix = f"_{parts[1]}" if len(parts) > 1 else ""
        return (len(_SUFFIX_ORDER), unknown_suffix, name)

    task_map = {}
    for suite_or_task in config.eval.suites_and_tasks:
        tasks = resolve_task_names(tm, [suite_or_task])
        tasks.sort(key=_task_sort_key)
        task_map[suite_or_task] = tasks

    task_map_dst = run_dir / "eh_task_map.json"
    task_map_dst.write_text(json.dumps(task_map, indent=2))
    logger.info(f"Task map written to: {task_map_dst}")

    # HF sync: write skip list for tasks already in HuggingFace
    hf_covered: dict[str, list[str]] | None = None
    if config.hf_result_repo:
        hf_covered = sync_hf_markers(
            run_dir=run_dir,
            manifest=manifest,
            task_map=task_map,
            suites=config.eval.suites_and_tasks,
            hf_repo=config.hf_result_repo,
            progress_dir=progress_dir,
        )

    # Mark jobs that are fully covered (disk + HF) as completed so that
    # submit skips them even when they weren't completed via HF alone.
    _mark_locally_complete(
        output_path=config.output_path,
        manifest=manifest,
        task_map=task_map,
        suites=config.eval.suites_and_tasks,
        hf_covered=hf_covered,
        progress_dir=progress_dir,
    )

    # Eval loop block: parallel (per-task) or sequential (per-suite)
    eval_format_vars = {
        "lm_eval_extra_args": build_lm_eval_extra_args(config),
        "model_args_string": build_model_args_string(config),
        "include_path_arg": include_path_arg,
    }
    if config.task_batch_size is not None:
        eval_loop_block = EVAL_LOOP_PARALLEL.format(
            parallel_tasks=config.parallel_tasks,
            task_batch_size=config.task_batch_size,
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

    # Display coverage summary
    display_run_summary(
        config.output_path, manifest, task_map,
        config.eval.suites_and_tasks, hf_covered,
    )

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
