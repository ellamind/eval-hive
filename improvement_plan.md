# eval-hive Redesign Analysis

## Context

eval-hive orchestrates large-scale lm-evaluation-harness runs on SLURM clusters. It works well for its purpose, but several design decisions create friction on Lustre filesystems and in offline-compute environments. This document identifies the most impactful areas for improvement, ranked by practical value.

---

## 1. Move job logic from generated Bash to a Python worker

**The problem**: `create_run.py` is 1306 lines, ~700 of which are raw bash templates (`SBATCH_TEMPLATE`, `SERVER_LIFECYCLE_BLOCK`, `EVAL_LOOP_PARALLEL`, etc.) assembled via Python `.format()` with `{{`/`}}` escaping. This bash-in-Python approach is:
- **Untestable** — you can't unit test the server health-check loop, the parallel batching logic, the signal handler, or the dedup scanning without actually running a SLURM job
- **Fragile** — a single mismatched brace or unescaped `$` breaks the entire script silently at runtime, far from where you can debug it
- **Hard to evolve** — adding a new feature (e.g., GPU utilization tracking) requires editing string constants and mentally parsing interleaved Python-format and bash syntax
- **Duplicated** — `_collect_completed_tasks()` is defined identically in both `create_run.py:157` and `status.py:40` because the bash script re-implements the same result-scanning logic in shell

**The fix**: Make the SBATCH script a thin ~40-line wrapper:
```bash
#!/bin/bash
#SBATCH ...
eval "$(pixi shell-hook ...)"
python -m eval_hive.worker "$RUN_DIR" "$EH_TASK_KEY"
```

Move **all** logic into `eval_hive/worker.py`:
- Server lifecycle (start, health check, shutdown) — testable with mock HTTP
- Parallel task batching — testable with no SLURM at all
- Result dedup scanning — shared code with `status.py` and `collect.py`
- Signal handling — `signal.signal(signal.SIGUSR1, ...)` in Python
- HF coverage filtering — reuse existing Python data structures

**Impact**: Cuts `create_run.py` from 1306 lines to ~300. Makes 80% of the job logic unit-testable. Eliminates the duplicated `_collect_completed_tasks()`. Template becomes a single Jinja2 file or a short string.

---

## 2. Reduce Lustre metadata pressure

**The problem**: Every evaluation batch creates a directory tree:
```
{output_path}/{model_key}/{label}/batch_{JOB_ID}_{NNN}/
    results_YYYY-MM-DDTHH-MM-SS.json    (~50-500KB)
    samples_taskname.jsonl               (~1-50MB per task)
    lm_eval.log                          (~10KB)
```

For a typical run (40 models x 8 parallel batches), that's 320+ directories with ~3 files each = ~1000 inodes created. The `compact` command exists purely to clean this up after the fact. Meanwhile:
- `status` calls `_collect_completed_tasks()` which does `base.iterdir()` + `subdir.glob("results_*.json")` + `json.loads()` for **every** manifest key — hundreds of `readdir` + `stat` + `open` calls per invocation
- `collect` does similar globbing across the entire output tree
- On Lustre, each `stat()` / `readdir()` is an MDS RPC — these are the bottleneck, not bandwidth

**Fixes (incremental, pick any subset)**:

### 2a. Write a completion index instead of scanning
When a job finishes, have it append a single line to a structured JSONL index:
```jsonl
{"task_key":"Model--ckpt_5000","tasks":["arc_easy","hellaswag"],"result_path":"batch_12345_001/results_2025-06-01.json","timestamp":"2025-06-01T12:00:00"}
```
Then `status`, `collect`, and runtime dedup read this index instead of walking the tree. This replaces hundreds of `stat()` calls with a single `read()`.

### 2b. Flatten output directories
Instead of `{model}/{label}/{batch}/results_*.json`, write:
```
{output_path}/{model_key}/{label}/results_{JOB_ID}_{BATCH}.json
```
One flat directory per model/checkpoint instead of N subdirectories. Eliminates nested `iterdir()` calls. The `compact` command becomes unnecessary.

### 2c. Samples as tar archives
Sample files (`.jsonl`) are write-once and only read during analysis. Write them into a tar archive per job instead of individual files:
```
{output_path}/{model_key}/{label}/samples_{JOB_ID}.tar
```
Reduces inode count dramatically for `--log_samples` runs.

---

## 3. Replace shared append-only logs with per-job marker files

**The problem**: `jobs_completed.log` and `jobs_failed.log` receive concurrent appends from potentially hundreds of SLURM jobs. On Lustre:
- Appends < stripe size (1MB default) go through a single OST — creating a serialization bottleneck
- Lustre does **not** guarantee POSIX atomic append semantics across multiple clients — concurrent `echo >> file` can interleave or lose data
- The `grep -qFx` check at job start reads the entire file sequentially

**The fix**: Use per-key marker files:
```
progress/completed/{Model--ckpt_5000}     # empty file, existence = done
progress/failed/{Model--ckpt_5000}.json   # failure metadata
```

Benefits:
- No concurrent writes to shared files — each job creates its own file
- `submit` checks completion via `os.path.exists()` (single MDS lookup) instead of reading + parsing a growing log
- Listing the `completed/` directory is a single `readdir()` — fast even on Lustre for flat directories with short names
- Failure metadata can be structured JSON instead of space-delimited text

The `submit` and `status` commands just do `set(os.listdir(completed_dir))` — one MDS call regardless of how many jobs have finished.

---

## 4. Make the online/offline boundary explicit

**The problem**: The boundary between internet-required and offline phases is implicit:
- `prepare` needs internet (clear)
- `create-run` optionally needs internet (HF sync), but there's no flag to control this
- `collect --upload` needs internet
- The job script sets `HF_HUB_OFFLINE=1` via `env_vars`, but this is user-configured, not enforced

If a user forgets `HF_HUB_OFFLINE`, a compute node could silently stall trying to download a tokenizer.

**Fixes**:

### 4a. Enforce offline mode in the worker
Hard-set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the worker/job script unconditionally, regardless of user config. These are always correct on compute nodes. The user's `env_vars` should only add to this, not override it.

### 4b. Separate HF sync from create-run
Extract HF sync into its own command (`eval-hive sync`) that can be run independently on a login node. `create-run --offline` would skip it entirely. This makes the workflow:
```
login node:   prepare → sync → create-run
compute:      submit → [jobs run offline]
login node:   collect → upload
```

### 4c. Pre-flight validation
Before submitting, verify that all required paths exist on the filesystem (model weights, eval suite YAML, datasets, request cache tarball). Currently, these fail at runtime — minutes or hours after submission, wasting allocation time.

---

## 5. Simplify the multi-step workflow

**The problem**: A full evaluation requires 5-7 manual steps:
```
validate-config → prepare → create-run → submit → [wait] → status → collect → compact
```

Each step requires remembering paths and flags. There's no single "just run it" command.

**Fixes**:

### 5a. `eval-hive go` command
A single command that runs the full pipeline: validate → create-run → submit. With `--watch`, it polls status and auto-collects when done. Smart about skipping steps that are already complete.

### 5b. Auto-compact on collect
`collect` should compact results as part of its normal operation (or offer `--compact`), eliminating the need for a separate step.

### 5c. SLURM job dependency for auto-collect
Submit a lightweight "collector" job with `--dependency=afterany:$JOB_IDS` that runs `collect` automatically when all evaluation jobs finish. No manual polling needed.

---

## 6. Structured state and better failure recovery

**The problem**:
- State is scattered across 6+ files (manifest, config, task map, HF covered, completed log, failed log)
- `jobs_failed.log` uses space-delimited format that's fragile to parse (what if a path has spaces?)
- No automatic retry — if a job fails due to a transient error (OOM, node failure), you must manually resubmit
- The SIGUSR1 handler resubmits on timeout, but other transient failures (server startup race, temporary disk full) don't auto-retry

**Fixes**:

### 6a. Structured failure records
Use JSON for failure records (trivial if using per-key marker files from item 3):
```json
{"reason": "health_check_failed", "job_id": "12345", "timestamp": "...", "attempt": 2, "exit_code": 1}
```

### 6b. Retry with backoff
`submit` should have a `--retry-failed` mode that resubmits tasks in the `failed/` directory, optionally filtering by reason (skip `invalid_task_key`, retry `health_check_failed`). The worker should track attempt count to implement per-task retry limits.

### 6c. Post-job hooks
Allow a configurable `post_job_command` in the config that runs after each job completes (e.g., send a Slack notification, write to a monitoring system, trigger incremental collection).

---

## 7. Config improvements

**The problem**:
- One config = one server topology. If you need to evaluate both a 70B model (TP=8, 2 nodes) and a 1.7B model (TP=1, 1 node), you need two separate configs and two separate runs. This is the most common source of complexity in practice.
- No config inheritance — configs for similar setups have lots of duplication.

**Fixes**:

### 7a. Per-model server overrides
Allow models to override scaling parameters:
```yaml
models:
  small_model:
    path: /path/to/1.7b
    scaling:                      # override top-level scaling
      num_inference_servers: 1
      num_nodes_per_inference_server: 1
      gres_per_node: "gpu:1"
  large_model:
    path: /path/to/70b
    scaling:
      num_inference_servers: 1
      num_nodes_per_inference_server: 2
      gres_per_node: "gpu:4"
```

This requires generating different SBATCH headers per model (or using SLURM's `--het-group`), which adds complexity, but eliminates the biggest workflow pain point.

### 7b. Config includes
Support `!include base_config.yaml` or a `base:` field for config inheritance. Common patterns (SLURM settings, eval suites, env setup) can be shared across configs.

---

## 8. Testing

**The problem**: Zero tests. The codebase has no unit tests, no integration tests, no CI. Config validation catches some errors, but the bash script logic, result parsing, aggregation math, and dedup logic are all untested.

**The fix**: With the Python worker (item 1), most logic becomes directly testable:
- **Config**: Pydantic models already validate — add edge case tests (empty models, duplicate keys, missing paths)
- **Manifest building**: Test checkpoint resolution, key sanitization, dedup detection
- **Result parsing**: Test with sample lm-eval JSON fixtures
- **Aggregation**: Test bottom-up computation with known hierarchies
- **Worker dedup**: Test task filtering with mock filesystem
- **HF sync**: Test coverage computation with mock parquet data

A `tests/` directory with pytest fixtures and ~200 lines of tests would catch most regressions.

---

## 9. Minor but worthwhile improvements

| Area | Current | Improvement |
|------|---------|-------------|
| Port selection | `RANDOM % 30000` — collision possible | Bind to port 0 and read assigned port, or use a lock file |
| Load balancer | Custom Python aiohttp proxy | Consider socat/HAProxy if available, or make LB auto-restartable |
| Logging | Mix of loguru and print | Standardize on structured JSON logging for machine parsing |
| Result format | lm-eval's nested JSON | Write a thin wrapper that outputs results in the target parquet schema directly, skip the JSON-to-parquet conversion |
| `prepare` command | Downloads datasets + builds cache | Add `--verify` mode that checks all artifacts exist without downloading |

---

## Priority ranking

If implementing incrementally:

1. **Python worker** (item 1) — highest leverage, unlocks testing, reduces maintenance burden
2. **Per-job marker files** (item 3) — small change, big Lustre reliability win
3. **Completion index** (item 2a) — eliminates the most expensive Lustre operations in status/collect
4. **Enforce offline mode** (item 4a) — one-line fix, prevents wasted compute hours
5. **Pre-flight validation** (item 4c) — catches errors before they waste allocation time
6. **Tests** (item 8) — essential for confident refactoring
7. **Flatten output dirs** (item 2b) — reduces inode pressure, eliminates compact
8. **`go` command** (item 5a) — quality of life
9. **Per-model scaling** (item 7a) — biggest UX win but most complex to implement
