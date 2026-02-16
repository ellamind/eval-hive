# eval-hive

Run [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) at scale on Slurm clusters.

eval-hive manages the full evaluation lifecycle. Configure models and eval suites, plan work, submit Slurm jobs, track progress, and avoid duplicate work.  Sibling project to [inference-hive](https://github.com/ellamind/inference-hive).

### Features

- **Batch evaluation** — evaluate many models and checkpoints against many eval suites with a single config file.
- **Slurm native** — automatic job script generation, array jobs, resource management, signal handling.
- **Any lm-eval backend** — works with vLLM, sglang, or serverless backends (hf, nemo). Any lm-eval task works out of the box.
- **Flexible server deployment** — single-node, multi-node, or multiple independent inference servers behind a built-in load balancer for high throughput eval.
- **Parallel task execution** — run multiple lm-eval processes concurrently to keep GPUs busy.
- **Request caching** — pre-build lm-eval request caches for fast, race-free distribution to compute nodes.
- **Result collection** — collect lm-eval results into a parquet file with automatic score aggregation for task groups and eval suites.
- **HuggingFace sync** — optionally push results to a HF dataset repo, with merge and dedup. Skip already-covered jobs at submit time with `--check-hf`.
- **Deduplication** — skips already-completed work at submit time, runtime, and optionally against a HF dataset. Resubmit safely after partial failures.
- **Failure handling** — per-task resumability, SIGUSR1 graceful shutdown, idempotent resubmission.
- **Pydantic config validation** — catches errors before any jobs are submitted.

## Installation

Requires [pixi](https://pixi.sh). Clone the repo and let pixi handle the environment:

```bash
git clone git@github.com:ellamind/eval-hive-2.git
cd eval-hive-2
pixi install
```

All commands below use `pixi run eval-hive` (alias for `python -m eval_hive`).

## Quickstart

```bash
# 1. Copy the template and fill in your cluster/model details
cp eh_config_template.yaml my_eh_config.yaml

# 2. Validate your config
pixi run eval-hive validate-config my_eh_config.yaml

# 3. Download datasets and build request caches (run on a login node with internet)
pixi run eval-hive prepare my_eh_config.yaml

# 4. Create a run directory with manifest and Slurm script
pixi run eval-hive create-run --config my_eh_config.yaml --output runs/my-run

# 5. Submit jobs
pixi run eval-hive submit runs/my-run

# 6. Monitor progress
pixi run eval-hive status runs/my-run

# 7. Collect results into parquet (with optional HF push)
pixi run eval-hive collect runs/my-run
pixi run eval-hive collect runs/my-run --push-to org/eval-scores
```

## Workflow

```
                  login node                          compute nodes
                (has internet)                      (no internet)

  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌───────────────────────────┐
  │ prepare  │───►│ create_run │───►│  submit  │───►│  slurm job per model      │
  │          │    │            │    │          │    │                           │
  │ download │    │ plan work  │    │ sbatch   │    │ (optional) start server(s)│
  │ datasets │    │ check done │    │ array    │    │ (optional) load balancer  │
  │ + caches │    │ write      │    │ jobs     │    │ run lm_eval suites        │
  │ + tarball│    │ manifest   │    │          │    │ write results             │
  └──────────┘    └────────────┘    └──────────┘    └───────────────────────────┘
                                         │
                                    ┌────┴─────┐
                                    │  status  │
                                    │ monitor  │
                                    │ progress │
                                    └────┬─────┘
                                         │
                                    ┌────┴─────┐
                                    │ collect  │
                                    │ parse +  │
                                    │ aggregate│
                                    │ → parquet│
                                    │ (→ HF)   │
                                    └──────────┘
```

### Commands

```bash
# Validate config (optional — prepare and create-run also validate)
pixi run eval-hive validate-config eh_config.yaml

# Prepare: download datasets and build request caches (login node, has internet)
pixi run eval-hive prepare eh_config.yaml
pixi run eval-hive prepare eh_config.yaml --refresh          # rebuild all caches
pixi run eval-hive prepare eh_config.yaml --workers 8        # parallel cache building

# Plan and generate run directory
pixi run eval-hive create-run --config eh_config.yaml --output runs/my-run
pixi run eval-hive create-run --config eh_config.yaml --output runs/my-run --force   # overwrite
pixi run eval-hive create-run --config eh_config.yaml --output runs/my-run --update  # update models/suites

# Submit jobs
pixi run eval-hive submit runs/my-run
pixi run eval-hive submit runs/my-run --dry                      # preview only
pixi run eval-hive submit runs/my-run --limit 5                  # submit at most 5 jobs
pixi run eval-hive submit runs/my-run --retry-interval 10        # retry every 10 min until all submitted
pixi run eval-hive submit runs/my-run --check-hf org/eval-scores # skip jobs covered in HF dataset

# Monitor progress
pixi run eval-hive status runs/my-run
pixi run eval-hive status runs/my-run --detailed                 # per-suite/task breakdown

# Collect results into parquet
pixi run eval-hive collect runs/my-run                           # writes scores.parquet
pixi run eval-hive collect runs/my-run -o my_scores.parquet      # custom output path
pixi run eval-hive collect runs/my-run --push-to org/eval-scores # merge + dedup + upload to HF

# Cancel all active jobs for a run
pixi run eval-hive cancel runs/my-run
```

## Configuration

Single YAML file per setup. Copy `eh_config_template.yaml` to get started. Models needing different server or Slurm settings get separate config files.


### Config design principles

- **Flat structure** — one config = one set of server/Slurm/eval settings. Models needing different parallelism or backends get separate configs.
- **`inference_server_command` is a plain string with placeholders**, not a structured config. Supports any inference server backend (vLLM, sglang) and container wrapping (singularity, apptainer) without schema changes. Placeholders substituted at runtime: `${EH_PORT}`, `${EH_MODEL_PATH}`.
- **`inference_server_command: null`** disables the server lifecycle entirely. Use this for lm-eval backends that load models directly (`hf`, `nemo`, etc.). The job skips server start/health-check/stop and runs `lm_eval` immediately.
- **`eval.model_args`** dict is joined as `key=value,key=value` for `lm_eval --model_args`. **`eval.lm_eval_args`** is a passthrough dict — each key becomes a CLI flag (`--key value`). No schema changes needed when lm-eval adds new flags.
- **Model entries carry metadata** — `path`, `display_name`, optional `model_key` override, checkpoint info, and training metadata (`train_batch_size` / `tokens_trained`). A model and its checkpoints can share the same `model_key` — they differ only by `step` in the parquet composite key `(model, step, task, metric, metric_filter)`.
- **Pydantic validation** catches config errors before any jobs are submitted.

### Model entries

Each model entry under `models:` defines a model or checkpoint series to evaluate:

```yaml
models:
  # Config dict keys must be unique but are only used internally.
  # Use model_key to control the parquet 'model' column.
  HPLT2c_eng_main:
    model_key: HPLT2c_eng          # parquet 'model' column (defaults to config key)
    path: "/path/to/model/main"
    display_name: "HPLT2c eng"     # required: human-friendly name
    tokens_trained: 100B           # recommended for non-checkpoint models (supports K, M, B, T)

  HPLT2c_eng_checkpoints:
    model_key: HPLT2c_eng          # same model_key → same model in parquet, different steps
    path: "/path/to/model"
    display_name: "HPLT2c eng"
    checkpoint_pattern: "checkpoint_{step}"
    steps: [5000, 10000, 20000]    # optional filter (default: discover all)
    train_batch_size: 2_097_152    # recommended for checkpoints (tokens_trained = batch_size × step)
```

| Field | Required | Description |
|-------|----------|-------------|
| `path` | yes | Local path or HuggingFace model ID |
| `display_name` | yes | Human-friendly name for result display |
| `model_key` | no | Parquet `model` column value. Defaults to config dict key. Set this to share a model key between a main model and its checkpoints. |
| `checkpoint_pattern` | no | Pattern for checkpoint subdirectories, e.g. `checkpoint_{step}` |
| `steps` | no | Filter to specific steps. Requires `checkpoint_pattern`. |
| `train_batch_size` | no | Batch size in tokens. Recommended for checkpoints (`tokens_trained = train_batch_size × step`). |
| `tokens_trained` | no | Total tokens trained. Supports human-readable suffixes: `100B`, `2T`, `500M`. Recommended for non-checkpoint models. |

## Prepare step

The prepare step (`prepare.py`) runs on a node with internet access and does two things:

1. **Downloads datasets** — loading each task triggers HuggingFace dataset downloads if not already in cache.
2. **Builds request caches** — calls `task.build_all_requests(cache_requests=True)` to pre-build and pickle the prompt instances lm-eval will need at eval time.

At the end, it packs all cache files into a compressed tarball (`<cache_dir>/cache.tar.gz`) for fast distribution to compute nodes.

### Why the tarball

Compute nodes on Lustre (or similar parallel filesystems) are slow at reading many small files due to metadata overhead. Eval jobs extract the tarball to a local temp directory and point `LM_HARNESS_CACHE_PATH` at it:

This also prevents race conditions. `lm-eval` has no file locking on cache writes, so if multiple jobs hit a cache miss simultaneously they'd all race to write the same pickle file. With per-job temp directories, each job has its own isolated copy.

### Tarball rebuild logic

The tarball is only rebuilt when needed:
- New cache files were created
- `--refresh` was used (all caches rewritten)
- The tarball doesn't exist yet

If all caches are already up to date, the tarball step is skipped.

### Parallel cache building

With `--workers N` (default 4), tasks are sharded across worker processes. Tasks sharing the same dataset are grouped together to avoid redundant downloads. A greedy bin-packing algorithm balances load across workers.

## Create-run step

The `create_run` step resolves the config into a concrete execution plan:

1. **Loads config** and validates with Pydantic.
2. **Builds manifest** — iterates all models, resolves checkpoint patterns, produces a flat mapping of `{model_key, label, model_path, display_name, train_batch_size, tokens_trained}` dicts keyed by `{model_key}--{label}`. The `model_key` comes from the entry's `model_key` field (or the config dict key if not set).
3. **Generates run directory** containing:
   - `eh_manifest.json` — the manifest (read by `jq` in the job script)
   - `eh_config.yaml` — frozen copy of the input config
   - `eh_job.slurm` — generated SBATCH script with all config baked in
   - `logs/` and `progress/` directories

The generated Slurm script handles the full job lifecycle:
- Reads its manifest entry via `jq .[${SLURM_ARRAY_TASK_ID}]`
- Sets `EH_MODEL_PATH`, `EH_PORT` (random 30000–59999), environment variables
- Optionally starts inference server, runs health checks, shuts down on completion
- Loops over all suites, skipping any with existing `results_*.json` (runtime dedup)
- Uses `--output_path` with `.json` suffix so lm-eval writes directly to the suite directory without creating model subdirectories
- Signal handling: SIGUSR1 (approaching timeout) triggers auto-resubmission, SIGTERM marks failure
- Records completion/failure to append-only logs

## Architecture

### Scaling model

Each Slurm job evaluates **one model** (or checkpoint) against **all its suites**, amortizing the cost of server startup. The job array maps each index to a (model, checkpoint) pair via a manifest.

```
Job array index 0 → model-A               → [suite_easy, suite_main]
Job array index 1 → model-A/ckpt-10000    → [suite_easy, suite_main]
Job array index 2 → model-A/ckpt-20000    → [suite_easy, suite_main]
Job array index 3 → model-B               → [suite_easy, suite_main]
```

### Parallel per-task execution (`parallel_tasks`)

By default (`parallel_tasks: 1`), suites are evaluated sequentially — one `lm_eval` process at a time. lm-eval tokenizes all requests on a single CPU core before sending any API calls, which causes a multi-minute stall with GPUs idle for large suites.

With `parallel_tasks: N` (N > 1), suites are expanded into their individual leaf tasks (e.g., `my_suite_easy` → `arc_easy_rc`, `hellaswag_rc`, ...) and up to N `lm_eval` processes run concurrently, each handling one task. This spreads tokenization across multiple CPU cores while the inference server handles concurrent requests from all processes.

`create_run` resolves suite/group names to leaf tasks and writes `eh_task_map.json` into the run directory. At runtime, the Slurm script reads it to build a deduplicated task list (tasks shared across suites are evaluated once).

Results and per-task logs are stored flat by task name:
```
{output_path}/{model_key}/{label}/{task_name}/results_*.json
{output_path}/{model_key}/{label}/{task_name}/lm_eval.log
```

The main Slurm log (`logs/`) only shows task start/finish messages. Full lm-eval output (progress bars, warnings, result tables) is captured exclusively in the per-task `lm_eval.log` files.

### Server scaling (num_nodes_per_inference_server × num_inference_servers)

Two independent scaling axes control how inference servers are deployed:

| Axis | Purpose | When to use |
|------|---------|-------------|
| `num_nodes_per_inference_server` | **Capacity** — fit a large model across multiple nodes | 70B model needs TP=8 across 2× 4-GPU nodes |
| `num_inference_servers` | **Throughput** — run independent instances behind load balancer | Fast eval with 4 parallel vLLM instances |

Total Slurm nodes per job = `num_nodes_per_inference_server × num_inference_servers`.

**Single server, single node** (default: `1×1`):
```
Slurm job (--nodes=1)

  Node 0:
    ├── vLLM server (:64444)
    └── lm_eval ──► http://localhost:64444/v1/completions
```

**Multi-node server** for large models (`2×1`):
```
Slurm job (--nodes=2)

  Node 0 (head):
    ├── vLLM server (:64444, TP=8 via Ray across nodes 0-1)
    └── lm_eval ──► http://localhost:64444/v1/completions

  Node 1: └── vLLM Ray worker (joins head node's server)
```

**Multiple independent servers** for throughput (`1×4`):
```
Slurm job (--nodes=4)

  Node 0 (coordinator):
    ├── vLLM server (:64444)
    ├── Load balancer (:8000) ──► least-connections to all 4 backends
    └── lm_eval ──► http://localhost:8000/v1/completions

  Node 1: └── vLLM server (:64444)
  Node 2: └── vLLM server (:64444)
  Node 3: └── vLLM server (:64444)
```

**Both combined** — large model with throughput scaling (`2×2`):
```
Slurm job (--nodes=4)

  Nodes 0-1: Server instance 0 (TP=8 across 2 nodes)
    Node 0 (coordinator): vLLM head + Load balancer (:8000) + lm_eval
    Node 1: vLLM Ray worker

  Nodes 2-3: Server instance 1 (TP=8 across 2 nodes)
    Node 2: vLLM head
    Node 3: vLLM Ray worker
```

**Why least-connections over round-robin**: lm-eval requests are non-uniform — a BPB loglikelihood takes ~10ms while a CoT generation takes seconds. Least-connections routes to the server with fewest in-flight requests, naturally adapting to uneven request costs.

The load balancer is a small async Python module (`eval_hive/load_balancer.py`) using `aiohttp`. No external dependencies (nginx, traefik) needed on compute nodes.

When `num_inference_servers: 1`, no load balancer is started — lm-eval connects directly to the server instance.

### Serverless backends (inference_server_command: null)

When `inference_server_command` is set to `null`, the job skips the entire server lifecycle (start, health check, load balancer, stop) and runs `lm_eval` directly. This supports lm-eval backends that load the model in-process (`hf`, `nemo`, etc.). Use a separate config with appropriate `eval.lm_eval_args.model` and `eval.model_args`.

Job script logic:

```bash
if inference_server_command is set:
    start_server → health_check → (load_balancer if num_inference_servers > 1)
fi

# parallel_tasks=1 (sequential, per-suite):
for suite in eval.suites_and_tasks:
    if results_*.json exists in output dir: skip
    lm_eval run --tasks $suite ... > {suite}/lm_eval.log 2>&1
done

# parallel_tasks>1 (parallel, per-task):
expand suites → deduplicated leaf tasks via eh_task_map.json
for task in all_tasks (up to N concurrent):
    if results_*.json exists in task dir: skip
    lm_eval run --tasks $task ... > {task}/lm_eval.log 2>&1 &
done; wait

if inference_server_command is set:
    stop_server
fi
```

### Deduplication

Three levels of deduplication.

**Submit-time dedup** (`submit.py`): before submitting a job, checks `progress/jobs_completed.log` and the Slurm queue, skipping manifest keys that are already completed or active.

**HF-based dedup** (`submit --check-hf`): optionally downloads a HF dataset parquet and skips manifest keys whose expected tasks are already fully covered. Useful for avoiding duplicate work across clusters sharing results via HuggingFace. Respects `HF_HUB_OFFLINE` for cached/offline access.

**Runtime dedup** (inside each job): before running each suite or task, checks if `results_*.json` already exists in the output directory. Skips work that already has results, enabling partial resumption.

Results are written to deterministic paths:

```
# sequential mode (parallel_tasks=1):
{output_path}/{model_key}/{label}/{suite_name}/results_*.json

# parallel mode (parallel_tasks>1):
{output_path}/{model_key}/{label}/{task_name}/results_*.json
```

Completion tracking uses a simple append-only log (`progress/jobs_completed.log`). Each completed job appends its task ID. Failed jobs are tracked in `progress/jobs_failed.log` with reasons.

### Collect and score aggregation

The `collect` command parses lm-eval result JSON files from a run directory and produces a single parquet file.

**Discovery**: walks `{output_path}/{model_key}/{label}/*/results_*.json` using the manifest to find all result files. For each manifest entry, the training step is extracted from the label (e.g. `checkpoint_0005000` → `5000`; `main` → `None`).

**Parsing**: each result JSON is parsed into `ScoreRow` records capturing the score, metric, task metadata, language, formulation type, and subtask structure.

**Aggregation**: after parsing leaf benchmark scores, group and suite scores are computed bottom-up from the YAML task hierarchy. For each group defined in the task YAMLs:

1. The group's `aggregate_metric_list` specifies which metrics to aggregate and how.
2. Children's scores are collected from leaf results or already-computed sub-group scores.
3. The aggregate is computed as a simple mean, or weighted by `n_samples` when `weight_by_size: true`.
4. A `subtask_tree` adjacency map tracks which tasks contribute to each group score.

Groups reachable from the configured `suites_and_tasks` are tagged as `eval_suite`; intermediate groups as `task_group`.

**HF push** (`--push-to`): downloads the existing parquet from a HuggingFace dataset repo, merges with local results, deduplicates on `(model, step, task, metric, metric_filter)` keeping the latest `eval_date`, and re-uploads. The local parquet is also updated with the merged result.

### Failure handling

- **Per-suite/task resumability**: each suite (or individual task in parallel mode) completes independently. Failures in one don't affect others.
- **SIGUSR1 signal handling**: Slurm sends SIGUSR1 before timeout. The job gracefully stops the server, marks progress, and can be resubmitted.
- **Idempotent resubmission**: `submit` only submits jobs for incomplete work. Run it again after partial completion.

### Run directory structure

Generated by `create-run`:

```
runs/my-run/
├── eh_config.yaml          # frozen copy of config
├── eh_manifest.json        # task_key → {model_key, label, model_path, display_name, ...}
├── eh_task_map.json        # suite → [leaf_tasks] (used by collect + submit --check-hf)
├── eh_job.slurm            # generated sbatch script
├── logs/                   # {model}-{checkpoint}-{jobid}.log (start/finish per task only)
└── progress/
    ├── jobs_completed.log  # append-only completion tracking
    └── jobs_failed.log     # failure tracking with reasons
```

## Project structure

```
eval-hive/
├── eval_hive/
│   ├── __init__.py
│   ├── __main__.py          # CLI entry point (subcommands)
│   ├── config.py            # Pydantic config models (EhConfig, ModelEntry, EvalSection)
│   ├── create_run.py        # Generate run directory, manifest, sbatch script
│   ├── prepare.py           # Download datasets, build request caches, pack tarball
│   ├── validate_config.py   # Validate config and display model/checkpoint table
│   ├── submit.py            # Submit jobs from manifest with dedup (+ --check-hf)
│   ├── collect.py           # Collect results into parquet with aggregation (+ --push-to)
│   ├── status.py            # Monitor run progress
│   ├── cancel.py            # Cancel active Slurm jobs for a run
│   ├── load_balancer.py     # Async least-connections reverse proxy
│   └── results/
│       ├── __init__.py
│       ├── schemas.py       # ScoreRow, EvalConfig, TaskConfig (Pydantic models)
│       ├── parse.py         # Parse lm-eval result JSONs into ScoreRows
│       ├── aggregate.py     # Compute group/suite scores from YAML hierarchy
│       └── hf.py            # HuggingFace parquet download, upload, merge + dedup
├── eh_config_template.yaml  # Annotated config template
├── pixi.toml                # Environment definition
└── README.md
```
