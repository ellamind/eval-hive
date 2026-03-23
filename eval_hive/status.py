import argparse
import json
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from eval_hive.config import load_config
from eval_hive.submit import get_active_jobs, get_completed_tasks


def get_failed_tasks(run_dir: Path) -> dict[str, dict]:
    """Parse jobs_failed.log into structured data.

    Returns dict mapping task_key to its most recent failure info.
    Format per line: TASK_KEY DATE TIME REASON JOB_ID INFO...
    """
    failed_file = run_dir / "progress" / "jobs_failed.log"
    if not failed_file.exists():
        return {}

    failed = {}
    for line in failed_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        task_key = parts[0]
        # parts[1] = date, parts[2] = time, parts[3] = reason, parts[4] = job_id
        failed[task_key] = {
            "timestamp": f"{parts[1]} {parts[2]}",
            "reason": parts[3],
            "job_id": parts[4],
            "info": " ".join(parts[5:]) if len(parts) > 5 else "",
        }
    return failed


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


def get_task_progress(
    completed: set[str],
    suites_and_tasks: list[str],
    task_map: dict[str, list[str]],
) -> dict[str, tuple[int, int]]:
    """Returns {suite_or_task: (done, total)} for each entry in suites_and_tasks."""
    result = {}
    for suite in suites_and_tasks:
        tasks = task_map[suite]
        done = sum(1 for t in tasks if t in completed)
        result[suite] = (done, len(tasks))
    return result


def get_unique_progress(
    completed: set[str],
    task_map: dict[str, list[str]],
    suites: list[str],
    hf_tasks: set[str] | None = None,
) -> tuple[int, int, int]:
    """Returns (local, hf_only, total) across all suites, deduplicating shared tasks."""
    all_tasks = set()
    for s in suites:
        all_tasks.update(task_map.get(s, []))
    hf_tasks = hf_tasks or set()
    local = 0
    hf_only = 0
    for t in all_tasks:
        if t in completed:
            local += 1
        elif t in hf_tasks:
            hf_only += 1
    return local, hf_only, len(all_tasks)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory created by create-run",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show per-suite/task breakdown",
    )


def run(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir.resolve()

    manifest_path = run_dir / "eh_manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    config = load_config(run_dir / "eh_config.yaml")

    suites_and_tasks = config.eval.suites_and_tasks
    job_name = config.job_name
    output_path = config.output_path

    # Load task map
    task_map_path = run_dir / "eh_task_map.json"
    if not task_map_path.exists():
        logger.error(f"Task map not found: {task_map_path}")
        return 1
    task_map = json.loads(task_map_path.read_text())

    # Compute unique task count for header
    all_unique = set()
    for s in suites_and_tasks:
        all_unique.update(task_map.get(s, []))
    unique_total = len(all_unique)

    # Load HF coverage
    hf_covered_path = run_dir / "eh_hf_covered.json"
    hf_covered = json.loads(hf_covered_path.read_text()) if hf_covered_path.exists() else {}

    # Gather state
    completed = get_completed_tasks(run_dir)
    failed = get_failed_tasks(run_dir)

    try:
        active_jobs = get_active_jobs(job_name)
    except RuntimeError:
        logger.warning("Could not query SLURM queue (squeue failed)")
        active_jobs = []

    active_by_key = {j["task_key"]: j for j in active_jobs}

    # Header
    print(f"Run: {run_dir} | Job: {job_name}")
    print(f"Suites: {', '.join(suites_and_tasks)} ({unique_total} unique tasks)")
    print()

    # Compute per-manifest-key state
    counts = {"completed": 0, "running": 0, "pending": 0, "failed": 0, "not_started": 0}
    sum_local = 0
    sum_hf = 0
    sum_total = 0
    rows = []

    # Cache completed tasks per (model_key, label) to avoid re-parsing
    _completed_cache: dict[tuple[str, str], set[str]] = {}

    for task_key, entry in manifest.items():
        model_key = entry["model_key"]
        label = entry["label"]

        cache_key = (model_key, label)
        if cache_key not in _completed_cache:
            base = output_path / model_key / label
            _completed_cache[cache_key] = _collect_completed_tasks(base)
        disk_completed = _completed_cache[cache_key]

        # Per-suite progress
        suite_progress = get_task_progress(disk_completed, suites_and_tasks, task_map)
        # Deduplicated overall progress
        hf_tasks = set(hf_covered.get(task_key, []))
        local, hf_only, total = get_unique_progress(
            disk_completed, task_map, suites_and_tasks, hf_tasks,
        )
        sum_local += local
        sum_hf += hf_only
        sum_total += total
        all_done = (local + hf_only) == total

        # Determine state
        if task_key in completed or all_done:
            state = "completed"
        elif task_key in active_by_key:
            job = active_by_key[task_key]
            state = "pending" if job["state"] == "PENDING" else "running"
        elif task_key in failed:
            state = "failed"
        else:
            state = "not_started"

        counts[state] += 1

        # SLURM info
        slurm_str = ""
        if task_key in active_by_key:
            job = active_by_key[task_key]
            slurm_str = f"{job['job_id']} {job['state']}"

        # Failed reason
        fail_str = ""
        if state == "failed":
            fail_str = failed[task_key]["reason"]

        progress_str = f"{local} / {hf_only} / {total}"
        rows.append([task_key, state, progress_str, slurm_str or fail_str])

        if args.detailed:
            for suite in suites_and_tasks:
                s_done, s_total = suite_progress.get(suite, (0, 0))
                rows.append(["  \u2514 " + suite, "", f"{s_done} / {s_total}", ""])

    print(tabulate(
        rows,
        headers=["Task Key", "Status", "Progress (local/HF/total)", "Info"],
        tablefmt="rounded_outline",
    ))

    # Summary
    print()
    sum_done = sum_local + sum_hf
    pct = (sum_done / sum_total * 100) if sum_total else 0
    parts_detail = []
    if sum_local:
        parts_detail.append(f"{sum_local} on disk")
    if sum_hf:
        parts_detail.append(f"{sum_hf} on HF")
    detail_str = f"  ({', '.join(parts_detail)})" if parts_detail else ""
    print(f"Tasks: {sum_done}/{sum_total} done, {sum_total - sum_done} remaining{detail_str}  ({pct:.0f}%)")
    parts = []
    for label_name, key in [("completed", "completed"), ("running", "running"),
                             ("pending", "pending"), ("failed", "failed"),
                             ("not started", "not_started")]:
        if counts[key] > 0:
            parts.append(f"{counts[key]} {label_name}")
    print(f"Jobs: {', '.join(parts)}" if parts else "Jobs: no tasks in manifest")
    return 0
