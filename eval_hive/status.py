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


def _has_results(path: Path) -> bool:
    return path.exists() and any(path.glob("results_*.json"))


def get_task_progress(
    output_path: Path,
    model_key: str,
    label: str,
    suites_and_tasks: list[str],
    task_map: dict[str, list[str]],
) -> dict[str, tuple[int, int]]:
    """Returns {suite_or_task: (done, total)} for each entry in suites_and_tasks."""
    base = output_path / model_key / label
    result = {}
    for suite in suites_and_tasks:
        tasks = task_map[suite]
        done = sum(1 for t in tasks if _has_results(base / t))
        result[suite] = (done, len(tasks))
    return result


def get_unique_progress(
    output_path: Path,
    model_key: str,
    label: str,
    task_map: dict[str, list[str]],
    suites: list[str],
) -> tuple[int, int]:
    """Returns (done, total) across all suites, deduplicating shared tasks."""
    base = output_path / model_key / label
    all_tasks = set()
    for s in suites:
        all_tasks.update(task_map.get(s, []))
    done = sum(1 for t in all_tasks if _has_results(base / t))
    return done, len(all_tasks)


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
    rows = []

    for task_key, entry in manifest.items():
        model_key = entry["model_key"]
        label = entry["label"]

        # Per-suite progress
        suite_progress = get_task_progress(output_path, model_key, label, suites_and_tasks, task_map)
        # Deduplicated overall progress
        done, total = get_unique_progress(output_path, model_key, label, task_map, suites_and_tasks)
        all_done = done == total

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
        progress_str = f"{done}/{total}"

        # SLURM info
        slurm_str = ""
        if task_key in active_by_key:
            job = active_by_key[task_key]
            slurm_str = f"{job['job_id']} {job['state']}"

        # Failed reason
        fail_str = ""
        if state == "failed":
            fail_str = failed[task_key]["reason"]

        rows.append([task_key, state, progress_str, slurm_str or fail_str])

        if args.detailed:
            for suite in suites_and_tasks:
                s_done, s_total = suite_progress.get(suite, (0, 0))
                rows.append(["  \u2514 " + suite, "", f"{s_done}/{s_total}", ""])

    print(tabulate(
        rows,
        headers=["Task Key", "Status", "Progress", "Info"],
        tablefmt="rounded_outline",
    ))

    # Summary
    print()
    parts = []
    for label_name, key in [("completed", "completed"), ("running", "running"),
                             ("pending", "pending"), ("failed", "failed"),
                             ("not started", "not_started")]:
        if counts[key] > 0:
            parts.append(f"{counts[key]} {label_name}")
    print(f"Total: {', '.join(parts)}" if parts else "Total: no tasks in manifest")
    return 0
