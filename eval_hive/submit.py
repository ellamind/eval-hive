import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from loguru import logger

from eval_hive.config import load_config

ACTIVE_JOB_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}


# ── Dedup helpers ─────────────────────────────────────────────


def get_completed_tasks(run_dir: Path) -> set[str]:
    """Read completed task keys from progress log."""
    completed_file = run_dir / "progress" / "jobs_completed.log"
    if not completed_file.exists():
        return set()
    return {
        line.strip()
        for line in completed_file.read_text().splitlines()
        if line.strip()
    }


def get_active_jobs(job_name: str) -> list[dict]:
    """Get active SLURM jobs matching the eval-hive job name pattern.

    Looks for jobs named "{job_name}-{manifest_key}".
    """
    cmd = ["squeue", "--me", "--json"]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"squeue failed: {e.stderr or e.stdout}") from e

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse squeue JSON output") from e

    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise RuntimeError("Unexpected squeue JSON structure: 'jobs' is not a list")

    prefix = f"{job_name}-"
    active = []
    for job in jobs:
        name = job.get("name", "")
        if not name.startswith(prefix) or len(name) <= len(prefix):
            continue

        task_key = name[len(prefix):]

        states = job.get("job_state", [])
        if isinstance(states, str):
            states = [states]
        if not any(s in ACTIVE_JOB_STATES for s in states):
            continue

        active.append({
            "task_key": task_key,
            "job_id": job.get("job_id"),
            "state": states[0],
            "name": name,
        })

    return active


def get_tasks_to_submit(
    manifest: dict,
    run_dir: Path,
    job_name: str,
    output_path: Path | None = None,
    suites: list[str] | None = None,
) -> list[str]:
    """Determine which manifest keys still need submission."""
    from tabulate import tabulate

    all_keys = list(manifest.keys())

    # Check completed (includes keys marked by HF sync in create-run)
    completed = get_completed_tasks(run_dir)

    # Check queue
    try:
        active = get_active_jobs(job_name)
    except Exception as e:
        logger.error(f"Error checking SLURM queue: {e}")
        raise

    in_queue = {j["task_key"] for j in active if j["task_key"] in manifest}
    active = [j for j in active if j["task_key"] in manifest]

    to_submit = [
        key for key in all_keys
        if key not in completed
        and key not in in_queue
    ]

    # Task-level coverage
    task_line = ""
    if output_path and suites:
        task_map_path = run_dir / "eh_task_map.json"
        if task_map_path.exists():
            from eval_hive.create_run import count_task_coverage

            task_map = json.loads(task_map_path.read_text())
            hf_covered_path = run_dir / "eh_hf_covered.json"
            hf_covered = json.loads(hf_covered_path.read_text()) if hf_covered_path.exists() else None

            all_tasks: set[str] = set()
            for s in suites:
                all_tasks.update(task_map.get(s, []))

            total, n_disk, n_hf_only, _, _ = count_task_coverage(
                output_path, manifest, all_tasks, hf_covered,
            )
            n_done = n_disk + n_hf_only
            n_remaining = total - n_done
            parts = []
            if n_disk:
                parts.append(f"{n_disk} on disk")
            if n_hf_only:
                parts.append(f"{n_hf_only} on HF")
            task_line = f"{n_done}/{total} done, {n_remaining} remaining"
            if parts:
                task_line += f"  ({', '.join(parts)})"

    # Status summary table
    rows = [
        ["Total", f"{len(all_keys)} jobs"],
        ["Completed", f"{len(completed)} jobs"],
    ]
    if active:
        state_counts = Counter(j["state"] for j in active)
        detail = ", ".join(f"{c} {s.lower()}" for s, c in sorted(state_counts.items()))
        rows.append(["In queue", f"{len(in_queue)} jobs  ({detail})"])
    else:
        rows.append(["In queue", "0 jobs"])
    rows.append(["To submit", f"{len(to_submit)} jobs"])
    if task_line:
        rows.append(["Tasks", task_line])

    print(tabulate(rows, tablefmt="rounded_outline"))

    return to_submit


# ── Submission ────────────────────────────────────────────────


def submit_tasks(
    tasks_to_submit: list[str],
    job_name: str,
    job_script: Path,
) -> tuple[int, list[str]]:
    """Submit tasks via sbatch. Returns (submitted_count, remaining_tasks)."""
    total = len(tasks_to_submit)
    submitted = 0
    for i, task_key in enumerate(tasks_to_submit):
        cmd = [
            "sbatch",
            f"--export=ALL,EH_TASK_KEY={task_key}",
            "--job-name", f"{job_name}-{task_key}",
            str(job_script),
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            submitted += 1
            job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "?"
            logger.info(f"  [{submitted}/{total}] {task_key} -> job {job_id}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or ""
            if "QOSMaxSubmitJobPerUserLimit" in error_msg or "Job violates accounting/QOS policy" in error_msg:
                remaining = tasks_to_submit[i:]
                logger.warning(f"QOS limit reached. Submitted {submitted}/{total}, {len(remaining)} remaining.")
                return submitted, remaining
            else:
                logger.error(f"sbatch failed for {task_key}: {error_msg}")
                raise RuntimeError(f"sbatch failed: {error_msg}") from e

    return submitted, []


# ── Main ──────────────────────────────────────────────────────


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the submit command."""
    parser.add_argument(
        "run_dir", type=Path,
        help="Path to the run directory created by create_run",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Preview what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of jobs to submit per cycle",
    )
    parser.add_argument(
        "--task-key", nargs="+", default=None,
        help="Only submit these specific task keys (exact match, as shown in status/dry run)",
    )
    parser.add_argument(
        "--retry-interval", type=int, default=None,
        help="Retry interval in minutes. Script will keep retrying until all tasks are submitted.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=None,
        help="Maximum number of retry cycles (default: unlimited)",
    )


def run(args: argparse.Namespace) -> int:
    config = load_config(args.run_dir / "eh_config.yaml")
    job_script = args.run_dir / "eh_job.slurm"
    manifest_path = args.run_dir / "eh_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if not job_script.exists():
        logger.error(f"Job script not found: {job_script}")
        return 1

    retry_count = 0

    while True:
        if retry_count > 0:
            logger.info(f"=== Retry cycle {retry_count} ===")

        tasks_to_submit = get_tasks_to_submit(
            manifest, args.run_dir, config.job_name,
            output_path=config.output_path,
            suites=config.eval.suites_and_tasks,
        )

        if args.task_key:
            unknown = set(args.task_key) - set(manifest.keys())
            if unknown:
                logger.error(f"Unknown task keys: {', '.join(sorted(unknown))}")
                return 1
            tasks_to_submit = [k for k in tasks_to_submit if k in set(args.task_key)]

        if not tasks_to_submit:
            logger.info("Nothing to submit.")
            return 0

        if args.limit:
            tasks_to_submit = tasks_to_submit[:args.limit]
            logger.info(f"Limited to {args.limit} jobs per cycle.")

        if args.dry:
            logger.info("Dry run — would submit:")
            for task_key in tasks_to_submit:
                entry = manifest[task_key]
                logger.info(f"  {task_key}: {entry['model_key']}/{entry['label']}")
            return 0
        submitted, remaining = submit_tasks(tasks_to_submit, config.job_name, job_script)

        if not remaining:
            logger.info(f"Done. Submitted {submitted} jobs.")
            if args.retry_interval:
                tasks_to_submit = get_tasks_to_submit(
                    manifest, args.run_dir, config.job_name,
                    output_path=config.output_path,
                    suites=config.eval.suites_and_tasks,
                )
                if not tasks_to_submit:
                    logger.info("All tasks submitted or in queue. Exiting.")
                    return 0
                remaining = tasks_to_submit
            else:
                return 0

        if remaining and args.retry_interval:
            retry_count += 1
            if args.max_retries and retry_count >= args.max_retries:
                logger.warning(
                    f"Max retries ({args.max_retries}) reached. "
                    f"{len(remaining)} tasks still pending."
                )
                return 1

            logger.info(
                f"Waiting {args.retry_interval} minutes before next retry... "
                f"({len(remaining)} tasks remaining)"
            )
            time.sleep(args.retry_interval * 60)
        elif remaining:
            raise RuntimeError(
                f"QOS limit reached. {len(remaining)} tasks not submitted. "
                f"Use --retry-interval to auto-retry."
            )


def main():
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(
        description="Submit eval-hive jobs from a run directory"
    )
    add_arguments(parser)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
