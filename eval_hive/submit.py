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


def get_tasks_to_submit(manifest: dict, run_dir: Path, job_name: str) -> list[str]:
    """Determine which manifest keys still need submission."""
    all_keys = list(manifest.keys())

    # Check completed
    completed = get_completed_tasks(run_dir)
    if completed:
        logger.info(f"Tasks completed: {len(completed)}")

    # Check queue
    try:
        active = get_active_jobs(job_name)
    except Exception as e:
        logger.error(f"Error checking SLURM queue: {e}")
        raise

    if active:
        active.sort(key=lambda j: j["task_key"])
        state_counts = Counter(j["state"] for j in active)
        logger.info(f"Found {len(active)} active jobs for {job_name}")
        for state, count in sorted(state_counts.items()):
            logger.info(f"    {count}/{len(active)} {state}")
    else:
        logger.info(f"No active jobs found for {job_name}")

    in_queue = {j["task_key"] for j in active}

    return [
        key for key in all_keys
        if key not in completed and key not in in_queue
    ]


# ── Submission ────────────────────────────────────────────────


def submit_tasks(
    tasks_to_submit: list[str],
    job_name: str,
    job_script: Path,
) -> tuple[int, list[str]]:
    """Submit tasks via sbatch. Returns (submitted_count, remaining_tasks)."""
    submitted = 0
    for i, task_key in enumerate(tasks_to_submit):
        logger.info(f"  {i + 1}/{len(tasks_to_submit)}: Submitting job for {task_key}...")
        cmd = [
            "sbatch",
            f"--export=ALL,EH_TASK_KEY={task_key}",
            "--job-name", f"{job_name}-{task_key}",
            str(job_script),
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"    {result.stdout.strip()}")
            submitted += 1
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or ""
            if "QOSMaxSubmitJobPerUserLimit" in error_msg or "Job violates accounting/QOS policy" in error_msg:
                logger.warning(
                    f"QOS limit reached after submitting {submitted} jobs. "
                    f"Remaining: {len(tasks_to_submit) - i}"
                )
                return submitted, tasks_to_submit[i:]
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

        tasks_to_submit = get_tasks_to_submit(manifest, args.run_dir, config.job_name)
        logger.info(f"Tasks to submit: {len(tasks_to_submit)}")

        if not tasks_to_submit:
            logger.info("No tasks to submit. All are either completed or in queue.")
            return 0

        if args.limit:
            logger.warning(f"Applying limit of {args.limit} per cycle.")
            tasks_to_submit = tasks_to_submit[:args.limit]

        if args.dry:
            logger.info("Dry run — would submit the following tasks:")
            for task_key in tasks_to_submit:
                entry = manifest[task_key]
                logger.info(f"  {task_key}: {entry['model_key']}/{entry['label']}")
            return 0

        logger.info(f"Submitting {len(tasks_to_submit)} jobs...")
        submitted, remaining = submit_tasks(tasks_to_submit, config.job_name, job_script)

        if not remaining:
            logger.info(f"Done. Submitted {submitted} jobs.")
            if args.retry_interval:
                tasks_to_submit = get_tasks_to_submit(manifest, args.run_dir, config.job_name)
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
