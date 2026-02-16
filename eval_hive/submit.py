import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import polars as pl
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


def _load_scores_parquet(source: str) -> pl.DataFrame | None:
    """Load a scores parquet from a local path or HF repo.

    If *source* points to an existing local file, reads it directly.
    Otherwise treats it as a HuggingFace dataset repo ID.
    """
    if Path(source).is_file():
        logger.info(f"Loading local parquet: {source}")
        return pl.read_parquet(source)

    from eval_hive.results.hf import download_hf_parquet

    return download_hf_parquet(source)


def get_hf_covered_keys(
    source: str,
    manifest: dict[str, dict],
    task_map: dict[str, list[str]],
    suites: list[str],
) -> set[str]:
    """Determine which manifest keys have all their tasks covered in existing data.

    *source* can be a local parquet path or a HuggingFace dataset repo ID.
    Downloads/reads the parquet once and checks each manifest key's expected
    leaf tasks against the data.
    """
    from eval_hive.collect import parse_step_from_label

    df = _load_scores_parquet(source)
    if df is None or len(df) == 0:
        logger.info("No existing score data available for dedup")
        return set()

    # Build lookup set of (model, step, task) tuples
    hf_tuples: set[tuple[str, int | None, str]] = set()
    for row in df.select("model", "step", "task").unique().iter_rows(named=True):
        step = None if row["step"] is None else int(row["step"])
        hf_tuples.add((row["model"], step, row["task"]))

    logger.info(f"HF data: {len(hf_tuples)} unique (model, step, task) tuples")

    # Collect all expected leaf tasks across all suites
    all_expected_tasks: set[str] = set()
    for suite in suites:
        all_expected_tasks.update(task_map.get(suite, []))

    covered_keys: set[str] = set()

    for mkey, entry in manifest.items():
        model_key = entry["model_key"]
        label = entry["label"]
        step = parse_step_from_label(label)

        # Check if ALL expected leaf tasks are present for this (model, step)
        if all((model_key, step, task) in hf_tuples for task in all_expected_tasks):
            covered_keys.add(mkey)

    if covered_keys:
        logger.info(f"HF dedup: {len(covered_keys)} manifest keys fully covered")

    return covered_keys


def get_tasks_to_submit(
    manifest: dict,
    run_dir: Path,
    job_name: str,
    hf_covered: set[str] | None = None,
) -> list[str]:
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

    # HF coverage (third filter layer)
    if hf_covered:
        logger.info(f"Tasks covered in HF dataset: {len(hf_covered)}")

    return [
        key for key in all_keys
        if key not in completed
        and key not in in_queue
        and (hf_covered is None or key not in hf_covered)
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
    parser.add_argument(
        "--check-hf", type=str, default=None,
        metavar="SOURCE",
        help="Skip manifest keys whose tasks are already covered. "
             "SOURCE can be a local parquet file or a HuggingFace dataset repo ID.",
    )


def run(args: argparse.Namespace) -> int:
    config = load_config(args.run_dir / "eh_config.yaml")
    job_script = args.run_dir / "eh_job.slurm"
    manifest_path = args.run_dir / "eh_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if not job_script.exists():
        logger.error(f"Job script not found: {job_script}")
        return 1

    # HF dedup: compute once before the retry loop
    hf_covered: set[str] | None = None
    if args.check_hf:
        task_map_path = args.run_dir / "eh_task_map.json"
        if task_map_path.exists():
            task_map = json.loads(task_map_path.read_text())
            hf_covered = get_hf_covered_keys(
                args.check_hf,
                manifest,
                task_map,
                config.eval.suites_and_tasks,
            )
        else:
            logger.warning(
                "Task map not found at %s. Skipping HF dedup. "
                "Re-run create-run to generate it.",
                task_map_path,
            )

    retry_count = 0

    while True:
        if retry_count > 0:
            logger.info(f"=== Retry cycle {retry_count} ===")

        tasks_to_submit = get_tasks_to_submit(
            manifest, args.run_dir, config.job_name, hf_covered
        )
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
                tasks_to_submit = get_tasks_to_submit(
                    manifest, args.run_dir, config.job_name, hf_covered
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
