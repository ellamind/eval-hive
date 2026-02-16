import argparse
import subprocess
from pathlib import Path

from loguru import logger

from eval_hive.config import load_config
from eval_hive.submit import get_active_jobs


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the cancel command."""
    parser.add_argument(
        "run_dir", type=Path,
        help="Path to the run directory created by create_run",
    )


def run(args: argparse.Namespace) -> int:
    """Cancel all active SLURM jobs for a run."""
    config = load_config(args.run_dir / "eh_config.yaml")

    try:
        active = get_active_jobs(config.job_name)
    except Exception as e:
        logger.error(f"Error checking SLURM queue: {e}")
        return 1

    job_ids = [str(j["job_id"]) for j in active if j.get("job_id")]

    if not job_ids:
        logger.info(f"No active jobs found for {config.job_name}.")
        return 0

    logger.info(f"Found {len(job_ids)} active job(s) for {config.job_name}")

    batch_size = 100
    for start in range(0, len(job_ids), batch_size):
        batch = job_ids[start:start + batch_size]
        cmd = ["scancel", *batch]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Cancelled {len(batch)} job(s): {' '.join(batch)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"scancel failed: {e.stderr or e.stdout}")
            return 1

    logger.info("Done.")
    return 0


def main():
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(description="Cancel eval-hive jobs for a run")
    add_arguments(parser)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    main()
