"""Continuous eval loop: collect, compact, create-run, submit, sleep, repeat."""

import argparse
import re
import subprocess
import sys
import time

from loguru import logger


def parse_interval(value: str) -> int:
    """Parse a human-readable interval like '30m', '2h', '3600s', or plain seconds."""
    match = re.fullmatch(r"(\d+)\s*([smh])?", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid interval: {value!r} (use e.g. 30m, 2h, 3600s)")
    n, unit = int(match.group(1)), match.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


def eval_hive(*args: str) -> None:
    """Run an eval-hive subcommand, raising on failure."""
    cmd = [sys.executable, "-m", "eval_hive", *args]
    logger.info("Running: {}", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_cycle(run_dir: str, config: str) -> None:
    eval_hive("collect", run_dir, "--upload")
    eval_hive("compact", run_dir)
    eval_hive("create-run", "--config", config, "--output", run_dir, "--force")
    eval_hive("submit", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the eval-hive YAML config")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory")
    parser.add_argument(
        "--interval",
        type=parse_interval,
        default=3600,
        help="Sleep interval between cycles, e.g. 30m, 2h, 3600s (default: 1h)",
    )
    args = parser.parse_args()

    cycle = 0
    while True:
        cycle += 1
        logger.info("=== Cycle {} start ===", cycle)
        try:
            run_cycle(args.run_dir, args.config)
        except subprocess.CalledProcessError as e:
            logger.error("Command failed (exit {}): {}", e.returncode, e.cmd)
        logger.info("=== Cycle {} done, sleeping {}s ===", cycle, args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
