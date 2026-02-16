import argparse
import re
import sys
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from eval_hive.config import load_config
from eval_hive.create_run import manifest_key


def _fmt_tokens(n: int | None) -> str:
    """Format a token count for display, e.g. 100_000_000_000 → '100B'."""
    if n is None:
        return "-"
    for suffix, divisor in [("T", 10**12), ("B", 10**9), ("M", 10**6)]:
        if n >= divisor:
            return f"{n / divisor:g}{suffix}"
    return f"{n:,}"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the validate-config command."""
    parser.add_argument(
        "config", type=str,
        help="Path to YAML configuration file",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the validate-config command with parsed arguments."""
    config_path = Path(args.config)

    if not config_path.exists():
        logger.error(f"Configuration file '{config_path}' does not exist.")
        return 1

    logger.info(f"Validating: {config_path}")

    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

    # Build model/checkpoint table
    rows = []
    for config_key, entry in config.models.items():
        effective_model_key = entry.model_key or config_key
        is_checkpoint = entry.checkpoint_pattern is not None
        display = entry.display_name
        model_paths = entry.resolve_model_paths()
        for label, path in model_paths:
            exists = path.is_dir() or path.is_file()
            key = manifest_key(effective_model_key, label)
            step = None
            if is_checkpoint:
                matches = re.findall(r"\d+", label)
                step = int(matches[-1]) if matches else None
            tokens = entry.tokens_trained
            if tokens is None and entry.train_batch_size is not None and step is not None:
                tokens = entry.train_batch_size * step
            rows.append([
                key,
                display,
                label,
                str(path),
                "yes" if exists else "NO",
                entry.train_batch_size or "-",
                _fmt_tokens(tokens),
            ])

    print()
    print(tabulate(
        rows,
        headers=["Manifest Key", "Display Name", "Checkpoint", "Path", "Exists", "Batch Size", "Tokens Trained"],
        tablefmt="rounded_outline",
    ))
    print()

    # Summary
    total_jobs = len(rows)

    logger.info(f"Jobs:             {total_jobs} (one per model/checkpoint)")
    logger.info(f"Suites and tasks: {', '.join(config.eval.suites_and_tasks)}")
    logger.info(f"Output path:      {config.output_path}")

    # Warn about missing paths
    missing = [r for r in rows if r[3] == "NO"]
    if missing:
        logger.warning(f"{len(missing)} model path(s) not found on disk (see table above)")

    logger.info("Configuration is valid!")
    return 0


def main():
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(description="Validate an eval-hive configuration file")
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
