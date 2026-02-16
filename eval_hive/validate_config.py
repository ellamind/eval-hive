import argparse
import sys
from pathlib import Path

from loguru import logger
from tabulate import tabulate

from eval_hive.config import load_config
from eval_hive.create_run import manifest_key


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
    for model_key, entry in config.models.items():
        model_paths = entry.resolve_model_paths()
        for label, path in model_paths:
            exists = path.is_dir() or path.is_file()
            key = manifest_key(model_key, label)
            rows.append([
                key,
                model_key,
                label,
                str(path),
                "yes" if exists else "NO",
            ])

    print()
    print(tabulate(
        rows,
        headers=["Manifest Key", "Model", "Checkpoint", "Path", "Exists"],
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
