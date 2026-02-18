import argparse
import sys

from eval_hive import cancel, collect, compact, create_run, prepare, status, submit, validate_config

COMMANDS = {
    "validate-config": (validate_config, "Validate a configuration file"),
    "prepare": (prepare, "Download datasets and build request caches"),
    "create-run": (create_run, "Create a run directory with manifest and SLURM script"),
    "submit": (submit, "Submit jobs from a run directory"),
    "status": (status, "Show progress of a run"),
    "cancel": (cancel, "Cancel all active jobs for a run"),
    "collect": (collect, "Collect results from a run into scores.parquet"),
    "compact": (compact, "Compact result files to reduce directory count"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="eval-hive",
        description="eval-hive: batch evaluation orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, (module, help_text) in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        module.add_arguments(sub)
        sub.set_defaults(func=module.run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
