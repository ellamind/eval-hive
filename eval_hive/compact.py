"""Compact result files to reduce directory count on the filesystem.

Merges all ``results_*.json`` files under each ``{model_key}/{label}/``
into a single ``compacted/results_compacted.json``, then removes the old
subdirectories.  The merged file is placed inside a ``compacted/``
subdirectory so that existing scan functions (which iterate subdirs and
glob for ``results_*.json``) pick it up automatically.
"""

import argparse
import json
import shutil
from pathlib import Path

from loguru import logger

from eval_hive.config import load_config

# Per-task dicts that should be merged across result files.
_MERGE_KEYS = (
    "results",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
    "task_hashes",
)


def compact_checkpoint(base: Path, *, dry: bool = False) -> tuple[int, int]:
    """Merge all result files under *base* into one compacted file.

    Returns ``(tasks_merged, dirs_removed)``.
    """
    if not base.is_dir():
        return 0, 0

    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        return 0, 0

    # Already fully compacted — nothing to do.
    if len(subdirs) == 1 and subdirs[0].name == "compacted":
        return 0, 0

    # Collect all result files, sorted by mtime (oldest first so that
    # later updates win when we merge via dict.update).
    result_files: list[Path] = []
    for subdir in subdirs:
        if subdir.name == "compacted":
            continue
        result_files.extend(subdir.glob("results_*.json"))

    if not result_files:
        return 0, 0

    result_files.sort(key=lambda f: f.stat().st_mtime)

    # Merge per-task dicts.
    merged: dict = {}
    for rf in result_files:
        try:
            data = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable file {}: {}", rf, exc)
            continue

        for key in _MERGE_KEYS:
            merged.setdefault(key, {}).update(data.get(key, {}))

        # File-level metadata: overwrite with each newer file.
        for key, value in data.items():
            if key not in _MERGE_KEYS:
                merged[key] = value

    task_count = len(merged.get("results", {}))

    if dry:
        dirs_to_remove = [d for d in subdirs if d.name != "compacted"]
        return task_count, len(dirs_to_remove)

    # Write merged file.
    compacted_dir = base / "compacted"
    compacted_dir.mkdir(exist_ok=True)
    compacted_file = compacted_dir / "results_compacted.json"
    compacted_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    # Remove old subdirectories.
    dirs_removed = 0
    for subdir in subdirs:
        if subdir.name == "compacted":
            continue
        shutil.rmtree(subdir)
        dirs_removed += 1

    return task_count, dirs_removed


# ── CLI ───────────────────────────────────────────────────────


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory created by create-run",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Preview what would be compacted without making changes",
    )


def run(args: argparse.Namespace) -> int:
    run_dir: Path = args.run_dir.resolve()

    manifest_path = run_dir / "eh_manifest.json"
    if not manifest_path.exists():
        logger.error("Manifest not found: {}", manifest_path)
        return 1

    manifest = json.loads(manifest_path.read_text())
    config = load_config(run_dir / "eh_config.yaml")
    output_path = config.output_path

    total_tasks = 0
    total_dirs_removed = 0
    entries_compacted = 0

    for mkey, entry in manifest.items():
        base = output_path / entry["model_key"] / entry["label"]
        tasks, dirs = compact_checkpoint(base, dry=args.dry)
        if dirs > 0:
            entries_compacted += 1
            total_tasks += tasks
            total_dirs_removed += dirs
            logger.info(
                "  {} — {} tasks, {} dirs {}",
                mkey, tasks, dirs,
                "would be removed" if args.dry else "removed",
            )

    action = "Would compact" if args.dry else "Compacted"
    logger.info(
        "{} {} entries: {} tasks merged, {} dirs removed",
        action, entries_compacted, total_tasks, total_dirs_removed,
    )
    return 0
