"""Collect lm-eval results from a run directory into scores.parquet."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from eval_hive.config import load_config


# ── Step extraction ───────────────────────────────────────────────────────────


def parse_step_from_label(label: str) -> int | None:
    """Extract training step from an eval-hive label.

    Returns the integer value of the last digit sequence in the label,
    or None if the label contains no digits (e.g., ``"main"``).

    Examples::

        >>> parse_step_from_label("checkpoint_0005000")
        5000
        >>> parse_step_from_label("main")
        >>> parse_step_from_label("v2_checkpoint_0005000")
        5000
    """
    matches = re.findall(r"\d+", label)
    if not matches:
        return None
    return int(matches[-1])


# ── Result discovery ──────────────────────────────────────────────────────────


@dataclass
class DiscoveredResult:
    """A result file discovered in the output directory."""

    manifest_key: str
    model_key: str
    label: str
    step: int | None
    model_path: str | None
    display_name: str | None
    train_batch_size: int | None
    tokens_trained: int | None
    result_path: Path


def discover_results(
    output_path: Path,
    manifest: dict[str, dict],
) -> list[DiscoveredResult]:
    """Walk the output directory to find result files for all manifest entries.

    Expects the directory structure produced by eval-hive::

        {output_path}/{model_key}/{label}/{task_or_suite}/results_*.json
    """
    results: list[DiscoveredResult] = []

    for mkey, entry in manifest.items():
        model_key = entry["model_key"]
        label = entry["label"]
        model_path = entry.get("model_path")
        display_name = entry.get("display_name")
        train_batch_size = entry.get("train_batch_size")
        tokens_trained = entry.get("tokens_trained")
        step = parse_step_from_label(label)

        base = output_path / model_key / label
        if not base.is_dir():
            logger.debug("No output directory for %s: %s", mkey, base)
            continue

        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue

            result_files = list(task_dir.glob("results_*.json"))
            if not result_files:
                continue

            latest = max(result_files, key=lambda f: f.stat().st_mtime)
            results.append(
                DiscoveredResult(
                    manifest_key=mkey,
                    model_key=model_key,
                    label=label,
                    step=step,
                    model_path=model_path,
                    display_name=display_name,
                    train_batch_size=train_batch_size,
                    tokens_trained=tokens_trained,
                    result_path=latest,
                )
            )

    return results


# ── Core collect logic ────────────────────────────────────────────────────────


def collect_from_run(
    run_dir: Path,
    output_parquet: Path,
) -> pl.DataFrame:
    """Collect all results from a run directory into a parquet file.

    1. Read ``eh_manifest.json`` and ``eh_config.yaml`` from *run_dir*.
    2. Discover result files under the configured ``output_path``.
    3. Parse each result file into ScoreRow records.
    4. Aggregate group/suite scores from the YAML hierarchy.
    5. Write combined leaf + aggregate rows to *output_parquet*.
    """
    from eval_hive.results.aggregate import aggregate_scores
    from eval_hive.results.parse import parse_result_file

    # Load manifest and config
    manifest = json.loads((run_dir / "eh_manifest.json").read_text())
    config = load_config(run_dir / "eh_config.yaml")
    output_path = config.output_path

    logger.info("Run directory: %s", run_dir)
    logger.info("Output path:   %s", output_path)
    logger.info("Manifest:      %d entries", len(manifest))

    # Discover results
    discovered = discover_results(output_path, manifest)
    logger.info("Discovered %d result files", len(discovered))

    # Parse results
    all_rows: list[dict] = []
    n_errors = 0

    for dr in discovered:
        is_checkpoint = dr.step is not None
        try:
            rows = parse_result_file(
                result_path=dr.result_path,
                model_key=dr.model_key,
                step=dr.step,
                model_path=dr.model_path,
                display_name=dr.display_name or dr.model_key,
                is_checkpoint=is_checkpoint,
                train_batch_size=dr.train_batch_size,
                tokens_trained=dr.tokens_trained,
            )
            all_rows.extend(r.model_dump() for r in rows)
        except Exception:
            n_errors += 1
            logger.warning("Failed to parse %s", dr.result_path, exc_info=True)

    if n_errors:
        logger.warning("%d files failed to parse", n_errors)

    if not all_rows:
        logger.warning("No results found to collect")
        df = pl.DataFrame()
        df.write_parquet(output_parquet)
        return df

    leaf_df = pl.DataFrame(all_rows)

    # Serialize subtask_tree (nested dicts from model_dump) to JSON strings
    if "subtask_tree" in leaf_df.columns:
        leaf_df = leaf_df.with_columns(
            pl.col("subtask_tree").map_elements(
                lambda v: json.dumps(v) if v is not None else None,
                return_dtype=pl.String,
            )
        )

    # Aggregate group/suite scores from YAML hierarchy
    task_dirs = _resolve_task_dirs(config)
    if task_dirs:
        # Filter to benchmark rows for aggregation input
        benchmarks = leaf_df.filter(pl.col("task_type") == "benchmark")
        # Deserialize subtask_tree back for aggregation (it needs dicts)
        if "subtask_tree" in benchmarks.columns:
            benchmarks = benchmarks.with_columns(
                pl.col("subtask_tree").map_elements(
                    lambda v: json.loads(v) if isinstance(v, str) else v,
                    return_dtype=pl.Object,
                )
            )
        agg_df = aggregate_scores(
            benchmarks,
            task_dirs,
            config.eval.suites_and_tasks,
        )
        if len(agg_df) > 0:
            # Serialize subtask_tree in aggregate rows
            if "subtask_tree" in agg_df.columns:
                agg_df = agg_df.with_columns(
                    pl.col("subtask_tree").map_elements(
                        lambda v: json.dumps(v) if v is not None else None,
                        return_dtype=pl.String,
                    )
                )
            leaf_df = pl.concat([leaf_df, agg_df], how="diagonal_relaxed")

    # Deterministic sort
    sort_cols = [c for c in ["model", "step", "task", "metric", "metric_filter"] if c in leaf_df.columns]
    if sort_cols:
        leaf_df = leaf_df.sort(sort_cols, nulls_last=False)

    leaf_df.write_parquet(output_parquet)
    logger.info("Wrote %d rows to %s", len(leaf_df), output_parquet)

    return leaf_df


def _resolve_task_dirs(config) -> list[Path]:
    """Resolve task/suite YAML directories from the eval config."""
    dirs: list[Path] = []
    if config.eval.eval_suite_path:
        suite_path = Path(config.eval.eval_suite_path)
        if suite_path.is_dir():
            dirs.append(suite_path)
            # Also check for tasks/ and suites/ subdirectories
            for subdir in ("tasks", "suites"):
                sub = suite_path / subdir
                if sub.is_dir():
                    dirs.append(sub)
    return dirs


# ── CLI ───────────────────────────────────────────────────────────────────────


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the collect command."""
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to the run directory created by create-run",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output parquet file (default: <run_dir>/scores.parquet)",
    )
    parser.add_argument(
        "--push-to",
        type=str,
        default=None,
        metavar="HF_REPO",
        help="HuggingFace dataset repo to push results to (e.g. 'ellamind/eval-scores')",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the collect command."""
    run_dir = args.run_dir.resolve()
    output = args.output if args.output else run_dir / "scores.parquet"

    for required in ("eh_manifest.json", "eh_config.yaml"):
        if not (run_dir / required).exists():
            logger.error("Required file not found: %s", run_dir / required)
            return 1

    df = collect_from_run(run_dir, output)

    if len(df) == 0:
        return 0

    if args.push_to:
        from eval_hive.results.hf import push_to_hf

        push_to_hf(output, args.push_to)

    return 0
