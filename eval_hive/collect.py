"""Collect lm-eval results from a run directory into scores.parquet."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger

from eval_hive.config import load_config


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
    hf_covered_keys: set[str] | None = None,
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
        step = entry.get("step")

        base = output_path / model_key / label
        if not base.is_dir():
            if not (hf_covered_keys and mkey in hf_covered_keys):
                logger.debug("No output directory for {}: {}", mkey, base)
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

    logger.info("Run directory: {}", run_dir)
    logger.info("Output path:   {}", output_path)
    logger.info("Manifest:      {} entries", len(manifest))

    # Load HF coverage to suppress warnings for entries fully covered by HF
    hf_covered_path = run_dir / "eh_hf_covered.json"
    hf_covered_keys = set(json.loads(hf_covered_path.read_text()).keys()) if hf_covered_path.exists() else None

    # Discover results
    discovered = discover_results(output_path, manifest, hf_covered_keys)
    logger.info("Discovered {} result files", len(discovered))

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
            logger.opt(exception=True).warning("Failed to parse {}", dr.result_path)

    if n_errors:
        logger.warning("{} files failed to parse", n_errors)

    if not all_rows:
        logger.warning("No results found to collect")
        df = pl.DataFrame()
        df.write_parquet(output_parquet)
        return df

    # Pre-serialize subtask_tree dicts to JSON strings BEFORE creating the
    # DataFrame.  Polars infers a Struct schema from the first non-null dict
    # it encounters, silently dropping keys that don't match that schema.
    # Storing as plain strings avoids this corruption.
    for row in all_rows:
        if row.get("subtask_tree") is not None:
            row["subtask_tree"] = json.dumps(row["subtask_tree"])

    leaf_df = pl.DataFrame(all_rows, infer_schema_length=None)

    # Aggregate group/suite scores from YAML hierarchy
    task_dirs = _resolve_task_dirs(config)
    if task_dirs:
        # Filter to benchmark rows for aggregation input
        benchmarks = leaf_df.filter(pl.col("task_type") == "benchmark")
        agg_df = aggregate_scores(
            benchmarks,
            task_dirs,
            config.eval.suites_and_tasks,
        )
        if len(agg_df) > 0:
            leaf_df = pl.concat([leaf_df, agg_df], how="diagonal_relaxed")

    # Reclassify: configured suites that were parsed as task_group → eval_suite.
    # This fixes misclassification when lm-eval wraps suites in an implicit
    # root group (making them appear as children in group_subtasks).
    suite_names = set(config.eval.suites_and_tasks)
    if suite_names and "task" in leaf_df.columns and "task_type" in leaf_df.columns:
        leaf_df = leaf_df.with_columns(
            pl.when(
                pl.col("task").is_in(suite_names) & (pl.col("task_type") == "task_group")
            )
            .then(pl.lit("eval_suite"))
            .otherwise(pl.col("task_type"))
            .alias("task_type")
        )

    # Deterministic sort
    sort_cols = [c for c in ["model", "step", "task", "metric", "metric_filter"] if c in leaf_df.columns]
    if sort_cols:
        leaf_df = leaf_df.sort(sort_cols, nulls_last=False)

    leaf_df.write_parquet(output_parquet)
    logger.info("Wrote {} rows to {}", len(leaf_df), output_parquet)

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
        "--upload",
        nargs="?",
        const=True,
        default=None,
        metavar="HF_REPO",
        help="Push results to HuggingFace. Uses hf_result_repo from config, "
             "or specify a repo to override (e.g. --upload org/other-scores)",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the collect command."""
    run_dir = args.run_dir.resolve()
    output = args.output if args.output else run_dir / "scores.parquet"

    for required in ("eh_manifest.json", "eh_config.yaml"):
        if not (run_dir / required).exists():
            logger.error("Required file not found: {}", run_dir / required)
            return 1

    df = collect_from_run(run_dir, output)

    if len(df) == 0:
        return 0

    if args.upload is not None:
        if isinstance(args.upload, str):
            hf_repo = args.upload
        else:
            config = load_config(run_dir / "eh_config.yaml")
            if not config.hf_result_repo:
                logger.error("--upload requires 'hf_result_repo' to be set in the config")
                return 1
            hf_repo = config.hf_result_repo
        from eval_hive.results.hf import push_to_hf

        push_to_hf(output, hf_repo)

    return 0
