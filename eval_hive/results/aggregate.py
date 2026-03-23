"""Compute aggregate scores for task groups and eval suites.

Reads the YAML task/suite hierarchy to determine group structure and
aggregation methods, then computes group scores bottom-up from leaf
benchmark scores.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import yaml

logger = logging.getLogger(__name__)


# ── YAML hierarchy parsing ───────────────────────────────────────────────────


@dataclass
class AggregateMetric:
    """One entry in a group's ``aggregate_metric_list``."""

    metric: str
    aggregation: str = "mean"
    weight_by_size: bool = False


@dataclass
class GroupInfo:
    """Parsed group definition from a YAML file."""

    name: str
    children: list[str]
    aggregate_metrics: list[AggregateMetric] = field(default_factory=list)


def build_task_hierarchy(task_dirs: list[Path]) -> dict[str, GroupInfo]:
    """Walk YAML files in *task_dirs* and build a ``{group_name: GroupInfo}`` index.

    Only entries with ``group:`` key and a ``task:`` list are included.
    """
    index: dict[str, GroupInfo] = {}

    for task_dir in task_dirs:
        if not task_dir.is_dir():
            continue
        for root, dirs, files in os.walk(task_dir):
            dirs.sort()
            for f in sorted(files):
                if not f.endswith(".yaml"):
                    continue
                path = os.path.join(root, f)
                try:
                    with open(path) as fh:
                        config = yaml.safe_load(fh)
                except Exception:
                    continue
                if not isinstance(config, dict):
                    continue
                if "group" not in config:
                    continue
                tasks = config.get("task")
                if not isinstance(tasks, list):
                    continue

                group_name = config["group"]

                # Resolve children: can be strings or dicts with "task" key
                children: list[str] = []
                for item in tasks:
                    if isinstance(item, str):
                        children.append(item)
                    elif isinstance(item, dict) and "task" in item:
                        children.append(item["task"])

                # Parse aggregate_metric_list
                agg_metrics: list[AggregateMetric] = []
                for entry in config.get("aggregate_metric_list", []):
                    if isinstance(entry, dict) and "metric" in entry:
                        agg_metrics.append(
                            AggregateMetric(
                                metric=entry["metric"],
                                aggregation=entry.get("aggregation", "mean"),
                                weight_by_size=entry.get("weight_by_size", False),
                            )
                        )

                index[group_name] = GroupInfo(
                    name=group_name,
                    children=children,
                    aggregate_metrics=agg_metrics,
                )

    return index


# ── Topological ordering ─────────────────────────────────────────────────────


def _topological_order(
    hierarchy: dict[str, GroupInfo],
    roots: list[str],
) -> list[str]:
    """Return group names in bottom-up order (leaves first, roots last).

    Only groups reachable from *roots* are included.
    """
    # Collect all reachable groups via BFS
    reachable: set[str] = set()
    queue = list(roots)
    while queue:
        name = queue.pop(0)
        if name in reachable:
            continue
        if name not in hierarchy:
            continue
        reachable.add(name)
        for child in hierarchy[name].children:
            if child in hierarchy:
                queue.append(child)

    # Topological sort via DFS post-order
    visited: set[str] = set()
    order: list[str] = []

    def _dfs(name: str) -> None:
        if name in visited or name not in reachable:
            return
        visited.add(name)
        if name in hierarchy:
            for child in hierarchy[name].children:
                _dfs(child)
        order.append(name)

    for root in roots:
        _dfs(root)

    return order


# ── Score aggregation ────────────────────────────────────────────────────────


def aggregate_scores(
    leaf_df: pl.DataFrame,
    task_dirs: list[Path],
    suites: list[str],
) -> pl.DataFrame:
    """Compute aggregated scores for groups and suites.

    Parameters
    ----------
    leaf_df:
        DataFrame of leaf benchmark ScoreRows (task_type == "benchmark").
    task_dirs:
        Directories containing task/suite YAML files.
    suites:
        Top-level suite names from the eval config's suites_and_tasks.

    Returns
    -------
    DataFrame of aggregate ScoreRows (task_type in {"task_group", "eval_suite"}).
    """
    hierarchy = build_task_hierarchy(task_dirs)

    if not hierarchy:
        logger.warning("No task groups found in YAML files")
        return pl.DataFrame()

    # Determine which groups are reachable from the configured suites
    order = _topological_order(hierarchy, suites)

    if not order:
        logger.info("No groups to aggregate")
        return pl.DataFrame()

    logger.info(
        "Aggregating %d groups (bottom-up): %s",
        len(order),
        ", ".join(order[:5]) + ("..." if len(order) > 5 else ""),
    )

    # Build a scores lookup: (model, step, task, metric, metric_filter) → row dict
    # Start with leaf scores, then add computed group scores as we go
    scores: dict[tuple, dict] = {}
    for row in leaf_df.iter_rows(named=True):
        key = (row["model"], row["step"], row["task"], row["metric"], row["metric_filter"])
        scores[key] = row

    # Unique (model, step) combos
    model_steps = leaf_df.select("model", "step").unique().rows()

    # Pre-compute which metrics each task can report, and the metric_filter
    # it uses for each metric.  Most tasks use "none" but code-execution
    # benchmarks use filters like "create_test" or "build_test".
    task_reported_metrics: dict[str, set[str]] = {}
    # (task, metric) → preferred metric_filter (first seen; "none" wins)
    task_metric_filter: dict[tuple[str, str], str] = {}
    for gname, ginfo in hierarchy.items():
        task_reported_metrics[gname] = {am.metric for am in ginfo.aggregate_metrics}
    for _, _, tname, mname, mfilter in scores:
        if tname not in task_reported_metrics:
            task_reported_metrics[tname] = set()
        task_reported_metrics[tname].add(mname)
        key = (tname, mname)
        if key not in task_metric_filter or mfilter == "none":
            task_metric_filter[key] = mfilter

    suite_set = set(suites)
    aggregate_rows: list[dict] = []

    for group_name in order:
        group = hierarchy[group_name]

        if not group.aggregate_metrics:
            logger.debug("Skipping %s: no aggregate_metric_list", group_name)
            continue

        task_type = "eval_suite" if group_name in suite_set else "task_group"

        # Determine parent (is this group a child of another group?)
        parent_task = None
        for other_name, other_group in hierarchy.items():
            if group_name in other_group.children:
                parent_task = other_name
                break

        for model, step_val in model_steps:
            for agg_metric in group.aggregate_metrics:
                metric_name = agg_metric.metric

                # Only children that can report this metric are applicable.
                applicable_children = [
                    c for c in group.children
                    if metric_name in task_reported_metrics.get(c, set())
                ]
                if not applicable_children:
                    continue

                # Collect children's scores for this metric
                child_scores: list[float] = []
                child_weights: list[int] = []
                subtask_children: list[str] = []

                for child_name in applicable_children:
                    # Use the filter the child actually reports for this
                    # metric (e.g. "create_test" for humaneval pass@1)
                    # rather than assuming "none".
                    child_filter = task_metric_filter.get(
                        (child_name, metric_name), "none"
                    )
                    child_key = (model, step_val, child_name, metric_name, child_filter)
                    child_row = scores.get(child_key)
                    if child_row is None or child_row.get("score") is None:
                        continue
                    child_scores.append(child_row["score"])
                    child_weights.append(child_row.get("n_samples") or 1)
                    subtask_children.append(child_name)

                if len(child_scores) < len(applicable_children):
                    if child_scores:
                        logger.warning(
                            "Skipping aggregate for %s (%s, step=%s, %s): "
                            "only %d/%d applicable children have scores",
                            group_name, model, step_val, metric_name,
                            len(child_scores), len(applicable_children),
                        )
                    continue

                # Compute aggregate
                if agg_metric.weight_by_size and any(w > 1 for w in child_weights):
                    total_w = sum(child_weights)
                    score = sum(s * w for s, w in zip(child_scores, child_weights)) / total_w
                else:
                    score = sum(child_scores) / len(child_scores)

                # Build subtask tree and collect propagatable fields from children
                subtask_tree: dict[str, list[str]] = {group_name: subtask_children}
                child_languages: set[str | None] = set()
                child_hib: set[bool | None] = set()
                child_eval_dates: list[str] = []
                child_model_path: str | None = None
                child_display_name: str | None = None
                child_train_batch_size: int | None = None
                child_tokens_trained: int | None = None

                for child_name in subtask_children:
                    child_filter = task_metric_filter.get(
                        (child_name, metric_name), "none"
                    )
                    child_key = (model, step_val, child_name, metric_name, child_filter)
                    child_row = scores.get(child_key)
                    if child_row is None:
                        continue
                    # Subtask tree propagation
                    if child_row.get("subtask_tree"):
                        tree = child_row["subtask_tree"]
                        if isinstance(tree, dict):
                            subtask_tree.update(tree)
                    # Collect fields to propagate
                    child_languages.add(child_row.get("language"))
                    child_hib.add(child_row.get("higher_is_better"))
                    if child_row.get("eval_date"):
                        child_eval_dates.append(child_row["eval_date"])
                    if child_row.get("model_path") and child_model_path is None:
                        child_model_path = child_row["model_path"]
                    if child_row.get("model_display_name") and child_display_name is None:
                        child_display_name = child_row["model_display_name"]
                    if child_row.get("train_batch_size") and child_train_batch_size is None:
                        child_train_batch_size = child_row["train_batch_size"]
                    if child_row.get("tokens_trained") and child_tokens_trained is None:
                        child_tokens_trained = child_row["tokens_trained"]

                # Propagate language if all children agree
                lang = None
                child_languages.discard(None)
                if len(child_languages) == 1:
                    lang = child_languages.pop()

                # Propagate higher_is_better if all children agree
                hib = None
                child_hib.discard(None)
                if len(child_hib) == 1:
                    hib = child_hib.pop()

                # Use latest eval_date from children
                eval_date = max(child_eval_dates) if child_eval_dates else None

                # Derive tokens_trained: prefer batch_size * step, fall back to child propagation
                tokens_trained = None
                if child_train_batch_size is not None and step_val is not None:
                    tokens_trained = child_train_batch_size * step_val
                elif child_tokens_trained is not None:
                    tokens_trained = child_tokens_trained

                row_dict = {
                    "model": model,
                    "step": step_val,
                    "task": group_name,
                    "metric": metric_name,
                    "metric_filter": "none",
                    "score": score,
                    "score_stderr": None,
                    "task_type": task_type,
                    "parent_task": parent_task,
                    "task_display_name": group_name,
                    "num_fewshot": None,
                    "task_formulation": None,
                    "higher_is_better": hib,
                    "language": lang,
                    "n_samples": sum(child_weights),
                    "subtask_tree": subtask_tree,
                    "train_batch_size": child_train_batch_size,
                    "tokens_trained": tokens_trained,
                    "eval_date": eval_date,
                    "result_source": None,
                    "eval_config": None,
                    "task_config": None,
                    "model_path": child_model_path,
                    "is_checkpoint": step_val is not None,
                    "model_display_name": child_display_name or model,
                }

                # Store so parent groups can use this score
                result_key = (model, step_val, group_name, metric_name, "none")
                scores[result_key] = row_dict

                aggregate_rows.append(row_dict)

    if not aggregate_rows:
        logger.info("No aggregate scores computed (no matching leaf data)")
        return pl.DataFrame()

    # Pre-serialize subtask_tree dicts to JSON strings before creating the
    # DataFrame.  Polars infers a Struct schema from the first non-null dict
    # it sees, silently dropping keys that don't match — corrupting every
    # other group's tree.  Storing as plain strings avoids this.
    for row in aggregate_rows:
        if row.get("subtask_tree") is not None:
            row["subtask_tree"] = json.dumps(row["subtask_tree"])

    logger.info("Computed %d aggregate score rows", len(aggregate_rows))
    return pl.DataFrame(aggregate_rows)
