"""Tests for group/suite aggregation completeness."""

from pathlib import Path

import polars as pl
import yaml

from eval_hive.results.aggregate import aggregate_scores


def _write_hierarchy(tmp_path: Path) -> list[Path]:
    groups = {
        "suite": {"group": "suite", "task": ["mc_group", "gen_task"],
                  "aggregate_metric_list": [{"metric": "acc"}]},
        "mc_group": {"group": "mc_group", "task": ["mc_a", "mc_b"],
                     "aggregate_metric_list": [{"metric": "acc"}]},
    }
    for name, cfg in groups.items():
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    return [tmp_path]


def _rows(tasks: dict[str, float], step: int = 100) -> pl.DataFrame:
    return pl.DataFrame([
        {"model": "m", "step": float(step), "task": t, "metric": "acc",
         "metric_filter": "none", "score": s, "task_type": "benchmark",
         "n_samples": 10, "language": "eng"}
        for t, s in tasks.items()
    ])


def test_complete_children_aggregate(tmp_path):
    dirs = _write_hierarchy(tmp_path)
    out = aggregate_scores(_rows({"mc_a": 0.4, "mc_b": 0.6, "gen_task": 0.2}), dirs, ["suite"])
    scores = dict(zip(out["task"], out["score"]))
    assert scores["mc_group"] == 0.5
    assert scores["suite"] == (0.5 + 0.2) / 2


def test_child_without_any_scores_blocks_aggregate(tmp_path):
    # gen_task has not been evaluated for any checkpoint yet: the suite must
    # not be averaged over mc_group alone.
    dirs = _write_hierarchy(tmp_path)
    out = aggregate_scores(_rows({"mc_a": 0.4, "mc_b": 0.6}), dirs, ["suite"])
    assert set(out["task"]) == {"mc_group"}


def test_missing_child_blocks_parents(tmp_path):
    dirs = _write_hierarchy(tmp_path)
    out = aggregate_scores(_rows({"mc_a": 0.4, "gen_task": 0.2}), dirs, ["suite"])
    assert len(out) == 0
