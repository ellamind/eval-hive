"""Parse a single lm-eval-harness result JSON into ScoreRow records."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .schemas import EvalConfig, ScoreRow, TaskConfig

logger = logging.getLogger(__name__)

# ── Language detection ────────────────────────────────────────────────────────
# Map task-name prefix → ISO 639-3 code.  Order matters: first match wins.
_LANG_PREFIXES: list[tuple[str, str]] = [
    ("deu_", "deu"),
    ("german_", "deu"),
    ("eng_", "eng"),
    ("fin_", "fin"),
]
_DEFAULT_LANG = "eng"  # unprefixed tasks are English


def _detect_language(task_name: str) -> str:
    for prefix, lang in _LANG_PREFIXES:
        if task_name.startswith(prefix):
            return lang
    return _DEFAULT_LANG


# ── Task formulation derivation ──────────────────────────────────────────────

_FORMULATION_MAP: dict[str, str] = {
    "multiple_choice": "rank_choice",
    "loglikelihood_rolling": "bits_per_byte",
    "generate_until": "generative",
}


def _derive_formulation(
    output_type: str | None,
    metric_names: set[str],
) -> str | None:
    """Map lm-eval output_type (+ metrics) → task_formulation."""
    if output_type is None:
        return None
    if output_type in _FORMULATION_MAP:
        return _FORMULATION_MAP[output_type]
    if output_type == "loglikelihood":
        # loglikelihood is ambiguous: bpb tasks compute bits_per_byte,
        # other tasks (rare) would be rank_choice
        if "bits_per_byte" in metric_names:
            return "bits_per_byte"
        return "rank_choice"
    return None


# ── Main parser ──────────────────────────────────────────────────────────────


def parse_result_file(
    result_path: Path,
    model_key: str,
    *,
    step: int | None = None,
    model_path: str | None = None,
    display_name: str | None = None,
    is_checkpoint: bool = False,
    train_batch_size: int | None = None,
    tokens_trained: int | None = None,
) -> list[ScoreRow]:
    """Parse one ``results_*.json`` and return a list of :class:`ScoreRow`."""
    with open(result_path) as f:
        data = json.load(f)

    results: dict[str, dict] = data.get("results", {})
    configs: dict[str, dict] = data.get("configs", {})
    group_subtasks: dict[str, list[str]] = data.get("group_subtasks", {})
    higher_is_better_map: dict[str, dict[str, bool]] = data.get(
        "higher_is_better", {}
    )
    n_shot: dict[str, int] = data.get("n-shot", {})
    n_samples_map: dict[str, dict] = data.get("n-samples", {})
    versions_map: dict[str, object] = data.get("versions", {})

    task_hashes: dict[str, str] = data.get("task_hashes", {})
    lm_eval_version: str | None = data.get("lm_eval_version")

    # ── Eval timestamp ────────────────────────────────────────────────────
    raw_date = data.get("date")
    eval_date: str | None = None
    if isinstance(raw_date, (int, float)):
        eval_date = datetime.fromtimestamp(raw_date, tz=timezone.utc).isoformat()

    # ── Per-file eval config struct ───────────────────────────────────────
    _tok_pad = data.get("tokenizer_pad_token")
    _tok_eos = data.get("tokenizer_eos_token")
    _tok_bos = data.get("tokenizer_bos_token")
    _total_time = data.get("total_evaluation_time_seconds")

    eval_config = EvalConfig(
        lm_eval_version=lm_eval_version,
        git_hash=data.get("git_hash"),
        transformers_version=data.get("transformers_version"),
        model_name=data.get("model_name"),
        model_source=data.get("model_source"),
        max_length=data.get("max_length"),
        total_eval_time_seconds=(
            float(_total_time) if _total_time is not None else None
        ),
        tokenizer_pad_token=_tok_pad[0] if isinstance(_tok_pad, list) and _tok_pad else None,
        tokenizer_eos_token=_tok_eos[0] if isinstance(_tok_eos, list) and _tok_eos else None,
        tokenizer_bos_token=_tok_bos[0] if isinstance(_tok_bos, list) and _tok_bos else None,
        eot_token_id=data.get("eot_token_id"),
        chat_template=data.get("chat_template"),
        chat_template_sha=data.get("chat_template_sha"),
        system_instruction=data.get("system_instruction"),
        system_instruction_sha=data.get("system_instruction_sha"),
    )

    # ── Build child → parent mapping ──────────────────────────────────────
    child_to_parent: dict[str, str] = {}
    for parent, children in group_subtasks.items():
        for child in children:
            child_to_parent[child] = parent

    # All tasks that appear as a child of some group
    all_children: set[str] = set(child_to_parent.keys())

    # ── Pre-compute which metrics each task reports ─────────────────────
    task_metrics: dict[str, set[str]] = {}
    for _tn, _md in results.items():
        _mnames: set[str] = set()
        for _k in _md:
            if _k != "alias" and "_stderr," not in _k:
                _mnames.add(_k.split(",")[0])
        task_metrics[_tn] = _mnames

    # ── Build metric-filtered subtask adjacency map (cached) ──────────
    _tree_cache: dict[tuple[str, str], dict[str, list[str]]] = {}

    def _build_subtree(task: str, metric: str) -> dict[str, list[str]]:
        """Build an adjacency map for *task*, keeping only children that
        report *metric*.

        Keys are group/suite names mapping to their immediate children.
        Leaf benchmarks appear only as list values, never as keys.
        """
        cache_key = (task, metric)
        if cache_key in _tree_cache:
            return _tree_cache[cache_key]
        children = group_subtasks.get(task)
        if children is None:
            _tree_cache[cache_key] = {}
            return _tree_cache[cache_key]
        # Keep only children that report this metric
        filtered = [c for c in children if metric in task_metrics.get(c, set())]
        if not filtered:
            _tree_cache[cache_key] = {}
            return _tree_cache[cache_key]
        adj: dict[str, list[str]] = {task: filtered}
        for child in filtered:
            if child in group_subtasks:
                adj.update(_build_subtree(child, metric))
        _tree_cache[cache_key] = adj
        return adj

    _display_name = display_name or model_key

    rows: list[ScoreRow] = []

    for task_name, metrics_dict in results.items():
        # ── task_type ─────────────────────────────────────────────────────
        if task_name in configs:
            task_type = "benchmark"
        elif task_name in group_subtasks and task_name not in all_children:
            # Top-level group that is not a subtask of anything
            task_type = "eval_suite"
        else:
            task_type = "task_group"

        # ── Task config (leaf benchmarks only) ────────────────────────────
        task_cfg = configs.get(task_name, {})
        output_type = task_cfg.get("output_type")

        # Collect non-stderr metric names for formulation detection
        metric_names: set[str] = set()
        for k in metrics_dict:
            if k != "alias" and "_stderr," not in k:
                metric_names.add(k.split(",")[0])

        formulation = _derive_formulation(output_type, metric_names)

        # ── Metadata lookups ──────────────────────────────────────────────
        hib = higher_is_better_map.get(task_name, {})
        ns = n_samples_map.get(task_name, {})
        effective_n = ns.get("effective") if isinstance(ns, dict) else None
        fewshot = n_shot.get(task_name)
        parent = child_to_parent.get(task_name)
        lang = _detect_language(task_name)
        version = versions_map.get(task_name)
        version_str = str(version) if version is not None else None
        task_hash = task_hashes.get(task_name)

        # Per-task config struct
        if task_cfg:
            _doc_to_choice = task_cfg.get("doc_to_choice")
            if _doc_to_choice is not None and not isinstance(_doc_to_choice, str):
                _doc_to_choice = json.dumps(_doc_to_choice)
            _gen_kwargs = task_cfg.get("generation_kwargs")
            if _gen_kwargs is not None and not isinstance(_gen_kwargs, str):
                _gen_kwargs = json.dumps(_gen_kwargs)
            task_config = TaskConfig(
                task_version=version_str,
                task_hash=task_hash,
                dataset_path=task_cfg.get("dataset_path"),
                output_type=output_type,
                dataset_name=task_cfg.get("dataset_name"),
                repeats=task_cfg.get("repeats"),
                doc_to_text=task_cfg.get("doc_to_text"),
                doc_to_target=task_cfg.get("doc_to_target"),
                doc_to_choice=_doc_to_choice,
                generation_kwargs=_gen_kwargs,
                should_decontaminate=task_cfg.get("should_decontaminate"),
            )
        else:
            task_config = None

        # Display name: use alias (stripped of leading " - ") or task name
        alias = metrics_dict.get("alias", task_name)
        display = alias.strip(" -") if alias else task_name

        # ── Emit one ScoreRow per (metric, filter) ───────────────────────
        for metric_key, value in metrics_dict.items():
            if metric_key == "alias":
                continue
            if "_stderr," in metric_key:
                continue  # stderr is paired with its primary metric below

            metric_name, metric_filter = metric_key.split(",", 1)

            stderr_key = f"{metric_name}_stderr,{metric_filter}"
            stderr_value = metrics_dict.get(stderr_key)

            # lm-eval sometimes writes "N/A" for stderr
            if isinstance(stderr_value, str):
                stderr_value = None
            if isinstance(value, str):
                value = None

            metric_hib = hib.get(metric_name)

            # Build metric-filtered subtask tree (only children that
            # report this metric contribute to the aggregated score)
            subtask_tree: dict[str, list[str]] | None = (
                _build_subtree(task_name, metric_name) or None
                if task_name in group_subtasks
                else None
            )

            rows.append(
                ScoreRow(
                    # identity
                    model=model_key,
                    step=step,
                    task=task_name,
                    metric=metric_name,
                    metric_filter=metric_filter,
                    # values
                    score=value,
                    score_stderr=stderr_value,
                    # task metadata
                    task_type=task_type,  # type: ignore[arg-type]
                    parent_task=parent,
                    task_display_name=display,
                    num_fewshot=fewshot,
                    task_formulation=formulation,  # type: ignore[arg-type]
                    higher_is_better=metric_hib,
                    language=lang,
                    n_samples=effective_n,
                    subtask_tree=subtask_tree,
                    # training metadata
                    train_batch_size=train_batch_size,
                    tokens_trained=tokens_trained,
                    # provenance
                    eval_date=eval_date,
                    result_source=str(result_path),
                    # nested metadata structs
                    eval_config=eval_config,
                    task_config=task_config,
                    # model metadata
                    model_path=model_path,
                    is_checkpoint=is_checkpoint,
                    model_display_name=_display_name,
                )
            )

    return rows
