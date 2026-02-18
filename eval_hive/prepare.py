import argparse
import json
import multiprocessing
import os
import queue
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from eval_hive.config import load_config


def _load_yaml(path):
    """Load a YAML file and return its contents as a dict."""
    import yaml

    return yaml.safe_load(Path(path).read_text())


def resolve_task_names(tm, names):
    """Resolve suite/group names to individual task names, recursively.

    Uses the Entry-based task_index API (lm-eval >= 0.4.8).
    Entry fields: name, kind (Kind enum), yaml_path, cfg (parsed YAML dict).
    """
    from lm_eval.tasks._index import Kind

    individual = []
    for name in names:
        entry = tm.task_index.get(name)
        if entry is None:
            individual.append(name)
            continue

        if entry.kind == Kind.GROUP:
            # Group — cfg["task"] contains the list of child names/dicts
            subtask_list = (entry.cfg or {}).get("task", [])
            if isinstance(subtask_list, list) and subtask_list:
                child_names = [
                    item if isinstance(item, str) else item["task"]
                    for item in subtask_list
                    if isinstance(item, (str, dict))
                ]
                individual.extend(resolve_task_names(tm, child_names))
            else:
                individual.append(name)
        elif entry.kind in (Kind.TASK, Kind.PY_TASK):
            individual.append(name)
        elif entry.kind == Kind.TAG:
            # Tags expand to their member tasks
            individual.extend(resolve_task_names(tm, sorted(entry.tags)))
        else:
            individual.append(name)
    return individual


def get_task_dataset(tm, task_name):
    """Read dataset_path from a task's YAML config without instantiating the task."""
    entry = tm.task_index.get(task_name)
    if entry is None:
        return "__unknown__"
    # Try cfg first (already parsed), fall back to loading YAML
    if entry.cfg:
        return entry.cfg.get("dataset_path", "__unknown__")
    if entry.yaml_path:
        config = _load_yaml(entry.yaml_path)
        return config.get("dataset_path", "__unknown__")
    return "__unknown__"


def shard_tasks(tm, task_names, num_workers):
    """Group tasks by dataset_path, then bin-pack groups into num_workers shards."""
    # Group by dataset
    dataset_groups = defaultdict(list)
    for name in task_names:
        ds = get_task_dataset(tm, name)
        dataset_groups[ds].append(name)

    # Greedy bin-packing: largest groups first, assign to lightest worker
    groups = sorted(dataset_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_shards = min(num_workers, len(groups))
    shards = [[] for _ in range(n_shards)]
    shard_datasets = [[] for _ in range(n_shards)]
    shard_sizes = [0] * n_shards

    for ds_name, task_list in groups:
        lightest = shard_sizes.index(min(shard_sizes))
        shards[lightest].extend(task_list)
        shard_datasets[lightest].append(f"{ds_name} ({len(task_list)})")
        shard_sizes[lightest] += len(task_list)

    # Log shard assignments
    for i, (shard, datasets) in enumerate(zip(shards, shard_datasets)):
        logger.info(f"  Worker {i}: {len(shard)} tasks — {', '.join(datasets)}")

    return shards


_progress_queue = None


def _init_worker(q):
    """Pool initializer — store the shared queue as a module global and suppress progress bars."""
    global _progress_queue
    _progress_queue = q

    # Monkeypatch tqdm to always be disabled in worker processes.
    # Env vars don't work because tqdm is already imported via fork.
    import tqdm as _tqdm_mod
    _orig_init = _tqdm_mod.tqdm.__init__

    def _disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _orig_init(self, *args, **kwargs)

    _tqdm_mod.tqdm.__init__ = _disabled_init


def worker_fn(args):
    """Load a shard of tasks and build request caches. Runs in a subprocess."""
    worker_id, shard, eval_suite_path, cache_dir, refresh, chat_args = args
    progress_queue = _progress_queue

    if cache_dir:
        os.environ["LM_HARNESS_CACHE_PATH"] = cache_dir

    from lm_eval.tasks import TaskManager
    import datasets
    datasets.logging.set_verbosity_warning()
    datasets.disable_progress_bar()

    tm = TaskManager(include_path=eval_suite_path)
    results = {}

    for task_name in shard:
        try:
            # Load task (downloads dataset if not cached)
            for attempt in range(3):
                try:
                    loaded = tm.load([task_name])
                    break
                except Exception as e:
                    if attempt < 2:
                        progress_queue.put((worker_id, task_name, "retry", str(e)))
                    else:
                        raise

            # Build request cache
            tasks = list(loaded["tasks"].items())
            if not chat_args.get("tokenizer_paths"):
                # Base model — model-independent cache key
                for _name, task in tasks:
                    task.build_all_requests(
                        cache_requests=True,
                        rewrite_requests_cache=refresh,
                    )
            else:
                # Chat template — build per tokenizer
                from transformers import AutoTokenizer

                for tok_path in chat_args["tokenizer_paths"]:
                    tokenizer = AutoTokenizer.from_pretrained(tok_path)
                    for _name, task in tasks:
                        task.build_all_requests(
                            cache_requests=True,
                            rewrite_requests_cache=refresh,
                            apply_chat_template=True,
                            chat_template=tokenizer.apply_chat_template,
                            tokenizer_name=tok_path,
                            system_instruction=chat_args.get("system_instruction"),
                            fewshot_as_multiturn=chat_args.get("fewshot_as_multiturn", False),
                        )

            results[task_name] = "ok"
            progress_queue.put((worker_id, task_name, "ok", None))
        except Exception as e:
            results[task_name] = f"error: {e}"
            progress_queue.put((worker_id, task_name, "error", str(e)))

    return results


def run_sequential(config, all_task_names, refresh, chat_args):
    """Load tasks and build caches sequentially (--workers=1 path)."""
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=config.eval.eval_suite_path)
    tasks = []

    t0 = time.monotonic()
    with tqdm(total=len(all_task_names), desc="  Loading tasks", unit="task", leave=True) as pbar:
        for task_name in all_task_names:
            for attempt in range(3):
                try:
                    loaded = tm.load([task_name])
                    break
                except Exception as e:
                    if attempt < 2:
                        tqdm.write(f"    {task_name}: attempt {attempt + 1} failed ({e}), retrying...")
                    else:
                        raise
            tasks.extend(loaded["tasks"].items())
            tqdm.write(f"    {task_name}")
            pbar.update(1)

    load_time = time.monotonic() - t0
    logger.info(f"Loaded {len(tasks)} task(s) in {load_time:.1f}s")

    t1 = time.monotonic()
    logger.info("Building request cache...")

    if not chat_args.get("tokenizer_paths"):
        for _name, task in tqdm(tasks, desc="  Caching requests", unit="task"):
            task.build_all_requests(
                cache_requests=True,
                rewrite_requests_cache=refresh,
            )
    else:
        from transformers import AutoTokenizer

        for tok_path in chat_args["tokenizer_paths"]:
            logger.info(f"  Tokenizer: {tok_path}")
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            for _name, task in tqdm(tasks, desc=f"  {Path(tok_path).name}", unit="task"):
                task.build_all_requests(
                    cache_requests=True,
                    rewrite_requests_cache=refresh,
                    apply_chat_template=True,
                    chat_template=tokenizer.apply_chat_template,
                    tokenizer_name=tok_path,
                    system_instruction=chat_args.get("system_instruction"),
                    fewshot_as_multiturn=chat_args.get("fewshot_as_multiturn", False),
                )

    cache_time = time.monotonic() - t1
    logger.info(f"Request caching took {cache_time:.1f}s")
    logger.info(f"Total: {load_time + cache_time:.1f}s (loading {load_time:.1f}s + caching {cache_time:.1f}s)")


def run_parallel(config, all_task_names, refresh, chat_args, num_workers):
    """Load tasks and build caches in parallel worker processes."""
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=config.eval.eval_suite_path)
    shards = shard_tasks(tm, all_task_names, num_workers)

    cache_dir = str(config.request_cache_dir) if config.request_cache_dir else None
    eval_suite_path = config.eval.eval_suite_path
    progress_queue = multiprocessing.Queue()

    worker_args = [
        (i, shard, eval_suite_path, cache_dir, refresh, chat_args)
        for i, shard in enumerate(shards)
    ]

    t0 = time.monotonic()
    logger.info(f"Launching {len(shards)} worker(s)...")
    total = len(all_task_names)

    with multiprocessing.Pool(len(shards), initializer=_init_worker, initargs=(progress_queue,)) as pool:
        async_result = pool.map_async(worker_fn, worker_args)

        # Per-worker bars + total bar at the bottom
        n = len(shards)
        ncols = 80
        # "worker 0:" and "Total:   " both 10 chars for alignment
        bar_fmt = "{desc} {bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        worker_bars = [
            tqdm(total=len(shards[i]), desc=f"Worker {i}:", unit="task",
                 position=i, leave=True, ncols=ncols, bar_format=bar_fmt)
            for i in range(n)
        ]
        total_bar = tqdm(total=total, desc="Total:   ", unit="task",
                         position=n, leave=True, ncols=ncols, bar_format=bar_fmt)

        completed = 0
        try:
            while completed < total:
                try:
                    worker_id, task_name, status, detail = progress_queue.get(timeout=2)
                except queue.Empty:
                    if async_result.ready():
                        break
                    continue

                if status == "ok":
                    total_bar.write(f"  [worker {worker_id}] {task_name}")
                    worker_bars[worker_id].update(1)
                    total_bar.update(1)
                    completed += 1
                elif status == "error":
                    total_bar.write(f"  [worker {worker_id}] {task_name}: FAILED — {detail}")
                    worker_bars[worker_id].update(1)
                    total_bar.update(1)
                    completed += 1
                elif status == "retry":
                    total_bar.write(f"  [worker {worker_id}] {task_name}: retrying ({detail})")
        finally:
            for bar in worker_bars:
                bar.close()
            total_bar.close()

        # Wait for workers to finish and propagate any exceptions
        all_results = async_result.get()

    elapsed = time.monotonic() - t0

    # Count from actual results (in case queue missed some)
    total_ok = sum(1 for r in all_results for s in r.values() if s == "ok")
    failures = [(n, s) for r in all_results for n, s in r.items() if s != "ok"]

    logger.info(f"Completed: {total_ok}/{total} tasks in {elapsed:.1f}s ({len(shards)} workers)")
    if failures:
        logger.error(f"{len(failures)} task(s) failed:")
        for name, status in failures:
            logger.error(f"  {name}: {status}")
        sys.exit(1)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the prepare command."""
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--refresh", action="store_true", help="Rebuild request cache even if it exists")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")


def run(args: argparse.Namespace) -> int:
    """Execute the prepare command with parsed arguments."""
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file '{config_path}' does not exist.")
        return 1

    config = load_config(config_path)

    # Set cache dir before importing lm_eval (reads env at module load)
    if config.request_cache_dir:
        config.request_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["LM_HARNESS_CACHE_PATH"] = str(config.request_cache_dir)

    from lm_eval.tasks import TaskManager
    from huggingface_hub import constants as hf_constants
    import datasets
    datasets.logging.set_verbosity_warning()

    logger.info(f"HuggingFace cache dir: {hf_constants.HF_HUB_CACHE}")
    if config.request_cache_dir:
        logger.info(f"Request cache dir:     {config.request_cache_dir}")
    else:
        from lm_eval.caching.cache import PATH as lm_cache_path
        logger.info(f"Request cache dir:     {lm_cache_path} (lm-eval default)")

    # 1. Resolve all task names (fast — YAML index only, no dataset loading)
    logger.info("Resolving task names...")
    tm = TaskManager(include_path=config.eval.eval_suite_path)

    all_task_names = []
    seen = set()
    for suite_or_task in config.eval.suites_and_tasks:
        task_names = resolve_task_names(tm, [suite_or_task])
        new_tasks = [t for t in task_names if t not in seen]
        skipped = len(task_names) - len(new_tasks)
        suffix = f" ({skipped} duplicates)" if skipped else ""
        logger.info(f"  {suite_or_task}: {len(new_tasks)} task(s){suffix}")
        seen.update(new_tasks)
        all_task_names.extend(new_tasks)

    logger.info(f"Total: {len(all_task_names)} task(s)")

    # 2. Build chat template args (shared across workers)
    apply_chat_template = bool(config.eval.lm_eval_args.get("apply_chat_template", False))
    chat_args = {}
    if apply_chat_template:
        # Collect unique tokenizer paths
        tokenizer_paths = []
        seen_tok = set()
        for model_key, entry in config.models.items():
            for label, _step, model_path in entry.resolve_model_paths():
                tok_path = str(model_path)
                if tok_path not in seen_tok:
                    seen_tok.add(tok_path)
                    tokenizer_paths.append(tok_path)
        chat_args = {
            "tokenizer_paths": tokenizer_paths,
            "system_instruction": config.eval.lm_eval_args.get("system_instruction"),
            "fewshot_as_multiturn": bool(config.eval.lm_eval_args.get("fewshot_as_multiturn", False)),
        }

    # 3. Resolve effective cache dir and count existing files
    if config.request_cache_dir:
        cache_dir = config.request_cache_dir
    else:
        from lm_eval.caching.cache import PATH as lm_cache_path
        cache_dir = Path(lm_cache_path)

    cache_dir = Path(cache_dir)
    caches_before = set(cache_dir.glob("*.pickle")) if cache_dir.is_dir() else set()

    # 4. Load tasks + build request caches
    if args.workers > 1:
        run_parallel(config, all_task_names, args.refresh, chat_args, args.workers)
    else:
        run_sequential(config, all_task_names, args.refresh, chat_args)

    # 5. Report cache stats
    caches_after = set(cache_dir.glob("*.pickle")) if cache_dir.is_dir() else set()
    new_caches = caches_after - caches_before
    refreshed = len(caches_before & caches_after) if args.refresh else 0
    reused = len(caches_after) - len(new_caches) - refreshed

    logger.info(f"  {len(new_caches)} new, {refreshed} refreshed, {reused} reused, {len(caches_after)} total")

    if args.refresh:
        logger.info(f"Refreshed {refreshed} request cache(s) in {cache_dir} ({len(new_caches)} new)")
    elif new_caches:
        logger.info(f"Wrote {len(new_caches)} new request cache(s) to {cache_dir} ({reused} reused)")
    else:
        logger.info(f"All {reused} request cache(s) already up to date in {cache_dir}")

    # 6. Pack cache into a tarball for fast distribution to eval jobs
    tarball_path = cache_dir / "cache.tar.gz"
    if new_caches or args.refresh or not tarball_path.exists():
        logger.info(f"Packing cache into {tarball_path}...")
        with tarfile.open(tarball_path, "w:gz") as tar:
            for f in sorted(cache_dir.glob("*.pickle")):
                tar.add(f, arcname=f.name)
        logger.info(f"Wrote {tarball_path} ({tarball_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        logger.info(f"Cache unchanged, skipping tarball rebuild ({tarball_path})")

    logger.info("Done!")
    return 0


def main():
    """Standalone entry point for backward compatibility."""
    parser = argparse.ArgumentParser(description="Download datasets and build request caches for eval-hive")
    add_arguments(parser)
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
