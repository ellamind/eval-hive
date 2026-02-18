"""Drop aggregate rows and apertus_8b rows from the ellamind/eval-scores HF dataset.

Aggregate rows (task_group, eval_suite) are derived from leaf benchmark
scores and will be recomputed by ``eval-hive collect``.  Removing them
ensures stale or incomplete aggregates don't persist.

apertus_8b rows had incorrect step values (token counts parsed instead of
training steps) and need to be re-collected with the fixed manifest.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "ellamind/eval-scores"
HF_FILENAME = "scores.parquet"
AGG_TYPES = ["task_group", "eval_suite"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop aggregate rows from HF eval-scores dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without uploading")
    args = parser.parse_args()

    # Download current dataset
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
    )
    df = pl.read_parquet(local_path)
    print(f"Downloaded {len(df)} rows from {HF_REPO}/{HF_FILENAME}")

    # Build combined drop mask
    drop_mask = pl.lit(False)

    if "task_type" in df.columns:
        agg_mask = pl.col("task_type").is_in(AGG_TYPES)
        n_agg = df.filter(agg_mask).height
        print(f"Aggregate rows to drop: {n_agg}")
        drop_mask = drop_mask | agg_mask
    else:
        print("No task_type column found — skipping aggregate cleanup.")

    if "model" in df.columns:
        apertus_mask = pl.col("model") == "apertus_8b"
        n_apertus = df.filter(apertus_mask).height
        print(f"apertus_8b rows to drop: {n_apertus}")
        drop_mask = drop_mask | apertus_mask
    else:
        print("No model column found — skipping apertus_8b cleanup.")

    n_drop = df.filter(drop_mask).height
    if n_drop == 0:
        print("Nothing to do.")
        return

    df_clean = df.filter(~drop_mask)
    print(f"Rows remaining: {len(df_clean)} (was {len(df)}, dropping {n_drop})")

    if args.dry_run:
        print("Dry run — no changes uploaded.")
        return

    # Upload cleaned dataset
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        df_clean.write_parquet(tmp_path)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=HF_FILENAME,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
        print(f"Uploaded cleaned dataset to {HF_REPO}/{HF_FILENAME}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
