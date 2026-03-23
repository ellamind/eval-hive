"""Drop invalid rows from the HF scores parquet and re-upload.

Drops:
  - arc_challenge (eng) rows
  - gsm_symbolic rows (all languages)
  - humaneval_fim* rows (all languages)

Usage:
    pixi run python scripts/drop_hf_rows.py                # dry run (default)
    pixi run python scripts/drop_hf_rows.py --apply         # actually upload
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "ellamind/eval-scores-ref"
HF_FILENAME = "scores.parquet"


def download_parquet(repo: str) -> pl.DataFrame:
    path = hf_hub_download(repo_id=repo, filename=HF_FILENAME, repo_type="dataset")
    return pl.read_parquet(path)


def upload_parquet(local_path: Path, repo: str) -> None:
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=HF_FILENAME,
        repo_id=repo,
        repo_type="dataset",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually upload the cleaned parquet to HF (default is dry run)",
    )
    args = parser.parse_args()

    df = download_parquet(HF_REPO)
    print(f"Total rows before: {len(df)}")

    # Rows to drop
    mask = (
        (pl.col("task").str.contains("arc_challenge") & (pl.col("language") == "eng"))
        | pl.col("task").str.contains("gsm_symbolic")
        | pl.col("task").str.contains("humaneval_fim")
    )
    invalid = df.filter(mask)
    clean = df.filter(~mask)

    print(f"Rows to drop:     {len(invalid)}")
    print(f"Rows after:       {len(clean)}")

    if len(invalid) == 0:
        print("Nothing to drop.")
        return

    # Show what will be removed
    summary = (
        invalid.select("model", "step", "task", "metric")
        .unique()
        .sort(["model", "step", "task", "metric"])
    )
    print(f"\nDistinct (model, step, task, metric) combinations to drop: {len(summary)}")
    with pl.Config(tbl_rows=50):
        print(summary)

    if not args.apply:
        print("\nDry run — pass --apply to upload the cleaned parquet.")
        return

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        clean.write_parquet(tmp_path)
        upload_parquet(tmp_path, HF_REPO)
        print(f"\nUploaded cleaned parquet ({len(clean)} rows) to {HF_REPO}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
