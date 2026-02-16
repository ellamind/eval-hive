"""HuggingFace dataset helpers for parquet download, upload, and merge."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

DEDUP_COLS = ["model", "step", "task", "metric", "metric_filter"]


def download_hf_parquet(
    hf_repo: str,
    filename: str = "scores.parquet",
) -> pl.DataFrame | None:
    """Download a parquet file from a HuggingFace dataset repo.

    Returns None if the repo or file does not exist.
    Respects ``HF_HUB_OFFLINE`` for offline / cached access.
    """
    from huggingface_hub import hf_hub_download

    try:
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
        )
        df = pl.read_parquet(local_path)
        logger.info("Downloaded %d rows from %s/%s", len(df), hf_repo, filename)
        return df
    except Exception as e:
        logger.info("Could not download %s from %s: %s", filename, hf_repo, e)
        return None


def upload_hf_parquet(
    local_path: Path,
    hf_repo: str,
    filename: str = "scores.parquet",
) -> None:
    """Upload a parquet file to a HuggingFace dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=hf_repo,
        repo_type="dataset",
    )
    logger.info("Uploaded %s to %s/%s", local_path, hf_repo, filename)


def merge_and_dedup(
    existing_df: pl.DataFrame,
    new_df: pl.DataFrame,
) -> pl.DataFrame:
    """Concatenate two DataFrames and deduplicate.

    Deduplication key: ``(model, step, task, metric, metric_filter)``.
    When duplicates exist, keep the row with the latest ``eval_date``.
    """
    combined = pl.concat([existing_df, new_df], how="diagonal_relaxed")
    before = len(combined)

    combined = (
        combined
        .sort("eval_date", nulls_last=False)
        .unique(subset=DEDUP_COLS, keep="last")
    )

    after = len(combined)
    if before != after:
        logger.info("Dedup: %d → %d rows (%d duplicates removed)", before, after, before - after)

    combined = combined.sort(DEDUP_COLS, nulls_last=False)
    return combined


def push_to_hf(
    local_parquet: Path,
    hf_repo: str,
    hf_filename: str = "scores.parquet",
) -> None:
    """Download existing HF parquet, merge with local, dedup, and re-upload.

    If no existing HF data is found, uploads the local parquet as-is.
    Also updates the local parquet file with the merged result.
    """
    local_df = pl.read_parquet(local_parquet)
    logger.info("Local parquet: %d rows", len(local_df))

    existing_df = download_hf_parquet(hf_repo, hf_filename)

    if existing_df is not None and len(existing_df) > 0:
        combined = merge_and_dedup(existing_df, local_df)
    else:
        combined = local_df

    # Write merged result to a temp file for upload
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        combined.write_parquet(tmp_path)
        upload_hf_parquet(tmp_path, hf_repo, hf_filename)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Update local file with merged data
    combined.write_parquet(local_parquet)
    logger.info("Updated local %s: %d rows", local_parquet, len(combined))
