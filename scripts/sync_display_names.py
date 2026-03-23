"""Sync model_display_name in HF parquet to match current run configs.

Reads all YAML files in examples/ref_runs/, builds a model_key → display_name
mapping, and updates any mismatched rows in the HF parquet.

Usage:
    pixi run python scripts/sync_display_names.py                # dry run
    pixi run python scripts/sync_display_names.py --apply        # upload
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl
import yaml
from huggingface_hub import HfApi, hf_hub_download

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "examples" / "ref_runs"
HF_FILENAME = "scores.parquet"


def load_display_names(configs_dir: Path) -> dict[str, str]:
    """Return {model_key: display_name} from all YAML run configs."""
    mapping: dict[str, str] = {}
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        models = cfg.get("models", {})
        if not models:
            continue
        for entry in models.values():
            key = entry.get("model_key")
            name = entry.get("display_name")
            if key and name:
                mapping[key] = name
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload the updated parquet to HF (default is dry run)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=CONFIGS_DIR,
        help="Directory with run config YAMLs (default: examples/ref_runs/)",
    )
    args = parser.parse_args()

    # Build model_key → display_name mapping from configs
    mapping = load_display_names(args.configs_dir)
    print(f"Display names from configs ({len(mapping)} models):")
    for k, v in sorted(mapping.items()):
        print(f"  {k} → {v}")

    # Discover HF repo from first config
    hf_repo = None
    for yaml_path in sorted(args.configs_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        hf_repo = cfg.get("hf_result_repo")
        if hf_repo:
            break
    if not hf_repo:
        print("ERROR: No hf_result_repo found in any config.")
        return

    print(f"\nHF repo: {hf_repo}")

    # Download current parquet
    path = hf_hub_download(repo_id=hf_repo, filename=HF_FILENAME, repo_type="dataset")
    df = pl.read_parquet(path)
    print(f"Total rows: {len(df)}")

    # Show current state per model_key
    current = (
        df.group_by("model")
        .agg(pl.col("model_display_name").first().alias("current_display_name"))
        .sort("model")
    )
    print(f"\nCurrent display names in HF:")
    for row in current.iter_rows(named=True):
        marker = ""
        if row["model"] in mapping and mapping[row["model"]] != row["current_display_name"]:
            marker = f"  ← will change to '{mapping[row['model']]}'"
        print(f"  {row['model']}: '{row['current_display_name']}'{marker}")

    # Apply renames
    rename_expr = pl.col("model_display_name")
    changes: list[tuple[str, str, str]] = []
    for model_key, new_name in mapping.items():
        # Check if there are rows that need updating
        mismatched = df.filter(
            (pl.col("model") == model_key) & (pl.col("model_display_name") != new_name)
        )
        if len(mismatched) > 0:
            old_name = mismatched["model_display_name"][0]
            changes.append((model_key, old_name, new_name))
            rename_expr = (
                pl.when((pl.col("model") == model_key))
                .then(pl.lit(new_name))
                .otherwise(rename_expr)
            )

    if not changes:
        print("\nAll display names already match. Nothing to do.")
        return

    print(f"\nChanges to apply ({len(changes)}):")
    for model_key, old, new in changes:
        affected = len(df.filter(pl.col("model") == model_key))
        print(f"  {model_key}: '{old}' → '{new}' ({affected} rows)")

    updated = df.with_columns(rename_expr.alias("model_display_name"))

    if not args.apply:
        print("\nDry run — pass --apply to upload.")
        return

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        updated.write_parquet(tmp_path)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=HF_FILENAME,
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"\nUploaded updated parquet to {hf_repo}.")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
