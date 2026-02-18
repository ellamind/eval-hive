#!/usr/bin/env python3
"""Download all revisions (branches + tags) of a HuggingFace repo."""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def get_all_revisions(repo_id: str, repo_type: str = "model") -> list[str]:
    """List all branches and tags for a repo."""
    api = HfApi()
    refs = api.list_repo_refs(repo_id, repo_type=repo_type)
    revisions = [b.name for b in refs.branches] + [t.name for t in refs.tags]
    return revisions


def _enable_hf_transfer() -> None:
    """Enable hf_transfer if available."""
    try:
        import hf_transfer  # noqa: F401

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("hf_transfer: enabled")
    except ImportError:
        print("hf_transfer: not installed, using default backend")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all revisions of a HuggingFace repo.")
    parser.add_argument("repo_id", help="HuggingFace repo id, e.g. 'meta-llama/Llama-2-7b'")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("downloads"),
        help="Target directory for downloads (default: ./downloads)",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repo (default: model)",
    )
    parser.add_argument(
        "--revisions",
        nargs="*",
        help="Only download these revisions (default: all branches + tags)",
    )
    parser.add_argument(
        "--no-hf-transfer",
        action="store_true",
        help="Disable hf_transfer (enabled by default if installed)",
    )
    args = parser.parse_args()

    if not args.no_hf_transfer:
        _enable_hf_transfer()

    repo_id: str = args.repo_id
    output_dir: Path = args.output_dir.resolve()

    print(f"Repo:       {repo_id}")
    print(f"Output dir: {output_dir}")

    if args.revisions:
        revisions = args.revisions
    else:
        print("Fetching revisions...")
        revisions = get_all_revisions(repo_id, repo_type=args.repo_type)

    if not revisions:
        print("No revisions found.")
        sys.exit(1)

    print(f"Found {len(revisions)} revision(s): {', '.join(revisions)}\n")

    success, failed = 0, 0
    for rev in revisions:
        rev_dir = output_dir / repo_id / rev
        print(f"[{rev}] downloading to {rev_dir} ...")
        try:
            snapshot_download(
                repo_id,
                revision=rev,
                repo_type=args.repo_type,
                local_dir=str(rev_dir),
            )
            print(f"[{rev}] done")
            success += 1
        except Exception as e:
            print(f"[{rev}] FAILED: {e}", file=sys.stderr)
            failed += 1

    print(f"\nFinished. success={success} failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
