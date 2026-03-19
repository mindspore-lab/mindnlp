#!/usr/bin/env python3
"""
Download the three models needed for the Llama3-merge-biomed-8b DARE-TIES merge.

Supports retry and mirror fallback (HF_ENDPOINT=https://hf-mirror.com).

Usage:
  # Download to default location (./models/ next to this script)
  python download_models.py

  # Download to a custom directory
  python download_models.py --output-dir /data/hf_models

  # Use mirror (useful inside mainland China)
  HF_ENDPOINT=https://hf-mirror.com python download_models.py

Environment variables:
  HF_TOKEN          - HuggingFace access token (required)
  HF_ENDPOINT       - HuggingFace mirror endpoint (optional)
"""

import argparse
import os
import sys
import time

DOWNLOADS = [
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "aaditya/Llama3-OpenBioLLM-8B",
]
MAX_RETRIES = 5
RETRY_DELAY = 30


def _read_token() -> str:
    """Read HF token from env var or .hf_token file."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if token:
        return token
    # Fallback: read from examples/wizard/.hf_token
    token_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", ".hf_token"
    )
    if os.path.isfile(token_file):
        with open(token_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return ""


def main():
    parser = argparse.ArgumentParser(description="Download models for DARE-TIES merge")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        help="Directory to download models into (default: ./models/)",
    )
    args = parser.parse_args()
    base_dir = args.output_dir

    token = _read_token()
    if not token:
        print(
            "ERROR: Set HF_TOKEN env var, or write your token into "
            "examples/wizard/.hf_token",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(base_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    endpoint = os.environ.get("HF_ENDPOINT", "")
    mirror = "https://hf-mirror.com"

    for repo_id in DOWNLOADS:
        local_dir = os.path.join(base_dir, repo_id)
        if os.path.isfile(os.path.join(local_dir, "config.json")):
            print(f"[skip] {repo_id} (already exists at {local_dir})")
            continue

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                use_mirror = attempt > MAX_RETRIES // 2 and not endpoint
                if use_mirror:
                    os.environ["HF_ENDPOINT"] = mirror
                    print(f"[{attempt}/{MAX_RETRIES}] {repo_id} (mirror: {mirror})")
                else:
                    print(f"[{attempt}/{MAX_RETRIES}] {repo_id}")

                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    token=token,
                )
                print(f"[done] {repo_id} -> {local_dir}")
                success = True
                if use_mirror and "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
                break
            except Exception as e:
                print(f"[fail] attempt {attempt}: {e}", file=sys.stderr)
                if use_mirror and "HF_ENDPOINT" in os.environ:
                    del os.environ["HF_ENDPOINT"]
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"       retry in {wait}s ...", file=sys.stderr)
                    time.sleep(wait)

        if not success:
            print(
                f"FATAL: could not download {repo_id} after {MAX_RETRIES} attempts",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\nAll models ready under {base_dir}")


if __name__ == "__main__":
    main()
