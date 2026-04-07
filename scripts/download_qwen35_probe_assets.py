from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-weights", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    if not args.full_weights:
        allow_patterns = [
            "*.json",
            "tokenizer*",
            "*.tiktoken",
            "*.txt",
            "*.model",
            "generation_config*",
            "special_tokens_map*",
            "config*",
        ]

    local_dir = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
    )

    manifest = {
        "model_id": args.model_id,
        "output_dir": str(output_dir),
        "resolved_dir": local_dir,
        "full_weights": args.full_weights,
        "hf_endpoint": os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_BASE_URL"),
        "downloaded_files": sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

