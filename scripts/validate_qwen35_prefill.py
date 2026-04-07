from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import torch
from accelerate import init_empty_weights

from future_seed_finetune import (
    ScalarFutureSeedConfig,
    apply_scalar_future_seed,
    freeze_except_future_seed,
    get_future_seed_runtime_stats,
    install_qwen35_upstream_compat_fixes,
    list_future_seed_parameters,
    load_qwen35_text_config,
)

def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype={name}")
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Future Seed shifts recurrent state across deep Gated DeltaNet layers.")
    parser.add_argument("--from-pretrained", action="store_true")
    parser.add_argument("--load-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        install_qwen35_upstream_compat_fixes()

        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    config = load_qwen35_text_config(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.from_pretrained:
        load_dtype = resolve_dtype(args.load_dtype)
        pretrained_kwargs = {
            "config": config,
            "torch_dtype": load_dtype,
            "device_map": None,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
        }
        if device.type == "cuda":
            pretrained_kwargs["device_map"] = {"": torch.cuda.current_device()}
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_dir,
            **pretrained_kwargs,
        )
        if pretrained_kwargs["device_map"] is None:
            model.to(device)
        run_forward = True
    else:
        with init_empty_weights():
            model = Qwen3_5ForCausalLM(config)
        run_forward = False

    fs_cfg = ScalarFutureSeedConfig(
        enabled=True,
        start_layer=max(1, config.num_hidden_layers - 8),
        prompt_only=True,
        detach_seed=True,
        rms_norm_seed=True,
        clip_value=1.0,
        alpha_init=0.25,
        reset_on_full_attention=True,
    )
    apply_scalar_future_seed(model, fs_cfg)
    trainable = freeze_except_future_seed(model)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    logits_shape = None
    runtime = None
    if run_forward:
        with torch.no_grad():
            out = model(**encoded, use_cache=False)
        logits_shape = list(out.logits.shape)
        runtime = get_future_seed_runtime_stats(model)

    result = {
        "status": "ok",
        "model_dir": str(model_dir),
        "loaded_from_pretrained": args.from_pretrained,
        "validation_mode": "pretrained_forward" if run_forward else "config_meta",
        "logits_shape": logits_shape,
        "future_seed_parameters": list_future_seed_parameters(model),
        "trainable_parameters": trainable,
        "runtime": runtime,
        "vocab_size": int(config.vocab_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "prompt_tokens": int(encoded["input_ids"].shape[1]),
        "load_dtype": args.load_dtype if args.from_pretrained else None,
        "low_cpu_mem_usage": bool(args.low_cpu_mem_usage) if args.from_pretrained else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
