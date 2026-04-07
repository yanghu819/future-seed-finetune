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
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Future Seed shifts recurrent state across deep Gated DeltaNet layers.")
    parser.add_argument("--from-pretrained", action="store_true")
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        install_qwen35_upstream_compat_fixes()

        from transformers import AutoTokenizer
        from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    raw = json.loads((model_dir / "config.json").read_text())
    text_config_dict = raw.get("text_config", raw)
    config = Qwen3_5TextConfig.from_dict(text_config_dict)
    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if isinstance(mlp_only_layers, AttributeError) or mlp_only_layers is None:
        config.mlp_only_layers = []
    else:
        config.mlp_only_layers = list(mlp_only_layers)

    if args.from_pretrained:
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.float32,
            device_map=None,
        )
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
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
