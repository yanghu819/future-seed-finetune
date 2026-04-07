from __future__ import annotations

import json
from pathlib import Path

import torch

from future_seed_finetune import (
    ScalarFutureSeedConfig,
    apply_qwen3next_scalar_future_seed,
    freeze_except_future_seed,
    get_future_seed_runtime_stats,
    list_future_seed_parameters,
)


def build_tiny_model():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM

    config = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        max_position_embeddings=128,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mlp_only_layers=[],
    )
    config.mlp_only_layers = []
    return Qwen3NextForCausalLM(config)


def main() -> None:
    torch.manual_seed(0)
    model = build_tiny_model()

    fs_cfg = ScalarFutureSeedConfig(
        enabled=True,
        start_layer=1,
        prompt_only=True,
        detach_seed=True,
        rms_norm_seed=True,
        clip_value=1.0,
        alpha_init=0.25,
        reset_on_full_attention=True,
    )
    apply_qwen3next_scalar_future_seed(model, fs_cfg)
    trainable = freeze_except_future_seed(model)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    input_ids = torch.randint(0, 256, (2, 12), dtype=torch.long)
    labels = torch.randint(0, 256, (2, 12), dtype=torch.long)

    model.train()
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    out.loss.backward()

    grad_norms = {
        name: (float(param.grad.norm().item()) if param.grad is not None else 0.0)
        for name, param in model.named_parameters()
        if name in trainable
    }
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    runtime = get_future_seed_runtime_stats(model)
    assert runtime is not None, "future-seed runtime stats missing"
    assert runtime["active"], "future-seed runtime did not activate"
    assert runtime["injection_count"] >= 1, f"expected at least one seed injection, got {runtime}"
    assert any(v > 0 for v in grad_norms.values()), f"no gradient flowed to fs_alpha params: {grad_norms}"

    result = {
        "status": "ok",
        "smoke_backend": "qwen3_next_proxy",
        "trainable_parameters": trainable,
        "future_seed_parameters": list_future_seed_parameters(model),
        "grad_norms": grad_norms,
        "runtime": runtime,
        "loss": float(out.loss.item()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
