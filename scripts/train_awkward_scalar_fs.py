from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

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


class JsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def collate_batch(rows, tokenizer, max_length: int):
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    input_ids = []
    labels = []
    for row in rows:
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
        target_ids = tokenizer(row["target"], add_special_tokens=False).input_ids
        ids = (prompt_ids + target_ids + [eos_id])[:max_length]
        y = ([-100] * len(prompt_ids) + target_ids + [eos_id])[:max_length]
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))

    max_len = max(x.numel() for x in input_ids)
    padded_inputs = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, (ids, y) in enumerate(zip(input_ids, labels)):
        padded_inputs[i, : ids.numel()] = ids
        padded_labels[i, : y.numel()] = y
        attention_mask[i, : ids.numel()] = 1

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask,
    }


def build_model(args, tokenizer):
    install_qwen35_upstream_compat_fixes()
    with contextlib.redirect_stdout(sys.stderr):
        from transformers.models.qwen3_5.modular_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_backend == "tiny":
        config = Qwen3_5TextConfig(
            vocab_size=max(tokenizer.vocab_size, tokenizer.eos_token_id + 1),
            hidden_size=args.tiny_hidden_size,
            intermediate_size=args.tiny_intermediate_size,
            num_hidden_layers=args.tiny_num_layers,
            num_attention_heads=args.tiny_num_heads,
            num_key_value_heads=args.tiny_num_kv_heads,
            head_dim=args.tiny_head_dim,
            linear_key_head_dim=args.tiny_head_dim,
            linear_value_head_dim=args.tiny_head_dim,
            linear_num_key_heads=args.tiny_num_heads,
            linear_num_value_heads=args.tiny_num_heads,
            max_position_embeddings=max(args.max_length, 128),
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"][: args.tiny_num_layers],
            mlp_only_layers=[],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        config.mlp_only_layers = []
        model = Qwen3_5ForCausalLM(config)
    else:
        load_dtype = resolve_dtype(args.load_dtype)
        config = load_qwen35_text_config(args.model_dir)
        pretrained_kwargs = {
            "config": config,
            "torch_dtype": load_dtype,
            "device_map": None,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
        }
        if device.type == "cuda":
            pretrained_kwargs["device_map"] = {"": torch.cuda.current_device()}
        model = Qwen3_5ForCausalLM.from_pretrained(
            args.model_dir,
            **pretrained_kwargs,
        )
        config = model.config
        model._future_seed_loaded_via_device_map = bool(pretrained_kwargs["device_map"] is not None)

    fs_cfg = None
    if not args.disable_future_seed:
        fs_cfg = ScalarFutureSeedConfig(
            enabled=True,
            start_layer=args.start_layer if args.start_layer >= 0 else max(1, config.num_hidden_layers - 8),
            prompt_only=True,
            detach_seed=True,
            rms_norm_seed=True,
            clip_value=1.0,
            alpha_init=args.alpha_init,
            reset_on_full_attention=True,
        )
        apply_scalar_future_seed(model, fs_cfg)

    if args.unfreeze_backbone:
        trainable = []
        for name, param in model.named_parameters():
            param.requires_grad = True
            trainable.append(name)
    else:
        if args.disable_future_seed:
            raise ValueError("disable-future-seed requires --unfreeze-backbone, otherwise no parameters are trainable")
        trainable = freeze_except_future_seed(model)

    return model, fs_cfg, trainable


def evaluate(model, tokenizer, rows, device, max_length: int, limit: int | None = None) -> dict[str, float]:
    model.eval()
    correct = 0
    eval_rows = rows[:limit] if limit is not None and limit > 0 else rows
    for row in eval_rows:
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
        encoded = {k: v[:, :max_length].to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        match = re.search(r"\d", decoded)
        pred = match.group(0) if match else ""
        correct += int(pred == row["target"])
    return {"exact_match": correct / max(1, len(eval_rows)), "num_examples": len(eval_rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-backend", choices=["tiny", "pretrained"], default="tiny")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--alpha-init", type=float, default=0.25)
    parser.add_argument("--start-layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--disable-future-seed", action="store_true")
    parser.add_argument("--load-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--tiny-hidden-size", type=int, default=32)
    parser.add_argument("--tiny-intermediate-size", type=int, default=64)
    parser.add_argument("--tiny-num-layers", type=int, default=4)
    parser.add_argument("--tiny-num-heads", type=int, default=4)
    parser.add_argument("--tiny-num-kv-heads", type=int, default=4)
    parser.add_argument("--tiny-head-dim", type=int, default=8)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model, fs_cfg, trainable = build_model(args, tokenizer)
        if not getattr(model, "_future_seed_loaded_via_device_map", False):
            model.to(device)

        dataset_dir = Path(args.dataset_dir)
        train_set = JsonlDataset(dataset_dir / "train.jsonl")
        valid_awkward = JsonlDataset(dataset_dir / "valid_awkward.jsonl")
        valid_friendly = JsonlDataset(dataset_dir / "valid_friendly.jsonl")

        loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda rows: collate_batch(rows, tokenizer, args.max_length),
        )
        batches = cycle(loader)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr)

        losses = []
        train_runtime = None
        model.train()
        for _step in range(args.max_steps):
            batch = next(batches)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, use_cache=False)
            loss = out.loss
            train_runtime = get_future_seed_runtime_stats(model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))

        eval_limit = args.eval_limit if args.eval_limit > 0 else None
        awkward_metrics = evaluate(model, tokenizer, valid_awkward.rows, device, args.max_length, eval_limit)
        friendly_metrics = evaluate(model, tokenizer, valid_friendly.rows, device, args.max_length, eval_limit)
        eval_runtime = get_future_seed_runtime_stats(model)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        alpha_values = {}
        for name, param in model.named_parameters():
            if name.endswith("fs_alpha"):
                alpha_values[name] = float(param.detach().float().cpu().item())
        (output_dir / "fs_alpha.json").write_text(json.dumps(alpha_values, indent=2, sort_keys=True) + "\n")

        result = {
            "status": "ok",
            "model_backend": args.model_backend,
            "future_seed_enabled": not args.disable_future_seed,
            "unfreeze_backbone": args.unfreeze_backbone,
            "train_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "loss_start": losses[0],
            "loss_end": losses[-1],
            "loss_min": min(losses),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "future_seed_parameters": list_future_seed_parameters(model),
            "trainable_parameters_preview": trainable[:20],
            "future_seed_config": asdict(fs_cfg) if fs_cfg is not None else None,
            "train_runtime": train_runtime,
            "eval_runtime": eval_runtime,
            "eval_awkward": awkward_metrics,
            "eval_friendly": friendly_metrics,
            "device": str(device),
            "load_dtype": args.load_dtype if args.model_backend == "pretrained" else None,
            "low_cpu_mem_usage": bool(args.low_cpu_mem_usage) if args.model_backend == "pretrained" else None,
            "eval_limit": args.eval_limit,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
