from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
import string
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from future_seed_finetune import (
    ScalarFutureSeedConfig,
    apply_scalar_future_seed,
    detect_qwen35_pretrained_architecture,
    freeze_except_future_seed,
    get_future_seed_runtime_stats,
    install_qwen35_upstream_compat_fixes,
    list_future_seed_parameters,
    load_qwen35_full_config,
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


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def normalize_qa_answer(text: str) -> str:
    text = normalize_answer(text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_prediction(decoded: str, target: str) -> str:
    target_norm = normalize_answer(target)
    if re.fullmatch(r"\d+", target_norm):
        match = re.search(r"\d+", decoded)
        return match.group(0) if match else ""
    if re.fullmatch(r"[A-Za-z0-9_]+", target_norm):
        match = re.search(r"[A-Za-z0-9_]+", decoded)
        return match.group(0).lower() if match else ""
    return normalize_answer(decoded)


def exact_match_score(prediction: str, answers: list[str]) -> float:
    pred = normalize_qa_answer(prediction)
    return float(any(pred == normalize_qa_answer(answer) for answer in answers))


def qa_f1_score(prediction: str, answers: list[str]) -> float:
    pred_tokens = normalize_qa_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_qa_answer(answer).split()
        if not gold_tokens:
            continue
        common = {}
        for token in pred_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def score_prediction(prediction: str, answers: list[str], metric: str) -> float:
    if metric == "qa_f1":
        return qa_f1_score(prediction, answers)
    return exact_match_score(prediction, answers)


def load_dataset_bundle(dataset_dir: Path) -> tuple[JsonlDataset, dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        train_rows = load_jsonl_rows(dataset_dir / "train.jsonl")
        eval_splits = {
            split_name: load_jsonl_rows(dataset_dir / f"{split_name}.jsonl")
            for split_name in metadata.get("eval_splits", {})
        }
        return JsonlDataset(dataset_dir / "train.jsonl"), eval_splits, metadata
    train_set = JsonlDataset(dataset_dir / "train.jsonl")
    eval_splits = {
        "awkward": JsonlDataset(dataset_dir / "valid_awkward.jsonl").rows,
        "friendly": JsonlDataset(dataset_dir / "valid_friendly.jsonl").rows,
    }
    return train_set, eval_splits, None


def compute_masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    active = shift_labels.ne(-100)
    if not active.any():
        return shift_logits.sum() * 0.0, 0
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    flat_active = active.view(-1)
    return (
        torch.nn.functional.cross_entropy(flat_logits[flat_active], flat_labels[flat_active], reduction="mean"),
        int(flat_active.sum().item()),
    )


def build_model(args, tokenizer):
    install_qwen35_upstream_compat_fixes()
    with contextlib.redirect_stdout(sys.stderr):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )
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
        architecture = detect_qwen35_pretrained_architecture(args.model_dir)
        if architecture == "conditional_generation":
            config = load_qwen35_full_config(args.model_dir)
            model_class = Qwen3_5ForConditionalGeneration
        else:
            config = load_qwen35_text_config(args.model_dir)
            model_class = Qwen3_5ForCausalLM
        pretrained_kwargs = {
            "config": config,
            "torch_dtype": load_dtype,
            "device_map": None,
            "low_cpu_mem_usage": args.low_cpu_mem_usage,
        }
        if device.type == "cuda":
            pretrained_kwargs["device_map"] = {"": torch.cuda.current_device()}
        model = model_class.from_pretrained(
            args.model_dir,
            **pretrained_kwargs,
        )
        config = getattr(model.config, "text_config", model.config)
        model._future_seed_loaded_via_device_map = bool(pretrained_kwargs["device_map"] is not None)

    fs_cfg = None
    if not args.disable_future_seed:
        fs_cfg = ScalarFutureSeedConfig(
            enabled=True,
            start_layer=args.start_layer if args.start_layer >= 0 else max(1, config.num_hidden_layers - 8),
            prompt_only=True,
            detach_seed=True,
            rms_norm_seed=True,
            clip_value=args.seed_clip_value,
            alpha_init=args.alpha_init,
            enable_delta_adapter=args.enable_delta_adapter,
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
            for _name, param in model.named_parameters():
                param.requires_grad = False
            trainable = []
        else:
            trainable = freeze_except_future_seed(model)

    return model, fs_cfg, trainable


def evaluate(
    model,
    tokenizer,
    rows,
    device,
    max_length: int,
    limit: int | None = None,
    eval_max_new_tokens: int = 8,
) -> dict[str, float]:
    model.eval()
    total_score = 0.0
    eval_rows = rows[:limit] if limit is not None and limit > 0 else rows
    for row in eval_rows:
        prompt_budget = max(1, max_length - eval_max_new_tokens)
        encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=False)
        encoded = {k: v[:, :prompt_budget].to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=eval_max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
        answers = [str(x) for x in row.get("answers", [row["target"]])]
        metric = str(row.get("metric", "exact_match"))
        pred = extract_prediction(decoded, answers[0])
        total_score += score_prediction(pred, answers, metric)
    return {
        "score": total_score / max(1, len(eval_rows)),
        "metric": str((eval_rows[0] if eval_rows else {}).get("metric", "exact_match")),
        "num_examples": len(eval_rows),
    }


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
    parser.add_argument("--seed-clip-value", type=float, default=1.0)
    parser.add_argument("--start-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--disable-future-seed", action="store_true")
    parser.add_argument("--enable-delta-adapter", action="store_true")
    parser.add_argument("--load-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=8)
    parser.add_argument("--optimize-in-eval-mode", action="store_true")
    parser.add_argument("--skip-nonfinite-loss", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--fs-alpha-clamp", type=float, default=0.0)
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
        train_set, eval_splits, dataset_metadata = load_dataset_bundle(dataset_dir)

        batches = None
        if len(train_set) > 0:
            loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda rows: collate_batch(rows, tokenizer, args.max_length),
            )
            batches = cycle(loader)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr) if params else None

        losses = []
        train_runtime = None
        skipped_nonfinite_losses = 0
        skipped_nonfinite_grads = 0
        skipped_empty_targets = 0
        last_grad_norm = None
        if args.optimize_in_eval_mode:
            model.eval()
        else:
            model.train()
        effective_steps = 0
        if optimizer is not None and args.max_steps > 0 and batches is not None:
            optimizer.zero_grad(set_to_none=True)
            for _step in range(args.max_steps):
                batch = next(batches)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss, active_targets = compute_masked_ce_loss(out.logits, batch["labels"])
                train_runtime = get_future_seed_runtime_stats(model)
                if active_targets == 0:
                    skipped_empty_targets += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if not torch.isfinite(loss):
                    skipped_nonfinite_losses += 1
                    optimizer.zero_grad(set_to_none=True)
                    if args.skip_nonfinite_loss:
                        continue
                    raise RuntimeError(f"non-finite loss encountered: {loss.item()}")
                loss.backward()
                grad_invalid = False
                if args.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                    last_grad_norm = float(grad_norm.detach().float().cpu().item())
                    grad_invalid = not torch.isfinite(grad_norm)
                else:
                    for param in params:
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            grad_invalid = True
                            break
                if grad_invalid:
                    skipped_nonfinite_grads += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                if args.fs_alpha_clamp > 0:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name.endswith("fs_alpha"):
                                param.clamp_(-args.fs_alpha_clamp, args.fs_alpha_clamp)
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.item()))
                effective_steps += 1

        eval_limit = args.eval_limit if args.eval_limit > 0 else None
        eval_metrics = {
            split_name: evaluate(
                model,
                tokenizer,
                split_rows,
                device,
                args.max_length,
                eval_limit,
                args.eval_max_new_tokens,
            )
            for split_name, split_rows in eval_splits.items()
        }
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
            "delta_adapter_enabled": bool(args.enable_delta_adapter and not args.disable_future_seed),
            "unfreeze_backbone": args.unfreeze_backbone,
            "train_steps": args.max_steps,
            "effective_train_steps": effective_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed_clip_value": args.seed_clip_value,
            "optimize_in_eval_mode": bool(args.optimize_in_eval_mode),
            "skip_nonfinite_loss": bool(args.skip_nonfinite_loss),
            "grad_clip_norm": args.grad_clip_norm,
            "fs_alpha_clamp": args.fs_alpha_clamp,
            "loss_start": losses[0] if losses else None,
            "loss_end": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "skipped_nonfinite_losses": skipped_nonfinite_losses,
            "skipped_nonfinite_grads": skipped_nonfinite_grads,
            "skipped_empty_targets": skipped_empty_targets,
            "last_grad_norm": last_grad_norm,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "future_seed_parameters": list_future_seed_parameters(model),
            "trainable_parameters_preview": trainable[:20],
            "future_seed_config": asdict(fs_cfg) if fs_cfg is not None else None,
            "dataset_metadata": dataset_metadata,
            "train_runtime": train_runtime,
            "eval_runtime": eval_runtime,
            "eval_results": eval_metrics,
            "device": str(device),
            "load_dtype": args.load_dtype if args.model_backend == "pretrained" else None,
            "low_cpu_mem_usage": bool(args.low_cpu_mem_usage) if args.model_backend == "pretrained" else None,
            "eval_limit": args.eval_limit,
            "eval_max_new_tokens": args.eval_max_new_tokens,
        }
        if "awkward" in eval_metrics:
            result["eval_awkward"] = eval_metrics["awkward"]
        if "friendly" in eval_metrics:
            result["eval_friendly"] = eval_metrics["friendly"]
        if "eval_longbench" in eval_metrics:
            result["eval_longbench"] = eval_metrics["eval_longbench"]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
