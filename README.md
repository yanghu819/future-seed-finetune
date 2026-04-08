# future-seed-finetune

Minimal repo for testing a prompt-only Future-Seed style adapter on the `Gated DeltaNet` sublayers of Qwen3.5.

Current scope:

- dense Qwen3.5 only
- prompt-only / prefill-only seed path
- deep-layer scalar seed adapter
- detached + RMS-normalized + clipped recurrent seed
- smoke run on a tiny randomly initialized `qwen3_5` model
- awkward-task tiny SFT smoke with `baseline` vs `scalar-FS` switches
- pretrained `Qwen3.5-0.8B` remote smoke on GPU with repo-root caches only
- remote full-validation comparison for deep-layer scalar-FS vs frozen control on `Qwen3.5-0.8B`

The first goal is not end-task quality. It is to verify that:

1. Qwen3.5 `prefill` no longer hardcodes `initial_state=None` for selected `Gated DeltaNet` layers.
2. A tiny trainable scalar `alpha_l` can gate the cross-layer seed path.
3. Only the scalar seed parameters can remain trainable while the backbone stays frozen.

## Layout

- `future_seed_finetune/qwen35_scalar_fs.py`: patching and adapter logic
- `scripts/smoke_qwen35_scalar_fs.py`: tiny random-model smoke
- `scripts/download_qwen35_probe_assets.py`: download Qwen3.5 probe assets or full weights
- `scripts/validate_qwen35_prefill.py`: validate real Qwen3.5 config or pretrained checkpoint
- `scripts/build_awkward_kv_dataset.py`: build a synthetic causally-awkward SFT dataset
- `scripts/build_public_retrieval_dataset.py`: build a public retrieval/long-context benchmark dataset bundle
- `scripts/train_awkward_scalar_fs.py`: train tiny or pretrained models on that dataset
- `setup.sh`: install deps with `uv`
- `down.sh`: create local artifact/cache layout
- `run.sh`: run smoke and write metadata/results under `runs/`
- `repo_env.sh`: force `uv` / Hugging Face / transformers / pip caches into `artifacts/cache/` under the repo root

## Quickstart

```bash
bash ./setup.sh
bash ./down.sh
bash ./run.sh
```

Probe the real 9B config/tokenizer:

```bash
DOWNLOAD_MODE=probe bash ./down.sh
RUN_MODE=validate-config bash ./run.sh
```

If you already have full weights locally:

```bash
DOWNLOAD_MODE=full bash ./down.sh
RUN_MODE=validate-pretrained bash ./run.sh
```

Run a tiny awkward-task training smoke:

```bash
DOWNLOAD_MODE=probe GENERATE_DATASET=awkward-kv bash ./down.sh
RUN_MODE=train-smoke bash ./run.sh
```

Run the matching tiny baseline without the Future-Seed path:

```bash
DOWNLOAD_MODE=probe GENERATE_DATASET=awkward-kv bash ./down.sh
RUN_MODE=train-smoke FS_MODE=disabled bash ./run.sh
```

Run the matching pretrained control on a GPU machine without training any parameters:

```bash
DOWNLOAD_MODE=probe GENERATE_DATASET=awkward-kv bash ./down.sh
RUN_MODE=train-pretrained FS_MODE=disabled LOAD_DTYPE=bfloat16 LOW_CPU_MEM_USAGE=1 EVAL_LIMIT=8 MAX_STEPS=1 BATCH_SIZE=1 bash ./run.sh
```

Build a public retrieval benchmark bundle from LongBench and the task's original train source:

```bash
DOWNLOAD_MODE=probe \
GENERATE_DATASET=public-retrieval \
DATASET_SOURCE=longbench \
DATASET_TASK=hotpotqa \
TRAIN_LIMIT=256 \
EVAL_LIMIT=64 \
bash ./down.sh
```

Run a pretrained frozen-control evaluation on that public bundle:

```bash
RUN_MODE=train-pretrained \
FS_MODE=disabled \
DATASET_DIR=$PWD/artifacts/datasets/public_retrieval/hotpotqa \
MODEL_DIR=$PWD/artifacts/models/qwen3_5_9b_base_probe \
MAX_STEPS=0 \
bash ./run.sh
```

## Current Status

- `qwen3_5` tiny smoke is passing and confirms cross-layer seed injection on selected DeltaNet layers.
- Real `Qwen3.5-9B-Base` config/tokenizer probe is passing.
- Tiny awkward-task SFT smoke now supports both `baseline` and `scalar-FS`:
  - baseline clean run: `runs/qwen35-train-smoke-baseline-clean-20260407T071851Z`
  - scalar-FS clean run: `runs/qwen35-train-smoke-clean-20260407T071851Z`
- At this stage both tiny runs land at the same exact-match (`0.0625` awkward / `0.0625` friendly). This is still a pipeline validation result, not a task-improvement claim.
- Low-memory pretrained loading flags are now wired in (`LOAD_DTYPE`, `LOW_CPU_MEM_USAGE`, `EVAL_LIMIT`), but a local `Qwen/Qwen3.5-0.8B` forward on this 16GB CPU machine still exited with `137`. Real pretrained runs should be moved to a higher-memory or GPU machine.
- Remote GPU runs now succeed end-to-end on `Qwen/Qwen3.5-0.8B`:
- The `0.8B` loading mismatch is fixed:
  - the correct runtime path is `Qwen3_5ForConditionalGeneration`, with Future-Seed attached only to `model.language_model`
  - the current code no longer shows the large `UNEXPECTED` / `MISSING` weight groups from the earlier text-only loading path
- Small remote pretrained runs on the fixed path now work:
  - validate: `runs/qwen35-0p8b-validate-pretrained-bf16-remote-rerun4-20260407T131510Z`
  - shallow FS 1-step smoke: `runs/qwen35-0p8b-train-pretrained-smoke-rerun4-20260407T131510Z`
  - deep-layer FS 1-step smoke: `runs/qwen35-0p8b-train-pretrained-smoke-rerun5-20260407T132258Z`
  - matching frozen control: `runs/qwen35-0p8b-train-pretrained-baseline-control-rerun4-20260407T131628Z`
- The important modeling lesson is that shallow FS is harmful here, while deep-layer FS is stable:
  - shallow FS from early layers: `awkward = 0.125`, `friendly = 0.0`
  - deep-layer FS from the last 8 layers: `awkward = 1.0`, `friendly = 1.0`
  - frozen disabled control: `awkward = 1.0`, `friendly = 1.0`
- Full validation on 64 awkward + 64 friendly examples is now complete:
  - deep-layer FS: `runs/qwen35-0p8b-train-pretrained-deepfs-fullval-rerun6-20260408T011643Z`
  - frozen disabled control: `runs/qwen35-0p8b-train-pretrained-baseline-fullval-rerun6-20260408T011643Z`
  - deep-layer FS reaches `awkward = 1.0`, `friendly = 0.984375`
  - frozen disabled control reaches `awkward = 1.0`, `friendly = 1.0`
- Retrieval-style `retrieval_digit_long` is now the first non-saturated benchmark line:
  - old `max_length=256` runs silently truncated answer tokens during training, which is why earlier `deepfs+delta` runs showed `skipped_empty_targets > 0` and `effective_train_steps = 0`
  - with `MAX_LENGTH=1024`, training now actually runs for the `deepfs+delta` path
  - baseline full-eval control: `runs/qwen35-0p8b-train-pretrained-baseline-retrievaldigit-max1024-rerun2-20260408T064845Z`
  - deepfs+delta train run: `runs/qwen35-0p8b-train-pretrained-deepfsdelta-retrievaldigit-max1024-rerun7-20260408T064845Z`
  - baseline reaches `awkward = 1.0`, `friendly = 1.0`
  - deepfs+delta reaches `awkward = 1.0`, `friendly = 0.984375`, with `effective_train_steps = 10`
- Current conclusion:
  - the Future-Seed deep-layer scalar adapter is technically valid on real pretrained Qwen3.5 weights
  - `scalar-FS only` was too weak, and `deepfs+delta` is the first version that both trains stably and preserves the real pretrained path
  - on the current synthetic awkward and retrieval-style tasks, it still does not beat the frozen baseline
  - so the current repo is a working research scaffold, not yet a positive quality result
- Public benchmark integration is now available:
  - `down.sh` supports `GENERATE_DATASET=public-retrieval`
  - `LongBench` task smoke has been exercised for `hotpotqa` and `passage_retrieval_en`
  - `train_awkward_scalar_fs.py` now reads `metadata.json` and reports generic `eval_results`

## Notes

- The target aistation root should be `/fangxueji/Projects/PG/future-seed-finetune`.
- This local implementation was developed on a machine where `/fangxueji` is read-only, so the equivalent local path was used during development.
- The repo now includes a local compatibility patch for the current `transformers main` `qwen3_5` constructor bugs, so tiny-model smoke can run directly on `qwen3_5`.
- Full `Qwen3.5-9B-Base` pretrained validation is supported, but it depends on downloading the full checkpoint and is expected to be run on a machine with enough RAM or GPU memory.
- Repo scripts now pin `uv`, Hugging Face, transformers, tokenizers, torch, and pip caches to `artifacts/cache/` so no runtime cache should land under `~`.
- The remote aistation workflow should always keep tools, caches, model weights, and datasets under `/fangxueji/Projects/PG/future-seed-finetune`.
