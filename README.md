# future-seed-finetune

Minimal repo for testing a prompt-only Future-Seed style adapter on the `Gated DeltaNet` sublayers of Qwen3.5.

Current scope:

- dense Qwen3.5 only
- prompt-only / prefill-only seed path
- deep-layer scalar seed adapter
- detached + RMS-normalized + clipped recurrent seed
- smoke run on a tiny randomly initialized `qwen3_5` model
- awkward-task tiny SFT smoke with `baseline` vs `scalar-FS` switches

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

## Current Status

- `qwen3_5` tiny smoke is passing and confirms cross-layer seed injection on selected DeltaNet layers.
- Real `Qwen3.5-9B-Base` config/tokenizer probe is passing.
- Tiny awkward-task SFT smoke now supports both `baseline` and `scalar-FS`:
  - baseline clean run: `runs/qwen35-train-smoke-baseline-clean-20260407T071851Z`
  - scalar-FS clean run: `runs/qwen35-train-smoke-clean-20260407T071851Z`
- At this stage both tiny runs land at the same exact-match (`0.0625` awkward / `0.0625` friendly). This is still a pipeline validation result, not a task-improvement claim.
- Low-memory pretrained loading flags are now wired in (`LOAD_DTYPE`, `LOW_CPU_MEM_USAGE`, `EVAL_LIMIT`), but a local `Qwen/Qwen3.5-0.8B` forward on this 16GB CPU machine still exited with `137`. Real pretrained runs should be moved to a higher-memory or GPU machine.

## Notes

- The target aistation root should be `/fangxueji/Projects/PG/future-seed-finetune`.
- This local implementation was developed on a machine where `/fangxueji` is read-only, so the equivalent local path was used during development.
- The repo now includes a local compatibility patch for the current `transformers main` `qwen3_5` constructor bugs, so tiny-model smoke can run directly on `qwen3_5`.
- Full `Qwen3.5-9B-Base` pretrained validation is supported, but it depends on downloading the full checkpoint and is expected to be run on a machine with enough RAM or GPU memory.
- Repo scripts now pin `uv`, Hugging Face, transformers, tokenizers, torch, and pip caches to `artifacts/cache/` so no runtime cache should land under `~`.
