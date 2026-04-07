# future-seed-finetune

Minimal repo for testing a prompt-only Future-Seed style adapter on the `Gated DeltaNet` sublayers of Qwen3.5.

Current scope:

- dense Qwen3.5 only
- prompt-only / prefill-only seed path
- deep-layer scalar seed adapter
- detached + RMS-normalized + clipped recurrent seed
- smoke run on a tiny randomly initialized `qwen3_5` model

The first goal is not end-task quality. It is to verify that:

1. Qwen3.5 `prefill` no longer hardcodes `initial_state=None` for selected `Gated DeltaNet` layers.
2. A tiny trainable scalar `alpha_l` can gate the cross-layer seed path.
3. Only the scalar seed parameters can remain trainable while the backbone stays frozen.

## Layout

- `future_seed_finetune/qwen35_scalar_fs.py`: patching and adapter logic
- `scripts/smoke_qwen35_scalar_fs.py`: tiny random-model smoke
- `scripts/download_qwen35_probe_assets.py`: download Qwen3.5 probe assets or full weights
- `scripts/validate_qwen35_prefill.py`: validate real Qwen3.5 config or pretrained checkpoint
- `setup.sh`: install deps with `uv`
- `down.sh`: create local artifact/cache layout
- `run.sh`: run smoke and write metadata/results under `runs/`

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

## Notes

- The target aistation root should be `/fangxueji/Projects/PG/future-seed-finetune`.
- This local implementation was developed on a machine where `/fangxueji` is read-only, so the equivalent local path was used during development.
- The repo now includes a local compatibility patch for the current `transformers main` `qwen3_5` constructor bugs, so tiny-model smoke can run directly on `qwen3_5`.
- Full `Qwen3.5-9B-Base` pretrained validation is supported, but it depends on downloading the full checkpoint and is expected to be run on a machine with enough RAM or GPU memory.
