# future-seed-finetune

Minimal repo for testing a prompt-only Future-Seed style adapter on the `Gated DeltaNet` sublayers of Qwen3.5.

Current scope:

- dense Qwen3.5 only
- prompt-only / prefill-only seed path
- deep-layer scalar seed adapter
- detached + RMS-normalized + clipped recurrent seed
- smoke run on a tiny randomly initialized `qwen3_next` proxy model

The first goal is not end-task quality. It is to verify that:

1. Qwen3.5 `prefill` no longer hardcodes `initial_state=None` for selected `Gated DeltaNet` layers.
2. A tiny trainable scalar `alpha_l` can gate the cross-layer seed path.
3. Only the scalar seed parameters can remain trainable while the backbone stays frozen.

## Layout

- `future_seed_finetune/qwen35_scalar_fs.py`: patching and adapter logic
- `scripts/smoke_qwen35_scalar_fs.py`: tiny random-model smoke
- `setup.sh`: install deps with `uv`
- `down.sh`: create local artifact/cache layout
- `run.sh`: run smoke and write metadata/results under `runs/`

## Quickstart

```bash
bash ./setup.sh
bash ./down.sh
bash ./run.sh
```

## Notes

- The target aistation root should be `/fangxueji/Projects/PG/future-seed-finetune`.
- This local implementation was developed on a machine where `/fangxueji` is read-only, so the equivalent local path was used during development.
- The repo intentionally avoids downloading a full base model in the first iteration.
- The patch logic for `qwen3_5` is included, but the current `transformers main` snapshot still has constructor bugs in `Qwen3_5TextConfig` / `Qwen3_5TextRotaryEmbedding`. The smoke run therefore validates the same seed path on `qwen3_next`, which shares the same `Gated DeltaNet` prefill contract.
