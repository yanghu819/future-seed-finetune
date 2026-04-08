## Qwen3.5-0.8B Full Validation Rerun 6

- Date: `2026-04-08`
- Repo: `future-seed-finetune`
- Machine: remote GPU host with `2 x NVIDIA A100-SXM4-80GB`
- Purpose: compare the corrected deep-layer scalar Future-Seed setup against a frozen disabled control on the full synthetic awkward-task validation split

### Preconditions

- The earlier checkpoint mismatch was fixed by loading `Qwen3_5ForConditionalGeneration` instead of forcing the `0.8B` checkpoint through `Qwen3_5ForCausalLM`.
- Future-Seed is attached only to `model.language_model`.
- The pretrained default `start_layer` is now `-1`, which resolves to the last 8 text layers.

### Runs

- Deep-layer scalar-FS:
  - `runs/qwen35-0p8b-train-pretrained-deepfs-fullval-rerun6-20260408T011643Z`
  - `commit_sha = 75f0a663a341f8256e3d8278903c76698abda3b6`
  - `exit_status = 0`
- Frozen disabled control:
  - `runs/qwen35-0p8b-train-pretrained-baseline-fullval-rerun6-20260408T011643Z`
  - `commit_sha = 75f0a663a341f8256e3d8278903c76698abda3b6`
  - `exit_status = 0`

### Settings

- Model: `Qwen/Qwen3.5-0.8B`
- Dtype: `bfloat16`
- `low_cpu_mem_usage = true`
- `batch_size = 1`
- `max_steps = 10`
- `eval_limit = 0` (full validation)
- Task splits:
  - `valid_awkward = 64`
  - `valid_friendly = 64`

### Result

- Deep-layer scalar-FS:
  - `eval_awkward.exact_match = 1.0`
  - `eval_friendly.exact_match = 0.984375`
  - `trainable_parameter_count = 6`
  - `future_seed_parameters` are only:
    - `model.language_model.layers.16.linear_attn.fs_alpha`
    - `model.language_model.layers.17.linear_attn.fs_alpha`
    - `model.language_model.layers.18.linear_attn.fs_alpha`
    - `model.language_model.layers.20.linear_attn.fs_alpha`
    - `model.language_model.layers.21.linear_attn.fs_alpha`
    - `model.language_model.layers.22.linear_attn.fs_alpha`
  - `train_runtime.injection_count = 4`
- Frozen disabled control:
  - `eval_awkward.exact_match = 1.0`
  - `eval_friendly.exact_match = 1.0`
  - `trainable_parameter_count = 0`
  - `effective_train_steps = 0`

### Interpretation

- The corrected deep-layer Future-Seed setup is stable on real pretrained Qwen3.5 weights.
- It no longer catastrophically degrades the task the way the earlier shallow-FS version did.
- But on this synthetic awkward task, it still does not beat the frozen baseline.
- The remaining gap is small and only shows on the friendly split:
  - deep-layer FS misses `1 / 64`
  - frozen control misses `0 / 64`

### Conclusion

- The repo now contains a technically correct, end-to-end working deep-layer scalar Future-Seed scaffold for Qwen3.5.
- On the current synthetic benchmark, the method is neutral-to-slightly-negative rather than beneficial.
- The next meaningful step is not more tuning on this exact toy task; it is moving to a harder causally awkward benchmark where the baseline is not already saturated at or near `1.0`.
