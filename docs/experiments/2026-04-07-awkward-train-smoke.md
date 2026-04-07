## Awkward Train Smoke

- Date: `2026-04-07`
- Repo: `future-seed-finetune`
- Purpose: validate a minimal awkward-task SFT loop for `baseline` vs `scalar-FS` on tiny `qwen3_5`

### Runs

- Baseline: `runs/qwen35-train-smoke-baseline-clean-20260407T071851Z`
- Scalar-FS: `runs/qwen35-train-smoke-clean-20260407T071851Z`

### Result

- Both runs finished with `exit_status = 0`
- Both runs reached the same exact-match:
  - `eval_awkward = 0.0625`
  - `eval_friendly = 0.0625`
- Loss is slightly lower with scalar-FS:
  - baseline `loss_end = 11.766023635864258`
  - scalar-FS `loss_end = 11.764974594116211`

### What This Means

- The comparison switch is working.
- The scalar Future-Seed path is active during training:
  - `train_runtime.injection_count = 1`
  - `captured_layers = [1, 2]`
  - `injected_layers = [2]`
- On this tiny synthetic smoke, there is no task-level lift yet. This should be treated as pipeline validation only.

### Next Step

- Move from tiny random-init smoke to a real `Qwen3.5-9B-Base` small-step run on a causally awkward task.
