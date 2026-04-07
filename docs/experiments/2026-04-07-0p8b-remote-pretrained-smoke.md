## Qwen3.5-0.8B Remote Pretrained Smoke

- Date: `2026-04-07`
- Repo: `future-seed-finetune`
- Machine: remote GPU host with `2 x NVIDIA A100-SXM4-80GB`
- Purpose: verify that the scalar Future-Seed patch can run on a real pretrained `Qwen3.5-0.8B` checkpoint with repo-root caches only, then compare against a frozen disabled control

### Environment

- Repo checkout on remote: `/fangxueji/Projects/PG/future-seed-finetune`
- Commit for scalar-FS rerun: `26b584ed950fd35e16599fa685469651a9560c35`
- Commit for frozen disabled control: `7ff4466a7f53ab4bcef899edd99b7f66a824ca7d`
- Repo-root caches/tools:
  - `artifacts/tools/uv/uv`
  - `artifacts/cache/uv`
  - `artifacts/cache/hf`

### Runs

- Validate pretrained scalar-FS:
  - `runs/qwen35-0p8b-validate-pretrained-bf16-remote-rerun2-20260407T124255Z`
  - `exit_status = 0`
- Train pretrained scalar-FS smoke:
  - `runs/qwen35-0p8b-train-pretrained-smoke-rerun2-20260407T124305Z`
  - `exit_status = 0`
- Train pretrained frozen disabled control:
  - `runs/qwen35-0p8b-train-pretrained-baseline-control-rerun3-20260407T125253Z`
  - `exit_status = 0`

### Result

- The real pretrained checkpoint now runs end-to-end on GPU.
- The scalar-FS path is active on pretrained weights:
  - `device = cuda`
  - `trainable_parameter_count = 17`
  - `train_runtime.injection_count = 11`
- The validate run also confirms prompt-time seed injection on a deep-layer subset:
  - `captured_layers = [16, 17, 18, 20, 21, 22]`
  - `injected_layers = [17, 18, 21, 22]`
  - `injection_count = 4`
- The frozen disabled control now runs without unfreezing the backbone:
  - `trainable_parameter_count = 0`
  - `effective_train_steps = 0`

### Quality

- On the small `eval_limit = 8` awkward/friendly split:
  - scalar-FS: `awkward = 0.0`, `friendly = 0.0`
  - frozen disabled control: `awkward = 0.0`, `friendly = 0.0`

### Load Mismatch

Both pretrained runs emit the same load report pattern:

- `UNEXPECTED`
  - `model.layers.{0...22}.linear_attn.in_proj_qkv.weight`
  - `model.layers.{0...22}.linear_attn.in_proj_a.weight`
  - `model.layers.{0...22}.linear_attn.in_proj_b.weight`
  - `model.layers.{0...22}.linear_attn.in_proj_z.weight`
  - `model.layers.{0...23}.mlp.gate_proj.weight`
  - `model.layers.{0...23}.mlp.up_proj.weight`
  - `model.layers.{0...23}.mlp.down_proj.weight`
- `MISSING`
  - `model.layers.{0...22}.linear_attn.in_proj_qkvz.weight`
  - `model.layers.{0...22}.linear_attn.in_proj_ba.weight`
  - `model.layers.{0...23}.mlp.gate.weight`
  - `model.layers.{0...23}.mlp.experts.gate_up_proj`
  - `model.layers.{0...23}.mlp.experts.down_proj`
  - `model.layers.{0...23}.mlp.shared_expert.*`

### Conclusion

- The pipeline is no longer blocked on environment, cache placement, model transfer, or GPU execution.
- The scalar Future-Seed patch is live on a real pretrained `Qwen3.5-0.8B` checkpoint.
- The current blocker for meaningful task comparison is checkpoint/module mismatch with the current `transformers` `qwen3_5` implementation.
- The next useful step is not more smoke on this exact mismatch; it is finding a checkpoint/config pair that loads cleanly enough for quality comparisons.
