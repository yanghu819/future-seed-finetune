## Qwen3.5-0.8B Local Validate

- Date: `2026-04-07`
- Machine: local 16GB CPU-only machine
- Model: `Qwen/Qwen3.5-0.8B`
- Purpose: verify whether a real pretrained `qwen3_5` checkpoint can complete a local forward with the Future-Seed patch

### Asset

- Full model downloaded to `artifacts/models/qwen3_5_0p8b_full`

### Runs

- `qwen35-0p8b-validate-pretrained-20260407T075006Z`
  - `torch_dtype=float32`
  - `exit_status=137`
- `qwen35-0p8b-validate-pretrained-bf16-20260407T075322Z`
  - `torch_dtype=bfloat16`
  - `low_cpu_mem_usage=true`
  - `exit_status=137`

### Conclusion

- The repo now supports low-memory pretrained loading flags.
- On this local machine, even the smaller real pretrained `Qwen3.5-0.8B` forward is still killed by the system.
- Real pretrained training or forward validation should be moved to a larger-memory or GPU machine.
