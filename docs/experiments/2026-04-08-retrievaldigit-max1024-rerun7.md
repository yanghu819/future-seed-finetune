## RetrievalDigit MAX_LENGTH=1024 Rerun7

This experiment closes the main training bug from the earlier retrieval runs.

- The earlier `retrieval_digit_long` runs used the default `max_length=256`.
- That silently truncated the answer span out of the train batch on long prompts.
- The symptom was:
  - `skipped_empty_targets > 0`
  - `effective_train_steps = 0`
- The repo now exposes `MAX_LENGTH` through `run.sh`, and the remote rerun used `MAX_LENGTH=1024`.

### Runs

- Baseline frozen control:
  - `runs/qwen35-0p8b-train-pretrained-baseline-retrievaldigit-max1024-rerun2-20260408T064845Z`
- Deep-layer Future-Seed + delta adapter:
  - `runs/qwen35-0p8b-train-pretrained-deepfsdelta-retrievaldigit-max1024-rerun7-20260408T064845Z`

### Results

- Baseline:
  - `awkward = 1.0`
  - `friendly = 1.0`
  - `effective_train_steps = 0`
- DeepFS + delta:
  - `awkward = 1.0`
  - `friendly = 0.984375`
  - `effective_train_steps = 10`
  - `skipped_empty_targets = 0`
  - `skipped_nonfinite_losses = 0`
  - `last_grad_norm = 14.632787704467773`

### Interpretation

- This is the first retrieval-style run where the trainable FS path actually optimized real steps on pretrained `Qwen3.5-0.8B`.
- The `delta` path and longer context fixed the two main blockers:
  - device mismatch
  - empty-target truncation
- Quality still does not beat the frozen control.
- The current state is:
  - training pipeline is now technically valid
  - method quality is still neutral-to-slightly-worse on this benchmark
