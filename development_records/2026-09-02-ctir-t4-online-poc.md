# CTIR T4 online proof of concept

## Motivation

The prior analysis treated a multi-step checkpoint delta as one update. This change adds a real online intervention at each Puzzle AdamW/DeepSpeed optimizer boundary so the experiment can test whether current-state Navigation-sensitive directions can redirect actual updates without changing their complete singular spectrum.

## Changes

- Added a CTIR trainer path and CLI/config surface without changing the default GRPO/CL trainer paths.
- Froze 32 Navigation probes with seed 142 and reused the Figure-B teacher-forced loss over each sample's complete `full_text_only_thought`; only valid assistant target tokens contribute and each sample is length-normalized.
- Added current-model Navigation tangent refreshes every five optimizer steps for the selected layer-9–26 attention/MLP matrices.
- Intercepted the real `DeepSpeedEngine.step` result, gathered one protected full matrix at a time under ZeRO-3, and measured the complete FP32 master-parameter delta.
- Used rank-8 raw singular frames only to choose Householder left/right rotations. The rotations are applied to the complete raw delta; no optimizer delta spectrum is rank-truncated.
- Selected beta from the frozen candidate grid by minimum Navigation harm subject to at least 90% of raw Puzzle descent, with strict beta-zero fallback.
- Added per-step mechanism logging, tangent-refresh logging, detached correctness/formal launchers, official checkpoint evaluation, and reproducible result plotting.

## Evaluation

- Correctness artifacts: `experiments/ctir_t4_60/correctness/beta0/` and `experiments/ctir_t4_60/correctness/spectrum/`.
- Formal run: `experiments/ctir_t4_60/logs/` and `checkpoints/Qwen3-VL-4B/CTIR-T4-60/training/Puzzle/`.
- Official evaluation and figures: `experiments/ctir_t4_60/eval/`, `experiments/ctir_t4_60/figures/`, and `experiments/ctir_t4_60/summary.md`.
- The result is mechanically valid but a behavioral no-go: step-60 Navigation was +4.60 points and Puzzle was -10.25 points relative to historical vanilla.

## Pitfalls

- With the formal 60-step cosine/warmup schedule, optimizer step 1 has zero learning rate. Correctness therefore stops after step 2 and measures the second, nonzero update; it does not initialize a two-step scheduler.
- FP32 SVD itself produced errors above the `1e-5` gate on large matrices. The exact correctness comparison converts the already-computed FP32 deltas to FP64 before `svdvals`; formal training uses cheap orthogonality and Frobenius certificates instead of repeated full SVDs.
- `raw_dominant_rank=8` controls the rotation frames only. It must not be interpreted as a rank-8 optimizer update.
- The final evaluator run is complete, but its log retains two failed startup attempts caused by GPU contention. Incomplete shards were discarded before the successful resume.
- The historical vanilla comparison is not a newly run paired control under the exact current dirty worktree, so it cannot establish a method-level causal claim from one seed.

## Validation and limits

- Beta-zero correctness: relative update error `0.0` on the second optimizer step.
- Exact full-spectrum correctness: relative errors `1.73e-9` (L9 `q_proj`) and `1.24e-9` (L9 `down_proj`) with FP64 measurement; both Frobenius ratios were within `2e-10` of one.
- Formal execution reached `global_step=60`, produced checkpoints 10/20/30/40/50/60, logged 60 optimizer steps and 12 × 126 tangent records, and generated all 28 official evaluation rows.
- No 300-step run, paired vanilla, static rotation, projection baseline, or multi-seed study has been validated.

Repository state: branch `main`, base commit `9429452cb536a9e713b73b91c0011b96df44962c`, intentionally dirty. This record and the implementation remain local and uncommitted; nothing was pushed.
