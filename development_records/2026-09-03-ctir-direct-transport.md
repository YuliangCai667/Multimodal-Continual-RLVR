# CTIR direct isospectral transport

## Motivation

The first online CTIR implementation preserved the singular spectrum of the complete optimizer delta, but lifted target rank-8 frames with sequential Householder reflections. A nontrivial Householder remains spectral-distance 2 from the identity even when the requested frame movement is arbitrarily small, and the chosen reflector chain can alter the raw delta tail in directions not specified by the rank-8 control problem. The controller also minimized old-task harm without charging for distance from the raw optimizer update.

## Changes

- Replaced the Householder lift with the identity-connected direct rotation on the source/target principal planes. It maps each raw dominant frame to its Procrustes-aligned target, acts as identity on the orthogonal complement, and approaches identity continuously as the target approaches the source.
- Re-orthonormalized transported FP32 frames and restored their Procrustes gauge in FP64 inside the small control space before forming the direct rotation. This makes the finite-precision map match its orthogonal/ispectral theory even when distributed frame Gram matrices are slightly imperfect.
- Kept the complete two-sided update `R_L @ delta_raw @ R_R.T`. Rank 8 only extracts the dominant raw/tangent frames and never truncates `delta_raw`.
- Added actual complete-update Frobenius distance, raw/redirected update cosine, direct-map operator distance, and maximum principal-angle logging.
- Replaced minimum-old-harm beta selection with a closest-feasible global controller: retain at least 90% of raw new-task descent, minimize positive old-harm constraint violation, and then minimize actual update distance. If the raw new-task score is nonpositive, preserve beta zero rather than applying a sign-ambiguous ratio rule.
- Added per-step diagnostics for all five beta candidates and explicit controller feasibility/status fields.
- Isolated probe-model construction and tangent estimation from Python, NumPy, CPU Torch, and current-device CUDA training RNG streams.
- Added the four-GPU, 300-step T4 launcher and frozen protocol config. The existing vanilla T4-300 trajectory matches the core seed, batch, optimizer, LR, warmup, scheduler, and checkpoint cadence and is reused as the schedule-paired comparator.
- Bound the formal launcher explicitly with `--include localhost:0,1,2,3`. A first preflight attempt showed that DeepSpeed ignores an inherited `CUDA_VISIBLE_DEVICES=2,3` when `--num_gpus 2` is present; it was stopped before any optimizer step, then relaunched correctly on physical GPUs 2–3. The formal launcher no longer admits that ambiguity.
- Added `scripts/CTIR-T4-300/launch_when_gpus_free.sh`, a one-shot detached tmux watcher for the shared server. It requires zero compute processes and at most 512 MiB used on each physical GPU 0–3, polls every two seconds, reconfirms after one second, and then replaces itself with the canonical formal launcher. It logs only state transitions and five-minute heartbeats to the ignored runtime log.

## Focused checks

- `env PYTHONPATH=. conda run -n trlQwen python scripts/CTIR-T4-60/test_orthogonal_redirect.py`
  - passed for rectangular complete deltas in both orientations;
  - full singular spectra and Frobenius norms preserved within the existing tolerance;
  - beta zero is exact;
  - direct maps reach the target frames and fix directions outside their active principal planes.
- `env PYTHONPATH=. conda run -n trlQwen python scripts/CTIR-T4-60/test_controller.py`
  - passed jointly feasible, infeasible-grid, already-safe raw update, and nonpositive raw-descent cases.
- Python compilation and `bash -n scripts/CTIR-T4-300/run.sh` passed.
- A detached two-GPU, two-step integration preflight used physical GPUs 2–3, all 32 full-CoT probes, the formal 300-step/0.1-warmup scheduler, seed 42, and forced beta 0.5. The nonzero step retained `95.19%` of raw new-task descent, improved an already-negative old-task directional score, preserved the global Frobenius norm to `9.14e-7` relative, and exercised the complete ZeRO-3 optimizer interception/writeback path.
- The integration preflight revealed a `2.43e-3` conservative orthogonality bound caused by imperfect FP32 input frames. After the FP64 re-orthonormalization fix, an adversarial regression reduced the corresponding bound from `8.79e-3` to `4.78e-8`.
- Post-fix GPU2/3 tests at `2560x9728` and `9728x2560` used less than 0.48 GiB per process. Sampled complete-spectrum FP64 relative errors were `4.99e-9` and `5.12e-9`; the largest map orthogonality bound was `7.83e-8`.
- `bash -n` passed for the GPU watcher, and its live `--status` probe correctly classified the occupied cards as `busy` with per-GPU memory/process counts. The detached tmux session and `caiyuliang`-owned worker were then verified.

## Experiment boundary

The two-step run above is an implementation preflight, not a shortened performance experiment. `EXP-CTIR-004` was queued in a detached watcher for a four-GPU 300-step Puzzle stage, stopped before launch at 2026-09-03 03:41:15 Asia/Shanghai, and explicitly requeued by the user at 04:55:15. The first queue attempt produced no formal-run data; the second was still waiting on occupied GPUs when verified. Runtime logs, checkpoints, and model weights are local artifacts and must not be pushed to GitHub. Poll-and-launch minimizes the scheduling window but cannot provide exclusive ownership against another user launching simultaneously; only a cluster scheduler or administrator-enforced exclusive mode can provide that guarantee.

## Publication boundary

The single-task direct-transport snapshot remains the private repository's new-history root on `main`. Publish the later T1–T5 multi-task extension as a separate commit on `exp/ctir-multitask-h100x-v4`, based on that private `main`; do not rewrite `main` and do not push to the original `MaolinLuo/CPO` remote. Include source, launch/configuration code, focused tests, and concise engineering/research records; exclude checkpoints, raw evaluations, generated figures, logs, sessions, and credentials.
