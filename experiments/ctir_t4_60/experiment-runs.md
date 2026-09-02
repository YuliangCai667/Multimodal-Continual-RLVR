# CTIR experiment ledger

| ID | Run | Status | Hardware | Primary artifacts |
|:---|:---|:---|:---|:---|
| EXP-CTIR-001 | beta=0 equivalence | completed/passed | GPU 0–3 | `experiments/ctir_t4_60/correctness/beta0/` |
| EXP-CTIR-002 | beta>0 exact spectrum | completed/passed (Householder + FP64 SVD) | GPU 0–3 | `experiments/ctir_t4_60/correctness/spectrum/` |
| EXP-CTIR-003 | T4-CTIR-Dynamic-60 | completed/evaluated; strict PoC no-go | GPU 0–3 train; GPU 2–3 eval | `experiments/ctir_t4_60/summary.md` |
| EXP-CTIR-004 | T4-CTIR-DirectRotation-300 | queued; detached GPU watcher active | GPU 0–3 | `configs/ctir_t4_300_direct.yaml` |

Common local code state before migration: `/home/caiyuliang/mrcl_cpo`, `main`, original base commit `9429452cb536a9e713b73b91c0011b96df44962c`, dirty with pre-existing user changes plus CTIR implementation.

Publishing authorization changed on 2026-09-03: create a new private repository under `YuliangCai667`, discard the original CPO commit history, and publish one code-only initial commit. Never push to `MaolinLuo/CPO`; exclude weights and runtime artifacts.

## EXP-CTIR-001 — beta-zero equivalence

- Status: completed/passed at 2026-09-02 13:31 Asia/Shanghai in tmux session `ctir_t4_60_20260902_132417`.
- Purpose: use the second, nonzero Puzzle optimizer update with CTIR enabled and forced beta 0; verify the interceptor is a strict post-optimizer no-op.
- Start: exact T3-final weights; fresh T4 optimizer.
- Protocol: initialize Trainer and cosine warmup scheduler with the formal `max_steps=60`, `warmup_ratio=0.1`; execute the zero-LR first optimizer step; use a CTIR debug stop after optimizer step 2 and validate only step 2. Four GPUs, force beta 0. This preserves the formal run's scheduler and first-step optimizer-state evolution exactly.
- Command: `bash scripts/CTIR-T4-60/run_correctness.sh beta0`
- Pass gate: protected final update relative error at or below `1e-5`; implementation also records whether FP32/BF16 rewrite was skipped.
- Result: step 1 was the expected zero-LR update. At step 2, `raw_update_fro_norm=0.02445651134`, beta-zero relative error `0.0`, `frob_ratio=1.00000000025`, and `new_descent_ratio=1.0`. The run stopped after step 2 as requested and passed the tightened finite/nonzero gate.

### Preserved failed attempts

- `beta0_failed_device_map_20260902_1239`: failed before probe-model load because the independent rank-0 model inherited Trainer's ZeRO-3 context while also passing `device_map`. Fixed by a scoped non-ZeRO load context followed by an explicit device transfer.
- `beta0_failed_rng_api_20260902_1245`: passed independent model loading, then failed while recording RNG metadata because this PyTorch build has no `torch.cuda.initial_seed_all`. Fixed by recording each rank's CUDA generator seed and hashed Python/NumPy/Torch RNG states.
- `beta0_failed_noncontiguous_20260902_1247`: completed all 32 full-CoT probes, then failed broadcasting the transposed right singular frame because NCCL requires contiguous tensors. Fixed at the shared broadcast boundary and verified with a one-rank NCCL regression test.
- `beta0_invalid_zero_delta_20260902_1254`: reached the Trainer optimizer callbacks but observed zero delta because Accelerate performs the real DeepSpeed step inside `accelerator.backward`; the Trainer callbacks surround a no-op optimizer wrapper. Replaced with an interceptor around the actual `DeepSpeedEngine.step` boundary.
- The 2026-09-02 13:05 retry proved the real-engine interceptor executes, but the first step still had zero delta because `warmup_ratio=0.1` with `max_steps=1` resolves to one warmup step and initial LR 0. The strengthened gate rejected `raw_update_fro_norm=0` and NaN ratios as intended.
- These attempts are invalid for scientific comparison and are retained only as engineering evidence; none produced a completed optimizer step.

## EXP-CTIR-002 — exact full-spectrum preservation

- Status: completed/passed at 2026-09-02 15:03 Asia/Shanghai in tmux session `ctir_t4_60_20260902_145613`.
- Purpose: force beta 1 and, on the second nonzero optimizer update, compare every singular value for one L9 attention q-projection and one L9 MLP down-projection.
- Start/protocol: same max-60 scheduler initialization and debug stop after step 2 as EXP-CTIR-001, except force beta 1. Redirect uses sequential Householder reflectors. The checker converts the FP32 raw/redirected deltas to FP64 before `svdvals`. Formal training does not repeat this FP64 SVD.
- Command: `bash scripts/CTIR-T4-60/run_correctness.sh spectrum`
- Pass gate: FP64 full-spectrum relative error and Frobenius-ratio error at or below `1e-5` for both sampled matrices. If this fails, stop and report; do not relax the gate and do not launch EXP-CTIR-003.
- User protocol 2026-09-02: no extra FP64 spectrum spot-check on the first real β>0 formal update; if the FP64 gate missed `1e-5`, stop rather than relax.
- Result: both sampled matrices passed. L9 `q_proj` spectrum relative error `1.73e-9`, Frobenius ratio `1-1.74e-10`. L9 `down_proj` spectrum relative error `1.24e-9`, Frobenius ratio `1+1.79e-12`. Online Householder certificate at step 2: `max_spectrum_error=5.09e-7`.

### Preserved failed attempts

- `spectrum_fp32_svd_failed_20260902_1332`: completed step 2 with force beta 1, but logged FP32 SVD. Relative spectrum errors were `1.15e-4` (L9 q_proj) and `1.64e-4` (L9 down_proj); Frobenius ratios `0.999983` and `1.000048`. Invalid as a Householder/FP64 gate.

## EXP-CTIR-003 — T4-CTIR-Dynamic-60

- Status: completed. Training ended at 2026-09-02 16:46 Asia/Shanghai at the intended step 60 (`train_runtime=6046.8s`); official evaluation ended at 2026-09-02 20:49:40 Asia/Shanghai with all 28 checkpoint/task rows present and finite.
- Purpose/idea: IDEA-CTIR-001; assess Navigation retention, Puzzle plasticity, online harm reduction, and tangent staleness.
- Protocol: `experiments/ctir_t4_60/config.yaml`; seed 42, `data_seed=None`, `full_determinism=False`; 60 optimizer steps; checkpoints every 10; no mid-run retuning. Training stays FP32; per-step logs use the Householder orthogonality-plus-Frobenius certificate only. No `--ctir_exact_spectrum_check`.
- Command: `bash scripts/CTIR-T4-60/run.sh`
- Artifacts: `experiments/ctir_t4_60/logs/formal_training.log`; `experiments/ctir_t4_60/logs/step_metrics.jsonl`; checkpoints under `checkpoints/Qwen3-VL-4B/CTIR-T4-60/training/Puzzle`
- Evaluation: official `VLMInference` + `eval.py`, data-parallel shards on GPU 2–3 only (restarted 2026-09-02 17:45 after the user requested the last two GPUs; resumed 2026-09-02 ~19:34 after step30 vLLM memory-gate crash). `temperature=0`, `top_p=1`, `repetition_penalty=1.05`, `max_completion_length=2048`, `batch_size=64`, `MRCL_GPU_MEMORY_UTILIZATION=0.6`. Checkpoints: step0 T3-final Puzzle only (MedBook/Navigation/We-Math2 reused from `T3_step300`); step10/20/30/40/50/60 on all four seen tasks. Command: `CUDA_VISIBLE_DEVICES=2,3 bash scripts/CTIR-T4-60/run_eval.sh`. Output: `experiments/ctir_t4_60/eval/performance.csv`. Completed step0/step10/step20 results are kept; incomplete step30 shards from the 18:03 crash were discarded and rerun.
- Decision criterion: Navigation recovery @60 >= +5 points and Puzzle cost @60 <= 3 points, together with the five mechanism conditions in the frozen plan.
- Result: at step 60, CTIR vs historical vanilla was MedBookVQA `+1.70` points, Navigation `+4.60`, We-Math2 `+1.50`, and Puzzle `-10.25`. Navigation missed its recovery gate by `0.40` point and Puzzle exceeded its allowed cost by `7.25` points, so the strict PoC decision is no-go.
- Mechanism result: among 59 nonzero updates, raw old harm was positive on 36 and beta was nonzero on the same 36. CTIR lowered all 36 harmful updates and reduced their summed proxy by `80.4%`; every nonzero update met the 0.90 Puzzle one-step descent constraint (minimum `0.9008`). Exact FP64 spectrum errors were `1.73e-9` and `1.24e-9`; the maximum online orthogonal-map certificate was `6.83e-7`. Mean tangent alignment vs step 0 reached `0.9707` at step 55 (minimum matrix `0.8530`).
- Artifacts/result narrative: `experiments/ctir_t4_60/summary.md`; figures under `experiments/ctir_t4_60/figures/`; machine-readable analysis at `experiments/ctir_t4_60/eval/analysis_summary.json`.
- Evaluation recovery history: evaluation first moved to GPU 2–3 at user request, then a step-30 vLLM startup failed its memory gate under contention. The incomplete shards were not used; the resumed run regenerated step 30 and completed steps 30–60. Final task totals match the official datasets.
- Confounders: historical comparison is not a newly run paired same-seed control under this exact dirty code state, and it also uses a materially different LR schedule. CTIR initialized cosine decay with `max_steps=60`, giving 6 warmup steps; historical vanilla is the first 60 steps of a `max_steps=300` run, giving 30 warmup steps. The summed logged LR over steps 1–60 is `1.5000e-4` for CTIR and `2.2591e-4` for historical vanilla. CTIR/vanilla LR mass ratios are `4.08` over steps 1–10, `1.53` over steps 1–30, and `0.21` over steps 31–60. Only one CTIR training seed exists. Timing claims are invalid under shared-GPU contention.
- Interpretation / next gate: the frozen historical-reference gate is a strict no-go, and the local proxy mechanism passed, but the behavioral retention–plasticity effect of CTIR is causally unresolved. Do not extend v1 to 300 steps. The first required behavioral control is a paired 60-step vanilla run with the exact CTIR scheduler/seed/config. Separately, IDEA-CTIR-002 identifies a nonlocal Householder lift that should be corrected before interpreting beta as rotation strength.

## EXP-CTIR-004 — T4-CTIR-DirectRotation-300

- Status: queued at 2026-09-03 01:38:58 Asia/Shanghai. A detached watcher is active; the formal run has not started because GPUs 0–3 are occupied by other users.
- Purpose/idea: test IDEA-CTIR-002 over the complete T4 stage after removing the Householder complement gauge and changing the controller from unconstrained old-harm minimization to closest feasible isospectral transport.
- Start/protocol: exact T3-final model; fresh Puzzle optimizer; four GPUs; 300 optimizer steps; checkpoints every 30 steps; seed 42; `data_seed=None`; `full_determinism=False`; 30-step warmup implied by `warmup_ratio=0.1`; all remaining GRPO settings match the existing vanilla T4-300 run.
- Geometry: 32 Navigation probes frozen with seed 142, full `full_text_only_thought` teacher-forced assistant-token NLL, rank-8 tangent and raw dominant frames, refresh every five optimizer steps, protected L9–26 attention/MLP matrices.
- Transport: identity-connected principal-plane direct rotations on both sides of the complete FP32 AdamW master delta. The maps are identity outside the source/target principal planes and preserve the full raw singular spectrum numerically. FP32 frames are re-orthonormalized and Procrustes-aligned in the small control space before constructing the maps, preventing distributed/frame Gram error from weakening orthogonality.
- Controller: global beta grid `[0, .25, .5, .75, 1]`; hard new-task descent ratio 0.90; lexicographic positive-old-constraint violation then actual full-update Frobenius distance.
- Comparator: the existing vanilla T4-300 trajectory has the same T3 start, seed 42, `data_seed=None`, world size 4, batch/accumulation semantics, LR `5e-6`, 300-step cosine schedule, warmup ratio 0.1, and checkpoints every 30 steps. It is therefore the schedule-paired comparator; no redundant vanilla rerun is planned unless a later audit finds a material mismatch.
- Commands/config: `scripts/CTIR-T4-300/run.sh`; `scripts/CTIR-T4-300/launch_when_gpus_free.sh`; `configs/ctir_t4_300_direct.yaml`.
- Queue runtime: tmux session `ctir_t4_300_wait` on socket `/tmp/ctir_t4_300_wait.sock`; worker owned by `caiyuliang`; log `experiments/ctir_t4_300/logs/gpu_waiter.log`. Trigger requires physical GPUs 0–3 each to have zero compute processes and at most 512 MiB allocated, sampled every two seconds and confirmed again one second later. Once confirmed, the watcher immediately `exec`s the canonical formal launcher in the same detached session.
- Queue verification: session and worker were live after launch. The first observation was correctly classified busy: GPU0 `48294 MiB/2 processes`, GPU1 `72826 MiB/2`, GPU2 `24367 MiB/1`, GPU3 `29209 MiB/1`. This state is queue evidence, not experiment start evidence.
- Operational limit: the watcher minimizes the check-to-launch gap but cannot reserve GPUs against a simultaneous launch by another user. No automatic retry is allowed after a partial formal startup; any such race must be recorded and inspected before retrying.
- Acceptance evidence before launch: CPU synthetic tests verify complete-spectrum/Frobenius preservation, exact beta-zero behavior, target-frame mapping, locality outside principal planes, noisy-FP32-frame re-orthogonalization, and controller selection semantics. GPU2/3 matrix tests on both Qwen projection orientations used less than 0.48 GiB each; exact sampled full-spectrum relative errors were `4.99e-9` and `5.12e-9`, maximum Frobenius-ratio error was `2.23e-10`, and maximum direct-map orthogonality bound was `7.83e-8`.

### Two-GPU implementation integration preflight

- Scope: implementation-only, not a performance result. Physical GPUs 2 and 3, world size 2, forced beta 0.5, full 32-probe Navigation tangent, and a callback stop after optimizer step 2. The scheduler was initialized with the formal `max_steps=300` and `warmup_ratio=0.1`; seed 42 and `data_seed=None` were recorded. The run was detached and all train/data processes were owned by `caiyuliang`.
- Step 1 was the expected zero-LR update. On the nonzero step 2, raw and redirected update norms were `0.00497915734` and `0.00497916189` (`frob_ratio=1.000000914`), relative update distance was `0.0980453`, and raw/redirected cosine was `0.995194`.
- With forced beta 0.5, new-task descent retained `0.951887` of raw, above the 0.90 floor. The old-task directional score changed from `-5.85270e-5` to `-1.07090e-4`; both were already feasible (`<=0`). Candidate beta 1 failed the new-task constraint (`0.882668`), while beta 0, 0.25, 0.5, and 0.75 passed. Therefore an unforced closest-feasible controller would correctly keep beta 0 on this particular already-safe update; beta 0.5 was forced only to exercise the redirect path.
- This preflight exposed an online orthogonality-bound maximum of `2.43e-3` from small Gram errors in transported FP32 frames. The final implementation now re-orthonormalizes and re-aligns frames in FP64 before constructing the 16-dimensional control rotation. An adversarial noisy-frame regression reduced the same class of bound from `8.79e-3` to `4.78e-8`; the post-fix GPU matrix checks above passed. The expensive end-to-end two-step run was not repeated because the change is isolated to that tested small-space construction.
- Artifacts are local-only: `experiments/ctir_t4_300/integration_2gpu_beta05_bound23/`. They are excluded from the code repository.
