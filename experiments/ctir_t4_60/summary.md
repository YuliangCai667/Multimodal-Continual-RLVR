# CTIR T4 first-60 result summary

- Experiment: `EXP-CTIR-003` / `T4-CTIR-Dynamic-60`
- Terminal state: training completed at step 60; official evaluation completed 2026-09-02 20:49:40 Asia/Shanghai.
- Decision: **strict historical-reference PoC no-go**. The mechanism checks passed, but the predeclared retention–plasticity gate did not. This is not a causal CTIR-vs-vanilla result because the two schedulers are not paired.
- Primary data: `eval/performance.csv`, `logs/step_metrics.jsonl`, and `logs/tangent_refresh.jsonl`.

## Official evaluation

All values are accuracy percentages. Step 0 is the exact shared T3-final checkpoint. Historical vanilla is available only at steps 30 and 60, so its dashed figure segments must not be read as measured intermediate points.

| Task | step 0 | CTIR 30 | vanilla 30 | delta 30 | CTIR 60 | vanilla 60 | delta 60 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| MedBookVQA | 81.83 | 80.48 | 79.12 | +1.36 pp | 81.32 | 79.63 | +1.70 pp |
| Navigation | 67.65 | 56.99 | 59.38 | -2.39 pp | 56.07 | 51.47 | +4.60 pp |
| We-Math2 | 62.54 | 61.00 | 60.29 | +0.71 pp | 61.63 | 60.12 | +1.50 pp |
| Puzzle | 35.00 | 36.50 | 36.75 | -0.25 pp | 35.75 | 46.00 | -10.25 pp |

At step 60, the two evaluated models were scored on exactly the same examples. Relative to historical vanilla, CTIR had 65 improved and 40 worsened Navigation examples (net +25/544), but 25 improved and 66 worsened Puzzle examples (net -41/400). This paired example accounting describes the two fixed checkpoints; it does not remove training-run variance.

The predeclared go/no-go criteria were:

| Criterion | Required | Observed | Result |
|:---|---:|---:|:---|
| Navigation recovery at step 60 | at least +5.00 pp | +4.60 pp | fail by 0.40 pp |
| Puzzle cost at step 60 | at most 3.00 pp | 10.25 pp | fail by 7.25 pp |

Puzzle briefly reached 38.75% at steps 40 and 50, then ended at 35.75%. The late drop is a real deterministic-evaluator observation, but a paired rerun is needed before assigning it causally to CTIR.

### Critical comparator mismatch

CTIR used a cosine schedule initialized with `max_steps=60` and `warmup_ratio=0.1` (6 warmup steps). Historical vanilla is the first 60 steps of a 300-step run using the same warmup ratio (30 warmup steps). Consequently, CTIR had 4.08 times as much cumulative LR in steps 1–10, but only 0.21 times as much in steps 31–60; across all 60 steps its LR sum was 66.4% of historical vanilla. This schedule mismatch can itself produce stronger early drift followed by weaker late Puzzle acquisition. The `+4.60` Navigation and `-10.25` Puzzle differences remain valid fixed-checkpoint descriptions, but they must not be interpreted as the causal effect of CTIR.

## Mechanism checks

Step 1 is the expected zero-update warmup step. Statistics below use the 59 nonzero updates.

| Check | Observation | Assessment |
|:---|:---|:---|
| Raw old-task harm occurs often | positive on 36/59 updates (61.0%) | pass |
| CTIR lowers predicted old harm | lowered all 36 harmful updates; summed harmful proxy reduced 80.4%; 7 became non-harmful | pass |
| Puzzle one-step descent constraint | minimum ratio 0.9008; all 59/59 at least 0.90; mean 0.9518 | pass |
| Nontrivial intervention | beta above zero on 36/59 updates (61.0%); counts: 0:23, 0.25:7, 0.5:25, 0.75:3, 1.0:1 | pass |
| Full-delta isospectrality | exact FP64 full-SVD errors 1.73e-9 and 1.24e-9; online max orthogonal certificate 6.83e-7; max Frobenius-ratio error 1.48e-9 | pass |

Although every per-step descent-ratio gate passed, the sum of the logged protected-matrix Puzzle descent proxies retained only 93.8% of raw, while final Puzzle accuracy paid 10.25 points. This is direct evidence that the v1 local first-order constraint is not a sufficient behavioral-plasticity guarantee.

## Tangent and probe behavior

- The 32-sample teacher-forced full-CoT Navigation probe NLL rose from 1.7390 at step 0 to 1.8153 at step 55 (+4.39%). CTIR reduced predicted incremental harm but did not freeze Navigation behavior.
- Across 126 protected matrices, mean left/right subspace alignment with step 0 was 0.9707 at step 55; the 10th percentile was 0.9490 and the worst matrix was 0.8530.
- Therefore the geometry drift is gradual and concentrated: a subset of matrices rotates materially, while the average tangent remains fairly close to step 0. This motivates testing static protection, but does not yet prove that five-step refresh is necessary.

## Interpretation

**Established facts:** the true full AdamW deltas were redirected isospectrally; beta was active on a majority of nonzero steps; the cached Navigation-gradient proxy was strongly reduced; all local Puzzle descent constraints passed; final Navigation was 4.60 points above the historical checkpoint and final Puzzle was 10.25 points below it.

**Current interpretation:** CTIR v1 is isospectral and the old-task proxy is directionally active, but the present experiment cannot determine its causal retention–plasticity effect because the comparator schedule differs. Within the CTIR trajectory, retaining 90% of a one-step gradient inner product was not sufficient to predict long-horizon probe/Puzzle behavior. A separate code-level diagnosis also shows that beta controls target-frame displacement but not the distance of the sequential-Householder orthogonal lift from identity; this makes the current implementation nonlocal even while preserving the complete spectrum. See `ideas.md`, IDEA-CTIR-002.

**Important limitation:** the comparator is the existing historical vanilla trajectory, not a newly executed paired control under this exact dirty code state, and its `max_steps=300` scheduler differs from CTIR's `max_steps=60` scheduler. There is one training seed. The result is sufficient only for the frozen historical-reference gate, not for a causal or statistical claim about CTIR.

## Decision gate

Do not extend this v1 configuration to 300 steps. If the goal is to diagnose rather than stop, the smallest next experiment is one paired 60-step vanilla run from the same T3-final weights with the exact recorded seed/RNG/training config. Only if that confirms the trade-off should the user choose among a stronger Puzzle functional guard, a tighter first-order constraint, or a narrower protected scope before spending compute on static/projection comparisons.

## Figures and analysis artifacts

- `figures/performance_curves.png`
- `figures/nav_recovery_vs_puzzle_cost.png`
- `figures/old_harm_by_step.png`
- `figures/beta_by_step.png`
- `figures/tangent_rotation.png`
- `figures/spectrum_error.png`
- `eval/analysis_summary.json`
- `eval/step60_paired_differences.csv`

Figures are reproducible with `python scripts/CTIR-T4-60/plot_results.py`.

## Follow-up method revision (2026-09-03)

The no-go above remains the correct conclusion for the frozen v1 experiment. It is not being reinterpreted. A v2 implementation now replaces the nonlocal sequential-Householder lift with an identity-connected, FP64-frame-stabilized principal-plane direct rotation and changes beta selection to closest feasible isospectral transport. A targeted two-GPU implementation preflight completed through the second optimizer step; post-fix matrix-level GPU checks preserve sampled complete spectra to about `5e-9`. The next performance experiment is a complete 300-step T4 run using the same 300-step scheduler and seed configuration as the existing vanilla trajectory, eliminating the v1 schedule mismatch; it is pending four-GPU availability. See `IDEA-CTIR-002` and `EXP-CTIR-004` in the adjacent research records.
