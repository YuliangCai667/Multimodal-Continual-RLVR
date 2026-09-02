# CTIR T4 research index

- Cycle: ongoing MRCL mechanism/method PoC; venue/deadline not specified.
- Direction: current-state tangent-aware isospectral update redirection during the T4 Puzzle forgetting shock.
- Objective: test whether isospectral redirection of real Puzzle optimizer steps can retain Navigation while preserving Puzzle plasticity, first in the 60-step mechanism PoC and then over the complete 300-step T4 stage.
- Repository: `/home/caiyuliang/mrcl_cpo`, branch `main`, base commit `9429452`; working tree intentionally dirty and uncommitted.

## Registry

| Type | Path | Role | Last material update |
|:---|:---|:---|:---|
| Protocol/config | `experiments/ctir_t4_60/config.yaml` | Frozen v1 settings and artifact paths | 2026-09-02 |
| Ideas | `experiments/ctir_t4_60/ideas.md` | v1 hypothesis plus gauge-fixed local isospectral transport candidate | 2026-09-02 |
| Experiment ledger | `experiments/ctir_t4_60/experiment-runs.md` | Correctness and formal-run lifecycle | 2026-09-02 |
| Result summary | `experiments/ctir_t4_60/summary.md` | Official performance, mechanism checks, interpretation, figures | 2026-09-02 |
| Engineering record | `development_records/2026-09-02-ctir-t4-online-poc.md` | Diff-grounded implementation and validation record | 2026-09-02 |
| V2 protocol/config | `configs/ctir_t4_300_direct.yaml` | Direct-rotation 300-step settings and comparator | 2026-09-03 |
| V2 engineering record | `development_records/2026-09-03-ctir-direct-transport.md` | Direct rotation, closest-feasible controller, and launch preparation | 2026-09-03 |

## Validated facts and gates

- T3 final model exists at `checkpoints/Qwen3-VL-4B/GRPO-CL/training/We-Math2`; T4 uses a fresh optimizer.
- Historical T4 baseline resolves to seed 42, `data_seed=None`, `full_determinism=False`, world size 4.
- Fixed Navigation geometry memory: 32 train samples, seed 142, complete `full_text_only_thought`, Figure-B assistant-token NLL.
- GPU beta-zero equivalence and exact FP64 full-spectrum correctness gates passed; the formal run reached step 60 and all 28 official evaluation rows completed.
- CTIR reduced summed predicted Navigation harm by 80.4% and kept every nonzero update above the 0.90 Puzzle descent constraint, but its step-60 comparison was Navigation `+4.60` points and Puzzle `-10.25` points vs historical vanilla.
- The frozen historical-reference v1 PoC is a strict no-go, but historical vanilla used a 300-step/30-warmup scheduler while CTIR used a 60-step/6-warmup scheduler. The causal behavioral effect is unresolved until an exact schedule-paired vanilla control is run.
- IDEA-CTIR-002 diagnoses the current sequential-Householder lift as nonlocal: it preserves the full spectrum but does not make actual rotation/update distance continuous in beta. A direct principal-plane rotation preserves the identical intended rank-8 redirect while removing arbitrary complement action.
- The gauge-fixed v2 implementation and targeted two-GPU preflight are complete. It uses an identity-connected principal-plane direct rotation, FP64 re-orthonormalization of the small transported control frames, actual complete-update distance, and the closest jointly feasible global-beta candidate instead of maximizing negative old harm.
- The forced-beta integration step retained `95.19%` of new-task descent with full-update Frobenius ratio `1.000000914`. Post-fix GPU matrix checks measured sampled full-spectrum errors near `5e-9` and map orthogonality bounds below `7.9e-8`.
- `EXP-CTIR-004` is queued in a verified detached watcher for four-GPU, complete 300-step T4 training. It will launch only after physical GPUs 0–3 are all below the strict idle threshold with no compute processes. The existing vanilla T4-300 trajectory is schedule-paired on the recorded core training configuration and will be reused rather than rerun.
- Per-matrix beta is deliberately deferred until the global-candidate and distance logs from the 300-step run show whether the extra constrained-allocation complexity is warranted.

Active runs: [experiment ledger](experiment-runs.md).
