# MRCL T4 前 60 步：CTIR 在线 PoC — Codex 执行计划

> **目标**：从 T4: Puzzle 的真实起点重新分叉，连续运行 **60 个真实 optimizer steps**，直接测试 **Current-State Tangent-Aware Isospectral Update Redirection (CTIR)**。
>
> 不再把 30-step checkpoint delta 当作一次 update。本实验必须发生在真实 GRPO optimizer dynamics 中。

## 1. 核心问题

只回答一个问题：

> 在 T4 前 60 步这个 Navigation 遗忘最严重的窗口里，能否根据**当前模型状态**下 Navigation 的敏感切线方向，把 Puzzle 的**真实 optimizer update**等谱转向，从而减轻 Navigation forgetting，同时基本保持 Puzzle plasticity？

当前 continual 时间轴：

```text
T1 MedBookVQA : global 0–300
T2 Navigation : global 300–600
T3 We-Math2   : global 600–900
T4 Puzzle     : global 900–1200
```

所以：

```text
T4 local step 0  = global 900
T4 local step 60 = global 960
```

现有曲线表明，T4 0→60 是 Navigation 最明显的快速遗忘区间，适合做第一次真实在线方法验证。

---

## 2. 起点和训练配置

### 2.1 起点

使用 **T3: We-Math2 final model**，即原始 T4 的真实起点：

\[
\theta_{T4,0}.
\]

不要从 T4 step30/60 resume。

### 2.2 Optimizer state

T4 是新 stage：

```text
load T3 final model weights
→ fresh T4 trainer / optimizer
→ train Puzzle
```

不要继承 T3 optimizer state。

### 2.3 保持当前本地 T4 baseline 规格

Codex 必须先读取**当前本地已经修改后的 T4 launcher**，不要把官方仓库默认参数重新覆盖回来。

只对实验分支修改：

```text
max_steps = 60
save_steps = 10
output_dir = 独立 CTIR 目录
```

其余保持当前 baseline 一致：数据、reward、prompt、batch semantics、gradient accumulation、num_generations、LR、scheduler、warmup、weight decay、BF16、DeepSpeed、temperature、top-p/top-k、epsilon、completion length、GPU 数量等。

---

## 3. 第一轮只跑 Dynamic CTIR

正式分支：

```text
T4-CTIR-Dynamic-60
```

先不同时跑 Vanilla / Projection / Static-IsoRotate。

已有原始 T4 0→60 曲线先作为历史参照。若 CTIR 有明显信号，再补 same-seed 的 Vanilla、Projection、Static-IsoRotate 做正式公平对照。

---

## 4. CTIR 操作的对象

CTIR 的 `isospectral` 对象必须是某个 optimizer step **实际施加到参数矩阵上的 parameter delta**：

\[
\Delta W_t^{raw}.
\]

正常 optimizer：

\[
W_{t+1}^{raw}=W_t+\Delta W_t^{raw}.
\]

CTIR：

\[
W_{t+1}=W_t+\widetilde{\Delta W}_t.
\]

要求：

\[
\sigma(\widetilde{\Delta W}_t)=\sigma(\Delta W_t^{raw})
\]

在数值误差范围内成立。

**不要把以下对象偷换成 CTIR update：**

- 权重矩阵 `W` 本身的 spectrum；
- 30-step cumulative delta；
- `-lr * raw_gradient`；
- ZeRO shard 的局部 spectrum。

如果当前 optimizer/ZeRO 栈无法安全截获完整矩阵的真实 parameter delta，先解决工程问题或报告阻塞，不要静默降级方法定义。

---

## 5. 第一版只保护 Navigation

T4 前已有：

```text
MedBookVQA
Navigation
We-Math2
```

但 CTIR v1 的 geometry signal **只来自 Navigation**。

原因：Navigation 是 T4 0→60 最强 forgetting 主案例，且当前机制证据主要围绕它建立。第一版先把单旧任务机制做干净。

但是 evaluation 必须同时测：

```text
MedBookVQA
Navigation
We-Math2
Puzzle
```

以检查 collateral damage。

---

## 6. Navigation geometry probes

固定从 Navigation train split 选择：

```text
32 samples
```

保存：

```text
experiments/ctir_t4_60/probes/navigation_probes.json
```

要求：

- 固定 sample id/source index；
- 固定 seed；
- 后续不重新抽；
- 不混入 Puzzle RL training batch；
- 只用于测量旧任务当前 geometry。

论文层面诚实称为：

```text
small old-task geometry memory / geometry probes
```

不要声称 replay-free。

---

## 7. Navigation differentiable loss

项目里已经存在用于之前 Figure B / directional-derivative / curvature 分析的 Navigation differentiable loss。

Codex 必须：

1. 找到已有实现；
2. 保持数学定义不变；
3. 封装为公共 `navigation_geometry_loss(...)`；
4. CTIR tangent estimation 复用同一 loss。

不要用 exact-match 0/1 reward 直接反向传播。

---

## 8. 第一版保护哪些参数

根据已有 causal rollback，第一版只处理：

```text
LLM layers 9–26
```

中的主要 Linear 权重：

```text
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

不干预：

```text
Vision tower
Merger
Embedding
LM head
Norm/Bias
L0–8
L27+
```

未保护参数完全使用 baseline optimizer update。

---

## 9. Current-state tangent proxy v1

第一版不要上 full NTK / Fisher / per-sample covariance。

### 9.1 每次 refresh

在当前模型 `θ_t` 上，用固定 32 个 Navigation probes 计算平均 differentiable old-task gradient。

对每个 protected matrix `W`：

\[
G_t^{old}(W)=\frac1N\sum_i \nabla_W L^{(i)}_{Nav}(\theta_t).
\]

### 9.2 低秩敏感 frames

对 `G_old(W)` 做 truncated/randomized SVD：

\[
G^{old}\approx Q_L S Q_R^T.
\]

v1 设置：

```text
tangent rank cap = 8
```

解释：

```text
Q_L = Navigation 当前输出侧主要敏感方向
Q_R = Navigation 当前输入侧主要敏感方向
```

这是低成本 current-state tangent proxy，不宣称是完整 NTK。

---

## 10. Tangent refresh cadence

v1：

```text
refresh_interval = 5 optimizer steps
```

refresh at：

```text
0,5,10,15,20,25,30,35,40,45,50,55
```

两次 refresh 间复用最近的 `Q_L/Q_R`。

必须记录相邻 refresh basis alignment，用于后续判断 static basis stale 的速度。

---

## 11. 每个真实 optimizer step 的执行顺序

### Step A — 正常 Puzzle GRPO

完全复用 baseline：

```text
rollout → reward → GRPO loss → backward → gradient accumulation
```

CTIR 只在真正 `optimizer.step()` 的 boundary 执行一次，不在 accumulation micro-step 上重复执行。

### Step B — 缓存 current new-task gradient

在 zero_grad 前缓存 protected matrices 的：

\[
G_t^{new}(W).
\]

用途：检查 CTIR 是否仍保留 Puzzle descent tendency。

### Step C — 必要时 refresh Navigation tangent

若：

```text
t % 5 == 0
```

则当前 `θ_t + 32 probes → G_old → Q_L/Q_R`。

### Step D — 截获真实 raw optimizer delta

对 protected matrix：

1. optimizer step 前保存 `W_before`；
2. 让原 AdamW/DeepSpeed optimizer 正常产生 proposed parameter update；
3. 得到 `W_raw_after`；
4. 计算：

\[
\Delta W^{raw}=W^{raw-after}-W^{before}.
\]

必须保留 base optimizer 的 momentum/preconditioning 语义；不要用 `-lr*grad` 近似。

### Step E — CTIR 等谱转向

构造：

\[
\widetilde{\Delta W}=R_L\Delta W^{raw}R_R^T
\]

其中：

\[
R_L^TR_L=I,\quad R_R^TR_R=I.
\]

因此理论上严格保持完整 singular spectrum 和 Frobenius norm。

### Step F — 选择 rotation strength

候选：

```text
β ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

`β=0` 必须严格等价于 raw update。

对于 raw delta dominant frames `U,V`，目标方向：

\[
U_{target}=orth[(I-\beta Q_LQ_L^T)U]
\]

\[
V_{target}=orth[(I-\beta Q_RQ_R^T)V].
\]

再通过 Orthogonal Procrustes / Householder / 等价低秩正交构造，得到真正 orthogonal 的 `R_L/R_R`。

**禁止用 projection+rescale 冒充 isospectral rotation。**

---

## 12. Puzzle plasticity constraint

原始 update 对当前 Puzzle gradient 的一阶 descent proxy：

\[
B_{raw}=-\langle G^{new},\Delta W^{raw}\rangle.
\]

candidate：

\[
B_\beta=-\langle G^{new},\widetilde{\Delta W}_\beta\rangle.
\]

v1 约束：

```text
B_beta >= 0.90 * B_raw
```

即 CTIR 不允许为了保护 Navigation，把当前 Puzzle 一阶下降能力压到 raw update 的 90% 以下。

---

## 13. Navigation harm score

使用最新 cached `G_old`：

\[
H_\beta=\langle G^{old},\widetilde{\Delta W}_\beta\rangle.
\]

选择规则：

```text
在满足 Puzzle descent >= 90% raw 的 candidates 中
选择 H_old 最小的 β
```

若所有 `β>0` 都违反新任务 constraint：

```text
fallback β=0
```

不要强制转向。

---

## 14. DeepSpeed / ZeRO-3 工程要求

这是第一轮最大工程风险。

完整矩阵的 singular spectrum 不能通过各 ZeRO shard 独立处理得到。

Codex 必须实现/验证一个：

```text
ProtectedParameterInterceptor
```

仅对 L9–26 protected matrices，在 optimizer boundary：

```text
gather one matrix
→ compute raw full-matrix delta
→ CTIR redirect
→ write back
→ release
→ next matrix
```

优先检查当前 DeepSpeed 版本的：

```text
deepspeed.zero.GatheredParameters
```

或等价安全机制。

禁止一次 gather 全部 L9–26；禁止 shard-wise “isospectral” 冒充 full-matrix isospectral。

如果该方案无法在当前 stack 稳定实现：停止 formal run，报告具体阻塞，不要静默改成 gradient rotation。

---

## 15. 允许的 v1 工程近似

允许：

- rank=8；
- randomized/truncated SVD；
- tangent 每 5 步刷新；
- 逐矩阵 gather/处理；
- geometry probes batch 化；
- basis 低精度存储、关键计算 FP32；
- 只由 rank0 写日志。

不允许：

- 30-step cumulative delta；
- static tangent 全程冒充 current-state；
- 改 Puzzle 训练配置；
- 改 reward/prompt；
- 隐式降低 update magnitude 后仍称等谱。

---

## 16. 正式跑前只做两个必要 correctness tests

### Test 1 — β=0 equivalence

只跑 1 optimizer step：

```text
CTIR enabled + force β=0
```

确认 protected matrices 的最终 parameter delta 与 base optimizer raw delta 一致。

目标相对误差：

```text
~1e-6 到 1e-5（按 BF16/ZeRO 数值情况记录）
```

### Test 2 — spectrum preservation

至少抽：

```text
1 attention matrix
1 MLP matrix
```

使用 `β>0`，检查：

\[
\epsilon_{spec}=\frac{\|\sigma(\widetilde{\Delta W})-\sigma(\Delta W)\|_2}{\|\sigma(\Delta W)\|_2+\epsilon}
\]

以及：

\[
\frac{\|\widetilde{\Delta W}\|_F}{\|\Delta W\|_F}\approx1.
\]

通过后直接跑正式 60 步，不再做大规模 smoke。

---

## 17. 正式 60-step run

建议输出：

```text
checkpoints/Qwen3-VL-4B/CTIR-T4-60/training/Puzzle/
```

设置：

```text
max_steps = 60
save_steps = 10
logging_steps = 1
```

保存：

```text
step10 step20 step30 step40 step50 step60
```

T4 start 作为 step0。

---

## 18. 每个 optimizer step 必须记录

写：

```text
experiments/ctir_t4_60/logs/step_metrics.jsonl
```

字段至少：

```text
local_step
global_continual_step   # 900 + local_step

grpo_loss
reward_summary

chosen_beta

raw_update_fro_norm
ctir_update_fro_norm
frob_ratio

raw_old_harm
ctir_old_harm
old_harm_reduction

raw_new_descent
ctir_new_descent
new_descent_ratio

mean_rotation_strength
max_rotation_strength

max_spectrum_error
mean_spectrum_error

tangent_age
```

---

## 19. 每次 tangent refresh 记录

写：

```text
experiments/ctir_t4_60/logs/tangent_refresh.jsonl
```

记录：

```text
step
navigation_probe_loss
protected_matrix
tangent_singular_values
effective_rank
Q_L_alignment_vs_previous
Q_R_alignment_vs_previous
Q_L_alignment_vs_step0
Q_R_alignment_vs_step0
```

这份日志以后直接用于验证：

> T4 前 60 步内 old-task sensitive geometry 是否快速 stale。

---

## 20. Evaluation

训练后对：

```text
step0,10,20,30,40,50,60
```

统一用当前官方 evaluator 测：

```text
MedBookVQA
Navigation
We-Math2
Puzzle
```

输出：

```text
experiments/ctir_t4_60/eval/performance.csv
```

第一轮先与已有 baseline 的：

```text
step0 / step30 / step60
```

对齐比较。

---

## 21. 第一轮关键指标

Navigation recovery：

\[
R_{Nav}(s)=Score^{CTIR}_{Nav}(s)-Score^{baseline}_{Nav}(s).
\]

Puzzle cost：

\[
C_{Puzzle}(s)=Score^{baseline}_{Puzzle}(s)-Score^{CTIR}_{Puzzle}(s).
\]

同时报告 MedBook / WeMath 差异。

PoC 的“值得继续”信号可暂定为：

```text
Navigation recovery @60 >= +5 points
Puzzle cost @60 <= 3 points
```

这个阈值只作为 go/no-go，不是统计显著性声明。

另外必须同时满足机制层面：

1. raw old harm 在相当一部分 step 为正；
2. CTIR 系统性降低 old harm；
3. new_descent_ratio 大部分 step >= 0.90；
4. spectrum error 接近数值误差；
5. chosen β 不是几乎永远为 0。

---

## 22. 失败模式要区分

### A. Navigation 没恢复，但 harm proxy 明显下降

说明当前 differentiable old-task proxy / tangent basis 不足以预测真实行为。

### B. Navigation 恢复，但 Puzzle 大幅下降

说明等谱不等于保留功能方向；需要更强 new-task constraint。

### C. 几乎每步 β=0

说明当前保护空间与 Puzzle descent 约束之间没有足够自由度；需要改 rotation space/module scope。

### D. 数值/训练不稳定

先排查 ZeRO 与 post-optimizer intervention，不能直接判方法失败。

---

## 23. 如果第一轮成功，下一步才补三条对照

### Paired Vanilla

同 T4 start、同 seed、同 60 steps，无 CTIR。

### Current Projection

用同样 current Navigation tangent，但传统删除 harmful component，而不是等谱转向。

### Static IsoRotate

只在 T4 step0 算一次 tangent，60 步不刷新。

这三条最终分别回答：

```text
CTIR 是否优于随机训练波动？
CTIR 是否优于“少更新一点”的 projection？
current-state refresh 是否优于 static protection？
```

---

## 24. 暂时不跑 stochastic q / Pass@32

如果 official eval 已看到明显 Navigation retention gain，再对：

```text
baseline step60
CTIR step60
```

做：

```text
128 Navigation prompts × 32 samples
```

分析 `q / Pass@32`。

没有 deterministic gain 时不要先烧这部分资源。

---

## 25. 推荐代码结构

新增：

```text
src/trainer/tangent_iso_grpo_trainer.py
src/ctir/
├── config.py
├── probe_dataset.py
├── geometry_loss.py
├── tangent_state.py
├── tangent_estimator.py
├── optimizer_interceptor.py
├── orthogonal_redirect.py
├── metrics.py
└── zero3_utils.py
```

修改：

```text
src/params.py
src/train/train_grpo.py
```

新增 launcher：

```text
scripts/CTIR-T4-60/run.sh
```

Trainer 设计：

```text
TangentIsoGRPOTrainer(GRPOTrainer)
```

只在 optimizer update boundary 加 intervention；不要重写 rollout/reward/GRPO loss/batching。

当前 naive GRPO path 在没有 `mask_path` 时本来就是普通 `GRPOTrainer`。CTIR 不应建立在 CPO 的 `CLGRPOTrainer` 上。

---

## 26. 建议新增 CLI 参数

```text
ctir_enable: bool = False
ctir_probe_path: Optional[str]
ctir_probe_count: int = 32

ctir_layer_start: int = 9
ctir_layer_end: int = 26
ctir_tangent_rank: int = 8
ctir_refresh_interval: int = 5
ctir_new_descent_ratio: float = 0.90
ctir_beta_candidates: str = "0,0.25,0.5,0.75,1.0"
ctir_log_dir: Optional[str]
```

必要时添加 target-module pattern。

---

## 27. Codex 实际执行顺序

### Phase 0 — Inspect local project

确认：

1. T3 final checkpoint；
2. 当前 T4 baseline launcher；
3. 当前本地修改后的训练超参；
4. DeepSpeed/GPU 配置；
5. Figure B Navigation differentiable loss 实现；
6. Qwen3-VL 参数名；
7. 当前 optimizer step / ZeRO hook 最安全的接入位置。

不要根据官方 repo 猜本地路径。

### Phase 1 — Implement CTIR core

完成：

```text
probe loader
geometry loss reuse
current tangent estimator
protected-module selector
real optimizer-delta interceptor
orthogonal redirector
beta candidate selector
metrics/logger
```

### Phase 2 — Two correctness tests

只做 β=0 equivalence + spectrum preservation。

### Phase 3 — Formal Dynamic CTIR 0→60

不中途根据结果改 rank、refresh interval、beta grid。

### Phase 4 — Eval step0/10/.../60

生成 performance 与机制日志图。

### Phase 5 — Go/No-Go summary

输出：

```text
experiments/ctir_t4_60/summary.md
```

只回答：

1. Navigation 是否少忘？
2. Puzzle 是否基本保持？
3. old-harm proxy 是否真的降低？
4. tangent 是否在 60 步内明显旋转？
5. 是否值得补 paired Vanilla / Projection / Static-Iso？

未经用户明确要求，不要直接跑完整 300-step / 五任务 CTIR。

---

## 28. 最终输出目录

```text
experiments/ctir_t4_60/
├── config.yaml
├── probes/navigation_probes.json
├── logs/
│   ├── step_metrics.jsonl
│   └── tangent_refresh.jsonl
├── eval/performance.csv
├── figures/
│   ├── performance_curves.png
│   ├── nav_recovery_vs_puzzle_cost.png
│   ├── old_harm_by_step.png
│   ├── beta_by_step.png
│   ├── tangent_rotation.png
│   └── spectrum_error.png
└── summary.md
```

checkpoints 可以保存在统一 checkpoint root，通过 manifest/symlink 指向，避免重复存储。

---

## 29. 这轮最重要的三张图

### Fig 1 — T4 0→60 performance

CTIR vs 原 baseline：

```text
Puzzle acquisition
Navigation retention
MedBook
WeMath
```

### Fig 2 — Online old-task harm

x = optimizer step 1...60

```text
raw predicted Navigation harm
CTIR predicted Navigation harm
```

### Fig 3 — Tangent rotation

每次 refresh 比较：

\[
Q_{Nav}(\theta_t) \text{ vs } Q_{Nav}(\theta_0).
\]

回答：在真正 forgetting shock 内，static protected basis 是否快速 stale。

---

## 30. 当前明确不要做

```text
multi-old-task tangent union
CPO mask
full NTK
Fisher/Hessian
head/neuron protection
LoRA
weight-spectrum freezing
online hyperparameter sweep
Pass@32 during training
full 300-step CTIR
T5 CTIR
```

先把 T4 前 60 步真实在线 PoC 做干净。

---

## 31. 一句话执行原则

> Start a fresh Puzzle stage from the exact T3-final model, run 60 real GRPO optimizer steps, refresh a Navigation-sensitive tangent proxy every 5 steps, intercept the actual AdamW/DeepSpeed full-matrix parameter delta on L9–26, redirect it with orthogonal left/right transformations that preserve the delta singular spectrum, choose the strongest safe rotation that retains at least 90% of the current Puzzle descent proxy, save every 10 steps, and evaluate all four seen tasks. Do not use 30-step cumulative checkpoint deltas as a substitute for optimizer-step dynamics.
