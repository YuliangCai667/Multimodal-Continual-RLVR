# DEV-2026-09-03-CTIR-MULTITASK-H100X

## Motivation

把已验证的单 Navigation CTIR 扩展为 T2–T5 的逐旧任务版本，并只向无网 TianheXY-AI 集群传输代码。旧 Slurm job body 会被复制到 `/tmp/slurmd/job*/`，因而不能再用其 `BASH_SOURCE[0]` 寻找共享目录脚本；新实验还必须使用 4×H100-80GB，并保持本地 4×PRO 6000 协议的 batch 8、梯度累积 4（名义全局 batch 128）。

## Changes

- 新增与 `ctir_enable` 隔离的 `ctir_multitask_enable` 路径。最终 v4 不再推测远端已有模块：它复制当前 `src/`、`scripts/`、`configs/` 等完整代码快照，再叠加多任务实现；冻结的 `EXP-CTIR-004` config/script 与当前源逐字节一致。
- 每个旧任务保存独立 probe manifest、当前态梯度和 rank-8 frame；refresh 前同步全部可训练参数到单一 rank-0 完整几何模型，再逐任务反传和清理。
- union 由 FP64 SVD 秩揭示，不保留重复的 `8K` 列；候选对完整 post-AdamW FP32 master delta 做一次全局直接旋转，不截断谱尾。
- 控制器保留每个 `H_k/Hhat_k`，用最坏正约束违反、更新距离、较小 β 的词典序选择；raw 新任务下降非正时保持 β=0，且 β=0 不调用任何参数写回。
- 新增 4-rank、1–4 节点 Slurm 流水线。提交 job body 使用固定共享 `BUNDLE_ROOT`，不依赖 Slurm 临时路径。ZeRO-3 参数/优化器 CPU offload、较小 bucket 和 expandable allocator 降低 80GB 峰值；同 allocation 的最坏 T5 两步 preflight 是正式 T1 的硬前置门。
- 新增单机联网 H100 的四卡入口。它复用同一 stage runner，以本地 `torchrun` 启动四个训练 rank，并以四个独立 TP=1 shard 并行评估；每个子进程只暴露一张物理卡，避免 DeepSpeed/vLLM 重复占用全部可见设备。
- complete-code v4 在统一 launcher 中显式导出 `CPO/src:CPO`，同时满足历史 `train.*` 与当前 `src.*` 导入；入口在启用 nounset 前加载 `/etc/profile`。安装器在写目标前执行训练/评估入口和 15 个算法/数据/集群运行契约，并用现有数据试生成 probes、检查被选图片。
- 生成可直接安装的完整 code-only archive；安装器先备份远端原有完整代码树，再覆盖完整当前代码快照，不携带或触碰数据、环境、模型、checkpoint、结果、实验输出和日志。

## Evaluation

正式运行应以 `step_metrics.jsonl` 的逐任务 `H_k/Hhat_k`、最坏违反、新任务下降比、β、完整更新范数/余弦，`union_rank.jsonl` 的左右有效秩，以及阶段后全部已见任务评估为证据。probe target 固定为：MedBookVQA/We-Math2 的 `conversations[1].value`，Navigation/Puzzle 的 `full_text_only_thought`。

## Pitfalls

- `per_device_train_batch_size=8` 在该 GRPO 脚本中对应每设备生成数；不要因为从 8 卡改为 4 卡而把梯度累积或 batch 补偿回旧 8 卡总量。
- 预检以 `max_steps=300` 构造正式训练器，但在第二个真实 optimizer step 后主动停止；它是内存/实现门禁，不是 2-step 科学结果。
- preflight 通过只能证明最坏 refresh 和两步更新在该 allocation 可运行，不能数学保证整个 1500-step 流程永不遇到数据相关峰值。
- 源 `/home/caiyuliang/mrcl_cpo` 是 `main@9429452` 上的已有 dirty snapshot；本交付位于非 Git worktree `/home/caiyuliang/continual_rlvr`，尚无可引用 commit。

## Validation and limits

- 离线集群版 15/15 CPU 测试在 `continual-rlvr` 容器、UID/GID 1013:1013 下通过：8 个算法/数据契约，加 7 个集群运行契约。联网私有分支把运行契约扩展到 11 个，另覆盖本地四卡 train/eval 映射、单卡进程可见性、非空正式输出拒绝覆盖，以及从联网入口经 preflight 到 T1–T5 全链路。
- 对真实 MRCL train JSON 完成 probe 生成审计：1374/1453/5725/6000 个有效样本；We-Math2 仅剔除空目标索引 4248；四任务均由 seed 142 固定 32 个 probe。
- 直接从 2026-08-30 的 11 GB 原始离线归档中选择性解出旧 CPO 代码树（确认其中无 `src/ctir/`）；在该旧代码副本和合成的完整 bundle fixture 上完成 v4 payload 校验、写入前入口/import/probe/image 检查、安装、15/15 测试及逐文件 snapshot/target 比较，所有 payload 文件一致。
- 2026-09-03 首个 archive 的安装器错误地要求目标已存在 `src/ctir/`，而 2026-08-30 离线包尚无该目录；用户安装尝试被前置检查安全拒绝，未写入目标。依赖补丁式 v2 随后也按用户要求被完整快照 v3 取代；v3 又因 job `241079` 的 runtime `PYTHONPATH` 缺陷被 v4 取代。前三版均已移出 canonical `dist/`。
- 2026-09-03 Slurm job `241079` 在 `hn35/hn39` 上于 probe 导入阶段失败：v3 未设置 runtime `PYTHONPATH`，报 `No module named 'src'`；未生成 probes、未进入 GPU preflight、未训练。v4 修复同时加入 `CPO/src` 与 `CPO`，并新增能复现该失败的入口与整链路测试。
- 2026-09-03 登录节点 `ln311` 实测 Slurm `24.05.1`，确认 `srun` 支持 `--gpus-per-task`、`--gpu-bind`、`--kill-on-bad-exit`；双根 `PYTHONPATH` 下 packed train/eval 两个入口的 `--help` 均通过。eval 环境的 `libcuda.so.1`/Triton 警告来自无 GPU 登录节点，尚不能替代计算节点 GPU preflight。
- Canonical archive: [`mrcl-ctir-multitask-t1-t5-h100x-4gpu-complete-code-v4-20260903.tar.gz`](../scripts/mrcl_cluster/dist/mrcl-ctir-multitask-t1-t5-h100x-4gpu-complete-code-v4-20260903.tar.gz)，221 个 payload 文件，SHA-256 `d7103ccee821a34cfb5d4f23a4c8334ef1bbcac8859e04fa340cbb316a1c06bf`。
- 尚未运行 GPU preflight 或正式实验；H100 显存结论仍待集群实测。
- 联网 H100 部署说明固定模型 `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` 与数据 `MaolinLuo/MRCL@0a4545c105f92073131ccb00c1fe358e039cce8b`，并记录双环境版本、解压、输入检查、日志和输出位置。
- 联网私有分支的 11 个运行契约与 8 个算法/数据契约（19/19）均在 `continual-rlvr` 容器内以 UID/GID 1013:1013 通过；Python 编译和所有相关 shell 脚本的 `bash -n` 同时通过。未在本机启动 GPU 进程。
