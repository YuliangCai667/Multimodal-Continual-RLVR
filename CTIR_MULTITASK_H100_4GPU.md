# CTIR T1–T5：单机 4×H100-80GB 部署与运行

这份说明用于一台可联网、单机四张 NVIDIA H100 80GB 的服务器。请始终用普通用户运行，不要用 root。实验顺序固定为 MedBookVQA → Navigation → We-Math2 → Puzzle → FinMME，每个任务 300 个 optimizer steps。

## 1. 拉取指定私有分支

```bash
git clone --branch exp/ctir-multitask-h100x-v4 --single-branch \
  git@github.com:YuliangCai667/Multimodal-Continual-RLVR.git
cd Multimodal-Continual-RLVR
git status --short
git rev-parse HEAD
```

`git status --short` 必须没有输出。仓库所有后续命令都从仓库根目录执行。

## 2. 创建两个隔离环境

下面的版本与已打包环境一致。把 `ENV_ROOT` 换成服务器上的持久化大盘目录。

```bash
export ENV_ROOT=/path/to/ctir-envs

conda create -p "$ENV_ROOT/trlQwen" python=3.12 -y
conda run -p "$ENV_ROOT/trlQwen" python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV_ROOT/trlQwen" python -m pip install psutil ninja einops
env MAX_JOBS=8 conda run -p "$ENV_ROOT/trlQwen" python -m pip install \
  flash-attn==2.8.3 --no-build-isolation
conda run -p "$ENV_ROOT/trlQwen" python -m pip install -r requirements.txt
conda run -p "$ENV_ROOT/trlQwen" python -m pip install deepspeed==0.16.4
conda run -p "$ENV_ROOT/trlQwen" python -c \
  "import nltk,sys,os; p=os.path.join(sys.prefix,'share','nltk_data'); os.makedirs(p,exist_ok=True); nltk.download('punkt_tab',download_dir=p)"

conda create -p "$ENV_ROOT/vllmQwen" python=3.12 -y
conda run -p "$ENV_ROOT/vllmQwen" python -m pip install \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -p "$ENV_ROOT/vllmQwen" python -m pip install -r requirements_eval.txt
conda run -p "$ENV_ROOT/vllmQwen" python -m pip install vllm==0.11.2
conda run -p "$ENV_ROOT/vllmQwen" python -c \
  "import nltk,sys,os; p=os.path.join(sys.prefix,'share','nltk_data'); os.makedirs(p,exist_ok=True); nltk.download('punkt_tab',download_dir=p)"
```

如服务器已有等价环境，可跳过创建，但正式运行前仍需执行第 4 节的检查。

## 3. 下载固定版本的模型与数据

先在任意联网 Python 环境安装 Hugging Face CLI，然后下载到持久化大盘。不要把模型或数据放进 Git 仓库。

```bash
python -m pip install -U huggingface_hub

export ASSET_ROOT=/path/to/ctir-assets
export BASE_MODEL="$ASSET_ROOT/models/Qwen3-VL-4B-Instruct"
export BASE_PATH="$ASSET_ROOT/datasets/MRCL"

hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir "$BASE_MODEL"

hf download MaolinLuo/MRCL --repo-type dataset \
  --revision 0a4545c105f92073131ccb00c1fe358e039cce8b \
  --local-dir "$BASE_PATH"

for task in MedBookVQA Navigation We-Math2 Puzzle FinMME; do
  unzip -q "$BASE_PATH/$task/images.zip" -d "$BASE_PATH/$task"
  test -d "$BASE_PATH/$task/images"
done
```

最终必须是下面的目录形状：

```text
$BASE_MODEL/config.json
$BASE_PATH/MedBookVQA/{images,jsons/train/data.json,jsons/test/data.json}
$BASE_PATH/Navigation/{images,jsons/train/data.json,jsons/test/data.json}
$BASE_PATH/We-Math2/{images,jsons/train/data.json,jsons/test/data.json}
$BASE_PATH/Puzzle/{images,jsons/train/data.json,jsons/test/data.json}
$BASE_PATH/FinMME/{images,jsons/train/data.json,jsons/test/data.json}
```

## 4. 登录 GPU 服务器后做只读检查

```bash
export TRAIN_PYTHON="$ENV_ROOT/trlQwen/bin/python"
export EVAL_PYTHON="$ENV_ROOT/vllmQwen/bin/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3

test "$(id -u)" -ne 0
test -x "$TRAIN_PYTHON"
test -x "$EVAL_PYTHON"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

PYTHONPATH="$PWD/src:$PWD" "$TRAIN_PYTHON" \
  scripts/CTIR-MULTI-T1-T5/prepare_multitask_probes.py --help
PYTHONPATH="$PWD/src:$PWD" "$TRAIN_PYTHON" src/train/train_grpo.py --help
PYTHONPATH="$PWD/src:$PWD" "$EVAL_PYTHON" src/eval/inference.py --help
```

必须恰好暴露四张 H100，每张总显存至少 75 GB，并确保没有别人的计算进程占卡。启动器会再次执行相同 GPU/入口检查。

## 5. 启动完整 T1–T5 实验

选择一个新的、只含字母数字和短横线的 run id。不要在已有正式输出的仓库副本里重新启动；stage runner 会拒绝复用非空的训练输出目录。

```bash
export MRCL_RUN_ID=ctir-t1-t5-$(date +%Y%m%d-%H%M%S)
mkdir -p logs

nohup env \
  BASE_MODEL="$BASE_MODEL" \
  BASE_PATH="$BASE_PATH" \
  TRAIN_PYTHON="$TRAIN_PYTHON" \
  EVAL_PYTHON="$EVAL_PYTHON" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  MRCL_RUN_ID="$MRCL_RUN_ID" \
  bash scripts/CTIR-MULTI-T1-T5/launch_online_h100_4gpu.sh \
  > "logs/$MRCL_RUN_ID.out" 2>&1 < /dev/null &

echo "$!"
tail -f "logs/$MRCL_RUN_ID.out"
```

启动器先生成固定 probe manifests，再用最坏的 T5/四旧任务几何路径做两个真实 optimizer steps。只有 `verify_preflight.py` 验证更新写回、有限数值、β=1 和谱保持记录全部通过后，才会开始正式 T1。该门禁显著降低 80GB OOM 风险，但不能替代完整 1500-step 实测。

训练协议固定为每卡 batch 8、四卡、梯度累积 4，名义全局 batch 为 128；不要为了模仿旧八卡任务而修改 batch 或累积步数。显存策略固定为 BF16、gradient checkpointing、ZeRO-3 参数与 optimizer CPU offload、小通信 bucket，以及 expandable CUDA allocator。

## 6. 日志与输出位置

- 总 stdout/stderr：`logs/$MRCL_RUN_ID.out`
- preflight：`experiments/ctir_multitask_t1_t5/preflight/job-$MRCL_RUN_ID/`
- probe manifests：`experiments/ctir_multitask_t1_t5/probes/`
- 每阶段 CTIR 指标：`experiments/ctir_multitask_t1_t5/logs/<TASK>/`
- checkpoints：`checkpoints/Qwen3-VL-4B/CTIR-MULTI-CL/training/<TASK>/`
- 逐阶段已见任务评估：`results/Qwen3-VL-4B/CTIR-MULTI-CL/<TRAIN_TASK>/<TEST_TASK>/`

完整成功时总日志最后一行是：

```text
EXP-CTIR-MULTI-T1-T5-001 complete
```

如 preflight 报 OOM 或任何验证失败，不要直接删掉门禁或启动正式 T1；保留总日志和对应 preflight 目录，先把错误发回。
