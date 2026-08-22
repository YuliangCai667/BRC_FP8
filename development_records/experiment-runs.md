# 实验运行台账

只记录可用于复现和横向比较的信息。实验结束后更新状态以及最后一次、最好一次 eval；训练日志中的随机策略 success 不作为最终结果。

## 汇总

| 日期 | Run ID | 状态 | 实验 | Seed / GPU | 代码 | Eval 结果 |
|---|---|---|---|---|---|---|
| 2026-08-21 | `t1eg59pz` | 完成 | 原始 BRC，固定 MetaWorld 配置 | 42 / GPU 1 | `codex/experiment-recorder`，运行时 `eec512f` + dirty，后整理为 `2ef1f94` | 最后 0.50；最好 0.62；末 3 次均值 0.56 |
| 2026-08-21 | `3rfih1il` | 完成 | 原始 BRC，固定 MetaWorld 配置 | 1 / GPU 2 | 同上 | 最后 0.68；最好 0.72；末 3 次均值 0.70 |
| 2026-08-21 | `0108ohl7` | 完成 | 原始 BRC，固定 MetaWorld 配置 | 123 / GPU 3 | 同上 | 最后 0.40；最好 0.50；末 3 次均值 0.37 |
| 2026-08-22 | `lwyfe3mv` | 运行中 | 原始 BRC，reset 重采样 | 42 / GPU 1 | `codex/experiment-recorder` @ `9891def` | 尚未到首次 eval（记录时 step 8,000） |
| 2026-08-22 | `ahg3l20l` | 运行中 | 论文对齐 BRC，reset 重采样 | 42 / GPU 2 | `codex/paper-alignment` @ `c41d510` | 尚未到首次 eval（记录时 step 7,000） |
| 2026-08-22 | `jvboukld` | 运行中 | DMC Dogs 四任务联合训练，原始 BRC | 42 / GPU 0 | `codex/experiment-recorder` @ `9891def` | 尚未到首次 eval（记录时 step 4,000） |
| 2026-08-22 | `de2aug47` | 运行中 | DMC Dogs 四任务联合训练，论文对齐 BRC | 42 / GPU 3 | `codex/paper-alignment` @ `c41d510` | 尚未到首次 eval（记录时 step 4,000） |
| 待 Dogs 完成 | `dmc_humanoids_original_s42_20260822` | 排队 | DMC Humanoids 三任务联合训练，原始 BRC | 42 / GPU 0 | 训练代码与 `e3a41da` 一致 | Dogs 正常到 500k 后启动 |
| 待 Dogs 完成 | `dmc_humanoids_paper_s42_20260822` | 排队 | DMC Humanoids 三任务联合训练，论文对齐 BRC | 42 / GPU 3 | `codex/paper-alignment` @ `c41d510` | Dogs 正常到 500k 后启动 |

## 已完成：原始 BRC 三种子基线

启动时间分别约为 2026-08-21 00:14、00:18、00:20，均训练至 500,000 env steps。三个进程启动时 metadata 记录的 Git HEAD 是 `eec512f`，工作树包含未提交的实验记录器修改；这批修改随后整理为 `2ef1f94`，因此不能把运行时状态严格表述为干净的 `2ef1f94`。

共同协议：MetaWorld 每个任务使用一个固定配置；eval 环境使用训练 seed + 42；每 25,000 步进行 10 episodes 确定性评估。

```bash
# t1eg59pz：GPU 1 / seed 42
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=METAWORLD_ALL --seed=42 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --run_root=runs --run_id=auto --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true

# 3rfih1il：GPU 2 / seed 1
CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=METAWORLD_ALL --seed=1 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --run_root=runs --run_id=auto --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true

# 0108ohl7：GPU 3 / seed 123
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=METAWORLD_ALL --seed=123 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --run_root=runs --run_id=auto --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

最终 eval 概况：

| Run ID | 最后 step | 最后 return / success | 最好 success | 末 3 次 success 均值 |
|---|---:|---:|---:|---:|
| `t1eg59pz` | 500,000 | 852.8 / 0.50 | 0.62 | 0.56 |
| `3rfih1il` | 500,000 | 1171.2 / 0.68 | 0.72 | 0.70 |
| `0108ohl7` | 500,000 | 667.8 / 0.40 | 0.50 | 0.37 |

## 运行中：reset 重采样对照

两次实验均为 seed 42，训练和 eval 环境分别维护独立 RNG，但都以 seed 42 初始化；每次 reset 重采样任务配置。除论文对齐开关外，其余主要超参数一致。

### 原始 BRC：`lwyfe3mv`

- 启动：2026-08-22 14:49
- 分支/版本：`codex/experiment-recorder` @ `9891def`
- W&B：`BRC-original-resample-seed42`

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=METAWORLD_ALL --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --metaworld_reset_mode=resample --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-original-resample-seed42
```

### 论文对齐 BRC：`ahg3l20l`

- 启动：2026-08-22 14:51
- 分支/版本：`codex/paper-alignment` @ `c41d510`
- W&B：`BRC-paper-aligned-resample-seed42`
- 相对原始分支启用：L1 task embedding、critic time-limit bootstrap、per-task empirical entropy correction。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=METAWORLD_ALL --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --metaworld_reset_mode=resample --paper_alignment=true --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-paper-aligned-resample-seed42
```

## 运行中：DMC Dogs 四任务联合训练

任务集合为 `dog-stand`、`dog-walk`、`dog-trot`、`dog-run`。四个环境并行交互，500,000 env steps 指每个任务的步数，对应总计 2,000,000 transitions。两次实验均为 seed 42，训练和 eval 环境使用相同初始 seed、独立 RNG；记录器和默认 checkpoint 策略均开启。DMC 没有 MetaWorld 式 success 指标，命令行中的 `success=0` 不用于判断效果；比较 eval return，论文汇总口径为 return / 1000。

### 原始 BRC：`jvboukld`

- 启动：2026-08-22 16:58
- tmux：`brc_dmc_dogs_old_s42`
- GPU：0（启动前已有其他进程占用约 14.6 GiB）
- 分支/版本：`codex/experiment-recorder` @ `9891def`
- W&B：[BRC-DMC-DOGS-original-seed42](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/jvboukld)

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_DOGS --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-DMC-DOGS-original-seed42 --run_root=runs --run_id=auto --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

### 论文对齐 BRC：`de2aug47`

- 启动：2026-08-22 16:58
- tmux：`brc_dmc_dogs_paper_s42`
- GPU：3
- 分支/版本：`codex/paper-alignment` @ `c41d510`
- W&B：[BRC-DMC-DOGS-paper-aligned-seed42](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/de2aug47)

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_DOGS --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --paper_alignment=true --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-DMC-DOGS-paper-aligned-seed42 --run_root=runs --run_id=auto --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

查看会话：

```bash
tmux attach -t brc_dmc_dogs_old_s42
tmux attach -t brc_dmc_dogs_paper_s42
```

## 排队：DMC Humanoids 三任务联合训练

任务集合为 `humanoid-stand`、`humanoid-walk`、`humanoid-run`。Figure 16 的 DMC-Hard 结果由 Dogs 4-task MT 和 Humanoids 3-task MT 的共 7 个任务聚合得到，并非单个 agent 混合训练两种 embodiment。以下接力任务只在对应 Dogs run 写入 `run_finished` 且达到 500,000 步后启动；Dogs 异常退出或训练代码发生变化时不会启动。

### 原始 BRC（排队）

- tmux：`brc_dmc_humanoids_old_queue`
- GPU：0
- 预定本地 run ID：`dmc_humanoids_original_s42_20260822`
- W&B 名称：`BRC-DMC-HUMANOIDS-original-seed42`

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_HUMANOIDS --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-DMC-HUMANOIDS-original-seed42 --run_root=runs --run_id=dmc_humanoids_original_s42_20260822 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

### 论文对齐 BRC（排队）

- tmux：`brc_dmc_humanoids_paper_queue`
- GPU：3
- 预定本地 run ID：`dmc_humanoids_paper_s42_20260822`
- W&B 名称：`BRC-DMC-HUMANOIDS-paper-aligned-seed42`

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_HUMANOIDS --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --paper_alignment=true --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-DMC-HUMANOIDS-paper-aligned-seed42 --run_root=runs --run_id=dmc_humanoids_paper_s42_20260822 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

查看等待状态：

```bash
tmux attach -t brc_dmc_humanoids_old_queue
tmux attach -t brc_dmc_humanoids_paper_queue
```

## 更新规则

- 启动实验时新增一行，记录开始时间、run ID、分支、commit 和完整命令。
- 结束后补充最终 eval、最好 eval，以及末 3 次 eval 均值；异常退出则记录停止 step 和原因。
- 对比前先检查 reset 模式、训练/eval seed 协议和代码 commit，协议不同的结果不直接合并统计。
