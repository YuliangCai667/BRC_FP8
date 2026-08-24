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
| 2026-08-22 | `jvboukld` | 完成 | DMC Dogs 四任务联合训练，原始 BRC | 42 / GPU 0 | `codex/experiment-recorder` @ `9891def` | 最后 eval return 776.4 |
| 2026-08-22 | `de2aug47` | 完成 | DMC Dogs 四任务联合训练，旧论文对齐 BRC（critic bootstrap） | 42 / GPU 3 | `codex/paper-alignment` @ `c41d510` | 最后 eval return 626.4 |
| 2026-08-23 | `ablation_bootstrap_only_s42_20260823` | 完成 | DMC Dogs 25k 消融：只改 critic bootstrap | 42 / GPU 1 | `codex/paper-alignment` @ `c41d510` | eval 关闭；25k train return 20.1；用于诊断尺度反馈 |
| 2026-08-23 | `validation_reward_mean_gbar_s42_20260823` | 完成 | DMC Dogs 10k 验证：paper preset 改用 reward-mean `Gbar` | 42 / GPU 1 | 运行时 `c41d510` + 修改，后固化为 `9295a2d` | eval 关闭；10k train return 13.3；`Gbar` 约 6–19 |
| 2026-08-22 | `dmc_humanoids_original_s42_20260822` | 运行中 | DMC Humanoids 三任务联合训练，原始 BRC | 42 / GPU 0 | 训练代码与 `e3a41da` 一致 | 正常接力启动，继续运行 |
| 2026-08-22 | `dmc_humanoids_paper_s42_20260822` | 主动停止 | DMC Humanoids，旧论文对齐 BRC（critic bootstrap） | 42 / GPU 3 | `codex/paper-alignment` @ `c41d510` | step 133k；125k eval return 3.61 |
| 2026-08-23 | `dmc_dogs_rewardmean_paper_s42_20260823` / `ttb7gnbz` | 运行中 | DMC Dogs，稳定化论文对齐 BRC | 42 / GPU 3 | `codex/paper-alignment` @ `9295a2d` | 正式 500k；完成后自动接 Humanoids |
| Dogs 完成后 | `dmc_humanoids_rewardmean_paper_s42_20260823` | 排队 | DMC Humanoids，稳定化论文对齐 BRC | 42 / GPU 3 | `codex/paper-alignment` @ `9295a2d` | 仅 Dogs 正常退出后启动 |
| 2026-08-24 00:14 | `brc_cheetah_run_d_target_fp8_resident_s0_smoke_final` | 完成 | `EXP-FP8-TARGET-SMOKE-D`：审查后常驻 FP8 目标 Critic 最终短跑验收 | 0 / GPU 3（另有约 8 GiB 外部任务） | `codex/blackwell-fp8-direct` @ `9c81c7d` + 本轮未提交实现 | 200 步、203 updates；无 NaN/Inf；实际 EMA 吞更新率 `8.01e-5`–`1.18e-4`；验收通过 |
| 2026-08-24 00:24 | `brc_dmc_dogs_c_target_fp8_resident_s42` / `trzg1mjw` | 主动停止 | `EXP-FP8-TARGET-C-S42`：在线 FP32、目标残差 kernel 常驻 FP8 | 42 / GPU 3 | `codex/blackwell-fp8-direct` @ `f88b662`，clean | 停止于 env step 150k / update 290003；125k eval `8.16`，朴素常驻 FP8 判定失败 |
| 2026-08-24 00:24 | `brc_dmc_dogs_d_target_fp8_resident_s42` / `fb02bf23` | 主动停止 | `EXP-FP8-TARGET-D-S42`：在线 FP8 Direct、目标残差 kernel 常驻 FP8 | 42 / GPU 3 | `codex/blackwell-fp8-direct` @ `f88b662`，clean | 00:32:38 按用户要求迁移 GPU；停止于 env step 10,057 / update 10,115；无 checkpoint，不作为完整正式结果 |
| 2026-08-24 00:33 | `brc_dmc_dogs_d_target_fp8_resident_s42_gpu1_r1` / `iwlomjbu` | 主动停止 | `EXP-FP8-TARGET-D-S42-R1`：D 在 GPU1 从头重启 | 42 / GPU 1（与既有任务共享） | `codex/blackwell-fp8-direct` @ `1ff6625`，clean；训练代码同 `f88b662` | 停止于 env step 137438 / update 264877；125k eval `7.23`，朴素常驻 FP8 判定失败 |
| 2026-08-24 01:16 | `brc_cheetah_run_target_fp8_direct_fp32_storage_s0_smoke` | 完成 | `EXP-FP8-TARGET-FWD-SMOKE`：目标 FP8 Direct 前向、FP32 存储短跑 | 0 / GPU 2（与既有任务共享） | `1eecc56` + 本轮未提交实现 | 200 steps / 203 updates；无 NaN/Inf；298 条 tensor stats；验收通过 |
| 2026-08-24 01:24 | `brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2` / `gij6jhgd` | 完成 | `EXP-FP8-TARGET-FWD-S42`：在线/目标 FP8 Direct，目标 FP32 存储与 EMA | 42 / GPU 2（与既有 MetaWorld 任务共享） | `codex/blackwell-fp8-direct` @ `186d153`，clean | 500k 完成；最终 return `757.81`，无数值失败；与 B 的 `816.37` 同量级，明显优于常驻 target 的 `7.23` |
| 2026-08-24 | `dogs_s42_step100k_lag_interleaved` | 完成 | `EXP-FP8-TARGET-OFFLINE-50K`：共享健康 teacher 的四路目标状态筛选 | 42 / GPU 3 | `codex/blackwell-fp8-direct` @ `50814df`，clean | 50k updates / 100 诊断点；lag target relative error `0.00667`，其余三路 `0.346`–`0.352` |
| 2026-08-24 | `dogs_s42_step100k_kahan_smoke100` | 完成 | `EXP-FP8-TARGET-KAHAN-SMOKE`：Kahan 双 E4M3 buffer 的 checkpoint 验收 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `50814df` + Kahan 未提交实现 | 100 updates / 2 诊断点；所有状态与指标有限，无 NaN/Inf |
| 2026-08-24 | `dogs_s42_step100k_kahan_lag_30k` | 完成 | `EXP-FP8-TARGET-KAHAN-OFFLINE-30K`：FP8 Kahan-momentum 与 lag-coded 同轨迹比较 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `50814df` + Kahan 未提交实现 | 30k / 60 点，无 NaN/Inf；Kahan state error `0.25858`，lag `0.00773` |
| 2026-08-24 16:32 | `brc_cheetah_run_target_fp8_lag_s0_smoke` | 完成 | `EXP-FP8-TARGET-LAG-SMOKE`：lag-coded target 闭环短跑 | 0 / GPU 1 | `2091e7e` + lag 未提交实现 | 200 steps / 203 updates；NaN/Inf `0/0`；498 条 tensor stats；HLO 与数值验收通过 |
| 待启动 | `brc_dmc_dogs_target_fp8_lag_s42` | 预登记 | `EXP-FP8-TARGET-LAG-S42`：在线 FP8 Direct、目标 lag 常驻 FP8 | 42 / 待选最空闲 GPU | 待 lag 干净提交 | Dogs 500k；只看闭环学习与稳定性，不设 return 自动停止阈值 |

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

## 已完成：DMC Dogs 四任务联合训练

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

## 已完成：critic bootstrap 单因素消融

- 时间：2026-08-23 00:03–00:23；GPU 1，与原有进程共享 GPU，未中断原任务。
- W&B：[BRC-ablation-DMC-DOGS-bootstrap-only-seed42](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/5mwzsn67)
- 对照：保留原版 L2 task embedding 与 target-entropy correction，仅将 time-limit return bootstrap 从 `reward_mean` 改为 `critic`。
- 为降低成本，运行 25,000 步，关闭 eval、profiling、tensor stats 和 checkpoint；保留每 1,000 步训练指标及资源记录。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_DOGS --seed=42 --eval_seed_offset=0 --max_steps=25000 --start_training=5000 --replay_buffer_size=50000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --paper_alignment=false --task_embedding_norm=l2 --return_bootstrap=critic --entropy_correction=target_entropy --offline_evaluation=false --eval_interval=25000 --eval_episodes=10 --render=false --log_to_wandb=true --wandb_name=BRC-ablation-DMC-DOGS-bootstrap-only-seed42 --run_root=runs --run_id=ablation_bootstrap_only_s42_20260823 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=0 --tensor_stats_interval=0 --analysis_checkpoint_interval=0 --recovery_checkpoint_interval=0 --keep_last_analysis_checkpoints=0 --keep_last_recovery_checkpoints=0 --save_replay_buffer=false
```

关键观察：critic bootstrap 在 step 7,000 已达到归一化支持上界约 10，使每任务 `Gbar` 从 step 6,000 的约 29–33 扩张到 step 10,000 的约 105–109，25,000 时约 113–116；同期真实 episode return 仍远低于该尺度。25,000 步的 train return 为：original 64.1、bootstrap-only 20.1、完整 paper-aligned 18.9。单种子短跑表明 bootstrap 是尺度反馈的主要触发器，经验熵修正会进一步放大该反馈；不能据此替代完整多种子最终性能比较。

### Reward-mean `Gbar` 验证

- 时间：2026-08-23 00:37–00:43；GPU 1，与原有任务共享 GPU且未中断原任务。
- W&B：[BRC-validation-DMC-DOGS-reward-mean-gbar-seed42](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/7egh3qkh)
- 配置：L1 task embedding、reward-mean return scale、per-task empirical entropy；10,000 步，关闭 eval、checkpoint、profiling 和 tensor stats。
- 结果：step 8,000 后 critic Q 仍饱和约 10，但 `Gbar` 到 step 10,000 仅为 `[19.1, 16.8, 11.0, 5.7]`；旧 critic-bootstrap + empirical-entropy 同期约为 `[294.6, 284.8, 287.3, 287.2]`。验证了尺度解耦，不代表最终性能结论。

## DMC Humanoids：original 运行中，旧 paper 主动停止

任务集合为 `humanoid-stand`、`humanoid-walk`、`humanoid-run`。Figure 16 的 DMC-Hard 结果由 Dogs 4-task MT 和 Humanoids 3-task MT 的共 7 个任务聚合得到，并非单个 agent 混合训练两种 embodiment。以下接力任务只在对应 Dogs run 写入 `run_finished` 且达到 500,000 步后启动；Dogs 异常退出或训练代码发生变化时不会启动。

### 原始 BRC（运行中）

- tmux：`brc_dmc_humanoids_old_queue`
- GPU：0
- 预定本地 run ID：`dmc_humanoids_original_s42_20260822`
- W&B 名称：`BRC-DMC-HUMANOIDS-original-seed42`

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python3 train.py --env_names=DMC_HUMANOIDS --seed=42 --eval_seed_offset=0 --max_steps=500000 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true --render=false --log_to_wandb=true --wandb_name=BRC-DMC-HUMANOIDS-original-seed42 --run_root=runs --run_id=dmc_humanoids_original_s42_20260822 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
```

### 旧论文对齐 BRC（step 133k 主动停止）

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

## 替换链：reward-mean `Gbar` Dogs → Humanoids

- 启动：2026-08-23 01:04；tmux `brc_dmc_paper_rewardmean_s42`；GPU 3；seed 42。
- 版本：`codex/paper-alignment` @ `9295a2d`；实际解析为 L1 task embedding、reward-mean return scale、per-task empirical entropy。
- Dogs：[W&B `ttb7gnbz`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/ttb7gnbz)，本地 run ID `dmc_dogs_rewardmean_paper_s42_20260823`。
- Humanoids 本地 run ID `dmc_humanoids_rewardmean_paper_s42_20260823`；Dogs 以退出码 0 完成 500k 后立即启动，Dogs 异常退出则不启动。
- 两段均使用正式配置：500k steps、start training 5k、buffer 1M、batch 1024、2 updates/step、critic width 4096、每 25k 步评估 10 episodes，并启用完整 recorder/checkpoint 策略。

## 常驻 FP8 目标 Critic 的 C/D 实验

短跑通过后，C/D 已从预登记状态切换为运行中。两者均从 `/home/caiyuliang/BRC_FP8_blackwell_fp8` 的干净提交 `f88b6628d39caec72f48e0e22d4d8dde815d1745` 启动，且该提交已先推送并与远端 `origin/codex/blackwell-fp8-direct` 核对一致。

既有严格对照采用相同的 DMC Dogs seed-42 正式协议：A（`dmc_dogs_rewardmean_paper_s42_20260823`）为在线/目标 Critic 均 FP32，最终 eval return `794.40`；B（`dmc_dogs_rewardmean_paper_fp8_direct_s42_20260823`）为在线 Critic `fp8_direct`、目标 Critic FP32，最终 eval return `816.37`。C 直接对照 A 以隔离目标常驻 FP8，D 直接对照 B 以隔离目标常驻 FP8；C/D 之间只用于观察在线 Critic 精度在同一常驻 FP8 目标下的差异。

### `EXP-FP8-TARGET-SMOKE-D` — GPU3 短跑验收

- 状态：审查后最终短跑完成并通过；2026-08-24 00:14:11–00:16:43 Asia/Shanghai，正常写入 `run_finished`。
- 目的：先覆盖最复杂的 D 路径，验证在线 `fp8_direct` 与目标 `fp8_resident` 可共同训练，并验证常驻 codes 写回和新增数值诊断链路。
- 实际运行：GPU 3；tmux `brc_fp8_target_smoke_d_final_gpu3`；本地 run ID `brc_cheetah_run_d_target_fp8_resident_s0_smoke_final`；W&B 关闭，无 W&B ID。运行基于 `9c81c7d` 加本轮未提交实现，因而只用于实现验收，不作为正式对照。GPU3 同时存在一个非本项目、约 8 GiB 的外部任务，故不读取本次 wall clock/吞吐作为性能证据。
- 协议：`cheetah-run`，seed 0，200 env steps，step 100 开始训练，replay 2,000，batch 256，2 updates/step，critic width 512；metrics 每 25 步，tensor stats 在 step 150 采集；evaluation、video、profiling、analysis/recovery checkpoint 和 replay-buffer checkpoint 均关闭。
- 精度：`critic_precision=fp8_direct`，`target_critic_precision=fp8_resident`。
- 验收：运行自身写入 `run_finished`；loss、梯度与 `update_nan_count`/`update_inf_count` 均有限且后两者为 0；目标 E4M3 codes 在 EMA 中确实逐步写回；EMA、activation 与 FP8/反量化参考 forward-error 统计均非空且有限。任一条件不满足则不启动正式 C/D。
- 结果：200 env steps、203 learner updates，受共享 GPU 与额外诊断编译影响总 wall time 为 `151.293 s`；5 个 post-update 指标点的 `update_nan_count`/`update_inf_count` 均为 0，critic loss `8.344`–`12.118`、critic grad norm `1.125`–`21.042`、actor loss `-1.327`–`4.565`、actor grad norm `2.508`–`12.004`，全部有限。
- 诊断：step 150 共写入 394 条 tensor stats，其中按 ensemble 拆开的原始 E4M3 code 8 条、kernel scale 8 条、目标 activation 36 条、最后一次真实写回的 EMA 标量 88 条和按 aggregate/ensemble 拆开的 forward-error 9 条，全部无 NaN/Inf。八个物理矩阵的 code unchanged fraction 为 `0.4316`–`0.5135`，真正 swallowed-update fraction 为 `8.01e-5`–`1.18e-4`，applied/intended L2 ratio 为 `0.999999`–`1.000005`，relative update error 为 `1.25e-4`–`2.23e-4`，sign agreement 不低于 `0.999958`。同一反量化权重参考下，按 BRC 实际 bootstrap 语义先平均 ensemble 概率后得到的 aggregate expected-Q MAE 为 `0.07494`、signed bias 为 `-0.01306`、101-bin probability JS divergence 为 `0.000651`；ensemble-0/1 的 MAE 分别为 `0.08925`/`0.12916`。这些数值只证明记录链路有效并给出机制量级，不构成正式学习效果或性能结论。
- 审查轨迹：此前 run `brc_cheetah_run_d_target_fp8_resident_s0_smoke` 使用了探针时反事实 EMA，随后 `..._smoke_reviewed` 虽改为真实 EMA，但 aggregate forward error 仍是逐成员误差的总体均值。两者都只用于发现并修正记录语义，已由本条 final run 取代，不能引用为最终机制证据。

```bash
tmux new-session -d -s brc_fp8_target_smoke_d_final_gpu3 \
  'env -u LD_LIBRARY_PATH bash -c "
    cd /home/caiyuliang/BRC_FP8_blackwell_fp8 &&
    export CUDA_ROOT=/usr/local/cuda-12.8 &&
    export CUDA_HOME=/usr/local/cuda-12.8 &&
    export PATH=/usr/local/cuda-12.8/bin:\$PATH &&
    export CUDA_VISIBLE_DEVICES=3 &&
    export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
    exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
      --env_names=cheetah-run \
      --seed=0 \
      --max_steps=200 \
      --start_training=100 \
      --replay_buffer_size=2000 \
      --batch_size=256 \
      --updates_per_step=2 \
      --width_critic=512 \
      --critic_precision=fp8_direct \
      --target_critic_precision=fp8_resident \
      --fp8_amax_history_length=1024 \
      --offline_evaluation=false \
      --eval_interval=0 \
      --render=false \
      --log_to_wandb=false \
      --run_root=runs \
      --run_id=brc_cheetah_run_d_target_fp8_resident_s0_smoke_final \
      --metrics_interval=25 \
      --metrics_flush_interval=25 \
      --system_metrics_interval_sec=10 \
      --profile_interval=0 \
      --tensor_stats_interval=150 \
      --analysis_checkpoint_interval=0 \
      --recovery_checkpoint_interval=0 \
      --keep_last_analysis_checkpoints=0 \
      --keep_last_recovery_checkpoints=0 \
      --save_replay_buffer=false
  "'
```

### `EXP-FP8-TARGET-C-S42` — 在线 FP32 / 目标常驻 FP8

- 状态：主动停止；2026-08-24 00:24:56 Asia/Shanghai 启动，2026-08-24 02:27:28 Asia/Shanghai 按用户要求停止。`run_interrupted` 与 `run_finished` 均已写入，tmux/训练进程消失且 GPU3 显存已释放。
- 目的：相对 A 仅改变目标 Critic 的四个逻辑残差 Dense kernel，将其常驻 E4M3、以 FP8 前向并逐步 FP8 EMA；在线 Critic 与 Adam 保持 FP32。
- 实际运行：GPU 3；tmux、本地 run ID 与 W&B 名称均为 `brc_dmc_dogs_c_target_fp8_resident_s42`；[W&B `trzg1mjw`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/trzg1mjw)。
- 协议：DMC Dogs 四任务，seed 42，500,000 env steps/task，step 5,000 开始训练，replay 1M/task，batch 1,024，2 updates/step，critic width 4,096；paper alignment、reward-mean bootstrap；每 25,000 步做 10 episodes 确定性 eval 和 tensor stats；analysis checkpoint 每 50,000 步、recovery checkpoint 每 100,000 步。
- 比较：直接对照 A（FP32/FP32，最终 eval return `794.40`）；同时报告 C/D，但不把单种子差异解释为稳定统计结论。
- 启动验证：metadata 为 clean `f88b662`，CUDA root/home 与 `ptxas 12.8.61` 均解析到 `/usr/local/cuda-12.8`；配置解析为 `critic_precision=fp32`、`target_critic_precision=fp8_resident`、残差 kernel 常驻范围、动态 current-amax per-tensor 权重/激活缩放。recorder 现有 schema 没有名为 `run_started` 的事件；等价的首事件 `wandb_initialization_finished` 与后续 `initialization_finished` 均成功。step 5,000 首次更新完成，critic loss `24.165`、critic grad norm `20.603`、actor loss `26.968`、actor grad norm `22.436`，`update_nan_count=0`、`update_inf_count=0`。
- 停止结果：终止边界为 `env_step=150000`、`global_transition=600000`、`update_step=290003`；最后完整 eval 为 step 125k，return `8.1607`（四任务 `17.5364/4.9029/5.0295/5.1738`）。最后训练指标仍为有限值且 NaN/Inf 为 `0/0`，但相对 A 在 125k 的 `427.68` 已形成决定性学习差距。保留 25k/50k/100k analysis checkpoint 与 100k recovery checkpoint；不续跑当前朴素方案。

```bash
tmux new-session -d -s brc_dmc_dogs_c_target_fp8_resident_s42 \
  'env -u LD_LIBRARY_PATH bash -c "
    cd /home/caiyuliang/BRC_FP8_blackwell_fp8 &&
    export CUDA_ROOT=/usr/local/cuda-12.8 &&
    export CUDA_HOME=/usr/local/cuda-12.8 &&
    export PATH=/usr/local/cuda-12.8/bin:\$PATH &&
    export CUDA_VISIBLE_DEVICES=3 &&
    export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
    exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
      --env_names=DMC_DOGS \
      --seed=42 \
      --eval_seed_offset=0 \
      --max_steps=500000 \
      --start_training=5000 \
      --replay_buffer_size=1000000 \
      --batch_size=1024 \
      --updates_per_step=2 \
      --width_critic=4096 \
      --critic_precision=fp32 \
      --target_critic_precision=fp8_resident \
      --fp8_amax_history_length=1024 \
      --paper_alignment=true \
      --return_bootstrap=reward_mean \
      --eval_interval=25000 \
      --eval_episodes=10 \
      --offline_evaluation=true \
      --render=false \
      --log_to_wandb=true \
      --wandb_name=brc_dmc_dogs_c_target_fp8_resident_s42 \
      --run_root=runs \
      --run_id=brc_dmc_dogs_c_target_fp8_resident_s42 \
      --metrics_interval=1000 \
      --metrics_flush_interval=1000 \
      --system_metrics_interval_sec=10 \
      --profile_interval=25000 \
      --profile_window=10 \
      --tensor_stats_interval=25000 \
      --analysis_checkpoint_interval=50000 \
      --recovery_checkpoint_interval=100000 \
      --keep_last_analysis_checkpoints=2 \
      --keep_last_recovery_checkpoints=1 \
      --save_replay_buffer=true
  "'
```

### `EXP-FP8-TARGET-D-S42` — 在线 FP8 Direct / 目标常驻 FP8

- 状态：主动停止；2026-08-24 00:24:55 Asia/Shanghai 启动，2026-08-24 00:32:38 按用户要求将 D 迁移到 GPU1 而停止。
- 目的：相对 B 仅把目标 Critic 的四个逻辑残差 Dense kernel 改为常驻 E4M3、FP8 前向和逐步 FP8 EMA；在线 Critic 继续使用现有 `fp8_direct`。
- 实际运行：GPU 3；tmux、本地 run ID 与 W&B 名称均为 `brc_dmc_dogs_d_target_fp8_resident_s42`；[W&B `fb02bf23`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/fb02bf23)。
- 协议：除精度开关外与 C 及既有 A/B 的正式 Dogs 协议逐项一致。
- 比较：直接对照 B（在线 FP8 Direct/目标 FP32，最终 eval return `816.37`），并以 C/D 的目标 EMA、activation、forward-error 和 eval 曲线检查数值稳定性与学习效果。
- 启动验证：metadata 为 clean `f88b662`，CUDA/`ptxas` 与 C 相同；配置解析为 `critic_precision=fp8_direct`、`target_critic_precision=fp8_resident` 和同一目标缩放/覆盖语义。`wandb_initialization_finished`、`initialization_finished` 与 step 5,000 的 `first_update_finished` 均成功；首次 critic loss `22.041`、critic grad norm `18.795`、actor loss `26.839`、actor grad norm `22.499`，`update_nan_count=0`、`update_inf_count=0`。
- 停止边界：`env_step=10057`、`global_transition=40228`、`update_step=10115`；事件流完整写入 `run_interrupted`（`KeyboardInterrupt`）和 `run_finished`。首个 checkpoint 门槛为 50k，因此没有可续跑 checkpoint；该短段保留为中断记录，不与新的 R1 拼接，也不作为完整正式结果。

```bash
tmux new-session -d -s brc_dmc_dogs_d_target_fp8_resident_s42 \
  'env -u LD_LIBRARY_PATH bash -c "
    cd /home/caiyuliang/BRC_FP8_blackwell_fp8 &&
    export CUDA_ROOT=/usr/local/cuda-12.8 &&
    export CUDA_HOME=/usr/local/cuda-12.8 &&
    export PATH=/usr/local/cuda-12.8/bin:\$PATH &&
    export CUDA_VISIBLE_DEVICES=3 &&
    export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
    exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
      --env_names=DMC_DOGS \
      --seed=42 \
      --eval_seed_offset=0 \
      --max_steps=500000 \
      --start_training=5000 \
      --replay_buffer_size=1000000 \
      --batch_size=1024 \
      --updates_per_step=2 \
      --width_critic=4096 \
      --critic_precision=fp8_direct \
      --target_critic_precision=fp8_resident \
      --fp8_amax_history_length=1024 \
      --paper_alignment=true \
      --return_bootstrap=reward_mean \
      --eval_interval=25000 \
      --eval_episodes=10 \
      --offline_evaluation=true \
      --render=false \
      --log_to_wandb=true \
      --wandb_name=brc_dmc_dogs_d_target_fp8_resident_s42 \
      --run_root=runs \
      --run_id=brc_dmc_dogs_d_target_fp8_resident_s42 \
      --metrics_interval=1000 \
      --metrics_flush_interval=1000 \
      --system_metrics_interval_sec=10 \
      --profile_interval=25000 \
      --profile_window=10 \
      --tensor_stats_interval=25000 \
      --analysis_checkpoint_interval=50000 \
      --recovery_checkpoint_interval=100000 \
      --keep_last_analysis_checkpoints=2 \
      --keep_last_recovery_checkpoints=1 \
      --save_replay_buffer=true
  "'
```

### `EXP-FP8-TARGET-D-S42-R1` — D 在 GPU1 从头重启

- 状态：主动停止；2026-08-24 00:33:14 Asia/Shanghai 启动，2026-08-24 02:27:28 Asia/Shanghai 按用户要求停止。由于旧 D 尚未达到首个 checkpoint，R1 使用相同 seed 和协议从 step 0 重新开始，不续接、覆盖或拼接旧 run。
- 实际运行：GPU 1；tmux、本地 run ID 与 W&B 名称均为 `brc_dmc_dogs_d_target_fp8_resident_s42_gpu1_r1`；[W&B `iwlomjbu`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/iwlomjbu)。GPU1 同时有一个既有 BRC 训练进程，因此本轮仍不读取 wall clock、功耗或吞吐作为性能证据。
- 版本：metadata 为 clean `1ff6625`；该提交相对正式实现 `f88b662` 只增加科研文档，训练代码不变。配置解析为 `critic_precision=fp8_direct`、`target_critic_precision=fp8_resident`、seed 42、width 4096、batch 1024 和每步 2 次更新。
- 启动验证：两个初始化事件和 step 5,000 的 `first_update_finished` 均已写入；首次 critic loss `24.505`、critic grad norm `20.890`、actor loss `26.848`、actor grad norm `22.670`，`update_nan_count=0`、`update_inf_count=0`。GPU 进程映射、tmux、run 目录和 W&B ID 均已核对；C 始终保留在 GPU3 且未重启。
- 停止结果：事件流终止于 `env_step=137438`、`global_transition=549752`、`update_step=264877`，并完整写入 `run_interrupted`/`run_finished`；tmux/训练进程消失且 GPU1 上对应约 17 GiB 显存已释放。最后完整 eval 为 step 125k，return `7.2317`（四任务 `14.7616/3.6172/4.6979/5.8502`）；最后落盘训练指标仍无 NaN/Inf。相对 B 在 125k 的 `411.66`，该朴素常驻方案已经无法支持继续消耗到 500k。保留 25k/50k/100k analysis checkpoint 与 100k recovery checkpoint供离线机制分析。

```bash
tmux new-session -d -s brc_dmc_dogs_d_target_fp8_resident_s42_gpu1_r1 \
  'env -u LD_LIBRARY_PATH bash -c "
    cd /home/caiyuliang/BRC_FP8_blackwell_fp8 &&
    export CUDA_ROOT=/usr/local/cuda-12.8 &&
    export CUDA_HOME=/usr/local/cuda-12.8 &&
    export PATH=/usr/local/cuda-12.8/bin:\$PATH &&
    export CUDA_VISIBLE_DEVICES=1 &&
    export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
    exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
      --env_names=DMC_DOGS --seed=42 --eval_seed_offset=0 \
      --max_steps=500000 --start_training=5000 \
      --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 \
      --width_critic=4096 --critic_precision=fp8_direct \
      --target_critic_precision=fp8_resident --fp8_amax_history_length=1024 \
      --paper_alignment=true --return_bootstrap=reward_mean \
      --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true \
      --render=false --log_to_wandb=true \
      --wandb_name=brc_dmc_dogs_d_target_fp8_resident_s42_gpu1_r1 \
      --run_root=runs --run_id=brc_dmc_dogs_d_target_fp8_resident_s42_gpu1_r1 \
      --metrics_interval=1000 --metrics_flush_interval=1000 \
      --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 \
      --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 \
      --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 \
      --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
  "'
```

终止核验已完成：C 与 D-R1 的 tmux 和训练进程均已消失，原 GPU3/GPU1 训练显存已释放；GPU2 上 `EXP-FP8-TARGET-FWD-S42` 不受影响并继续运行。C/D 的部分数据保留为朴素常驻 FP8 失败与时间分辨率诊断证据，不作 wall clock、功耗或吞吐结论。

### `EXP-FP8-TARGET-FWD-SMOKE` — 目标 FP8 计算 / FP32 存储短跑

- 状态：完成；2026-08-24 01:16 Asia/Shanghai 在 GPU2 前台运行。GPU2 同时有既有 MetaWorld seed-123 任务，因此 wall clock 与吞吐不作为证据。
- 目的：验证目标 Critic 四个残差 Dense 使用 `fp8_direct` 原生前向时，参数和 `tau=0.005` EMA 仍为 FP32，input/kernel delayed-scaling 状态能随 bootstrap 更新，且不引入目标 backward。
- 版本：`1eecc56` 加本轮未提交实现；只作实现验收。配置解析为 `critic_precision=fp8_direct`、`target_critic_precision=fp8_direct`、目标 FP8 compute scope 为 residual Dense、全部目标参数存储为 FP32。
- 结果：`cheetah-run` seed 0、width 512、batch 256、200 steps、203 learner updates 正常写入 `run_finished`。全部训练采样点 `update_nan_count=0`、`update_inf_count=0`；step 150 写入 298 条 tensor stats，四层目标 input/kernel scale/amax 与 FP32-reference error 均非空且有限。aggregate expected-Q MAE `0.0704`、signed bias `-0.0441`、probability JS divergence `9.09e-4`，仅作为 smoke 数值证据。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false /home/caiyuliang/anaconda3/envs/brc/bin/python train.py --env_names=cheetah-run --seed=0 --eval_seed_offset=0 --max_steps=200 --start_training=100 --replay_buffer_size=2000 --batch_size=256 --updates_per_step=2 --width_critic=512 --critic_precision=fp8_direct --target_critic_precision=fp8_direct --fp8_amax_history_length=1024 --paper_alignment=true --return_bootstrap=reward_mean --eval_interval=0 --eval_episodes=1 --offline_evaluation=true --render=false --log_to_wandb=false --run_root=runs --run_id=brc_cheetah_run_target_fp8_direct_fp32_storage_s0_smoke --metrics_interval=25 --metrics_flush_interval=25 --system_metrics_interval_sec=10 --profile_interval=0 --tensor_stats_interval=150 --analysis_checkpoint_interval=0 --recovery_checkpoint_interval=0 --keep_last_analysis_checkpoints=0 --keep_last_recovery_checkpoints=0 --save_replay_buffer=false
```

### `EXP-FP8-TARGET-FWD-S42` — 目标 FP8 计算 / FP32 存储正式对照

- 状态：完成；2026-08-24 01:24:39 Asia/Shanghai 在 GPU2 的独立 tmux 中启动，正常训练至 env step 500k / learner update 990003 并写入 `run_finished`。
- 目的：在线 Critic 保持现有 FP8 Direct；目标 Critic 的四个残差 Dense 同样采用 FP8 Direct，但全部目标参数与逐步 EMA 保持 FP32。相对 B 只增加目标 FP8 前向；相对 D-R1 只移除常驻 E4M3 存储与 requantized EMA。
- 协议：逐项复用 A/B/C/D 的 Dogs seed-42 正式协议：500k steps、start 5k、replay 1M、batch 1024、2 updates/step、width 4096、paper alignment、reward-mean bootstrap、25k eval/tensor stats、50k analysis checkpoint、100k recovery checkpoint。
- 运行：GPU2，与既有 MetaWorld seed-123 训练共享；tmux、run ID 与 W&B name 均为 `brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2`；[W&B `gij6jhgd`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/gij6jhgd)。因此只比较数值稳定性和学习/eval，不比较 wall clock、功耗或吞吐。
- 决策门：没有预设 return 阈值；完成后先核对协议和有限值，再以 B → 本组判断目标 bootstrap 计算误差，以本组 → D-R1 判断持久 FP8 存储/EMA 的增量影响。
- 启动验证：metadata 为 clean `186d153`；CUDA root/home 与 `ptxas 12.8.61` 均来自 `/usr/local/cuda-12.8`。配置解析为 online/target `fp8_direct`、四个目标残差 Dense FP8 compute、全部目标参数 FP32、delayed per-tensor scaling、仅 bootstrap 前向推进 input/kernel scale state。tmux、PID `71842`、物理 GPU2 绑定、run 目录、W&B ID 和两个初始化事件均已核对。step 5,000 的首次更新完成：critic loss `23.059`、critic grad norm `19.649`、actor loss `26.913`、actor grad norm `22.356`，`update_nan_count=0`、`update_inf_count=0`。
- 最终结果：四任务 return 为 `960.56 / 912.40 / 791.48 / 366.79`，均值 `757.81`。单 seed 下相对在线 FP8、目标 FP32 baseline `816.37` 有一定下降，但仍处于同一学习量级；与 naive resident D-R1 在 125k 的 `7.23` 形成数量级差距，支持将主要失败归因于常驻 E4M3 target 状态与逐步 requantized EMA，而不是目标 FP8 前向本身。

```bash
tmux new-session -d -s brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2 \
  'env -u LD_LIBRARY_PATH bash -c "
    cd /home/caiyuliang/BRC_FP8_blackwell_fp8 &&
    export CUDA_ROOT=/usr/local/cuda-12.8 &&
    export CUDA_HOME=/usr/local/cuda-12.8 &&
    export PATH=/usr/local/cuda-12.8/bin:\$PATH &&
    export CUDA_VISIBLE_DEVICES=2 &&
    export XLA_PYTHON_CLIENT_PREALLOCATE=false &&
    exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
      --env_names=DMC_DOGS --seed=42 --eval_seed_offset=0 \
      --max_steps=500000 --start_training=5000 \
      --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 \
      --width_critic=4096 --critic_precision=fp8_direct \
      --target_critic_precision=fp8_direct --fp8_amax_history_length=1024 \
      --paper_alignment=true --return_bootstrap=reward_mean \
      --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true \
      --render=false --log_to_wandb=true \
      --wandb_name=brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2 \
      --run_root=runs --run_id=brc_dmc_dogs_target_fp8_direct_fp32_storage_s42_gpu2 \
      --metrics_interval=1000 --metrics_flush_interval=1000 \
      --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 \
      --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 \
      --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 \
      --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
  "'
```

## FP8 目标状态离线共享轨迹模拟

### `EXP-FP8-TARGET-OFFLINE-SMOKE` — 100-update GPU 验收

- 状态：完成；100 learner updates，全部方法状态和诊断有限，teacher 与无 shadow 的独立更新逐叶一致。
- 输入：`EXP-FP8-TARGET-FWD-S42` 的 100k 完整 recovery；已用硬链接固定为 `runs/offline_inputs/dogs_target_fwd_s42_step100000`，源 checkpoint 的 eval return 为 `340.61`。
- 目的：验证健康 teacher、lag-coded、naive per-tensor、naive block-scale 与 interleaved block 在 width 4096 下可共同编译，所有状态和功能指标有限，且 teacher 不受 shadow 影响。
- 协议：100 learner updates，checkpoint 原始 batch 1024、2 updates/group、seed/task order/normalizer/replay/RNG 全部恢复；block `128x128`；每 50 updates 诊断。无环境、W&B、return eval 或性能结论。

### `EXP-FP8-TARGET-OFFLINE-50K` — 双方法主筛选

- 状态：完成；输出 `runs/offline_target_simulations/dogs_s42_step100k_lag_interleaved`，50,000 updates、100 个诊断点、teacher/shadow 无 NaN/Inf。
- 输入与 teacher：与 smoke 完全相同；冻结 replay 与 normalizer 统计，25,000 次两-batch 采样组顺序产生 50,000 learner updates。teacher 完整更新 Actor、online Critic、FP32 target 与 temperature；四个 shadow 逐步读取同一 `online_old/online_new`，不反向影响 teacher。
- 方法：主方法为 per-tensor E4M3 lag-coded target 与 `128x128` 动态 block-scale 的确定性交错 EMA；控制为 naive per-tensor resident 与 naive per-step block scale。不实现 Kahan、随机舍入或 sub-ULP 状态。
- 诊断：每 500 updates，共 100 点；记录累计参数方向/范数、target drift、codes/block 事件、动态 scale、FP32 reference expected-Q/101-bin JS、完整 categorical Bellman target 误差，以及 teacher FP8 Direct 计算噪声底。
- 判断：interleaved 只与 naive block 比较时间调度增量；lag 必须接近 FP32 teacher，而非仅优于已知失败的 naive per-tensor。无预设成功阈值，且本轮不自动启动闭环训练。
- 结果：lag-coded 终点 target relative error `0.006665`、displacement ratio `1.000354`、cosine `0.999808`、expected-Q MAE `0.001160`。naive per-tensor、naive block 和 interleaved 的 relative error 分别为 `0.345703/0.352283/0.346894`；interleaved 相对 naive block 的改善很小，首版时间调度方案不继续推进。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/caiyuliang/anaconda3/envs/brc/bin/python scripts/simulate_fp8_target_updates.py \
  --checkpoint=runs/offline_inputs/dogs_target_fwd_s42_step100000 \
  --num_updates=50000 --block_size=128 --diagnostic_interval=500 \
  --output_root=runs/offline_target_simulations \
  --run_id=dogs_s42_step100k_lag_interleaved
```

### `EXP-FP8-TARGET-KAHAN-OFFLINE-30K` — FP16 SAC Kahan-momentum 的 FP8 基线

- 状态：完成。2026-08-24 15:11:15–15:33:12 Asia/Shanghai，GPU1，tmux `brc_fp8_target_kahan_offline_30k_gpu1`；teacher step `190003→220003`，30,000 updates、60 个诊断点，完成事件和完整输出均存在，teacher/shadow NaN/Inf 为 `0/0`。先行 100-update smoke 也完成且有限。
- 输入与 teacher：复用固定的 `dogs_target_fwd_s42_step100000` recovery，恢复相同 replay、normalizer、模型和 RNG；30,000 learner updates，不进行环境交互。
- 方法：在相同四个残差 kernel 上增加 scaled Kahan-momentum shadow。按论文使用 `C=1e4`，`C*target` 与 compensation 分别持久存为 E4M3 + ensemble 独立 current-amax per-tensor scale；不保留 FP32 target master。
- 比较：主要比较 `kahan_momentum ↔ lag_coded`，并保留原 naive/block shadows。由于 Kahan 需要两张 FP8 矩阵而 lag 只需一张，除误差外也报告状态开销差异。
- 诊断：每 500 updates，共 60 点；与 50K screen 使用完全相同的参数位移、expected-Q、101-bin JS、Bellman target 和非有限值指标。
- 限制：动态 scale 会使 `C` 的纯放大作用在数学上抵消；本实验检验的是 Kahan compensation 能否在 FP8 状态下恢复更新，而不是把 `C=1e4` 当作新的可调收益来源。
- 结果：30k 终点 lag/Kahan 的 target relative error 为 `0.007725/0.258575`，displacement L2 ratio 为 `1.000292/0.207277`，cosine 为 `0.999540/0.030080`；expected-Q MAE 为 `0.001216/0.141415`，Bellman expected-Q MAE 为 `0.001204/0.140001`，probability JS 为 `3.26e-6/0.012536`。
- 归因：Kahan 与 naive per-tensor 在全部 60 点几乎重合；两者 target relative error 的最大绝对差仅 `6.26e-7`，expected-Q MAE 最大差 `2.30e-6`。compensation 并非未更新：终点八张物理矩阵的 compensation amax 为 `0.00124–0.00144`，但它没有改变可见 E4M3 target 轨迹。Kahan 还需要两份 E4M3 matrix state，lag 只需一份。
- 可比性：本次 run 内五个 shadow 严格共享同一 teacher，因此 Kahan↔lag 归因有效。它与早先 50k run 虽从同一 checkpoint/协议恢复，但 GPU replay 并非 bitwise 相同；不把跨 run 的逐点差异解释为方法差异。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/caiyuliang/anaconda3/envs/brc/bin/python scripts/simulate_fp8_target_updates.py \
  --checkpoint=runs/offline_inputs/dogs_target_fwd_s42_step100000 \
  --num_updates=30000 --block_size=128 --diagnostic_interval=500 \
  --output_root=runs/offline_target_simulations \
  --run_id=dogs_s42_step100k_kahan_lag_30k
```

## Lag-coded FP8 目标闭环

### `EXP-FP8-TARGET-LAG-SMOKE`

- 状态：完成；2026-08-24 16:32 Asia/Shanghai 在 GPU1 前台运行，200 env steps / 203 learner updates，正常写入 `run_finished`。
- 版本：`2091e7e` 加待提交 lag 实现；只作实现验收，不作为正式学习或性能证据。
- 配置：`cheetah-run`、seed 0、width 512、batch 256、2 updates/step、start 100；在线 `fp8_direct`、目标 `fp8_lag`；W&B/eval/video/profile/checkpoint 关闭，tensor stats 在 step 150 记录。
- 验收：训练 NaN/Inf `0/0`，loss、梯度、重建 target、lag codes/scale 和 metadata 全部有限；498 条 tensor stats。八张物理矩阵的 lag underflow 为 `1.14e-5–5.39e-5`，applied/intended L2 为 `0.999993–1.000002`，相对更新误差为 `1.09e-4–1.55e-4`。
- HLO：Blackwell/CUDA 12.8 target-only 优化后 HLO 含恰好四个 `__cublas$lt$matmul$f8`；RHS 明确执行 lag E4M3 解码、乘 scale、加 FP32 online、再转 E4M3；无 E5M2 target backward。

```bash
env -u LD_LIBRARY_PATH CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false /home/caiyuliang/anaconda3/envs/brc/bin/python train.py --env_names=cheetah-run --seed=0 --eval_seed_offset=0 --max_steps=200 --start_training=100 --replay_buffer_size=2000 --batch_size=256 --updates_per_step=2 --width_critic=512 --critic_precision=fp8_direct --target_critic_precision=fp8_lag --fp8_amax_history_length=1024 --paper_alignment=true --return_bootstrap=reward_mean --eval_interval=0 --eval_episodes=1 --offline_evaluation=true --render=false --log_to_wandb=false --run_root=runs --run_id=brc_cheetah_run_target_fp8_lag_s0_smoke --metrics_interval=25 --metrics_flush_interval=25 --system_metrics_interval_sec=10 --profile_interval=0 --tensor_stats_interval=150 --analysis_checkpoint_interval=0 --recovery_checkpoint_interval=0 --keep_last_analysis_checkpoints=0 --keep_last_recovery_checkpoints=0 --save_replay_buffer=false
```

### `EXP-FP8-TARGET-LAG-S42`

- 状态：预登记；仅在 lag 实现测试、留痕、提交、推送且 worktree clean 后启动。
- 目的：首轮闭环可行性。比较 B `816.37`、target-forward `757.81` 与 naive resident D-R1 的 125k `7.23`；单 seed 不作统计显著性或吞吐结论。
- 协议：Dogs seed 42、500k、start 5k、replay 1M、batch 1024、2 updates/step、width 4096、paper alignment、reward-mean bootstrap、25k eval/tensor stats、50k analysis、100k recovery。在线 `fp8_direct`，目标 `fp8_lag`。除 NaN/Inf 或运行故障外不提前停止。
- 名称：tmux、run ID 与 W&B name 统一为 `brc_dmc_dogs_target_fp8_lag_s42`；GPU 在启动前按空闲显存选择并在启动记录中补齐。

## 更新规则

- 启动实验时新增一行，记录开始时间、run ID、分支、commit 和完整命令。
- 结束后补充最终 eval、最好 eval，以及末 3 次 eval 均值；异常退出则记录停止 step 和原因。
- 对比前先检查 reset 模式、训练/eval seed 协议和代码 commit，协议不同的结果不直接合并统计。
