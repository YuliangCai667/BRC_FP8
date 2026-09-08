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
| 2026-08-24 16:47 | `brc_dmc_dogs_target_fp8_lag_s42` / `nu2d5b90` | 完成 | `EXP-FP8-TARGET-LAG-S42`：在线 FP8 Direct、目标 lag 常驻 FP8 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `1c135af`，clean | 500k 最终 / 最佳 / 末 3 次平均 return `806.10 / 806.73 / 799.81`；`run_finished` |
| 2026-09-01 11:09 | `brc_dmc_dogs_online_fp8_resident_s42-20260901-110927` / `fm76bnld` | 主动停止 | `EXP-FP8-ONLINE-RESIDENT-DIAG-S42-A`：online residual kernel E4M3 常驻机制诊断 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `124de8a` + 未提交实现 | 停止于 env step 276144；275k eval `374.51`；critic pnorm 已增至约 6.9k，明显高于匹配 control |
| 2026-09-01 15:18 | `brc_dmc_dogs_online_fp8_resident_s42-20260901-151810` / `vv3x4xuh` | 完成 | `EXP-FP8-ONLINE-RESIDENT-DIAG-S42-B`：增加 optimizer/quantization-error 径向 cosine 后从头重跑 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `124de8a` + 未提交实现 | 500k 最终 / 最佳 / 末 3 次平均 return `505.98 / 530.14 / 512.29`；critic pnorm `11428.8`，诊断为 scale-dominated norm runaway |
| 2026-09-02 05:19 | `brc_dmc_dogs_online_fp8_resident_s42-20260902-051913` / `ozhbrc8v` | 完成 | `EXP-FP8-ONLINE-RESIDENT-MECH-S42`：angular/norm/decay/scale-code 机制诊断增强 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `124de8a` + 未提交诊断实现 | 500k 最终 / 最佳 / 末 3 次 `667.78 / 667.78 / 625.50`；pnorm `9641.4`；NaN/Inf `0/0` |
| 2026-09-02 05:23 | `brc_dmc_dogs_online_fp8_resident_s1-20260902-052342` / `zymfcugz` | 完成 | `EXP-FP8-ONLINE-RESIDENT-MECH-S1`：同协议第二种子 | 1 / GPU 0 | `codex/blackwell-fp8-direct` @ `124de8a` + 同一未提交诊断实现 | 500k 最终 / 最佳 / 末 3 次 `564.03 / 564.03 / 534.04`；pnorm `19714.7`；NaN/Inf `0/0` |
| 2026-09-03 03:11 | `brc_dmc_dogs_online_fp8_resident_canonical_s42-20260903-031138` / `fsd8vwfb` | 主动停止（moving-anchor implementation failure） | `EXP-FP8-ONLINE-CANON-S42`：moving-anchor FP8 Residency | 42 / GPU 2（共享） | `codex/blackwell-fp8-direct` @ `5500f6a` + 未提交 moving-anchor 实现 | 停止于 env step 55152 / update 100305；55k pnorm `2041.5`；resident norm 25k→50k `804.1→1833.9`；不属于 fixed-anchor 结果 |
| 2026-09-03 03:11 | `brc_dmc_dogs_online_fp8_resident_canonical_s1-20260903-031138` / `97nf6rie` | 主动停止（moving-anchor implementation failure） | `EXP-FP8-ONLINE-CANON-S1`：同协议第二种子 | 1 / GPU 3（共享） | 同一 dirty source snapshot | 停止于 env step 47628 / update 85257；47k pnorm `1763.0`；不属于 fixed-anchor 结果 |
| 2026-09-03 05:04 | `brc_dmc_dogs_online_fp8_resident_fixed_anchor_s42-20260903-050448` / `crwzhu6b` | 25k 完成；机制门通过 | `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-25K`：fixed-anchor 机制门 | 42 / GPU 3（共享） | `codex/blackwell-fp8-direct` @ `5500f6a` + 未提交 fixed-anchor 实现 | checkpoint NumPy float64：8/8 ratio `0.9999168–1.0002157`；25k pnorm `359.29`；允许 recovery 续跑 500k |
| 2026-09-03 05:36 | 同一 run / W&B | 主动停止（backward-invalid negative control） | `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-R500K`：25k recovery→500k | 42 / GPU 3（共享） | 同一 dirty fixed-anchor source snapshot | 15:15:06 停止于 env `452791` / update `895583`；450k eval `608.65`；anchor 稳定但 resident kernel backward 全零 |
| 2026-09-03 15:37 | `brc_dmc_dogs_online_fp8_resident_scaled_backward_s42-20260903-153745` / `z6n3ca1s` | 运行中 | `EXP-FP8-ONLINE-SCALED-BWD-S42-500K`：scale-aware repaired backward、无 canonicalization | 42 / GPU 3 | `codex/blackwell-fp8-direct` @ `5500f6a` + 未提交 repaired-backward 实现 | 150k eval `177.48`；75k norm runaway 已复现，梯度非零、NaN/Inf `0/0` |
| 2026-09-03 17:16 | `brc_dmc_dogs_online_fp8_resident_fixed_anchor_s42-20260903-171625` / `p7osuon0` | 运行中；100k 已否定充分性 | `EXP-FP8-ONLINE-SCALED-BWD-FIXED-ANCHOR-S42-500K`：repaired backward + fixed-initial anchor | 42 / GPU 0 | 同一 HEAD + 未提交 repaired-backward/fixed-anchor 实现 | 100k eval `134.44`，低于 repaired-only/direct 的 `157.98/340.61`；8/8 anchor 独立 ratio `0.9999815–1.0004089`，梯度非零 |
| 2026-09-03 18:47 | `brc_dmc_dogs_online_fp8_current_master_s42-20260903-184747` / `vgd8apty` | 运行中；100k 因果结论成立 | `EXP-FP8-ONLINE-CURRENT-MASTER-S42-150K`：current-amax FP8 compute、scale-aware backward、FP32 online master | 42 / GPU 1 | 同一 HEAD + 未提交 current-master 隔离对照 | 100k return `244.89`，高于 repaired/fixed `157.98/134.44`，direct `340.61`；内部轨迹贴近 direct，run 继续 150k |
| 2026-09-04 02:33 / 02:41 | `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_smoke_s42-20260904-023355` / `...-024134` | 完成 | `EXP-FP8-ONLINE-CARRY-S42-SMOKE-5K`：CARRY-FP8 工程 smoke 与相邻步诊断 | 42 / GPU 1 | `codex/blackwell-fp8-direct` @ `5500f6a` + 未提交 CARRY-FP8 与既有研究改动 | 5001 env / 5 updates；四层梯度非零；carry saturation `0`；checkpoint 恢复后的下一批更新重建误差降低 `39.65x–43.32x`；无 NaN/Inf |
| 2026-09-04 02:47 | `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_s42` / `a4h7dp4d` | 主动停止；机制无效 | `EXP-FP8-ONLINE-CARRY-S42-500K`：pre-barrier negative control | 42 / GPU 1 | 同一 HEAD + recorder 保存的当前 dirty source snapshot | 04:23 停止于 env 136418 / update 262837；GPU JIT 旁路 FP8 narrowing/widening，carry 8/8 全零，不能作为 CARRY treatment |
| 2026-09-04 02:47 | `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_s1` / `ddivbdzt` | 主动停止；机制无效 | `EXP-FP8-ONLINE-CARRY-S1-500K`：pre-barrier negative control | 1 / GPU 2 | 同一 HEAD + recorder 保存的当前 dirty source snapshot | 04:23 停止于 env 136339 / update 262679；同一 code-side barrier 缺失，不能作为 CARRY treatment |
| 2026-09-04 04:48 | `...jitbarrier_v1_s42` / `7mdayzva`; `...jitbarrier_v1_s1` / `hxf61n84` | seed42 主动停止；seed1 完成 | `EXP-FP8-ONLINE-CARRY-JITBARRIER-S42/S1-500K`：修复后双 seed 正式实验 | 42 / GPU 1；1 / GPU 2 | `codex/blackwell-fp8-direct` @ `5500f6a` + 未提交修复/研究改动 | seed42 于 454409 停止以替换 target-lag，450k eval/best `832.74`；seed1 500k final/best/末3均值 `782.86/807.60/793.28`；全程 NaN/Inf 0 |
| 2026-09-04 10:32 | `...jitbarrier_v1_target_lag_s42` / `5o2266m9` | 运行中；25k 机制/学习门通过 | `EXP-FP8-ONLINE-CARRY-TARGET-LAG-S42-500K`：online CARRY-FP8 + target lag-coded FP8 | 42 / GPU 1 | 同一 branch/HEAD + 当前未提交实现；fresh initialization | 25k eval `58.07`；两侧 8/8 持久码非零，925 条 tensor stats NaN/Inf 0；继续观察 50k/75k |

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

- 状态：完成；2026-08-24 16:47:19 Asia/Shanghai 从干净提交 `1c135af` 在 GPU1 的独立 tmux 中从头启动，2026-08-24 运行至 500k 并写入 `run_finished`。
- 目的：首轮闭环可行性。比较 B `816.37`、target-forward `757.81` 与 naive resident D-R1 的 125k `7.23`；单 seed 不作统计显著性或吞吐结论。
- 协议：Dogs seed 42、500k、start 5k、replay 1M、batch 1024、2 updates/step、width 4096、paper alignment、reward-mean bootstrap、25k eval/tensor stats、50k analysis、100k recovery。在线 `fp8_direct`，目标 `fp8_lag`。除 NaN/Inf 或运行故障外不提前停止。
- 运行：tmux、run ID 与 W&B name 统一为 `brc_dmc_dogs_target_fp8_lag_s42`，PID `1599548`，物理 GPU1；[W&B `nu2d5b90`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/nu2d5b90)。
- 启动验证：提交与远端均为 `1c135aff90d031eb785d55d320f8bdf2044dd56e`，metadata `git_status` 为空；CUDA root/home 与 `ptxas 12.8.61` 均来自 `/usr/local/cuda-12.8`。解析配置明确记录 online `fp8_direct`、target `fp8_lag`、E4M3 lag 常驻、FP32 online+lag 重建、无 derived cache。step 5,000 首更完成：critic loss `23.401`、critic grad norm `19.977`、actor loss `26.865`、actor grad norm `22.586`，NaN/Inf `0/0`。
- 结果：20 次定期 eval 均完整；500k 最终 return `806.0978`，最佳 `806.7286`，末 3 次平均 `799.8090`，125k 为 `422.0946`。终点 checkpoint 和 recovery checkpoint 完整。单 seed 不建立统计等价，但相对 naive resident D-R1 的 125k `7.23` 已恢复学习，并与 B 最终 `816.37` 同量级。
- 解释：lag-coded 修复了 target 绝对权重存储的时间分辨率问题；它仍依赖 FP32 online anchor，因此不能被当作 online/full-chain 解法。

```bash
tmux new-session -d -s brc_dmc_dogs_target_fp8_lag_s42 \
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
      --target_critic_precision=fp8_lag --fp8_amax_history_length=1024 \
      --paper_alignment=true --return_bootstrap=reward_mean \
      --eval_interval=25000 --eval_episodes=10 --offline_evaluation=true \
      --render=false --log_to_wandb=true \
      --wandb_name=brc_dmc_dogs_target_fp8_lag_s42 \
      --run_root=runs --run_id=brc_dmc_dogs_target_fp8_lag_s42 \
      --metrics_interval=1000 --metrics_flush_interval=1000 \
      --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10 \
      --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 \
      --recovery_checkpoint_interval=100000 --keep_last_analysis_checkpoints=2 \
      --keep_last_recovery_checkpoints=1 --save_replay_buffer=true
  "'
```

## 运行中：Online Critic E4M3 常驻诊断

### `EXP-FP8-ONLINE-RESIDENT-DIAG-S42-A/B`

- 状态：A 于 2026-09-01 11:09 Asia/Shanghai 在 GPU1 启动，按用户要求于 15:15 主动停止；事件流终止于 env step 276144 / update 542289，并写入 `run_interrupted` 与 `run_finished`。B 于 15:18 从头启动并于 2026-09-01 22:41:54 Asia/Shanghai 正常完成 500k / update 990003；最终 checkpoint 和 `run_finished` 完整。
- 目的 / Idea：Idea 002；隔离 online Critic 的 E4M3 权重写回，判别 update resolution、scale coupling、Q-function sensitivity 和 bootstrap feedback。
- 拟定范围：四个 online residual-Dense kernel 常驻 E4M3 并直接作为 FP8 GEMM RHS；无 FP32 weight master；Adam moments、target 参数/EMA、其余 Critic leaves、Actor 保持 FP32，target residual GEMM 使用 `fp8_direct`。
- 比较：主控制是已完成的 `EXP-FP8-TARGET-FWD-S42`（online/target 均 `fp8_direct`，target 存储/EMA FP32，最终 `757.81`）；新臂只把 online 改为 `fp8_resident`。B（online `fp8_direct` + 全 FP32 target，`816.37`）是次要上限对照。`target_critic_precision=fp8_resident/fp8_lag` 均关闭。
- 诊断：在现有 25k tensor-stat 间隔的最后一次真实更新上，记录 intended L2、applied/intended L2、cosine、swallowed/code-unchanged fraction、weight-relative error、scale log2 ratio 和 fixed-old-scale 反事实；同 batch 只返回 candidate 对 resident write 的 expected-Q、JS 与 loss 标量误差。不重复记录 raw codes 或完整更新张量。B 额外记录 `intended_update_radial_cosine`（旧物理权重与 optimizer intended update）和 `quantization_error_radial_cosine`（candidate 与 resident write error），用于把范数增长归因到 optimizer 子步骤或 FP8 投影子步骤。
- 正式协议：Dogs seed 42、width 4096、500k、batch 1024、2 updates/step、25k eval/tensor stats；dynamic per-tensor current-amax scale。GPU 由用户运行时选择，除非 NaN/Inf 或运行故障，不设 return 自动停止阈值。
- A 证据：run `brc_dmc_dogs_online_fp8_resident_s42-20260901-110927` / W&B `fm76bnld`；275k eval return `374.5096`，最后周期 train metric（276k）critic pnorm `6898.855`，无 update NaN/Inf。四个 resident kernel 在 250k 贡献 critic 范数平方的 98.4%；单步写回 expected-Q error 很小，因此该结果支持长期 resident 状态漂移，但尚未证明 bootstrap 因果机制。
- B 运行：run `brc_dmc_dogs_online_fp8_resident_s42-20260901-151810` / [W&B `vv3x4xuh`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/vv3x4xuh)；GPU1，tmux `brc_fp8_radial_s42`，PID 启动时为 `3895504`。命令为 `scripts/run_dogs_online_fp8_resident.sh 1 42`；工作树为 `codex/blackwell-fp8-direct` @ `124de8a` 加未提交实现。按用户明确要求，新增指标后未做 smoke、测试或额外校验，直接启动正式实验。
- B 结果：最终 / 最佳 / 末 3 次平均 eval return 为 `505.9815 / 530.1448 / 512.2925`，匹配 target-forward-only control 为 `757.8079 / 786.6346 / 774.9461`；四任务最终 return 均更低。最终 critic pnorm `11428.8`，control 为 `2712.0`；四个 resident kernels 占 critic 范数平方 `99.0%`，无 update NaN/Inf。A/B 在 25k–275k 的 pnorm 和 eval 轨迹近似复现，排除单次偶然曲线。
- 机制诊断：160 个稀疏 layer-member 样本中，optimizer radial cosine 均值 `+0.00392`、按 intended-L2 加权 `+0.00131`、正号率 `52.5%`，不支持稳定 optimizer 径向外推；quantization-error radial cosine 均值 `+0.0833`、正号率 `51.9%`，单点正负幅度很大，不能仅凭 unweighted cosine 宣称单向偏置。原有 applied/intended L2 均值 `1.00002`、update cosine `0.999991`；swallowed fraction 从前 100k 平均 `0.052%` 增至 400k 后 `0.731%`，不是主失败机制。expected-Q MAE 均值 `8.10e-6`、JS 约 `9.89e-9`、loss relative error `1.62e-7`，排除单步 Q 敏感性。
- Scale 证据：25k→500k，最大 kernel scale 从 `8.22e-4` 增至 `2.98e-2`；四个 runaway member 的典型 scale 倍率为 `16.4x/36.2x/16.8x/22.8x`。其中三个主要 runaway member 的 25k→500k code cosine 为 `0.9998/1.0000/1.0000`，code norm 几乎不变，而物理 norm 与 scale 同比例增长；最极端 `Block0/Dense1/ensemble1` 的 physical norm `264.2→9573.8`。这支持 LayerNorm 尺度自由度上的 smooth scale-dominated runaway，而非 code-space 爆炸或突发 scale jump。稀疏径向样本不足以断言每步 FP8 projection 都有正偏，bootstrap amplification 仍未被因果证明。
- 实现记录：[2026-08-28-online-fp8-resident-critic.md](2026-08-28-online-fp8-resident-critic.md)；各 run 已按启动时的实际 commit/dirty state 记录，不能用后来发布的分支提交替代其 provenance。
- 后续因果门：若闭环失败但单步权重→Q 误差小，再设计匹配的 FP32 teacher-target open-loop 对照；不仅凭时序相关宣称 bootstrap amplification。

### `EXP-FP8-ONLINE-RESIDENT-MECH-S42/S1`

- 状态：2026-09-02 05:19:13 Asia/Shanghai 在 GPU1、tmux
  `brc_fp8_mechanism_s42` 启动；run
  `brc_dmc_dogs_online_fp8_resident_s42-20260902-051913`，W&B
  `ozhbrc8v`。
- 代码：`codex/blackwell-fp8-direct` @ `124de8a9e916` 加未提交诊断实现；
  算法、quantizer、optimizer 和正式超参数不变。
- 配置：复用上一轮 Dogs resident seed42/width4096/batch1024/2 updates
  /500k/25k eval+tensor-stat 协议；online `fp8_resident`，target 参数与 EMA
  FP32 常驻、target residual forward 为 `fp8_direct`，actor 与 optimizer state
  FP32。
- 启动验收：W&B 和模型初始化完成，step 5000 首次真实 update 完成；critic
  loss/gnorm/pnorm `7.79685 / 17.9412 / 328.690`，训练 NaN/Inf `0/0`。
- 启动命令：`scripts/run_dogs_online_fp8_resident.sh 1 42`。
- 第二种子：2026-09-02 05:23:42 在 GPU0、tmux
  `brc_fp8_mechanism_s1_gpu0` 启动；run
  `brc_dmc_dogs_online_fp8_resident_s1-20260902-052342`，W&B
  `zymfcugz`。step 5000 首次 update 正常完成，critic
  loss/gnorm/pnorm `5.27780 / 12.4531 / 328.688`，NaN/Inf `0/0`。启动命令
  为 `scripts/run_dogs_online_fp8_resident.sh 0 1`。
- 完成：两者均在 2026-09-02 12:05 Asia/Shanghai 正常写入 500k、
  update 990003、20 次 eval、20 次 tensor stats、最终 analysis/recovery
  checkpoint 和 `run_finished`。seed42 最终 / 最佳 / 末 3 次 return 为
  `667.78 / 667.78 / 625.50`，seed1 为
  `564.03 / 564.03 / 534.04`；最终 pnorm 分别为
  `9641.4 / 19714.7`，所有训练 NaN/Inf 为 `0/0`。
- Angular：resident-kernel 全局 intended angular step 从 25k 到 500k
  分别下降 `19.9x / 33.4x`，而实际/intended angular retention 的全样本
  中位数为 `1.0000005 / 1.0000019`，95% 绝对偏差仅
  `4.31e-5 / 7.38e-5`。seed42 tangential L2 `0.171→0.186`，下降几乎全由
  norm 分母造成；seed1 tangential L2 `0.195→0.136`，但 norm
  `847.5→19675.7` 仍是主要因素。
- Norm 分解：25k→500k resident norm-square 增量要求平均每次 update 增加
  `96.2 / 406.8`；稀疏诊断中的二阶项平均仅 `0.0404 / 0.0365`，量级相差
  `2.4e3x / 1.1e4x`，排除二阶切向累积为主因。诊断时点的一阶项并非
  持续为正，因此现有 25k 采样只能说明长期增长必由未采样时刻累计的一阶
  径向事件承担，不能声称每一步都有固定 outward bias。
- Decay：按 intended radial 加权，FP8 write 后 decay retention 为
  `0.829 / 0.937`，说明有轻度削弱但单点值受同量级量化 residual 干扰。
  原 AdamW `lr*wd=3e-8/update` 即使完全保留，990k updates 也只提供约
  `2.93%` 累计收缩，故 decay 削弱不可能解释 `21.7x / 23.2x` norm 增长。
- Code/scale：25k→500k，runaway member 的 scale 增长最高
  `39.9x / 71.5x`；除 seed42 一个 member 的 code L2 降至 `0.898x` 外，
  code L2 基本保持在 `0.99x–1.01x`。code unchanged fraction 从
  `0.628→0.870 / 0.640→0.991`，但 sampled write 的 angular retention
  仍近 1；因此是长期 scale-dominated gauge drift 与 effective angular
  step collapse，不是单步 directional write failure。单步 aggregate
  expected-Q MAE 均值仅 `6.50e-6 / 1.07e-5`。
- 数值说明：直接做两个大 FP32 norm-square 的差在后期出现 cancellation，
  部分 member 记录为零；误差仍位于估计的 FP32 cancellation bound 内。
  机制判断使用稳定的 `2<W,delta>+||delta||^2` 分解，而不使用后期 direct
  difference 的相对误差。
- 对照与限制：匹配 seed42 FP8-direct/FP32-weight control 的最终 / 最佳 /
  末 3 次为 `757.81 / 786.63 / 774.95`，pnorm `2712.0`；本轮 seed42
  仍较差。seed1 没有匹配 control，因此两个 treatment seed 不能形成两种子
  paired quality estimate。上一轮相同 seed42/config 的 resident run 最终
  `505.98`，而本轮 `667.78`；source snapshot 除诊断代码外一致，说明长期
  return 对 GPU 数值/轨迹扰动敏感，但 norm/scale/angular 机制跨三次 run
  保持复现。

## 已停止：Moving-Anchor FP8 Residency（implementation failure）

### `EXP-FP8-ONLINE-CANON-S42/S1`

这两条 run 只检验 previous-physical-norm moving anchor，不是 fixed-initial-
anchor canonicalization 的结果；其目录、W&B 和停止记录全部保留。

- 状态：两者于 2026-09-03 03:11:38 Asia/Shanghai 直接启动 500k。按用户
  明确要求，没有等待现有 GPU 进程结束，并跳过 10k/Phase C 50k 预跑。
  在确认主要 norm invariant 失败后，按用户要求于 04:12:35 向两个 tmux
  会话发送 `Ctrl-C`。seed42 在 env step `55152` / update `100305`、seed1 在
  env step `47628` / update `85257` 分别写入 `run_interrupted`（
  `KeyboardInterrupt`）与 `run_finished`；04:13:25 复核时两个 tmux、原 PID
  `4136808/4136812` 及其 GPU compute process 均已不存在。
- 目的 / Idea：Idea 002；以 per-update norm canonicalization 直接干预已诊断的
  `scale-gauge norm runaway → effective angular-step collapse`，检验机制和最终
  return 是否同时恢复。
- 代码：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，分支
  `codex/blackwell-fp8-direct`，HEAD `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958`
  加未提交实现与研究记录；变更说明见
  [2026-09-03-gauge-fixed-fp8-residency.md](2026-09-03-gauge-fixed-fp8-residency.md)。
- 协议：Dogs、500k、seed `42/1`、eval seed offset 0、width 4096、batch 1024、
  每环境步 2 updates、25k eval/tensor stats、10 eval episodes；online
  `fp8_resident` + canonicalization，target `fp8_direct` + FP32 persistent params，
  FP32 AdamW moments，current-amax E4M3 code 不变。
- 对照：seed42 对比 `EXP-FP8-TARGET-FWD-S42` 和
  `EXP-FP8-ONLINE-RESIDENT-MECH-S42`；seed1 对比
  `EXP-FP8-ONLINE-RESIDENT-MECH-S1`，但没有 matched FP32-weight seed1 control。
- 资源：登记时 GPU2 已有 PID `3248781` 使用约 24.5 GiB，GPU3 已有 PID
  `3250250` 使用约 29.2 GiB；故不使用 wall clock、吞吐或能耗作方法证据。
- seed42：tmux `brc_fp8_canonical_s42_gpu2`，PID `4136808`，run
  `brc_dmc_dogs_online_fp8_resident_canonical_s42-20260903-031138`，
  [W&B `fsd8vwfb`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/fsd8vwfb)，
  目录 `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_canonical_s42-20260903-031138`；
  命令 `scripts/run_dogs_online_fp8_resident_canonical.sh 2 42`。
- seed1：tmux `brc_fp8_canonical_s1_gpu3`，PID `4136812`，run
  `brc_dmc_dogs_online_fp8_resident_canonical_s1-20260903-031138`，
  [W&B `97nf6rie`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/97nf6rie)，
  目录 `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_canonical_s1-20260903-031138`；
  命令 `scripts/run_dogs_online_fp8_resident_canonical.sh 3 1`。
- 启动验收：03:11:49 两个 tmux 和训练 PID 均存在；NVIDIA-SMI 将 PID
  `4136808/4136812` 分别映射到 GPU2/GPU3；两个 run 的 config 均解析为
  `max_steps=500000`、正确 seed、online `fp8_resident`、
  `fp8_resident_canonicalization=true`、resolved canonicalization
  `preserve_previous_kernel_norm_with_paired_bias`、target `fp8_direct`；W&B、
  recorder 和 model initialization 均完成，未见立即错误。
- 2026-09-03 04:11 Asia/Shanghai 机制核验：两条 tmux/PID 仍存活，但主要
  norm invariant 已失败。seed42 的 `critic_pnorm` 在 env step
  `5k/10k/25k/50k` 为 `328.7/347.9/842.9/1862.4`；四个 resident kernel
  合并 norm 从 `804.1` 增至 `1833.9`（`2.281x`）。matched naive seed42
  同期为 `441.1→794.1`（`1.800x`），所以不是总 pnorm 中其他 FP32 leaves
  的口径假象。seed1 在 `5k/10k/25k` 的 pnorm 为
  `328.7/366.3/782.3`，同样未被固定。
- 矛盾诊断：seed42 的 sampled per-write
  `post_to_old_kernel_norm_ratio` 仍落在
  `0.99999988–1.00000012`，但 25k→50k 跨 50,000 learner updates 的合并
  resident norm 增长 `2.281x`，等价于平均每 update 约 `+1.65e-5`。code L2
  基本稳定而 stored scale 漂移；例如 `Block0/Dense1/ensemble1` 的 norm
  `623.7→1542.1`、scale `0.001900→0.004697`，code L2
  `328314→328320`。这排除开关未启用，支持 previous-physical-norm 的 FP32
  reduction/scale recurrence 累积数值漂移；同一计算图内的单步 ratio 因与
  canonicalization 共享 reduction，不能作为独立的长期 invariant 验证。
- 当前解释：这两个 run 已证伪“无需 reference norm 即可递推保持初始化半径”
  的预注册主张，并作为该有限精度递推的负结果保留。更稳健的候选是每个
  kernel/member 持久保存一个固定初始化 radius（共 8 个 FP32 scalar），每次
  直接从该 anchor 和 next code norm 计算最终 scale；该修订尚未实现或启动。
- 若继续固定-anchor 修订，其成功判断应改为独立检查每个 kernel/member 相对
  固定初始化 radius 的长期误差，而不能再以同一写入图内的 post/old ratio
  代替；同时要求 stored scale 不再长期承担 norm 膨胀，angular diagnostics
  与 Q/function-write 指标有限，并比较 seed42 的 matched return。

## Fixed-Initialization-Anchor FP8 Residency

### `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-25K`

- 状态：实现与本地验证完成；2026-09-03 05:04:48 Asia/Shanghai 已在
  GPU3 启动 seed42 25k 机制门。tmux
  `brc_fp8_fixed_anchor_s42_gpu3_25k`，训练 PID `330805`，run
  `brc_dmc_dogs_online_fp8_resident_fixed_anchor_s42-20260903-050448`，W&B
  `crwzhu6b`。启动后 tmux/PID/GPU process、run 目录、初始化事件均存在；
  resolved method 为 `fixed_initial_kernel_norm_with_paired_bias`。
- 目的 / Idea：Idea 002 的 fixed-anchor 修订。每个 resident kernel/member 持久
  保存初始化物理 norm（4 kernels × 2 members = 8 个 FP32 scalar）；每次只
  保留 current-amax 选出的 code，并直接以 `rho0 / ||code||` 写 scale。
- 代码：`codex/blackwell-fp8-direct` @ `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958`
  加未提交 fixed-anchor 实现；变更记录见
  [2026-09-03-fixed-anchor-fp8-residency.md](2026-09-03-fixed-anchor-fp8-residency.md)。
- 协议：Dogs、seed42、GPU3、width4096、batch1024、2 updates/env step；先到
  25k，完成后从同一 final recovery checkpoint 继续到 500k。其余协议与已停止
  moving-anchor seed42 run 一致。
- 机制门：用 final checkpoint 中实际序列化的 code/scale/rho0，以 NumPy
  float64 独立计算 8 个 `scale * ||code|| / rho0`。八个成员均存在且没有长期
  scale/norm 漂移后才允许续跑；不以同一训练图内 JAX FP32 ratio 作为唯一依据。
- 验证：完整 CPU FP8/checkpoint suite 29 tests 通过（160.334s）；GPU3 上
  10,000-write 与真实 resident update smoke 通过。启动脚本：
  `scripts/run_dogs_online_fp8_resident_fixed_anchor.sh 3 42 25000`。
- 25k 结果：2026-09-03 05:35:06 Asia/Shanghai 正常写入
  `final_checkpoints_finished` 和 `run_finished`，env step `25000` / update
  `40003`；最终 train `critic_pnorm=359.2903`，eval return `48.0575`，更新
  NaN/Inf `0/0`。analysis/recovery 两个 checkpoint 的独立 NumPy float64
  报告完全一致，8/8 member 的 `anchor_norm_ratio` 为
  `0.9999168158–1.0002157129`，最大绝对偏差 `2.1571294e-4`。相对固定
  `rho0=90.5096664` 未见 moving-anchor 的单调尺度漂移，机制门通过。
- 续跑：`EXP-FP8-ONLINE-FIXED-ANCHOR-S42-R500K` 将从同一 run 的
  `checkpoints/recovery_step_000000025000` 恢复 optimizer、replay 和八个 anchor，
  命令 `scripts/run_dogs_online_fp8_resident_fixed_anchor.sh 3 42 500000
  <recovery-checkpoint>`。
- 续跑启动验收：2026-09-03 05:36:44 Asia/Shanghai 在 GPU3 启动，tmux
  `brc_fp8_fixed_anchor_s42_gpu3_500k`，PID `453470`。同一 W&B `crwzhu6b`
  与 run 目录已恢复；事件流写入 env step 25000 / update 40003 的
  `resume_reset`、`initialization_finished(resumed=true)`，并完成 env step 25001
  / update 40005 的首次真实更新。GPU3 compute process 正常存在。
- 425k outcome readout（observed）：eval mean 从 25k 的 `48.06` 上升到
  400k 的 `572.84`，425k 回落到 `499.04`；严格 matched
  `EXP-FP8-TARGET-FWD-S42` 在 400k/425k 为 `766.69/764.57`。fixed-anchor
  不是完全不学习，但没有关闭 return gap。425k 的四任务 return 为
  `931.64/600.54/243.99/219.96`，差距主要来自 walk/trot/run。
- Anchor/write check（observed）：400k checkpoint 的独立 NumPy float64
  ratio 为 `0.9996764–1.0035631`；25k/50k/350k/400k 的最大偏差分别为
  `2.16e-4/2.01e-3/1.93e-3/3.56e-3`，上下波动而非递归单调漂移。425k
  四个 resident kernel 合并 physical norm 为 `255.995`（理论固定值约
  `256.0`），而总 critic pnorm `1127.25` 的增长主要来自非 resident FP32
  kernel（合并 `1079.40`）。最后一次 function-write expected-Q MAE
  `2.93e-5`，只说明 candidate→canonical write 保真。
- Backward failure（observed）：fixed-anchor run 已有 17 个 tensor-stat
  时点，四个 resident kernel 的 gradient 共 `68/68` 条均为
  `l2_norm=0, zero_fraction=1`；完成的 naive resident seed42 同样为
  `80/80` 全零，而严格 matched FP8-direct control 的 `80/80` 条均非零。
  同一 425k batch 的 resident Dense bias gradient 非零，故不是整个 critic
  loss 或 recorder 失效，而是 kernel FP8 dot operand 的反向路径失效。
- Root cause（code evidence + isolated probe）：`ResidentFp8Dense` 直接依赖
  raw `lax.dot_general(E4M3,E4M3)` 的自动微分，并在 dot 外乘 activation/kernel
  scale；它没有 Flax `Fp8DirectDotGeneralOp` 的 custom VJP、E5M2 output-gradient
  dynamic scaling 与反向 dequantization。因而未缩放 cotangent 先被转换为 FP8，
  再除 scale，典型梯度在转换前已下溢为零。使用 resident 实际量级
  `activation_scale=0.5, kernel_scale=2e-4` 的独立 GPU probe，在 output-gradient
  sigma `1e-3` 时 resident kernel gradient 为全零，而数学等价 FP32 gradient
  L2 为 `151.17`；sigma `1e-2` 时仍有 `90.23%` 坐标为零，显示 cliff/spike
  而非正常低精度反向。
- Interpretation：fixed anchor 已修复它承诺的 norm recurrence，但原先
  “scale-gauge runaway 是 return gap 主因”的闭环因果解释被本次干预否定。
  现在的首要 failure mode 是 online resident 手写 GEMM 缺少 scaled backward；
  大矩阵只会从偶发越过 FP8 梯度阈值的 spike/残留 Adam moment 获得更新，
  input/output projection、bias、LayerNorm 和 residual skip 仍可学习，所以 reward
  会缓慢上升而达不到 FP8-direct control。此前 parameter/function-write 指标位于
  optimizer candidate 之后，不能验证 candidate 之前的 backward，因此没有暴露
  这一故障。
- 停止记录：按用户要求于 2026-09-03 15:15:06 Asia/Shanghai 向 tmux
  `brc_fp8_fixed_anchor_s42_gpu3_500k` 发送 `Ctrl-C`。run 在 env step
  `452791` / update `895583` 写入 `run_interrupted(KeyboardInterrupt)` 与
  `run_finished`；tmux 和 PID `453470` 均已退出。最后完整 eval 为 450k
  `608.6501`，最后 train metric 为 452k，NaN/Inf `0/0`。目录、400k recovery
  与 450k analysis checkpoint 均保留；该 run 可作为“只固定 norm 不能修复
  backward-invalid operator”的 negative control，但不能评价正确 backward 下
  fixed-anchor 的 return 效果。

## Scale-Aware Online-Resident FP8 Backward

### `EXP-FP8-ONLINE-SCALED-BWD-S42-500K`

- 状态：2026-09-03 15:32 Asia/Shanghai 完成实现与预运行数值门；15:37:45
  已在 GPU3 直接启动正式 seed42 500k。tmux
  `brc_fp8_scaled_backward_s42_gpu3`，训练 PID `2393746`，run
  `brc_dmc_dogs_online_fp8_resident_scaled_backward_s42-20260903-153745`，
  [W&B `z6n3ca1s`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/z6n3ca1s)。
  旧 fixed-anchor run 已于 15:15:06 停止并保留为 backward-invalid negative
  control。
- 目的 / Idea：Idea 002；只修复 resident Dense backward。online 大 kernel
  继续以 E4M3 code + FP32 scale 常驻、AdamW moments 保持 FP32、candidate
  继续 current-amax E4M3 单次写回；canonicalization 明确关闭，target 继续
  `fp8_direct` + FP32 persistent parameters。
- Backward：复用 Flax `quantized_dot` custom VJP 和与 FP8-direct 相同的
  E5M2 output-gradient delayed-amax scaling/FP32 accumulation；直接返回 physical
  FP32 kernel/input gradient。四层×两 member 的 output-gradient scale/history
  纳入 checkpoint，actor 的 critic backward 同样推进这组 metadata。
- 代码：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，分支
  `codex/blackwell-fp8-direct`，HEAD
  `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958` 加未提交实现与记录；没有提交
  或推送。变更说明见
  [2026-09-03-scale-aware-resident-fp8-backward.md](2026-09-03-scale-aware-resident-fp8-backward.md)。
- Independent probe（observed）：`sx=0.5, sw=2e-4`，output-gradient sigma
  `1e-3/1e-2`。repaired resident 对 FP8-direct 的 kernel cosine/L2 ratio 均
  `1.0/1.0`，input gradient 也等价，zero fraction 均 `0`；对 FP32 cosine
  约 `0.9986`。机器记录：`scale_aware_backward_independent_probe.json`。
- Real replay gate（observed）：从 matched Dogs seed42 recovery replay 取 256
  transitions、width512、共同 physical weights。四个 kernel 对 direct 的
  cosine `0.99927–0.99994`、L2 ratio `0.99977–1.00081`，全部非零；`dQ/da`
  cosine `0.98988`、ratio `1.00182`、zero fraction `0`。机器记录：
  `scale_aware_backward_real_batch_probe.json`。
- Smoke（observed）：GPU0 `cheetah-run` 到 env 200 / update 203 正常结束，
  update NaN/Inf 始终 `0/0`。step150 四个 resident kernel gradient L2 为
  `0.36380/0.41178/0.29192/0.33431`，zero fraction
  `0/0.00883/0/0.02454`，均无 NaN/Inf；actor gradient 有限非零。run
  `brc_cheetah_run_online_fp8_resident_scaled_backward_s0_smoke`，未上传 W&B。
- 测试：resident-focused CPU tests `3/3` 通过；修正 checkpoint transition
  guard 后，完整 FP8 suite `27/27` 通过（166.881s）；`git diff --check` 通过。
- 正式协议：Dogs seed42、GPU3、width4096、batch1024、2 updates/env step，
  直接 500k，25k eval/tensor stats、50k analysis、100k recovery。用户于启动前
  取消单独 25k gate；仅做启动验收，不持续监视，发现问题时由用户手动停止。
  seed1 必须等 seed42 完成，不并行启动。
- 启动脚本：`scripts/run_dogs_online_fp8_resident_scaled_backward.sh 3 42
  500000`。
- 启动验收（observed）：tmux/PID 与 GPU3 compute process 存在，W&B 已开始
  同步；run config 解析为 `max_steps=500000`、seed42、width4096、online
  `fp8_resident`、`resolved_online_fp8_backward=scale_aware_e5m2_delayed_amax_custom_vjp`、
  `resolved_online_fp8_canonicalization=disabled`、无 FP32 weight master、target
  `fp8_direct`。按用户要求此后不做长期自动监视，异常由用户手动停止。
  `initialization_finished` 于启动后 12.78s 正常写入；移交前已推进至 env
  step 4000，未见立即错误。
- 94k readout（observed，2026-09-03 17:13 Asia/Shanghai）：run/tmux/PID 仍
  正常，update NaN/Inf 为 `0/0`。25k/50k/75k eval return 为
  `45.94/101.41/129.44`；matched FP8-direct control 为
  `19.92/120.40/220.44`，到 75k 已落后 `91.00`。75k 四任务 return 为
  repaired `299.42/70.68/78.21/69.44`，control
  `403.69/225.05/154.85/98.17`。
- Backward/write check（observed）：25k/50k/75k 四个 resident kernel
  gradient L2 均非零；其 zero fraction 与 FP8-direct 一样会因 E5M2/网络结构
  呈层间差异，不能重现旧实现的四层全零指纹。sampled candidate→resident
  write cosine 约为 `0.999997–1.0`，applied/intended L2 ratio 约
  `0.999994–1.000054`，所以 backward 修复和单步写回均在工作。
- Norm evidence（observed）：critic pnorm 在 25k/50k/75k/90k 为
  `733.80/1930.97/3046.30/3740.79`，matched FP8-direct 为
  `495.37/756.40/958.63/1064.44`。独立反序列化 25k/50k checkpoint 得到四层
  八 member 合并 physical norm `691.53→1905.03`；sampled 75k 为
  `3022.31`，相对初始化合并 norm 约 `256` 已为 `11.8x`。member norm 在
  50k 已分化到 `32.44–1511.29`，stored scale 为
  `9.94e-5–4.42e-3`；75k sampled intended effective angular step 均值比
  25k 低约 `5x`。
- Interpretation / next gate：修复后的非零梯度没有自行消除 scale-gauge norm
  runaway，反而使其比旧 zero-gradient resident 更快；当前差 reward 与快速
  norm/scale 分化同步，但尚不能仅凭相关性宣称因果。建议的新独立 arm 是从头
  运行 `scale-aware backward + fixed-initial anchor`，与本 run 和 matched
  FP8-direct 做同 seed 对照。旧 fixed-anchor run 因 backward-invalid 不能替代
  该实验。该 arm 尚未启动，等待用户决定。

### `EXP-FP8-ONLINE-SCALED-BWD-FIXED-ANCHOR-S42-500K`

- 状态：`running`；用户于 2026-09-03 17:15 Asia/Shanghai 明确选择从头在
  GPU0 启动，17:16:25 实际启动。不是从 unanchored checkpoint 恢复，也不是
  旧 backward-invalid fixed-anchor run 的续跑。
- 目的 / Idea：Idea 002；检验 scale-aware repaired backward 恢复真实梯度后，
  fixed-initial anchor 是否通过消除八个 resident member 的 scale-gauge norm
  runaway 恢复 angular motion 与 return。
- 代码：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，分支
  `codex/blackwell-fp8-direct`，HEAD
  `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958` 加未提交 repaired-backward、
  fixed-anchor、测试与研究记录；实际 dirty source snapshot 由 recorder 保存。
- 协议：Dogs、seed42、从随机初始化开始、GPU0、width4096、batch1024、
  2 updates/env step、500k；online `fp8_resident`、scale-aware E5M2 custom
  backward、`fp8_resident_canonicalization=true`、八个 fixed-initial anchor，
  target `fp8_direct` + FP32 persistent parameters，FP32 AdamW moments；25k
  eval/tensor stats、50k analysis、100k recovery。
- 比较：同 seed repaired-only `EXP-FP8-ONLINE-SCALED-BWD-S42-500K` 与
  matched FP8-direct `EXP-FP8-TARGET-FWD-S42`。旧
  `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-R500K` 只作为 backward-invalid negative
  control，不作为该组合方法的结果。
- 运行：tmux `brc_fp8_scaled_backward_fixed_anchor_s42_gpu0`，PID `2656519`；
  命令 `scripts/run_dogs_online_fp8_resident_fixed_anchor.sh 0 42 500000`。run
  `brc_dmc_dogs_online_fp8_resident_fixed_anchor_s42-20260903-171625`，
  [W&B `p7osuon0`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/p7osuon0)。
- 决策指标：八个 serialized `anchor_norm_ratio`、四层 kernel gradient
  L2/zero fraction、actor gradient、critic pnorm、eval return、NaN/Inf；不使用
  wall clock/throughput 作为方法证据。
- 启动验收（observed）：tmux/PID 与 GPU0 compute process 存在，run directory
  和 W&B 已建立；config 为 seed42、`max_steps=500000`、`resume_from=''`、
  `resolved_online_fp8_backward=scale_aware_e5m2_delayed_amax_custom_vjp`、
  `resolved_online_fp8_canonicalization=fixed_initial_kernel_norm_with_paired_bias`、
  target `fp8_direct`、无 FP32 weight master。`initialization_finished` 于启动后
  12.38s 写入，验收时已到 env step 2000，未见立即错误。
- 100k 因果读数（observed）：eval return `134.4419`，对同 seed
  repaired-only `157.9756`、matched FP8-direct `340.6060`；对应 critic pnorm
  `519.075/4171.116/1129.211`，三者 NaN/Inf 均为 `0/0`。组合方法没有恢复
  reward，且低于 repaired-only；单 seed 不能给出最终统计显著性，但差距已经
  足以否定“修 backward + 固定初始化半径必然关闭早期 gap”的机制预期。
- 机制排除（observed）：100k serialized checkpoint 的 NumPy float64 独立审计
  给出 8/8 `anchor_norm_ratio=0.9999815–1.0004089`；四层 kernel gradient L2
  为 `0.002290/0.007208/0.000322/0.007310`，均非零。最后一次写回的
  expected-Q MAE `1.7975e-5`、critic-loss relative error `9.2689e-8`、JS
  divergence `1.0268e-8`，所以不是 anchor recurrence、旧全零 backward、
  非有限值或单步函数突变。
- 剩余机制（observed + inference）：100k 写回的 8-member code unchanged
  fraction 均值 `0.80694`（范围 `0.38953–0.94172`），但 physical exact-zero
  swallowed fraction 仅 `3.12e-4`，因为 global scale 仍会移动。fixed run
  50k→100k 的多个 kernel code-direction cosine 为 `0.0458–0.3246`，表明并非
  整体冻结；相反 repaired-only 100k→150k 多数 cosine 为
  `0.999999–1.0`，与 scale runaway 后方向冻结一致。fixed anchor 把每 member
  固定在初始 norm `90.51`，而 direct 100k 四层 combined-member norm 已为
  `241.87–669.69`。因此 hard initial-radius projection 虽对 LayerNorm 前向近似
  gauge-invariant，却不保持 AdamW 的参数化与有效角学习率；当前证据把主因上移
  到无 FP32 master 的逐步 E4M3 参数转移/优化轨迹。online current-amax 与
  direct delayed-amax 仍是下一因果控制需要消除的混杂。
- Continued exclusion checks（observed）：同一 100k serialized physical
  weights 与保留 probe batch 上，resident-vs-FP32 loss relative error
  `6.54e-6`，四层 kernel-gradient cosine `0.98384–0.98892`、L2 ratio
  `0.88758–0.92540`，`dQ/da` cosine/L2 ratio
  `0.9999979/0.9999878`。100k E5M2 current/history amax ratio 约
  `0.36–0.44`，不是旧 raw-FP8 backward 的全零 cliff。fixed online-target
  kernel cosine `0.9999933–0.9999999`、relative gap
  `0.000552–0.003675`；8 个 resident matrices 的
  `code→physical→code` 共 `134,217,728` 元素 mismatch 为 0。
- Optimizer geometry（observed + inference）：fixed100 Adam 一阶 moment 与
  weight cosine 量级仅 `6.9e-5–1.3e-3`，排除“大量陈旧径向 momentum 每步被
  丢掉”；由现有 write scalars 独立重建的 intended/applied tangential cosine
  在数值误差内为 1。更窄的解释是 hard initial-radius projection 改写了
  LayerNorm 前 scale-gauge 上的 Adam 有效角步长：direct 100k 四层
  combined-member norm 已自然增长至 `241.87–669.69`，fixed 始终约 `128`，
  因而取消了健康 control 中随 norm growth 发生的隐式角学习率退火。该解释
  针对 fixed arm；masterless E4M3 累计路径与 current/delayed-amax 混杂仍需
  matched FP32-master/current-amax control 才能完全分离。
- 进程处置：本轮用户只要求判断效果；未停止或重启任何实验。记录时 GPU0
  tmux/PID 仍在正常运行，后续仍由用户手动决定是否停止。

### `EXP-FP8-ONLINE-CURRENT-MASTER-S42-150K`

- 状态：`running`；2026-09-03 18:47:47 Asia/Shanghai 在 GPU1 从随机初始化
  启动，计划运行至 150k；25k 首个闭环点与 tensor stats 已完成。
- 目的 / Idea：Idea 002 的单因素隔离。在线四个 residual Dense 保留与
  repaired resident 相同的 activation/weight current-amax E4M3 前向、FP32
  accumulate 和 scale-aware E5M2 custom backward，但参数与 AdamW moments
  持久为 FP32；不做 resident 写回或 fixed anchor。若它恢复 matched direct
  曲线，则剩余主因位于 masterless E4M3 参数转移/优化轨迹；若仍接近 resident，
  则 current-amax compute policy 本身是主要嫌疑。
- 协议：Dogs、seed42、width4096、batch1024、2 updates/env step、target
  `fp8_direct` + FP32 parameters、paper alignment/reward-mean bootstrap、25k
  eval/tensor stats、50k analysis、100k recovery，与现有三条比较曲线一致。
- 运行：tmux `brc_fp8_current_master_s42_gpu1`，启动 PID `2931188`；run
  `brc_dmc_dogs_online_fp8_current_master_s42-20260903-184747`，
  [W&B `vgd8apty`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/vgd8apty)。
- 启动前验证（observed）：定向 unittest 通过；四层初始 current-master 与
  resident logits 逐元素相同；online params/AdamW 浮点 state 全为 FP32；连续
  三次 learner update 后参数变化且四层 output-gradient history 推进。启动脚本
  `bash -n`、Python 编译和 `git diff --check` 均通过。
- 启动验收（observed）：tmux/PID、W&B 与本地 run directory 已建立；命令解析为
  `--critic_precision=fp8_current_master --target_critic_precision=fp8_direct`
  和 `--fp8_resident_canonicalization=false`。
- 25k readout（observed）：mean return `49.53`，对 repaired resident
  `45.94`、fixed `34.78`、matched direct `19.92`；该早期点没有复现 resident
  reward deficit，但曲线尚未拉开，不能单独作为最终归因。critic pnorm
  `472.85`，接近 direct 的 `495.37`，而 repaired/fixed 为 `733.80/359.29`。
  serialized checkpoint 的四层 combined-member FP64 norms 为
  `209.31/272.91/145.85/152.09`，也接近 direct 的
  `213.62/302.42/154.87/154.93`，明显不同于 repaired 的
  `555.28/328.57/206.87/138.30` 与 fixed 的约 `128`。
- 25k operator health（observed）：四层 kernel-gradient L2 为
  `0.02531/0.01065/0.000415/0.009906`，全部非零；update NaN/Inf 为 `0/0`，
  291 条 tensor stats 与 analysis checkpoint 正常写入。继续以 50k/100k
  return 是否贴近 direct 为主因果门。
- Aligned deterministic probe（observed）：width16 的 current-master 与
  resident 从完全相同的 online physical params、target state 和零 Adam state
  出发。一次更新后 resident write 产生最大 kernel relative L2 `0.000897`，但
  FP8 logits 仍逐元素相同；10 次后两者为 `0.008423/0.027712`。这直接验证
  off-lattice FP32 位置的逐步丢失能在局部 forward 一致时累计成函数轨迹分叉。
  原始数值保存在 `development_records/current_master_aligned_pair_probe.json`。
- 50k readout（observed）：return `113.17`，达到 direct `120.40` 的 `94.0%`，
  并高于 repaired/fixed `101.41/87.12`。四任务 current/direct 分别为
  `261.89/73.86/60.67/56.28` 与 `271.14/77.23/65.91/67.32`，没有由单任务
  抵消造成的假接近。critic pnorm 为 `720.93`，direct/repaired/fixed 为
  `756.40/1930.97/412.72`；四层 physical kernel norms current
  `391.76/412.33/197.91/180.21` 也贴近 direct
  `385.68/452.56/217.95/196.32`，远离 repaired
  `1680.26/744.36/482.33/138.30` 和 fixed 的约 `128`。update NaN/Inf
  `0/0`，tensor stats/checkpoint 正常。50k 已支持 masterless E4M3 write 是
  主要分叉源；75k/100k 用于检查该结论在曲线明显拉开后是否保持。
- Same-weight compute probe（observed）：在健康 direct 500k 的相同 FP32
  weights 与 256-sample probe 上，将保存的 delayed-amax 与 current-amax 直接
  比较，logits relative L2 `0.004814`、expected-Q MAE `0.003916`；四层
  kernel VJP cosine `0.97565–0.99818`、L2 ratio `0.99079–1.00024`，zero
  fraction 也近似一致。current/delayed 是真实但次级的累计轨迹混杂，不是
  resident 量级的局部算子崩坏；原始值保存于
  `development_records/current_vs_delayed_direct500_probe.json`。
- 75k readout（observed）：current-master return `178.43`，比
  repaired/fixed `129.44/129.26` 高 `37.9%/38.0%`，但仍低于 direct
  `220.44` 约 `19.1%`。四任务 current 均逐项高于两条 resident arm，排除
  单任务均值抵消；相对 direct 的主要缺口来自 walk/trot。该点表明保留 FP32
  master 已消除 resident gap 的主要部分，但 current/delayed compute 小差异和
  单 seed 闭环敏感性仍可能贡献剩余差距。100k 是最终主判据。
- 100k causal result（observed）：current-master return `244.89`，对
  repaired/fixed/direct `157.98/134.44/340.61`；相对两条 resident arm 提升
  `55.0%/82.2%`，但仍比 direct 低 `28.1%`。四任务 current 为
  `597.35/156.11/109.37/116.74`，均高于 repaired
  `393.30/68.02/85.12/85.46` 和 fixed
  `293.34/96.43/83.98/64.01`；direct 的额外优势主要在 walk/trot。
- 100k internal trajectory（observed）：current/direct/repaired/fixed critic
  pnorm 为 `1090.48/1129.21/4171.12/519.08`；四层 physical norms current
  `668.29/616.10/277.23/216.72` 对 direct
  `669.69/642.33/316.31/241.87`，而 repaired 为
  `3543.41/1375.18/1654.27/138.65`、fixed 全部约 `128`。5k–100k 全段
  critic-pnorm SMAPE：current 对 direct `3.63%`，对 repaired/fixed
  `77.11%/47.97%`；Q-prediction/actor-pnorm SMAPE 对 direct 也最低
  (`3.77%/6.43%`)。四层 current gradient 全部非零，形态与 direct 同量级；
  update NaN/Inf `0/0`，analysis/recovery checkpoint 均成功。
- Causal decision：masterless resident write 每步丢失 off-lattice FP32 参数位置，
  是 fixed/repaired reward 不升的确定且大幅因素；unanchored 的 scale runaway/
  角步冻结和 fixed anchor 的过强投影/角学习率失配是该状态替换的两种不同后果。
  但 current-master 仍未达到 delayed-amax direct，因此 current/delayed scaling
  的小算子差异经 RL 闭环累积是实质性第二因素。不能再把全部 gap 归给 anchor
  或全部归给单步写回。run 继续到 150k，不需要为当前因果结论等待结束。

### `EXP-FP8-ONLINE-CARRY-S42-SMOKE-5K`

- 状态：`completed`；2026-09-04 02:33:55 与 02:41:34 Asia/Shanghai 在
  空闲 GPU1 以独立 tmux 会话从全新随机初始化完成。两次均为本地记录
  （`BRC_LOG_TO_WANDB=false`），没有向 W&B 上传；smoke 与正式 run 使用
  不同目录，正式 run 不 resume。
- 目的 / Idea：Idea 002 的 CARRY-FP8 工程门。持久 online resident 状态为
  `Theta=s(C+R/16)`，其中主码 `C` 与 carry `R` 都是 E4M3、共享现有
  current-amax FP32 scale；forward/backward 只使用 `sC`，AdamW 和 target
  EMA 使用临时 logical reconstruction。主比较仍是 current-amax FP32-master。
- 代码：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，分支
  `codex/blackwell-fp8-direct`，HEAD `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958`
  加未提交 CARRY-FP8 和此前未提交研究改动；无 commit/push。
- 协议：Dogs、seed42、width4096、batch1024、2 updates/env step、5k env
  steps、online `fp8_resident` + scale-aware backward + carry、fixed anchor
  明确关闭、target `fp8_direct` + FP32 persistent parameters、paper alignment
  与 reward-mean bootstrap。tensor/analysis/recovery cadence 在 5k 触发。
- 启动前验收（observed）：CPU `tests.test_fp8` 为 `35/35`，完整
  `unittest discover -s tests` 为 `63/63`；GPU1 上 7 项 carry 核心单测为
  `7/7`。脚本 `bash -n`、Python compile 和 `git diff --check` 均通过。
  GPU1/2 启动前显存均为 `3 MiB`，未复用或停止任何现有实验。
- 命令：首次为
  `BRC_LOG_TO_WANDB=false scripts/run_dogs_online_fp8_resident_carry.sh 1 42 5000`；
  相邻步复核为
  `BRC_LOG_TO_WANDB=false BRC_TENSOR_STATS_INTERVAL=1 BRC_ANALYSIS_CHECKPOINT_INTERVAL=5001 BRC_RECOVERY_CHECKPOINT_INTERVAL=5001 scripts/run_dogs_online_fp8_resident_carry.sh 1 42 5001`。
- 工程门：四层 gradient 非零；main/carry dtype 均 E4M3；carry saturation
  为 0；logical reconstruction error 小于 main-only；无 NaN/Inf；analysis 与
  recovery checkpoint 可读且 carry 可恢复；随后才启动全新 500k run。
- 结果（observed）：第二次 run 正常结束于 env `5001` / learner update `5`，
  `run_finished` wall time `136.94 s`。两个 tensor 点共 `1466` 行，参数、梯度
  与 update 的 NaN/Inf 均为 `0/0`；四层 resident kernel gradient L2 为
  `7.8689/15.1037/9.0506/16.5058`，均非零；JAX peak bytes-in-use 约
  `5.94 GiB`。main/carry dtype 均为 `float8_e4m3fn`，所有 8 个
  layer/member 的 carry saturation fraction 为 `0`。
- 边界读数（observed）：最初四次 learner update 的 candidate 恰已位于主
  current-amax lattice（main write relative L2 仅 `2.33e-8–6.53e-8`），因此
  carry 为零且 reduction ratio 约为 `1`；这不是非零主误差下的失败。加载
  5001 recovery checkpoint 后，按已保存 RNG/replay/normalizer 取下一批并执行
  两次 update，main-only relative L2 为 `0.008859–0.012317`，carry logical
  relative L2 为 `0.000212–0.000311`，error reduction 为
  `39.65x–43.32x`，saturation 为 `0`，四个含 ensemble 轴的 carry array 均约
  `33.55M/33.55M` 非零元素；update NaN/Inf 为 `0/0`。这同时完成实际
  width-4096 recovery restore 与 next-update 验证。
- Checkpoint / 状态体积（observed）：analysis/recovery 分别为
  `...-024134/checkpoints/analysis_step_000000005001` 与
  `...-024134/checkpoints/recovery_step_000000005001`。四个 main kernel 和
  carry 分别都是 `134,217,728` bytes，weight scale 合计 `32` bytes，新增
  full-size FP32 carry 为 `0`。analysis `critic.msgpack` 为 `281,873,411`
  bytes；与相同结构的 repaired resident `147,655,495` bytes 相比增加
  `134,217,916` bytes（payload 增量恰为 `128 MiB`，其余为序列化开销）。

### `EXP-FP8-ONLINE-CARRY-S42/S1-500K` (pre-barrier invalid controls)

- 状态：`stopped; mechanism-invalid`；2026-09-04 02:47 Asia/Shanghai 在旧
  smoke 工程门与
  recovery-next-update 复核后，从两个全新目录并行启动，不从任何
  smoke/naive/fixed checkpoint resume。W&B 初始化均成功。
- 协议：Dogs、500k、start 5k、replay 1M、batch 1024、2 updates/env step、
  width 4096、paper alignment、reward-mean bootstrap；online resident main
  current-amax E4M3 + E4M3 carry + repaired scale-aware backward，fixed anchor
  关闭；target `fp8_direct` 且参数/EMA 为 FP32；25k eval/tensor、50k analysis、
  100k recovery。
- Seed / resource：seed42 / GPU1 / tmux `brc_fp8_carry_s42_gpu1` / PID
  `4112431` / W&B [`a4h7dp4d`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/a4h7dp4d)；
  seed1 / GPU2 / tmux `brc_fp8_carry_s1_gpu2` / PID `4112436` / W&B
  [`ddivbdzt`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/ddivbdzt)。
  用户已于 2026-09-04 明确允许 W&B 遥测上传。
- 命令：`scripts/run_dogs_online_fp8_resident_carry.sh 1 42 500000` 与
  `scripts/run_dogs_online_fp8_resident_carry.sh 2 1 500000`。
- 输出：`runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_s42`
  与 `..._s1`；checkpoint 子目录均为各自 `checkpoints/`。
- 启动验收（observed）：两条 config 均为 carry `true`、canonicalization
  `false`、resolved state `carry_e4m3_shared_scale_gain16`、target
  `fp8_direct`、resume 空、500k/25k/50k/100k cadence 正确。GPU UUID 与 PID
  绑定已核验；seed42/seed1 在 env 5000 / update 3 的 critic loss 为
  `12.6415/8.7781`、gradient norm 为 `19.5278/13.5810`，update NaN/Inf
  均为 `0/0`，JAX peak bytes-in-use 均约 `5.12 GiB`。
- 2026-09-04 02:51 progress snapshot（observed）：seed42 已到 env `8000` /
  update `6003`，seed1 已到 env `7000` / update `4003`；最新 update NaN/Inf
  均为 `0/0`，对应 `critic_pnorm=340.58/365.79`。两个 tmux、训练 PID 与
  GPU1/GPU2 映射均仍存活。
- 2026-09-04 03:36 health audit（observed）：两条 run 已到 env `68000` /
  update `126003`，tmux/PID/GPU 均存活，最新 update NaN/Inf 均为 `0/0`，
  peak JAX bytes-in-use 为 `11.15/11.02 GiB`；25k/50k analysis checkpoint
  完整。因此运行时和数值稳定性本身正常。
- CARRY mechanism failure（observed）：两种子在 25k 与 50k 的 8/8
  layer/member carry 均为 E4M3 但 `zero_fraction=1`、`carry_prequant_absmax=0`、
  saturation `0`；main/logical relative error 相同且仅为
  `1.57e-8–7.09e-8`，reduction ratio 约 `1`，physical/logical norm 完全相同。
  直接解码两个 50k `critic.msgpack` 再次确认四个含 ensemble 轴的 carry
  array 各有 `33,554,432` 个元素且非零计数全部为 `0`，排除 tensor logger
  误报。四层 gradient L2 在两个点仍全部非零，所以这不是旧的 zero-backward
  failure，而是 carry 持久状态在真实长程训练中没有生效。
- Trajectory evidence（observed）：seed42 的 25k/50k/68k critic pnorm 为
  `782.90/1839.92/2605.80`，接近 repaired resident 的
  `733.80/1930.97/2608.39`，远高于 current-master 的
  `472.85/720.93/约851@65k`。25k/50k return 为 `47.22/80.02`，对 repaired
  `45.94/101.41`、current-master `49.53/113.17`；早期 return 有随机性，不能
  单独定性，但没有显示 CARRY 收益。seed1 的 25k/50k return 为
  `30.44/84.02`，68k pnorm 已为 `3594.74`。
- Health decision（inference）：工程进程仍健康，但预注册的 CARRY 重建门与
  trajectory-preservation 假设已经失败；这两条 partial run 不能回答有效
  CARRY-FP8 的 return 问题。按原计划应停止并修复机制，而不是继续用其作为
  CARRY treatment；本次只读健康审计未擅自停止，等待用户决定。
- 2026-09-04 03:49 root cause（observed）：最小 `4x4` 与 `4096x4096` 探针
  都证明，当前 GPU JIT 会让同一 compiled graph 内 `float32(code)` 的消费
  路径看到 FP8 cast 之前的 FP32 值；持久返回的 E4M3 code 本身仍真实量化。
  因此 `candidate/s-float32(code)` 被编译成 0，carry 全零，同图内 main error
  也被错误报告为约 `1e-8`。CPU JIT 与 GPU eager 正常。4096² plain probe
  的 main relative error/nonzero carry/prequant absmax 为
  `2.684e-8 / 0 / 0`；在 E4M3 code 上插入 `jax.lax.optimization_barrier`
  后为 `0.0264938 / 16,777,205 / 255.995`。stop-gradient、bitcast detour、
  FP32 candidate 侧 barrier 均无效。环境为 JAX/JAXLIB `0.6.0`、Flax
  `0.10.4`、driver `580.95.05`、RTX PRO 6000 Blackwell SM 12.0。
- 2026-09-04 03:51 live follow-up（observed）：两条 run 均已到 env `90000` /
  update `170003`，PID/GPU 仍存活，update NaN/Inf 均为 `0/0`；critic pnorm
  已到 `3421.95/4855.63`。75k 第三个 tensor 点再次给出 8/8
  `carry_prequant_absmax=0` 和约 `1.81e-8–8.57e-8` 的伪 main relative error，
  与最小 GPU-JIT reproduction 完全一致。
- Root-cause decision（inference）：真实缺陷是量化器没有在 code cast 后建立
  GPU-JIT 可观察边界；不是 replay、checkpoint、Actor metadata merge、carry
  gain 或 Adam candidate 恰落 lattice。现有 same-graph write-fidelity 诊断同样
  失真。修复必须在所有需要立即 widen 新 code 的路径放置 code-side barrier，
  并新增 GPU-JIT materialization regression test 后从 fresh initialization
  重跑；当前进程未因本次诊断被停止或修改。
- 2026-09-04 04:22–04:28 corrective implementation（observed）：共享 E4M3
  quantizer 已在 main code cast 后加入 `jax.lax.optimization_barrier`；独占 GPU
  gate 进一步发现 carry 自身的直接 cast 也需要 barrier。两层持久化边界均修复
  后，actual carry kernel 的 vmap+JIT regression 在 CPU/GPU1 均通过，并验证
  same-graph main/logical reconstruction 与返回 main/carry code 第二次 dispatch
  完全一致。4096² probe 的 main/carry relative error 为
  `0.00122739/3.2394553e-5`，nonzero carry
  `16,767,853/16,777,216`、saturation `0`。此前记录的 carry error
  `3.598577e-8` 是未 barrier carry cast 的第二个伪读数，已废弃。完整 CPU
  suite `64/64` 通过。config 新增 materialization 语义版本并拒绝修复前
  resident checkpoint resume；launcher 支持 `BRC_RUN_TAG=jitbarrier_v1`。
- 2026-09-04 04:24 invalid-run follow-up（observed）：100k tensor point 第四次
  确认两 seed 的 8/8 carry 全零、prequant absmax `0`、reduction ratio `1`；
  125k eval return 为 `163.17/159.64`。124k pnorm `4647.94/6863.59`，
  NaN/Inf `0/0`，GPU 利用率约 `63%`。因此仍是运行健康但机制无效。四张卡
  当前均被占用；共享卡上的 full-model probe 遇到 cuSolver 初始化失败或进程
  终止，不能替代独占 GPU gate。
- 2026-09-04 04:23 stop（observed）：用户授权后通过各 tmux pane `Ctrl-C`
  优雅停止；seed42/seed1 终止于 env `136418/136339`、update
  `262837/262679`。两条均写入 `run_interrupted` 和 `run_finished`；训练/W&B
  PID 与 tmux 均退出，GPU1/GPU2 回到 `3 MiB`。目录和 100k recovery 保留为
  pre-barrier negative controls，不允许恢复成 corrected treatment。
- 预注册停止条件：只因 NaN/Inf、resident gradient 全零、持续非零 carry
  saturation、dtype/restore 错误、nonzero main error 下 carry 无重建收益、
  OOM 或进程故障停止；不因早期 return 偏低停止。IDs、PID、W&B URL 与状态
  已记录；25k/50k/75k/100k 与最终读数待后续追加。

### `EXP-FP8-ONLINE-CARRY-JITBARRIER-SMOKE-S42`

- 状态：`completed; all mechanism and recovery gates passed`。GPU1 / tmux
  `brc_fp8_carry_jitbarrier_smoke_s42_gpu1`，fresh run ID
  `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_smoke_jitbarrier_v1_s42-20260904-043900`。
- 协议：与 formal 相同的 Dogs、width4096、batch1024、2 updates/step、online
  resident main+carry E4M3、scale-aware backward、target FP8-direct；W&B 关闭，
  tensor interval 1，analysis/recovery 在 5001。命令为
  `BRC_RUN_TAG=jitbarrier_v1 BRC_LOG_TO_WANDB=false BRC_TENSOR_STATS_INTERVAL=1 BRC_ANALYSIS_CHECKPOINT_INTERVAL=5001 BRC_RECOVERY_CHECKPOINT_INTERVAL=5001 scripts/run_dogs_online_fp8_resident_carry.sh 1 42 5001`。
- 结果（observed）：env `5001` / update `5` 正常结束，wall `127.24 s`。
  8/8 carry 为 E4M3 且 zero fraction `0.0001186–0.0001390`；main/logical
  relative error `0.02645–0.02654 / 0.000646–0.000756`，改善
  `35.0x–40.9x`；saturation `0`。四层 resident gradient L2
  `0.1475/0.3902/0.1965/0.3765`；733 条 tensor stats NaN/Inf 总数 `0/0`。
- Recovery（observed）：从完整 5001 recovery 在 tmux
  `brc_fp8_carry_jitbarrier_resume_s42_gpu1` 恢复到 env `5002` / update `7`；
  `resume_reset`、`resumed=true`、first update、analysis/recovery COMPLETE 和
  `run_finished` 均写入。5002 carry 仍非零，error reduction `40.8x–48.1x`，
  saturation/733 条 NaN/Inf 为 `0/0`。manifest 包含 optimizer/replay，main/
  carry payload 各 `134,217,728` bytes、FP32 carry `0`。

### `EXP-FP8-ONLINE-CARRY-JITBARRIER-S42/S1-500K`

- 状态：`running`；2026-09-04 04:48:50 Asia/Shanghai 并行 fresh 启动。只在上述 CPU/GPU、width4096、
  Dogs 与 recovery gates 全部通过后，从 fresh initialization 启动；不恢复
  pre-barrier negative-control 或 smoke checkpoint。
- 计划 run ID：
  `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_s42` 与
  `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_s1`；tmux 为
  `brc_fp8_carry_jitbarrier_s42_gpu1` / `brc_fp8_carry_jitbarrier_s1_gpu2`。
- 协议：Dogs 500k、start5k、replay1M、batch1024、2 updates/step、width4096、
  paper alignment/reward-mean bootstrap；online resident main+carry E4M3、
  carry gain16、fixed anchor off、scale-aware backward；target FP8-direct with
  FP32 parameters；eval/tensor/analysis/recovery cadence `25k/25k/50k/100k`。
- 命令：`BRC_RUN_TAG=jitbarrier_v1 BRC_LOG_TO_WANDB=true scripts/run_dogs_online_fp8_resident_carry.sh 1 42 500000`
  与 GPU2/seed1 对应命令。W&B 上传已获用户授权。必须核验 fresh config 中
  materialization semantic、resume 为空、tmux/PID/GPU UUID/W&B URL，并在
  5k train metrics 与 25k tensor gate 继续检查 finite gradient 和 nonzero carry。
- 实际资源（observed）：seed42 / GPU1 / tmux
  `brc_fp8_carry_jitbarrier_s42_gpu1` / PID `199383` / W&B
  [`7mdayzva`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/7mdayzva)；
  seed1 / GPU2 / tmux `brc_fp8_carry_jitbarrier_s1_gpu2` / PID `199386` /
  W&B [`hxf61n84`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/hxf61n84)。
  两条 config 均确认 `resume_from: ''`、materialization v1、500k、carry true、
  fixed anchor false、target FP8-direct 和 `25k/50k/100k` cadence；W&B events
  均为 enabled/initialized true。
- 启动健康门（observed）：两条均通过 initialization 与 env `5000` 首次更新，
  并继续到 env `9000` / update `8003`。seed42/seed1 的 9k critic loss
  `0.01979/0.02648`、gnorm `0.02754/0.03099`、pnorm `326.61/345.78`，更新
  NaN/Inf 都是 `0/0`；吞吐 `106.83/106.91 transitions/s`。tmux、PID 与
  GPU1/GPU2 映射持续存活。
- 25k carry 机制门（observed，2026-09-04 05:18 Asia/Shanghai）：两条均生成
  `733` 条 tensor stats 和完整 analysis checkpoint；各自 8/8 carry code 张量
  非零。seed42/seed1 carry zero fraction 分别为
  `5.69e-5–1.23e-4` / `5.38e-5–9.81e-5`；main-write relative L2 为
  `0.02574–0.03941` / `0.02639–0.03751`，加 carry 后 logical reconstruction
  relative L2 降至 `5.77e-5–3.33e-4` / `1.15e-4–2.43e-4`，对应误差改善
  `103.18x–582.45x` / `108.47x–322.27x`。16 个 layer-member 的 carry
  saturation 全为 `0`，两组 733 条记录的 NaN/Inf 总数均为 `0/0`。
- 当前运行状态（observed，2026-09-04 05:18 Asia/Shanghai）：两条均推进到
  env `43000` / update `76003`，累计 update NaN/Inf 均为 `0/0`，吞吐约
  `105.62/104.74 transitions/s`；tmux、PID `199383/199386` 与 GPU1/GPU2
  映射仍存活。25k eval return 为 seed42 `72.14`、seed1 `32.39`；这只是首次
  单 seed 早期学习点，不作为最终效果结论。
- 75k follow-up（observed，2026-09-04 05:41 Asia/Shanghai）：两条均到 env
  `75000` / update `140003`，eval return 轨迹分别为 seed42
  `72.14→125.16→185.73`、seed1 `32.39→68.43→189.56`；四个 Dogs 子任务在
  两个 seed 的 75k eval 中都高于各自 25k。25k/50k/75k 的所有 carry 均持续
  非零，75k zero fraction 为 `2.27e-5–1.43e-4`，logical reconstruction
  relative L2 为 `5.88e-5–6.63e-4`，误差改善 `42.85x–668.98x`，carry
  saturation 和每次 733 条 tensor stats 的 NaN/Inf 都是 `0`。seed42 75k
  critic pnorm `1212.55`，高于匹配 target-FP8-direct/FP32-storage control 的
  `958.63`，但远低于无 carry repaired-resident 的 `3046.30`；75k return
  `185.73` 也已进入 direct/current-master 对照的同一早期学习量级
  (`220.44/178.43`)，并高于无 carry repaired-resident 的 `129.44`。
  这是“已经开始持续学习”的强早期证据，但不是 500k 最终效果结论。tmux、
  PID `199383/199386` 和 GPU1/GPU2 映射在 05:41 仍正常，累计 update
  NaN/Inf 为 `0/0`。
- seed42 停止与成熟结果（observed，2026-09-04 10:02 Asia/Shanghai）：按用户
  要求为组合实验释放 GPU1，经 tmux Ctrl-C 优雅停止于 env `454409` / update
  `898819`；事件流完整写入 `run_interrupted(KeyboardInterrupt)` 和
  `run_finished`。停止前 450k eval return/best 为 `832.7369`，四任务为
  `963.91/962.28/891.17/513.59`；450k 的 8/8 carry 仍非零，重建误差改善
  `66.64x–1130.49x`，carry saturation 与 733 条 tensor stats NaN/Inf 均为 0。
  该 run 因主动替换未到 500k，不能记为 completed，但其 18 个 eval 和 450k
  analysis checkpoint 保留为有效部分结果。同期 seed1 未受停止操作影响，后于
  10:44 正常完成 500k / update 990003，final/best/tail-three return
  `782.8582/807.5967/793.2795`；完整 500k analysis/recovery checkpoints、
  `final_checkpoints_finished` 与 `run_finished` 均已核验，累计 update NaN/Inf
  `0/0`。

### `EXP-FP8-ONLINE-CARRY-TARGET-LAG-S42-500K`

- 状态：`running`；用户于 2026-09-04 10:01 Asia/Shanghai 明确要求停止当前
  corrected-carry seed42，并在释放的 GPU1 上以 seed42 从头启动本组合。
- 目的 / Idea：检验已验证的 online CARRY-FP8 与既有 target lag-coded FP8
  能否组合。四个 online residual-Dense kernel 以 main/carry E4M3 + FP32 scale
  常驻，四个 target residual-Dense kernel 以相对 online logical parameter 的
  lag E4M3 + FP32 scale 常驻；optimizer moments、非覆盖参数及必要元数据仍为
  FP32，因此这里的“整体 FP8 critic”专指 online/target 两侧的大残差 kernel
  都不保留 FP32 weight master，而不是声称 Critic 每个标量均为 FP8。
- 代码：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，分支
  `codex/blackwell-fp8-direct`，HEAD
  `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958` 加当前未提交 carry 修复、研究改动
  和 launcher 的 `BRC_TARGET_CRITIC_PRECISION` 参数。
- 协议：fresh Dogs seed42、500k、start5k、replay1M、batch1024、2 updates/step、
  width4096、paper alignment/reward-mean bootstrap；online `fp8_resident` + carry
  gain16 + scale-aware backward，target `fp8_lag`；eval/tensor/analysis/recovery
  cadence `25k/25k/50k/100k`，W&B 开启。不从 target-FP8-direct run 恢复。
- 运行计划：GPU1；tmux `brc_fp8_carry_target_lag_s42_gpu1`；run/W&B name
  `brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_target_lag_s42`；
  命令为 `BRC_RUN_TAG=jitbarrier_v1_target_lag BRC_TARGET_CRITIC_PRECISION=fp8_lag BRC_LOG_TO_WANDB=true scripts/run_dogs_online_fp8_resident_carry.sh 1 42 500000`。
- 对照：同 seed 的当前 CARRY + target FP8-direct run（停止边界待记录），以及已完成
  online FP8-direct + target lag run `nu2d5b90`。两者与本组合分别隔离 target lag
  和 online carry 的增量，但最终解释仍需核对运行时 dirty source snapshot。
- 验收：先以 W&B 关闭的 5001-step fresh smoke 验证首次更新、online carry、target
  lag codes/scale、EMA 诊断和 checkpoint 均有限非空；正式 run 在 25k 检查两套
  持久状态及首次 eval。除 NaN/Inf、持久状态退化、OOM 或运行故障外不按早期
  return 自动停止。
- Smoke（observed）：fresh run
  `...smoke_jitbarrier_v1_target_lag_s42-20260904-101452` 在 GPU1 正常完成 env
  `5001` / update `5`，写入 `run_finished` 及 COMPLETE analysis/recovery
  checkpoints。925 条机制统计中 online carry 8/8 非零，main/logical relative
  L2 为 `0.02645–0.02659 / 0.000619–0.000653`，改善 `40.53x–42.79x`，
  saturation 0；target lag codes 8/8 非零，zero/underflow fraction 约
  `1.19e-7–7.75e-7`。训练 loss、梯度与 update NaN/Inf 均有限/为 0。
  首步 lag quantization relative error 为 `2.30%–2.61%`，相对很小的
  `tau=0.005` intended EMA step 放大为 update relative error `4.58–5.20`；这是
  后续 25k 必须监测的数值风险，而不是启动前隐藏掉的通过指标。
- Smoke 诊断修复（observed）：online function-write JS 的 3 个 Inf 来自两个
  极小 softmax 概率求 midpoint 时下溢为 0；不在训练路径，其他状态有限。正式
  启动前改为对 log 输入使用 dtype tiny 下界，并新增 midpoint-underflow JIT
  regression；目标测试 `1/1`、launcher `bash -n` 和 `git diff --check` 通过。
- 正式启动（observed）：2026-09-04 10:32:00 Asia/Shanghai 在 GPU1 fresh
  启动；tmux `brc_fp8_carry_target_lag_s42_gpu1`，PID `1363708`，W&B
  [`5o2266m9`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/5o2266m9)。
  config 确认 `resume_from: ''`、online carry、target `fp8_lag`、materialization
  v1 和 500k/cadence；W&B 与模型初始化、env 5000 首次更新均完成。10:43 已到
  env `13000` / update `16003`，critic loss/gnorm/pnorm
  `2.6929/1.3367/360.02`，累计 update NaN/Inf `0/0`；首个正式机制/eval 门为
  25k。
- 25k 机制/学习门（observed）：eval return `58.0742`（四任务
  `146.287/24.977/30.956/30.076`），与同 seed corrected-carry / 既有 standalone
  target-lag 的 25k `72.14/76.99` 同属早期学习量级，尚不能下最终结论。925 条
  tensor stats 全部 NaN/Inf `0/0`。online carry 8/8 非零，zero fraction
  `2.99e-5–1.48e-4`，main/logical relative L2
  `0.02613–0.03858 / 8.27e-5–1.65e-4`，重建误差改善
  `199.80x–430.67x`，saturation 0；target lag 8/8 非零，code zero fraction
  `2.68e-6–6.66e-5`、lag quantization relative error `0.00845–0.02728`。
  target 的相对 intended EMA update error 仍为 `1.68–5.43`，applied/intended
  ratio `1.44–4.66`、cosine `-0.830–0.857`，所以继续以 50k/75k 判断这些逐步
  噪声能否在闭环中平均掉；当前没有数值或机制退化理由提前停止。

## 更新规则

- 启动实验时新增一行，记录开始时间、run ID、分支、commit 和完整命令。
- 结束后补充最终 eval、最好 eval，以及末 3 次 eval 均值；异常退出则记录停止 step 和原因。
- 对比前先检查 reset 模式、训练/eval seed 协议和代码 commit，协议不同的结果不直接合并统计。

## EXP-V1-BLOCK-20260908 — 三格式双项 online residual

用户要求按附件实现并启动 GPU0 MXFP8、GPU1 NVFP4、GPU2 MXFP4，三个均从头
seed42/500k；三组启动验收后停止监控。实现和 smoke 见
[变更记录](2026-09-08-v1-block-compute.md)。原始证据与实时启动身份保存在
`/home/caiyuliang/brc_v1_audit`，正式输出在 `/home/caiyuliang/brc_v1_runs`。

- MXFP8：GPU0，16 update / eval / restore smoke 通过，待冻结后立即启动。
- NVFP4：GPU1，待从 V1 准确提交扩展原生后端并通过本卡 smoke。
- MXFP4：GPU2，同一 FP4 提交，待原生后端和本卡 smoke。
- GPU1/2 已有其他进程；按本轮计划只等待对应 GPU，不停止其他任务，不更换 GPU。
- 所有 early return 均不作为重配或停止依据；仅工程/数值错误停止。
