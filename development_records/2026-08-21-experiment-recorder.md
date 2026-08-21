# 统一实验记录器与 Checkpoint

日期：2026-08-21

状态：已实现，作为 FP32 基线及后续 FP8 实验的公共设施。

## 动机

全量多任务训练成本高，原实现主要依赖稀疏命令行输出和 W&B，缺少完整的 episode 数据、性能拆分、数值分布及可恢复 checkpoint。此次改造建立本地优先、低频同步、可恢复且便于 FP32/FP8 对比的统一记录路径。

## 本次修改

- 新增 `ExperimentRecorder`：每次运行写入 `runs/<env>/<run_id>/`，本地 JSONL/CSV 为权威数据，W&B 仅作镜像；外部日志或监控失败不终止训练。
- episode 按任务累计 return、success、length，缓冲后批量写入；训练指标每 1,000 步在既有同步点采样一次。
- 记录吞吐、wall-clock、JIT、评估、日志、profiling 和 checkpoint 耗时；后台每 10 秒采集 CPU、RAM 和 GPU 设备指标。
- 每 25,000 步基于固定 probe batch 保存参数、梯度、优化器状态、activation、policy mean/std 和 critic logits 的摘要，不保存完整张量。
- 新增原子 checkpoint：分析 checkpoint 保存模型，恢复 checkpoint 额外保存优化器、RNG、normalizer、recorder 和 replay buffer 有效区域；支持 `--resume_from`。
- 修复模型加载器混用、缺失 step/RNG/optimizer 状态及 one-hot dtype 警告；replay buffer 支持有效区间分块保存和环绕恢复。
- 预留 FP8 统计 provider，可扩展 amax、scale、饱和/下溢率、SQNR、relative L2 和 cosine similarity。

默认频率：训练指标 1k、系统指标 10 秒、评估/张量统计 25k、分析 checkpoint 50k、恢复 checkpoint 100k。低频工作耗时单列，不混入 steady-state 吞吐。

## 评估时看什么

| 目的 | 主要指标/文件 |
|---|---|
| 任务效果 | `eval_metrics.jsonl` 的 deterministic success/return 及 per-task 分解 |
| 数据收集行为 | `episodes.jsonl`、训练窗口 success/return；与 eval 分开解释 |
| 训练稳定性 | loss、梯度/参数 norm、Q prediction/target、reward、NaN/Inf |
| FP32/FP8 数值差异 | `tensor_stats.jsonl`；重点比较 absmax、P99.9、零值、饱和/下溢及 FP8 扩展指标 |
| 速度 | 排除 JIT、eval、profiling、checkpoint 后的 transitions/s 与 updates/s |
| 显存 | 优先比较 JAX `bytes_in_use`/peak；NVML 设备占用作为资源环境参考 |
| 恢复能力 | `events.jsonl`、checkpoint manifest、恢复前后固定 probe 输出 |

## 容易误读的地方

- 训练 success 来自带噪声的数据收集策略；eval 默认是 `temperature=0` 的确定性均值策略，两者不是同一指标。
- `temperature` 训练指标是 SAC 熵系数 alpha，不是 policy std，也不是评估时传入的采样 temperature。
- `action_std` 混合了任务、动作维度和策略均值差异；判断分布尖锐程度应看 `outputs/policy_std`，且该值是 tanh 前尺度。
- `nvidia-smi`/NVML 包含 CUDA 上下文和分配器缓存。即使关闭预分配，缓存池也可能阶梯式增长；它不等于活跃训练张量显存。
- 周期训练指标是该记录点的最新 device sample，不是最近 1,000 步的均值；episode 指标才是窗口汇总。
- 固定 probe 便于纵向比较，但只代表被抽中的状态，不应替代真实 eval 轨迹和 per-task 分析。
- 恢复会重置环境并丢弃未完成 episode；模型、优化器和 replay 连续，但不保证环境轨迹逐 bit 重现。

## 验证与后续

已覆盖 episode 映射、JSONL、W&B/监控容错、checkpoint 原子性、参数/优化器/RNG/normalizer/replay round-trip，以及短程中断恢复。启动正式 FP8 对照前，应固定依赖、seed、记录频率和 allocator 设置，并用相同 checkpoint/初始状态补充 deterministic 与 stochastic 配对评估。
