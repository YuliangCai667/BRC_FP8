# BRC FP8 研究索引

- 周期：ICLR 投稿周期；2026-08-23 时用户说明距截稿约一个月，精确日期待补。
- 方向：在 Blackwell 上把 BRC 从原生 FP8 Critic 计算逐步扩展到低精度持久训练状态。
- 代码范围：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，`codex/blackwell-fp8-direct`。
- 当前常驻目标实现：`f88b6628d39caec72f48e0e22d4d8dde815d1745`，已推送到对应远端分支。
- 当前目标 FP8 前向 / FP32 存储实现：`186d153ea8f715b6a2aa18be6c50881b3d03f3dc`，已推送到对应远端分支。
- 当前目标：已确认朴素常驻 FP8 目标的逐步 EMA 存在时间分辨率失配；并行筛选能够保留累计更新、同时限制 bootstrap 误差传播的自适应方法。
- 当前离线实现：`50814df` 加未提交的 scaled Kahan-momentum FP8 baseline；30k screen 已完成，Kahan 的 `C*target` 与 compensation 均为 E4M3 动态缩放状态。

## 文档登记

| 类型 | 文档 | 作用 | 最近实质更新 |
|---|---|---|---|
| Idea | [fp8-research-ideas.md](fp8-research-ideas.md) | 假设、机制、实验门槛和论文叙事 | 2026-08-24 |
| 性能技巧 | [fp8-performance-optimization-backlog.md](fp8-performance-optimization-backlog.md) | 待测性能优化，不等同于采用方案 | 2026-08-23 |
| Baseline 比较 | [baseline-comparisons.md](baseline-comparisons.md) | 已知方法与当前方案的简短证据、机制差异和论文叙事提示 | 2026-08-24 |
| 实验台账 | [experiment-runs.md](experiment-runs.md) | 命令、运行状态、结果和混杂因素 | 2026-08-24 |
| 在线 FP8 改动 | [2026-08-23-blackwell-fp8-direct.md](2026-08-23-blackwell-fp8-direct.md) | 当前 A/B 计算路径及验证证据 | 2026-08-23 |
| 常驻目标 FP8 改动 | [2026-08-23-persistent-fp8-target-critic.md](2026-08-23-persistent-fp8-target-critic.md) | 本轮实现、验证与限制 | 2026-08-24 |
| 目标 FP8 前向改动 | [2026-08-24-fp8-target-forward.md](2026-08-24-fp8-target-forward.md) | FP32 目标存储下的 FP8 bootstrap 前向消融 | 2026-08-24 |
| 目标状态离线模拟 | [2026-08-24-fp8-target-offline-simulation.md](2026-08-24-fp8-target-offline-simulation.md) | 共享 teacher 轨迹下的 lag-coded / interleaved block 方法筛选 | 2026-08-24 |
| FP8 Kahan 基线 | [2026-08-24-fp8-kahan-target-offline.md](2026-08-24-fp8-kahan-target-offline.md) | FP16 SAC Kahan-momentum 的 FP8 适配、验证与限制 | 2026-08-24 |

## 已验证事实

- 在线 Critic 四个残差 Dense 使用原生 FP8 计算时，已有任务未见明显精度损失；这不代表持久 FP8 参数状态同样稳定。
- DMC Dogs seed-42 的严格对照为 A（在线/目标 FP32，最终 eval return `794.40`）和 B（在线 FP8、目标 FP32，最终 `816.37`）。
- 目标 Critic 每次以 `tau=0.005` 吸收最新在线 Critic，常驻 FP8 后会首次把这一小更新暴露给 E4M3 存储分辨率。
- Blackwell 优化后 HLO 已确认四个目标残差 GEMM 直接以常驻 E4M3 参数作为 RHS，并输出 FP32；不存在每次前向的 RHS FP32→FP8 转换。
- 目标 `fp8_direct` 控制保持全部目标参数和 EMA 为 FP32，只把四个目标残差 GEMM 转为 E4M3×E4M3→FP32；其 input/kernel delayed-scaling 状态只在训练 bootstrap 前向推进，不引入目标反向。
- Dogs 50k eval 中，目标 FP8 前向/FP32 存储达到 `120.40`，与仅在线 FP8 的 `124.50` 接近；常驻 FP8 C/D 仅 `15.16/14.93`。C/D 在 125k 仍只有 `8.16/7.23`，已主动停止。
- 100k C/D checkpoint 的直接下一步 EMA 探针显示 `99.9947%`–`99.9994%` FP8 codes 不变，实际/预期更新 L2 仅 `0.195%`–`0.992%`（C）和 `0.240%`–`0.519%`（D）。现有精确零吞更新率与稀疏单点诊断不足以描述累计更新保真度。
- 当前归因是“目标 FP8 状态的时间分辨率失配 → 目标无法跟踪在线网络 → bootstrap 反馈放大学习停滞”；无 NaN/Inf，且 FP8 前向相对同一反量化权重的误差很小，因此不是普通溢出或 GEMM 前向误差。
- 固定健康 checkpoint 的 50k open-loop screen 中，lag-coded 终点 target relative error 为 `0.006665`，naive per-tensor / block / interleaved 为 `0.345703/0.352283/0.346894`；lag 已通过第一轮机制筛选，首版 interleaved 只带来很小改善。
- 同 checkpoint 的 30k Kahan screen 中，lag/Kahan target relative error 为 `0.007725/0.258575`，expected-Q MAE 为 `0.001216/0.141415`；Kahan 与 naive per-tensor 在 60 个点上几乎重合，且需要两份而非一份 FP8 matrix state。

## 当前实验与决策门

- `EXP-FP8-TARGET-SMOKE-D`：203 updates 无 NaN/Inf，但 width-512 的单个稀疏 EMA 样本不能代表 width-4096 的累计更新；保留为实现 smoke，不再作为更新生存率结论。
- `EXP-FP8-TARGET-C-S42`：主动停止于 env step 150k / update 290003；最后完整 125k eval `8.16`，W&B `trzg1mjw`，GPU3 已释放。
- `EXP-FP8-TARGET-D-S42`：GPU3 旧段按迁移要求主动停止于 env step 10,057 / update 10,115；W&B `fb02bf23`，无 checkpoint，不作为完整正式结果。
- `EXP-FP8-TARGET-D-S42-R1`：主动停止于 env step 137438 / update 264877；最后完整 125k eval `7.23`，W&B `iwlomjbu`，GPU1 对应进程/显存已释放。
- `EXP-FP8-TARGET-FWD-SMOKE`：GPU2 已完成；200 steps / 203 updates，无 NaN/Inf，目标 FP8 scale/amax 与 FP32-reference 误差统计完整。
- `EXP-FP8-TARGET-FWD-S42`：完成；在线/目标均 FP8 Direct，目标参数与 EMA 保持 FP32，W&B `gij6jhgd`；500k 最终 return `757.81`，无数值失败。
- `EXP-FP8-TARGET-OFFLINE-SMOKE` / `EXP-FP8-TARGET-OFFLINE-50K`：均完成；lag-coded 明显接近 FP32 teacher，首版 interleaved 未恢复时间分辨率。
- `EXP-FP8-TARGET-KAHAN-OFFLINE-30K`：完成；30,000 updates / 60 点，全部有限。FP8 scaled Kahan 未改善 naive target 漂移，保留为 prior-art baseline。
- 下一决策门：lag-coded 已通过两个共享-teacher open-loop screen，可进入闭环 target-state/forward 设计；Kahan 不启动正式闭环实验。bootstrap 仍只做功能空间诊断，不修改学习目标。
