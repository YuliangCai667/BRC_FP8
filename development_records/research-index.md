# BRC FP8 研究索引

- 周期：ICLR 投稿周期；2026-08-23 时用户说明距截稿约一个月，精确日期待补。
- 方向：在 Blackwell 上把 BRC 从原生 FP8 Critic 计算逐步扩展到低精度持久训练状态。
- 代码范围：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，`codex/blackwell-fp8-direct`。
- 当前常驻目标实现：`f88b6628d39caec72f48e0e22d4d8dde815d1745`，已推送到对应远端分支。
- 当前目标 FP8 前向 / FP32 存储实现：`186d153ea8f715b6a2aa18be6c50881b3d03f3dc`，已推送到对应远端分支。
- 当前目标：主线调整为“全链路 FP8 常驻约束下的稳定 RL 训练”。第一阶段只将 online Critic 的四个 residual-Dense kernel 作为 E4M3 常驻学习权重；online/target residual GEMM 均保留 FP8，target 使用 `fp8_direct`、参数与 EMA 保持 FP32，并关闭 target 常驻与 lag-coded，用 FP32 Adam moments 隔离 online 权重写回问题。
- 当前离线实现：`2091e7e`；scaled Kahan-momentum FP8 baseline 与 30k screen 已完成并推送。
- 当前闭环实现：`1c135af`；`fp8_lag` 已验证并推送，正式 Dogs seed-42 已从该干净提交完成 500k。
- 当前 online 常驻实现与机制诊断已完成并纳入本分支；正式运行使用
  `124de8a` 加当时未提交的实现，实验源码快照保留了精确 provenance。
  2026-09-02 两个 Dogs seed 均完成，naive resident 出现可复现的
  scale-gauge norm runaway 与 effective angular-step collapse。

## 文档登记

| 类型 | 文档 | 作用 | 最近实质更新 |
|---|---|---|---|
| Idea | [fp8-research-ideas.md](fp8-research-ideas.md) | 假设、机制、实验门槛和论文叙事 | 2026-09-02 |
| 性能技巧 | [fp8-performance-optimization-backlog.md](fp8-performance-optimization-backlog.md) | 待测性能优化，不等同于采用方案 | 2026-08-23 |
| Baseline 比较 | [baseline-comparisons.md](baseline-comparisons.md) | 已知方法与当前方案的简短证据、机制差异和论文叙事提示 | 2026-09-02 |
| 实验台账 | [experiment-runs.md](experiment-runs.md) | 命令、运行状态、结果和混杂因素 | 2026-09-02 |
| 在线 FP8 改动 | [2026-08-23-blackwell-fp8-direct.md](2026-08-23-blackwell-fp8-direct.md) | 当前 A/B 计算路径及验证证据 | 2026-08-23 |
| 常驻目标 FP8 改动 | [2026-08-23-persistent-fp8-target-critic.md](2026-08-23-persistent-fp8-target-critic.md) | 本轮实现、验证与限制 | 2026-08-24 |
| 目标 FP8 前向改动 | [2026-08-24-fp8-target-forward.md](2026-08-24-fp8-target-forward.md) | FP32 目标存储下的 FP8 bootstrap 前向消融 | 2026-08-24 |
| 目标状态离线模拟 | [2026-08-24-fp8-target-offline-simulation.md](2026-08-24-fp8-target-offline-simulation.md) | 共享 teacher 轨迹下的 lag-coded / interleaved block 方法筛选 | 2026-08-24 |
| FP8 Kahan 基线 | [2026-08-24-fp8-kahan-target-offline.md](2026-08-24-fp8-kahan-target-offline.md) | FP16 SAC Kahan-momentum 的 FP8 适配、验证与限制 | 2026-08-24 |
| Lag-coded 目标闭环 | [2026-08-24-lag-coded-fp8-target.md](2026-08-24-lag-coded-fp8-target.md) | lag 常驻状态、统一重建、原生 FP8 前向与闭环验证 | 2026-08-24 |
| Online 常驻 FP8 改动 | [2026-08-28-online-fp8-resident-critic.md](2026-08-28-online-fp8-resident-critic.md) | online E4M3 payload、FP32 Adam 控制、机制诊断与手动 Dogs 入口 | 2026-09-02 |

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
- `fp8_lag` 闭环 smoke 完成 200 env steps / 203 learner updates，loss/梯度/lag/metadata 的 NaN/Inf 均为 0；实际 EMA 的 L2 ratio 为 `0.999993–1.000002`，相对误差为 `1.09e-4–1.55e-4`。
- `EXP-FP8-TARGET-LAG-S42` 已在 2026-08-24 完成 500k；最终 / 最佳 / 末 3 次平均 eval return 为 `806.10 / 806.73 / 799.81`，与 online-FP8/target-FP32 基线 `816.37` 同量级。这支持 lag-coded 能修复 naive resident target 的时间分辨率失配，但它依赖 FP32 online anchor，不解决 online 常驻更新。
- `EXP-FP8-ONLINE-RESIDENT-DIAG-S42-B` 已完成 500k；naive online resident 最终 / 最佳 / 末 3 次平均 eval return 为 `505.98 / 530.14 / 512.29`，匹配的 FP8-direct/FP32-weight control 为 `757.81 / 786.63 / 774.95`。最终 critic pnorm 为 `11428.8` 对 `2712.0`，四个 resident kernels 占 squared norm 的 99.0%。
- Online resident 的 applied/intended L2、update cosine 和单步 expected-Q fidelity 都保持良好；两个稀疏 radial cosine 没有稳定单向 optimizer 偏置。checkpoint 显示多个 runaway member 的 FP8 code norm/direction 几乎不变，而 per-ensemble scale 与物理 norm 增长 `16x–36x`，支持 LayerNorm 尺度自由度上的 smooth scale-dominated runaway。
- `EXP-FP8-ONLINE-RESIDENT-MECH-S42/S1` 均完成 500k；resident norm 增长
  `21.7x/23.2x` 导致 intended angular step 下降 `19.9x/33.4x`，但实际角步
  retention 近乎 1。二阶项和 decay attenuation 均不足以解释增长；code
  freezing 与最高 `39.9x/71.5x` 的 scale drift 跨 seed 复现。当前主机制为
  scale-gauge norm runaway 引发 effective angular-step collapse。

## 当前实验与决策门

- `EXP-FP8-TARGET-SMOKE-D`：203 updates 无 NaN/Inf，但 width-512 的单个稀疏 EMA 样本不能代表 width-4096 的累计更新；保留为实现 smoke，不再作为更新生存率结论。
- `EXP-FP8-TARGET-C-S42`：主动停止于 env step 150k / update 290003；最后完整 125k eval `8.16`，W&B `trzg1mjw`，GPU3 已释放。
- `EXP-FP8-TARGET-D-S42`：GPU3 旧段按迁移要求主动停止于 env step 10,057 / update 10,115；W&B `fb02bf23`，无 checkpoint，不作为完整正式结果。
- `EXP-FP8-TARGET-D-S42-R1`：主动停止于 env step 137438 / update 264877；最后完整 125k eval `7.23`，W&B `iwlomjbu`，GPU1 对应进程/显存已释放。
- `EXP-FP8-TARGET-FWD-SMOKE`：GPU2 已完成；200 steps / 203 updates，无 NaN/Inf，目标 FP8 scale/amax 与 FP32-reference 误差统计完整。
- `EXP-FP8-TARGET-FWD-S42`：完成；在线/目标均 FP8 Direct，目标参数与 EMA 保持 FP32，W&B `gij6jhgd`；500k 最终 return `757.81`，无数值失败。
- `EXP-FP8-TARGET-OFFLINE-SMOKE` / `EXP-FP8-TARGET-OFFLINE-50K`：均完成；lag-coded 明显接近 FP32 teacher，首版 interleaved 未恢复时间分辨率。
- `EXP-FP8-TARGET-KAHAN-OFFLINE-30K`：完成；30,000 updates / 60 点，全部有限。FP8 scaled Kahan 未改善 naive target 漂移，保留为 prior-art baseline。
- `EXP-FP8-TARGET-LAG-SMOKE`：完成；GPU1，203 updates，498 条 tensor stats，HLO 与数值验收通过。
- `EXP-FP8-TARGET-LAG-S42`：完成；GPU1，clean `1c135af`，W&B `nu2d5b90`；500k 最终 return `806.10`，20 次 eval 有限，运行写入 `run_finished`。
- `EXP-FP8-ONLINE-RESIDENT-DIAG-S42-A/B`：A 主动停止于 276144，B 正常完成 500k；两次前 275k 轨迹近似复现。B 的 W&B 为 `vv3x4xuh`，所有更新指标有限。
- `EXP-FP8-ONLINE-RESIDENT-MECH-S42/S1`：均正常完成 500k / 20 次 eval / 20
  次机制诊断，W&B `ozhbrc8v/zymfcugz`；最终 return `667.78/564.03`，最终
  critic pnorm `9641.4/19714.7`。
- 下一决策门：是否实现并运行 LayerNorm-aware norm-canonicalized resident write，以每个 layer/member 一个 FP32 reference norm 锚定物理权重；optimizer tangent projection 作为后续可选消融。bootstrap amplification 仍需另行 open-loop 因果对照，不从本轮闭环相关性直接宣称。
