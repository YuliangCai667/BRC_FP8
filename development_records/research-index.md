# BRC FP8 研究索引

- 周期：ICLR 投稿周期；2026-08-23 时用户说明距截稿约一个月，精确日期待补。
- 方向：在 Blackwell 上把 BRC 从原生 FP8 Critic 计算逐步扩展到低精度持久训练状态。
- 代码范围：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，`codex/blackwell-fp8-direct`。
- 当前常驻目标实现：`f88b6628d39caec72f48e0e22d4d8dde815d1745`，已推送到对应远端分支。
- 当前目标 FP8 前向 / FP32 存储实现：`186d153ea8f715b6a2aa18be6c50881b3d03f3dc`，已推送到对应远端分支。
- 当前目标：修复版 CARRY-FP8 已在 Dogs 上形成成熟双 seed 证据：seed42 于 450k 达到 `832.74` 后按用户要求在 env 454409 停止，seed1 达到 500k final/best/tail-three `782.86/807.60/793.28`，全程 update NaN/Inf 为 0；450k carry 仍非零并提供最高 `1130x` 重建改善。下一实验把 online CARRY-FP8 与既有 target lag-coded FP8 组合，检验 online/target 两侧大残差 kernel 都无 FP32 weight master 是否可行；width4096 smoke 和正式 seed42 的 25k 机制/学习门均已通过，25k eval `58.07`，两侧持久码 8/8 非零且 tensor stats NaN/Inf 为 0。target intended-EMA 相对误差仍为 `1.68–5.43`，因此继续观察 50k/75k。current/delayed scaling 仍作为独立第二因素。
- 当前离线实现：`2091e7e`；scaled Kahan-momentum FP8 baseline 与 30k screen 已完成并推送。
- 当前闭环实现：`1c135af`；`fp8_lag` 已验证并推送，正式 Dogs seed-42 已从该干净提交完成 500k。
- 当前 online 常驻实现与机制诊断已完成并纳入本分支；正式运行使用
  `124de8a` 加当时未提交的实现，实验源码快照保留了精确 provenance。
  2026-09-02 两个 Dogs seed 均完成，naive resident 出现可复现的
  scale-gauge norm runaway 与 effective angular-step collapse。

## 文档登记

| 类型 | 文档 | 作用 | 最近实质更新 |
|---|---|---|---|
| Idea | [fp8-research-ideas.md](fp8-research-ideas.md) | 假设、机制、实验门槛和论文叙事 | 2026-09-04 |
| 性能技巧 | [fp8-performance-optimization-backlog.md](fp8-performance-optimization-backlog.md) | 待测性能优化，不等同于采用方案 | 2026-08-23 |
| Baseline 比较 | [baseline-comparisons.md](baseline-comparisons.md) | 已知方法与当前方案的简短证据、机制差异和论文叙事提示 | 2026-09-04 |
| 实验台账 | [experiment-runs.md](experiment-runs.md) | 命令、运行状态、结果和混杂因素 | 2026-09-04 |
| 在线 FP8 改动 | [2026-08-23-blackwell-fp8-direct.md](2026-08-23-blackwell-fp8-direct.md) | 当前 A/B 计算路径及验证证据 | 2026-08-23 |
| 常驻目标 FP8 改动 | [2026-08-23-persistent-fp8-target-critic.md](2026-08-23-persistent-fp8-target-critic.md) | 本轮实现、验证与限制 | 2026-08-24 |
| 目标 FP8 前向改动 | [2026-08-24-fp8-target-forward.md](2026-08-24-fp8-target-forward.md) | FP32 目标存储下的 FP8 bootstrap 前向消融 | 2026-08-24 |
| 目标状态离线模拟 | [2026-08-24-fp8-target-offline-simulation.md](2026-08-24-fp8-target-offline-simulation.md) | 共享 teacher 轨迹下的 lag-coded / interleaved block 方法筛选 | 2026-08-24 |
| FP8 Kahan 基线 | [2026-08-24-fp8-kahan-target-offline.md](2026-08-24-fp8-kahan-target-offline.md) | FP16 SAC Kahan-momentum 的 FP8 适配、验证与限制 | 2026-08-24 |
| Lag-coded 目标闭环 | [2026-08-24-lag-coded-fp8-target.md](2026-08-24-lag-coded-fp8-target.md) | lag 常驻状态、统一重建、原生 FP8 前向与闭环验证 | 2026-08-24 |
| Online 常驻 FP8 改动 | [2026-08-28-online-fp8-resident-critic.md](2026-08-28-online-fp8-resident-critic.md) | online E4M3 payload、FP32 Adam 控制、机制诊断与手动 Dogs 入口 | 2026-09-02 |
| Moving-anchor online FP8 负结果 | [2026-09-03-gauge-fixed-fp8-residency.md](2026-09-03-gauge-fixed-fp8-residency.md) | previous-norm 递推实现、失败证据与限制 | 2026-09-03 |
| Fixed-anchor online FP8 改动 | [2026-09-03-fixed-anchor-fp8-residency.md](2026-09-03-fixed-anchor-fp8-residency.md) | 固定初始化半径、checkpoint 独立验收与验证 | 2026-09-03 |
| Scale-aware resident backward | [2026-09-03-scale-aware-resident-fp8-backward.md](2026-09-03-scale-aware-resident-fp8-backward.md) | E5M2 output-gradient scaling、physical FP32 VJP、matched-gradient 与 formal gate | 2026-09-03 |
| Current-amax FP32-master 对照 | [2026-09-03-current-amax-fp32-master-control.md](2026-09-03-current-amax-fp32-master-control.md) | 隔离 current-amax 计算与 masterless E4M3 写回的因果实验 | 2026-09-03 |
| CARRY-FP8 online residency | [2026-09-04-carry-fp8-residency.md](2026-09-04-carry-fp8-residency.md) | E4M3 carry 表示、logical AdamW/EMA、验证、状态体积与正式入口 | 2026-09-04 |

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
  freezing 与最高 `39.9x/71.5x` 的 scale drift 跨 seed 复现。这些仍是有效
  描述，但 fixed-anchor 干预已否定其为 return gap 的首要原因。
- 2026-09-03 backward audit：fixed-anchor 的 `68/68`、naive resident seed42
  的 `80/80` resident-kernel gradient 记录全部为零，matched FP8-direct
  control 的 `80/80` 全部非零；同批 bias gradient 非零。独立 GPU probe 复现
  raw FP8 dot 在代表性 scale 下把非零 FP32 kernel gradient 全部下溢为零。
  当前 online resident 的首要 failure mode 是缺少 Flax FP8-direct 式 scaled
  custom backward；candidate 后的 write fidelity 指标没有覆盖这个阶段。
- 2026-09-03 backward repair：独立 GPU probe 的 repaired-resident 与
  FP8-direct kernel gradient cosine/L2 ratio 在 `sigma=1e-3/1e-2` 均为
  `1.0/1.0`，input gradient 也等价且不再全零。保留 Dogs replay 的 matched
  batch 上，四个 kernel cosine 为 `0.99927–0.99994`、L2 ratio 为
  `0.99977–1.00081`；`dQ/da` cosine `0.98988`、ratio `1.00182`。203-update
  GPU smoke 的四层 kernel gradient 均非零且无 NaN/Inf。
- 2026-09-03 repaired-backward + fixed-anchor 100k 因果读数：return
  `134.44`，对 repaired-only `157.98`、matched FP8-direct/FP32-weight
  `340.61`；critic pnorm `519.08/4171.12/1129.21`。独立 NumPy float64
  checkpoint 审计的 8/8 `anchor_norm_ratio` 为
  `0.9999815–1.0004089`，四层 resident kernel gradient 均非零，更新
  NaN/Inf 为 `0/0`，所以结果既不是 anchor recurrence，也不是旧的全零
  backward。单步 candidate→stored expected-Q MAE 仅 `1.80e-5`、critic-loss
  relative error `9.27e-8`，但 8 个 kernel/member 平均 `80.69%` code 在该次
  write 不变；fixed run 50k→100k 的多枚 kernel 方向 cosine 已降到
  `0.0458–0.3246`，而 repaired-only 100k→150k 多数为
  `0.999999–1.0`。这否定“只修 backward 和 norm 即恢复 reward”，并暴露
  两种相反的优化病态：无 anchor 时 scale runaway 令 code/direction 近冻结；
  固定初始半径时又取消了 direct control 中真实存在的中等 norm growth，改变
  了 scale-invariant LayerNorm 层上的 AdamW 有效角步长与长期方向轨迹。
- 100k continued diagnosis：在同一 serialized physical weights 和 probe batch
  上，resident 对全 FP32 的 loss relative error 为 `6.54e-6`，四层 kernel
  gradient cosine `0.98384–0.98892`、L2 ratio `0.88758–0.92540`，`dQ/da`
  cosine `0.9999979`、ratio `0.9999878`；这是有限的 FP8 梯度范数损失，不是
  方向崩坏。fixed online-target kernel cosine 为
  `0.9999933–0.9999999`、relative gap `0.000552–0.003675`，排除 target EMA
  跟不上。8 个 `code→physical→code`、共 `134,217,728` 元素全部 round-trip
  一致。100k Adam 一阶 moment 对权重 cosine 仅约
  `6.9e-5–1.3e-3`，并非大量径向 moment 被球面投影丢弃；从写回标量重建的
  intended/applied tangential cosine 也在数值误差内为 1。结合 direct 在
  100k 已将四层 combined-member norm 自然增长至 `241.87–669.69`，最支持的
  fixed-arm 解释是 Adam 对 scale reparameterization 不变但绝对步长不随半径
  同比例缩放，固定初始小半径取消了 norm growth 带来的有效角步长退火。
- 2026-09-03 current-master 100k 因果隔离：保持 current-amax E4M3 前向和
  repaired E5M2 backward、只恢复 FP32 online master 后，return 从
  repaired/fixed `157.98/134.44` 提升到 `244.89`，四任务逐项改善；direct 为
  `340.61`。current/direct critic pnorm `1090.48/1129.21`，5k–100k SMAPE
  `3.63%`，对 repaired/fixed 为 `77.11%/47.97%`。严格物理初态对齐探针显示
  resident write 的 off-lattice 状态丢失从第 1 次更新开始，到第 10 次形成
  `2.77%` logits relative L2；同一 direct500 权重的 current/delayed 算子
  probe 则为 logits relative L2 `0.48%`、kernel VJP cosine
  `0.97565–0.99818`。因此 masterless 写回是大因素，scaling policy 是较小但
  经闭环放大的第二因素。
- 2026-09-04 CARRY-FP8 工程门：online persistent state 为
  `main E4M3 C + shared FP32 s + carry E4M3 R`，optimizer/target EMA 临时读取
  `s(C+R/16)`，forward/backward 只读取 `sC`。CPU `35/35` FP8 tests、完整
  `63/63` suite 和 GPU1 七项核心测试均通过。Dogs width-4096 smoke 的四层
  gradient 非零、saturation 与 NaN/Inf 为零，checkpoint 可恢复；从 recovery
  取下一批更新时 main/logical relative error 为
  `0.008859–0.012317/0.000212–0.000311`，降低 `39.65x–43.32x`。main/carry
  payload 各 `134,217,728` bytes，新增 full-size FP32 state 为零。
- 2026-09-04 03:36 formal health audit：seed42/seed1 的 25k/50k tensor stats
  与 50k checkpoint 均确认 carry 全零，physical/logical norm 相同，重建收益
  不存在；梯度非零、NaN/Inf 为零。seed42 25k/50k/68k pnorm
  `782.90/1839.92/2605.80` 接近 repaired resident，而不是 current-master。
  这推翻了从单个 recovery-next-update probe 外推“formal carry 已生效”的判断。
- 2026-09-04 03:49 root cause probe：CPU JIT 与 GPU eager 的
  `FP32→E4M3→FP32` 均保留真实量化误差，但 GPU JIT 把同图内的 widen 消费
  路径化简为原 FP32 值，导致 carry residual 精确为零；持久返回的 FP8 code
  仍是量化值。4096² plain/barrier-on-code 的 main relative error 分别为
  `2.684e-8/0.0264938`，非零 carry 为 `0/16,777,205`。因此现有 carry 与
  write-fidelity 单测/诊断漏掉了 production GPU-JIT materialization 语义。
- 2026-09-04 04:22–04:41 corrective gate：main 与 carry E4M3 code 均加入
  `optimization_barrier`。实际 vmapped carry kernel 的 GPU-JIT same-graph 与
  returned-code 二次反量化/重建完全一致；4096² main/carry relative error 为
  `0.00122739/3.2394553e-5`，carry 非零 `16,767,853/16,777,216`、saturation
  为 0。width4096 双 update 的 8 行 error reduction `42.0x–42.7x`。Dogs
  5001 及 recovery→5002 的 reduction `35.0x–48.1x`，carry 非零、gradient
  非零、saturation/NaN/Inf 为 0，checkpoint 完整且 FP32 carry bytes 为 0。
  config/checkpoint 记录 materialization semantic，修复前 resident/lag 不允许
  静默 resume；CPU 全套 `64/64` 通过。

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
- `EXP-FP8-ONLINE-CANON-S42/S1`：moving-anchor FP8 Residency 正式 500k
  intervention 于 2026-09-03 03:11:38 Asia/Shanghai 分别在 GPU2/GPU3
  启动，W&B `fsd8vwfb/97nf6rie`；Phase C 50k 预跑按用户要求跳过。seed42
  resident norm 在 25k→50k 增长 `2.281x`，previous-norm recurrence 的主要
  机制不变量失败。两条 run 已按用户要求于 04:12:35 主动停止，终止于 env
  step `55152/47628`、update `100305/85257`，均写入 `run_interrupted` 与
  `run_finished`，进程与 tmux 已复核退出；它们是 moving-anchor implementation
  failure，不属于 fixed-anchor canonicalization 结果。
- `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-25K`：2026-09-03 05:35:06
  Asia/Shanghai 正常完成 25k / update 40003。analysis/recovery checkpoint 的
  NumPy float64 报告一致，8/8 `anchor_norm_ratio` 为
  `0.9999168–1.0002157`；25k critic pnorm `359.29`，机制门通过。
- `EXP-FP8-ONLINE-FIXED-ANCHOR-S42-R500K`：2026-09-03 15:15:06
  Asia/Shanghai 按用户要求主动停止于 env `452791` / update `895583`，
  `run_interrupted`/`run_finished` 完整，tmux/PID 已退出。450k eval `608.65`；
  保留为 backward-invalid negative control，不能回答正确 backward 下的
  fixed-anchor return 因果问题。
- `EXP-FP8-ONLINE-SCALED-BWD-S42-500K`：2026-09-03 15:37:45 已在 GPU3
  启动（W&B `z6n3ca1s`）；scale-aware backward 的首个正式 run，online
  naive resident persistence、canonicalization 关闭、target
  FP8-direct。用户取消单独 25k gate，seed42 直接 500k 并保留正常 25k
  eval/tensor cadence；仅做启动验收，后续由用户手动判断是否停止。seed1 不与
  seed42 并行。2026-09-03 17:13 核验至 94k：梯度与写回正常，但 75k
  resident norm 已约为初始化的 `11.8x`，return `129.44` 对 matched direct
  `220.44`。后续已到 150k eval `177.48`；run 仍在 GPU3 运行。
- `EXP-FP8-ONLINE-SCALED-BWD-FIXED-ANCHOR-S42-500K`：用户于
  2026-09-03 17:15 Asia/Shanghai 接受，17:16:25 已从随机初始化在 GPU0
  启动；W&B `p7osuon0`。组合 repaired E5M2 backward 与 fixed-initial anchor，
  并与 repaired-only 和 matched FP8-direct 做同 seed 对照；启动验收通过。
  100k eval `134.44`，而 repaired-only/direct 为 `157.98/340.61`；8/8 独立
  anchor ratio `0.9999815–1.0004089`、四层梯度非零、NaN/Inf `0/0`。该读数
  已否定 fixed-initial anchor 是 backward 修复后的充分 reward 修复；run 仍在
  GPU0 运行，未因本次只读诊断停止。
- `EXP-FP8-ONLINE-CURRENT-MASTER-S42-150K`：2026-09-03 18:47:47 在 GPU1
  从头启动，W&B `vgd8apty`。online residual Dense 使用与 repaired resident
  相同的 current-amax E4M3 前向和 scale-aware E5M2 backward，但持久参数与
  AdamW state 均为 FP32，无 canonicalization；target 仍为 `fp8_direct`。初始
  resident/current-master logits 逐元素一致，三次更新 smoke 与 metadata 推进
  通过。100k return `244.89`，显著高于 repaired/fixed
  `157.98/134.44`，但低于 direct `340.61`；critic pnorm `1090.48` 接近
  direct `1129.21`，而后两者为 `4171.12/519.08`。因果主结论成立，run
  继续 150k。物理初态严格对齐的 width16
  探针进一步显示：一次 resident write 后 logits 仍相同，但到第 10 次 update
  kernel/logit relative L2 已累计为 `0.008423/0.027712`。
- `EXP-FP8-ONLINE-CARRY-S42-SMOKE-5K`：本地 GPU1/tmux 工程检查完成；主 run
  到 env 5001/update 5 正常退出，analysis/recovery 均成功。初始零 main-error
  边界点的 carry 为零；实际 recovery-next-update 在非零 main error 下取得
  `39.65x–43.32x` 重建收益，saturation/NaN/Inf 为零。
- `EXP-FP8-ONLINE-CARRY-JITBARRIER-S42/S1-500K`：修复版 W&B
  `7mdayzva/hxf61n84`。seed42 在 450k eval/best `832.74` 后按用户要求优雅
  停止于 env 454409，为 target-lag 组合释放 GPU1；seed1 已正常完成 500k，
  final/best/tail-three `782.86/807.60/793.28`，500k analysis/recovery 与
  `run_finished` 完整。两者累计 update NaN/Inf 为 0，正式 carry 机制
  从 25k 持续有效到 450k。
- `EXP-FP8-ONLINE-CARRY-TARGET-LAG-S42-500K`：width4096 fresh smoke 在
  env 5001 正常完成，online carry 和 target lag 均 8/8 非零、checkpoint 完整；
  target lag 首步相对 intended EMA update 的误差为 `4.58–5.20`，列为 25k
  风险门。正式 run 于 2026-09-04 10:32 在 GPU1/tmux
  `brc_fp8_carry_target_lag_s42_gpu1` fresh 启动，PID `1363708`、W&B
  `5o2266m9`；25k eval `58.07`，online carry/target lag 均 8/8 非零、925 条
  tensor stats NaN/Inf 为 0。carry 重建改善 `199.80x–430.67x`；target
  intended-EMA relative error `1.68–5.43`，继续观察 50k/75k。
