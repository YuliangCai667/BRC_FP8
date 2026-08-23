# BRC FP8 研究索引

- 周期：ICLR 投稿周期；2026-08-23 时用户说明距截稿约一个月，精确日期待补。
- 方向：在 Blackwell 上把 BRC 从原生 FP8 Critic 计算逐步扩展到低精度持久训练状态。
- 代码范围：`/home/caiyuliang/BRC_FP8_blackwell_fp8`，`codex/blackwell-fp8-direct`。
- 当前常驻目标实现：`f88b6628d39caec72f48e0e22d4d8dde815d1745`，已推送到对应远端分支。
- 当前目标：验证目标 Critic 残差核心常驻 FP8 后，逐步 EMA 是否被量化格点吞掉，以及 bootstrap 前向误差是否影响稳定性。

## 文档登记

| 类型 | 文档 | 作用 | 最近实质更新 |
|---|---|---|---|
| Idea | [fp8-research-ideas.md](fp8-research-ideas.md) | 假设、机制、实验门槛和论文叙事 | 2026-08-24 |
| 性能技巧 | [fp8-performance-optimization-backlog.md](fp8-performance-optimization-backlog.md) | 待测性能优化，不等同于采用方案 | 2026-08-23 |
| 实验台账 | [experiment-runs.md](experiment-runs.md) | 命令、运行状态、结果和混杂因素 | 2026-08-24 |
| 在线 FP8 改动 | [2026-08-23-blackwell-fp8-direct.md](2026-08-23-blackwell-fp8-direct.md) | 当前 A/B 计算路径及验证证据 | 2026-08-23 |
| 常驻目标 FP8 改动 | [2026-08-23-persistent-fp8-target-critic.md](2026-08-23-persistent-fp8-target-critic.md) | 本轮实现、验证与限制 | 2026-08-24 |

## 已验证事实

- 在线 Critic 四个残差 Dense 使用原生 FP8 计算时，已有任务未见明显精度损失；这不代表持久 FP8 参数状态同样稳定。
- DMC Dogs seed-42 的严格对照为 A（在线/目标 FP32，最终 eval return `794.40`）和 B（在线 FP8、目标 FP32，最终 `816.37`）。
- 目标 Critic 每次以 `tau=0.005` 吸收最新在线 Critic，常驻 FP8 后会首次把这一小更新暴露给 E4M3 存储分辨率。
- Blackwell 优化后 HLO 已确认四个目标残差 GEMM 直接以常驻 E4M3 参数作为 RHS，并输出 FP32；不存在每次前向的 RHS FP32→FP8 转换。

## 当前实验与决策门

- `EXP-FP8-TARGET-SMOKE-D`：审查后 GPU3 最终短跑已通过；203 updates 无 NaN/Inf，step 150 最后一次真实 EMA 写回在八个目标矩阵上的吞更新率为 `8.01e-5`–`1.18e-4`，applied/intended L2 ratio 为 `0.999999`–`1.000005`。这只是 width-512 单点机制证据。
- `EXP-FP8-TARGET-C-S42`：运行中；在线 FP32、目标残差核心常驻 FP8，GPU3，W&B `trzg1mjw`。
- `EXP-FP8-TARGET-D-S42`：GPU3 旧段按迁移要求主动停止于 env step 10,057 / update 10,115；W&B `fb02bf23`，无 checkpoint，不作为完整正式结果。
- `EXP-FP8-TARGET-D-S42-R1`：运行中；在线 FP8 Direct、目标残差核心常驻 FP8，在 GPU1 从头重启，W&B `iwlomjbu`；step 5,000 首次更新有限且无 NaN/Inf。
- 下一决策门：只在 C/D 证明存在明显吞更新或训练不稳定后，选择更新频率匹配、Kahan、随机舍入或其他修正；当前不预埋任何修正。
