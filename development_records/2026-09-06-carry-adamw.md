# CARRY-AdamW：分块双码 optimizer moments

## 动机与实现

按用户冻结方案，保留健康蓝线 residual-only CARRY-FP8 权重、scale-aware
backward、高精度 input/head 与 lag-coded target，只压缩四个 residual kernel
的 Adam m/v。每个 ensemble member 内沿连续方向分为 128 元素块；m/v 独立
FP32 scale，E4M3FN main + signed E4M3FN carry，重建 `s*(C+R/16)`。
零块 scale=1；没有权重量化器的绝对 scale 下限。FP32 subnormal scale 向上
取下一可表示值，归一化前同时用 2^64 提升分子/分母，避免极小 scale 舍入造成
FP8 overflow。Blackwell 可保留最小 subnormal，CPU 遵循其原有 flush-to-zero
算术。主码与 carry 均有 JAX
optimization barrier，防止 GPU 编译器消除同图 narrowing/widening 的真实误差。

`jaxrl/optimizers.py` 的包装层解码旧状态，直接调用本地 Optax 0.2.5
`scale_by_adam` 得到尚未量化的新 moments 和本次 update，最后编码新 moments。
weight decay 和 learning-rate transform 也直接复用 Optax。默认超参数保持
`lr=3e-4, b1=.9, b2=.999, eps=1e-8, eps_root=0, weight_decay=1e-4`；
调用方仍把逻辑 CARRY 参数传给 AdamW，权重写回和 target 递推没有改动。

`--critic_optimizer_state={fp32,bf16,fp8,fp8_carry}` 保留四个成本对照；默认
仍为 FP32。正式入口 `scripts/run_dogs_carry_adamw.sh` 固定 residual-only
权重 scope 与 lag target，默认选择 fp8_carry。初始化事件、recovery manifest
新增 optimizer 各 leaf 的 dtype/shape/bytes，resume 配置检查拒绝静默改变
moment 表示。既有 tensor-stat cadence 增加重建 v 非负性、非有限值和 carry
统计。全套回归发现已有 `_is_resident_kernel` 新签名与离线模拟器旧调用不兼容，
以 optional model 恢复旧 residual-only 调用行为，显式 model 的 edge scope 不变。

## 验证与证据

- 9 项新增 optimizer/learner 测试在 CPU 和共享 GPU2 上通过，覆盖 block/ensemble
  隔离、padding、零/小状态、signed v carry、GPU 实际舍入、无损 Optax 对齐、
  同步旧状态 oracle、非零 decay、schedule、eps_root、状态 bytes、真实 UTD2 与
  checkpoint 后下一步。加载时全部 optimizer bytes 数值相同；下一步 FP8 codes
  逐元素一致。独立 GPU 编译的未压缩 FP32 moment 允许与既有测试相同的
  `rtol=2e-7, atol=1e-12`，观察到的差异为约一个 FP32 ULP。
- CPU 全套首次运行的 10 项离线 target 测试因上述旧接口失败；修复后
  10/10 定向复测通过，最终全套 **76/76** 通过（320.716 s）。没有改变训练
  公式以绕过失败。日志归档在 `runs/carry_adamw_validation/`。
- `scripts/probe_carry_adamw.py` 用健康 early/late recovery 的真实梯度做各 256
  步同梯度四臂探针；方法与结果详见
  [实验台账](experiment-runs.md#exp-carry-adamw-s42--optimizer-moments-压缩2026-09-06)。
  最终版本 `early_v2/late_v2` 各 256 步完成。早期最终 u relative L2：BF16 `0.0615859`、single FP8 `1.0639391`、
  dual FP8 `0.0212123`。双码最差 cosine `0.9997278`、carry saturation=0、
  v<0=0；32768 个坐标中 v 低于参考一半的数量最多 4，最大 update peak
  为该步参考最大值的 `2.016x`。单码对应峰值达到 `1064.727x`。
  双码平均误差较小并不代表坐标误差均匀，也不证明长期 RL 收益。
  后期 BF16/single FP8/dual FP8 分别为 `0.0424896/0.0990562/0.0161540`；
  双码最差 cosine `0.9998680`，v<0/非有限/carry saturation 均为 0。
  后期 raw-v 低估统计包含 Adam eps 主导的 subnormal 坐标，不等同于有效分母低估。
- `late_v1` 因极小 scale 编码溢出而失败并保留；上述边界修复及新版本重跑消除了
  该问题。width4096 learner smoke、正式 seed42 500k：执行状态由台账维护。
- width4096 smoke 实际更新 4 次，恢复后再更新 2 次，两次各 1005 条 tensor records
  均有限，四个大 kernel 梯度非零，moments carry 非零且 saturation=0、v 非负。
  初始化、checkpoint 原始 bytes 独立审计、恢复后的 inventory 一致：residual
  moments **520 MiB**，没有完整 FP32 mu/nu 副本。Critic checkpoint 为
  `853942277` bytes（蓝线 `1382423893`），全模型持久状态为 `1004275976`
  bytes（蓝线 `1532758280`）。审计报告位于 validation 目录。
- seed42 500k 正式 run `g5m77goj` 已完成，20 次 eval、最终 analysis/recovery
  checkpoint 与 `run_finished` 均完整；原协议配置与健康蓝线相同，`resume_from=''`。
  最终/最佳/末3次 return 为 `808.6335/827.0607/811.6903`，蓝线为
  `785.1083/797.5496/787.4880`。单 seed 支持本设置下的闭环可行性，尚无跨 seed
  不确定性估计。
- `scripts/report_carry_adamw.py` 从本地日志生成
  `runs/carry_adamw_validation/run_comparison.json`，覆盖 final/best/tail3 return、
  同步 profile 窗口、JAX allocator peak、analysis/recovery 大小及其 replay/model
  分项。首个 profile 窗口包含诊断图编译，单独保留而非误当稳定 update 耗时。
  2026-09-07 读取两条完整运行的本地记录后返回 `comparison_complete=true`。
  蓝线后续窗口每 learner update 平均/中位为 `26.607/28.898 ms`，本实验为
  `41.325/41.373 ms`；本实验共享 GPU2，不能用这组时间作受控的加速或减速归因。
  完整终点 manifest 仍为 520 MiB residual moments：512 MiB E4M3 codes/carry、
  8 MiB FP32 scales，未记录完整 FP32 mu/nu 副本。

## 完整运行的内存记录（2026-09-07）

| 口径 | FP32 Adam 蓝线 | CARRY-AdamW | 减少 |
|---|---:|---:|---:|
| 覆盖 residual moments，MiB | 1024 | 520 | 504 |
| 全模型持久状态，MiB | 1461.7522 | 957.7522 | 504 |
| 500k train 采样的 JAX bytes_in_use，MiB | 6886.5508 | 5235.4790 | 1651.0718 |
| 日志中 JAX peak_bytes_in_use 最大值，MiB | 11246.5461 | 9859.2896 | 1387.2566 |

前两行统计状态载荷；后两行还包括运行期缓冲区和诊断开销。基线未记录最后一次
checkpoint 之后的 allocator peak，表中比较的是已有日志所观测的最大值。不能把
1387.2566 MiB 全部归因于 moments 载荷压缩，也不能把它当作纯训练算子的峰值。
两次运行均设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，allocator 使用默认值。
当前后端的 reserved counters 为 0，不能据此推断进程没有分配池；系统 recorder
采集的是整卡显存，共享 GPU 时不能替代进程内 allocator 统计。

进一步降低显存的优先线索是诊断张量生命周期：`train.py` 在 tensor-stat 分支中
保存 `diagnostic_trees`，其中含重建的 physical/logical/target kernels、梯度与旧
optimizer state；汇总后变量仍在 main 的局部作用域内，直到下次赋值。CARRY-AdamW
在 25k/26k 的活动分配为 1135.7466/4833.7363 MiB，存在与首次诊断阶段同步的
上升。应先用作用域释放和逐项汇总验证实际可回收量，再考虑更新入口的 buffer
donation、融合 moment 解码/Adam/编码、activation rematerialization。这些是下一步
候选优化，本次 Adam 提交不改变训练或诊断语义。

## 成本与限制

四个 `(2,4096,4096)` kernel 的 m/v 理论持久状态从 1024 MiB 变为
520 MiB（main/carry 512 MiB + scales 8 MiB），减少 504 MiB。
BF16 对照为 512 MiB，单码 block-FP8 为 264 MiB。实际全模型 bytes、
checkpoint 文件大小以 recovery inventory 和文件为准，包含未压缩其他参数。

首版采用 JAX/XLA 组合算子，未编写 fused GPU kernel；更新时可以临时物化
FP32 数组。持久状态压缩不等于峰值显存下降或加速，正式实验记录 allocator
peak、同步 update 时间与 checkpoint 大小；GPU2 共享带来 timing 混杂。
上述 matched probe 冻结网络与 metadata，采样完整块，不是长程 RL 质量证明。

2026-09-07 用户要求在当前分支提交 Adam 修改。提交包含 moment codec/Optax
包装层、learner/CLI/checkpoint 接入、测试、探针、汇总工具和本记录；既有独立
input/head FP8 scope 改动与 PPT 提纲保留在工作区。正式实验仍以运行时的源码
快照为准，提交提取的 residual-only Adam 路径另做独立验证。
该待提交树已独立导出并通过 CPU 全套 74/74 测试（291.391 s），不依赖工作区
尚未提交的 edge-scope 实现；原完整工作区的 76 项测试另含 edge-scope 测试。
