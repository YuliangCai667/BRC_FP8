# 与 Baseline 的比较

本文件只保留可用于论文写作的简短对照逻辑。完整协议和数据以
[实验台账](experiment-runs.md) 为准，方法演化以
[idea 文档](fp8-research-ideas.md) 为准。

## Lag-coded target vs. FP8 Kahan-momentum

- **Baseline**：按 FP16 SAC 的 Kahan-momentum 使用 `C=1e4` 放大 target，
  并维护 Kahan compensation；在 FP8 适配中，两份状态都以 E4M3 +
  ensemble 独立动态 per-tensor scale 保存，不保留 FP32 master。
- **公平比较**：从同一个健康 Dogs 100k checkpoint 出发，在同一次 run
  中共享 teacher online-Critic 轨迹，离线模拟 30,000 learner updates。
- **结果**：lag/Kahan 的 target relative error 为
  `0.007725/0.258575`，位移 cosine 为 `0.999540/0.030080`，
  expected-Q MAE 为 `0.001216/0.141415`。Kahan 与 naive FP8 EMA 在
  60 个诊断点上几乎重合，但需要两份 FP8 矩阵状态；lag 只需一份。
- **为什么 Kahan 不如 lag**：Kahan 补偿“低精度加法丢失的余量”，但
  修正量最终仍要写回绝对 target 权重的粗糙 E4M3 lattice。动态 scale
  又会同步吸收 `C=1e4` 的放大，因此放大本身不增加尾数分辨率。
  Lag-coded 改为存储幅值更小的 `target-online`，直接改变表示坐标，
  让同样的 E4M3 位数对应更细的绝对更新分辨率。
- **论文提示**：可以将 Kahan 描述为“在 FP16 中有效的补偿算术”，而
  FP8 target EMA 需要进一步解决状态表示问题；我们的收益来自改变
  被量化对象，而不是堆叠更强的累加技巧。
- **限制**：当前证据来自单 checkpoint 的 open-loop 机制实验；lag 的
  闭环训练和更多 benchmark 仍需验证。这里比较的是带动态 scale 的
  FP8 Kahan 适配，不等同于论文原始的固定范围 FP16 实现。

## Naive online FP8 residency vs. FP8-direct compute with FP32 weights

- **Baseline / treatment**：control 的 online/target residual GEMM 都使用
  FP8 Direct，但在线和目标权重保持 FP32；treatment 只把四个 online
  residual-Dense kernel 改成 E4M3 payload + FP32 per-ensemble scale，目标
  参数/EMA 和 FP32 Adam moments 不变且不保留 FP32 master weight。
- **公平比较**：Dogs seed 42、width 4096、500k、batch 1024、每环境步两次
  learner update、同一 eval 协议；两臂只在 online residual kernel 的
  persistent representation/write path 上存在预定差异。完整 ID 见
  [实验台账](experiment-runs.md)。
- **决定性结果**：naive resident 与 control 的最终 / 最佳 / 末 3 次平均
  return 为 `505.98/530.14/512.29` 与
  `757.81/786.63/774.95`；最终 critic pnorm 为
  `11428.8` 与 `2712.0`。resident 的四个覆盖 kernel 贡献 99.0% 的
  squared norm，且前 275k 轨迹在一次独立重跑中近似复现。
- **机制差异**：单步 applied/intended L2 与 update cosine 仍接近 1，
  expected-Q MAE 约 `8.1e-6`，因此不是普通 update swallowing 或单步
  Q-function 敏感性。多个 runaway member 在 25k→500k 间保持几乎相同
  的 FP8 code norm/direction，但 scale 和物理 norm 同步增长 `16x–36x`。
  这支持紧随 LayerNorm 的矩阵沿尺度自由度进入 smooth scale-dominated
  runaway；稀疏径向 cosine 尚不能证明每次投影都有固定向外偏置。
- **增强诊断复现**：两个新的 treatment seed 均完成 500k。resident norm
  增长 `21.7x/23.2x`，intended angular step 下降 `19.9x/33.4x`，而
  actual/intended angular retention 几乎严格为 1。二阶更新平方项比解释
  norm 增长所需量小 `2.4e3x/1.1e4x`；weight-decay write retention 虽仅
  `0.829/0.937`，但完整 AdamW decay 在全程也只有约 2.93% 收缩能力。
  因而主因收窄为 scale-gauge norm runaway 压低有效角步长，而不是
  directional write loss、二阶切向累积或 decay 丢失。
- **论文提示**：online resident FP8 的困难不是“单步写不进去”，而是
  动态 scale 成为未受函数损失约束的连续 gain state，并在长期闭环中
  形成层/ensemble 不对称漂移。候选修复应利用 LayerNorm gauge 做 norm
  canonicalization，而不只是增加 bit width 或 full-size residual。
- **限制**：目前有两个 resident treatment seed，但只有 seed42 有严格匹配
  control，不能报告两种子 paired quality estimate。相同 seed42/config 的
  resident 重跑最终 return 从 `505.98` 到 `667.78`，显示长程轨迹敏感；
  bootstrap amplification 未经 open-loop 因果对照证明，吞吐与能耗也不应
  从共享运行条件推断。
