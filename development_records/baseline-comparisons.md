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
