# BRC 论文实现对齐与 MetaWorld 协议

日期：2026-08-22
分支：`codex/paper-alignment`

## 动机

复现的 MetaWorld-50 eval success 明显低于论文 Figure 16。核查发现公开代码与论文附录在任务嵌入范数、截断 return bootstrap、熵修正三处不一致；同时当前 MetaWorld 版本会冻结构造时的 `rand_vec`，而作者早期 BRO 封装会在 episode 结束后重建环境。需要把算法差异和环境协议拆开验证，不能直接归因于论文结果。

## 修改

- 新增 `--paper_alignment` preset；开启后解析为 L1 task embedding、critic bootstrap、per-task empirical entropy。三项仍可用独立 flag 覆盖，便于逐项消融；关闭时保持公开代码的 L2、reward-mean bootstrap 和 target-entropy `/2` 修正。
- 截断时用 policy 动作和 target critic 的 categorical expectation 估计边界价值，并乘回当前 reward normalization scale 后接入原始 Monte Carlo return。只在 truncation 时增加一次前向和同步。
- 经验熵直接复用 actor 更新中的 `-log π(a|s,i)`，按 task 聚合并下一步使用，不额外执行 actor 前向；相关 RNG、熵和计数进入 checkpoint。
- `--metaworld_reset_mode={frozen,recreate}` 独立于算法。`recreate` 每个 episode 用确定性 seed 流重建 MetaWorld env，从而改变冻结的任务实例；训练和 eval 使用各自 seed 流。
- 新增 `scripts/run_metaworld_alignment.sh`，提供 `legacy_frozen`、`legacy_recreate`、三项单独修正和 `paper_recreate` 六种可比运行方式。

论文没有说明 bootstrap 使用 online/target critic、策略动作的具体估计器。本实现选择 SAC 中更稳定且一致的 target critic + policy sample；这是明确记录的实现解释，不应称为作者唯一实现。

## 评估时看什么

- `config.yaml`：确认 `resolved_*` 三项及 `metaworld_reset_mode`，不要只看 `paper_alignment` 原始 flag。
- `eval_metrics.jsonl`：主指标是 deterministic `success_mean` 及 `success_by_task`；用相同 seed、reset mode 和 eval 频率比较。
- `train_metrics.jsonl`：检查 `task_entropy_by_task`、`entropy_correction_by_task`、`reward_denominator_by_task` 和当前熵样本计数。
- `episodes.jsonl`：截断 episode 会记录 normalized/raw bootstrap，便于发现尺度或异常值问题。
- 建议先比 `legacy_frozen` vs `legacy_recreate` 隔离环境协议，再比 `legacy_recreate` vs `paper_recreate`；三项单独 variant 用于归因。

## 容易误导的点

- MetaWorld task 名相同不代表 task instance 相同；`rand_vec` 会改变目标/物体初始配置，但不改变奖励公式和动力学定义。
- 训练 success 来自随机采样策略，论文曲线应与确定性 eval 对齐，两者不可混用。
- empirical entropy 是连续分布的微分熵，可能为负；应同时看熵修正和最终 denominator，不能把“更小/更尖锐”直接等同于更好。
- critic 输出处于归一化价值尺度，未反变换就 bootstrap 会产生单位错误。
- 第一个完整 episode 前 `Ḡ` 未建立，公开实现的 normalized reward 为 0；本实现仅对首次 bootstrap 的反变换使用有限的单位尺度回退。

## 验证

- CPU 单元测试：16 项通过，覆盖 preset、L1/L2、两类 bootstrap、论文熵公式、环境重建、per-task 熵及 checkpoint round-trip。
- Legacy preset 与修改前代码在固定输入上的单步模型/温度/RNG SHA-256 完全一致，确认关闭修正时数值路径未改变。
- MetaWorld 实测：相同 task 连续重建后的 `_last_rand_vec` 发生变化。
- 端到端短跑：paper preset 运行 202 步并在第 200 步完成 truncation/bootstrap；legacy preset 独立短跑通过。测试未占用正在运行的训练 GPU。
