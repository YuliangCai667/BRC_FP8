# MetaWorld reset 改为原位重采样

日期：2026-08-22
分支：`codex/paper-alignment`

## 动机

上一版通过每个 episode 重建 MetaWorld env 来绕过 goal-observable constructor 的固定 `rand_vec`。该方法符合 BRO 的生命周期行为，但 50 任务每轮比普通 reset 多约 2.45 秒。核查官方 TD-MPC2 后确认其复用 env，并设置 `_freeze_rand_vec=False` 使每次 reset 重采样；当前安装的 MetaWorld 也支持该行为。

## 修改

- `--metaworld_reset_mode` 从 `{frozen,recreate}` 改为 `{frozen,resample}`。
- `resample` 只构造一次 MetaWorld env，初始化时解除 `_freeze_rand_vec`；每次 episode reset 使用独立、可复现的 seed 流，不再 close 或重新执行 constructor。
- 非 MetaWorld 环境及 `frozen` 路径保持原行为；若未来 MetaWorld 不再提供该私有字段，启动时明确报兼容性错误。
- 消融脚本 variant 更新为 `legacy_resample` 和 `paper_resample`。旧 `recreate` 命令被拒绝，避免名称与实际行为不一致。

## 评估与易错点

- 从 `config.yaml` 确认 `metaworld_reset_mode=resample`；比较环境协议时使用 `legacy_frozen` vs `legacy_resample`，比较算法时使用 `legacy_resample` vs `paper_resample`。
- `resample` 保持任务/奖励/动力学定义不变，只改变 episode 的物体和目标配置；训练曲线不能与旧 frozen 运行当作同一数据分布直接拼接。
- MetaWorld 官方 MT50 固定位置协议与 TD-MPC2/BRO 使用的 goal-observable 任务集合并非完全相同。本修改跟随论文比较代码的跨 episode 随机化语义，不宣称适用于所有 MT50 实验。
- `_freeze_rand_vec` 是 MetaWorld 私有接口，升级依赖后必须重新运行兼容性测试。
- 旧 `recreate` checkpoint 保留原配置，应使用提交 `56be7d2` 恢复；不要以 `resample` 名义续跑后拼接曲线。

## 验证

- CPU 单元测试 17 项通过；覆盖 resample 复用对象、解除冻结、seed 更新及拒绝旧模式。
- 真实 MetaWorld：连续 reset 的 `_last_rand_vec` 不同，同一顶层 seed 的配置序列完全复现，env 对象 ID 不变。
- 当前机器 50 任务 reset 中位数约 0.676 秒；上一版重建约 3.13 秒，周期额外开销已消除。
- paper preset + resample 完成 202 步 CPU 短跑，并正常经过第 200 步 truncation/critic bootstrap；未占用训练 GPU。
