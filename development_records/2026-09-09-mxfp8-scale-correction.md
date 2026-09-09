# MXFP8 scale 取整修正（2026-09-09）

旧 `floor(log2(amax))-8` 会使部分块最大值超出 E4M3 范围，例如 240
在 scale=0.5 时被截成 224。现在直接读取 FP32 exponent，检查当前尺度是否
容纳 amax，必要时将 exponent 加一；有效范围内等价于
`ceil(log2(amax/448))`，保留零块和最小尺度处理。

尺度通过 FP32 位模式精确构造，避免 JIT 下 `exp2`/`log2` 近似影响边界；
`unpack` 使用同样的精确构造。block32、E4M3/E8M0、RTN、两项 CARRY 计算、
Adam block128/gain16 及 target LAG 路径保持原配置。

新增两项打包回归：已有 E4M3 码在两种 reduction 方向上逐元素保持；
块最大值与紧邻尺度边界值均被容纳，同时覆盖零块和不规则 padding。
CPU 上两项及 VJP 代数测试通过；GPU0 上两项及两项原生测试通过。
原生 GEMM 包含 1024×4096×4096 和 4096×1024×4096，量化操作数 oracle
相对 L2 最大误差为 4.03e-8。这些检查验证数值/实现，不证明长期 return 改善。

GPU0 的全宽 learner smoke 通过：16 次更新、状态及指标有限、dQ/da 范数
3.05219；保存恢复后的下一步逐状态最大误差为 0，RNG 完全一致。

训练配置新增 `resolved_online_compute_scale_policy`，使 W&B 能区分当前
工作树中的 scale 修正。按用户后续要求，取消额外的源码哈希记录与校验。
已有 W&B entity/project 和 shell 环境覆盖修改被保留。

根据用户指示，已停止 GPU0 的旧进程 3850557；旧 run 为
`brc_v1_mxfp8_twoterm_s42_500k_20260909_125308`，中断时 env_step=26030、
update_step=42063。原始日志、checkpoint 和 W&B run 保留。

本次验证与启动记录位于 `/home/caiyuliang/brc_v1_audit/mxfp8_scale_correction/`
及 `/home/caiyuliang/brc_v1_audit/mxfp8_scale_correction_launch.json`。
新版使用独立 run ID，从头训练 seed42 500k，不续训旧 checkpoint。
用户随后指定 GPU2；GPU0 的新版初始化进程已停止，复用已通过的数值与
16 次更新检查，在 GPU2 重新启动。
W&B 运行：[mxfp8-ceilscale-s42-gpu2-20260909_134232](https://wandb.ai/cai200661-sun-yat/FP8%20RL/runs/mxfp8-ceilscale-s42-gpu2-20260909_134232)。
