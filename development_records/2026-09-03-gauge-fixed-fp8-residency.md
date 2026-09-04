# Gauge-Fixed FP8 Residency

> Historical moving-anchor implementation record. This formulation failed its
> long-horizon invariant and is not evidence about the later fixed-initial-anchor
> canonicalization; see
> [2026-09-03-fixed-anchor-fp8-residency.md](2026-09-03-fixed-anchor-fp8-residency.md).

## Motivation

两个 500k Dogs mechanism run 显示，naive online E4M3 resident kernel 的
物理 norm 增长 `21.7x/23.2x`，intended effective angular step 同期下降
`19.9x/33.4x`；code 方向和 norm 基本不变而 FP32 scale 持续增长。由于四个
resident Dense 后都紧接 LayerNorm，本轮直接固定这一冗余的尺度自由度，检验
它是否是 return gap 的可干预原因。

## Changes

- 新增 opt-in `--fp8_resident_canonicalization`，且只允许与 online
  `critic_precision=fp8_resident` 配合；默认关闭时保留原 naive write。
- 每次 AdamW candidate 仍只做一次 current-amax E4M3 quantization。对每个
  kernel/ensemble member 计算
  `alpha = ||W_old||_F / max(||dequant(code, raw_scale)||_F, 1e-12)`，保留
  code，将 stored scale 写为 `raw_scale * alpha`，并将配对 FP32 bias 写为
  `candidate_bias * alpha`。
- AdamW moments 不缩放，不新增 FP32 master、reference norm 或 full-size
  residual；previous physical norm 通过递推把半径保持在初始化附近。
- quantizer-only 诊断继续使用 raw dequantized candidate；actual geometry、
  scale/code decomposition 和 function-write 诊断使用最终 persisted state。
  新增 factor、canonicalization 前后 norm、post/old ratio 和 correction L2。
- resume 配置记录并校验该开关；新增 matched Dogs 运行脚本
  `scripts/run_dogs_online_fp8_resident_canonical.sh`。

## Evaluation

正式实验以 `post_to_old_kernel_norm_ratio`、stored scale、intended/actual
angular step、expected-Q MAE、critic-loss relative error 和 eval return 为主。
seed42 与既有 matched FP8-direct/FP32-weight control、naive resident 比较；
seed1 提供第二个 intervention seed，但目前没有 matched FP32-weight control。

## Pitfalls

- `weight_relative_error`、`quantization_error_radial_cosine` 和
  `dynamic_vs_fixed_relative_difference` 刻意只描述 raw quantizer，不能把它们
  解释为 canonicalized function 的总扰动。
- bias 必须与同一 member 的 kernel 使用同一个 `alpha`，否则 LayerNorm 前的
  affine function 不再只是整体缩放。
- 当前运行来自 dirty `5500f6a` 工作树；不能用未来整理出的 commit 替代启动
  provenance，也不能从共享 GPU 条件推断吞吐或能耗。

## Validation and limits

- `git diff --check` 与新 shell 脚本语法检查通过。
- 最终精简实现的 5 个聚焦 CPU 测试通过（39.521s）：helper 代数、真实
  resident update norm、off-path 等价、canonical checkpoint next-step、resume
  配置策略。
- 强制 `JAX_PLATFORMS=cuda` 的 GPU 2 canonical resident update smoke 通过
  （22.283s），覆盖 CUDA JIT 与实际 update 路径。
- 按用户要求未运行完整测试集、10k smoke 或 Phase C 50k 预跑；长程机制和
  return 结论留给 500k 正式实验。
