# Actor CARRY / QAT / FP8 deployment — 2026-09-09

本分支 `codex/actor-qat-fp8-export-v1` 从实际 V1 提交
`1dd21cb8b420f32c3c3bc33d8a0cd8f11a86ce6d` 继承，并导入本地已经验证的
MXFP8 `ceil_amax_over_448_no_overflow` 修正。基线工作树保留原有未提交状态。
直接用户要求指定 GPU1，覆盖附件的空闲卡建议；使用 UUID
`GPU-280b96e2-67fc-577e-ba63-1ccd9493c231` 的剩余资源，不停止其他进程。
按照用户此前要求，不新增源码或 checkpoint 内容哈希；记录已有 Git commit
和原生后端缓存标识。

Actor 保持真实基线的 width=256、depth=1：两个残差 Dense 使用 E4M3
main + carry / 16，直接复用 V1 两项 MXFP8 Fprop/Dgrad 和单次逻辑 Wgrad。
输入投影和 mean/log_std 两个头以 BF16 kernel 常驻，从第一次前向执行
FP8 weight QAT。所有参数先构造临时 FP32 逻辑 tree，AdamW 更新和 weight
decay 作用于未假量化的逻辑权重，随后将边界 candidate 写回 BF16、主体写回
原 CARRY codec。Actor AdamW 的 FP32 moments、lr=3e-4、betas=.9/.999、
eps=1e-8、weight_decay=1e-4 保持基线，没有常驻 FP32 kernel master。

`BRC.set_env_step` 在 rollout 前及 learner update 入口统一选择 phase。
从 env_step=450000 起所有 actor Dense 使用相同的单码权重 codec 和 W8A16：
输入/解码权重 BF16，累加、输出与 bias FP32。主阶段 actor 的 MXFP8 activation
打包此时退出；CARRY 状态和 Adam 不清空，critic/target/temperature 继续更新。
Checkpoint 保存 phase、环境步、阈值；450k recovery 单独保留。中断尝试保存
有限状态恢复点，非有限状态另存排查并保留此前有效恢复点。

共享 codec `e4m3fn_block32_fp32scale_rtn_v1` 按 [K,N] 的 K 维 block32，
每块每列 FP32 scale=amax/448；零块 scale=1，仅 K 补零。它不是 E8M0
MXFP8 存储。显式 E4M3 ties-to-even 位编码避免当前 XLA CPU FP32→E4M3
转换的二次舍入（例如 -151.99255）；FP8/BF16 边界带 optimization barrier。
非法 kernel/scale 在 eager 和 JIT 都报错。GPU FP32 scale 除法与 NumPy
可相差 1 ULP；测试单独检查这一点，不改变部署路径一致性的固定阈值。

独立 `deployment.load_actor_export` 仅依赖导出目录及 JAX/Flax/Distrax。
所有 Dense 恰好一份 FP8 code + scale，NPZ 禁用 pickle，码以 uint8 位模式
保存。Bias、LayerNorm、最终 critic 中的任务 embedding 小表保持原精度。
Manifest 记录 task ID、L1 embedding 归一化、flatten 顺序、右侧零 padding、
action mask/rescaling、log_std 变换等。实际基线没有 observation normalizer，
因此明确记录 identity；reward normalizer 不属于策略输入预处理。
Loader 不运行 quantizer、不常驻解码后的 BF16 kernel cache，支持确定性动作、
随机动作及注入相同 Gaussian noise。确定性动作直接使用 tanh(mean)。

训练完成后从同一个完整 recovery checkpoint 导出，独立进程验证，再用两套
同 seed 新环境进行训练 actor/部署 actor 配对评估。正式协议为每任务 10 episodes。
`training_complete`、`export_validated`、`deployment_eval_complete` 分别记录，
导出失败不标记完成。部署相关脚本与 runtime 一同纳入启动时源码快照，最终
验证/评估使用该快照，避免长时间训练期间工作树编辑改变导出流程。

验证证据保存在本机：

- `/home/caiyuliang/brc_v1_audit/actor_qat_cpu_tests.log`：9 项 actor codec/storage/phase/failure 测试及 3 项原 checkpoint 测试通过。
- `/home/caiyuliang/brc_v1_audit/actor_qat_gpu_codec_tests.log`：GPU1 的 5 项 codec/STE 测试通过；RTN midpoint、padding、非法输入与 FP32 物理梯度均覆盖。
- `/home/caiyuliang/brc_v1_audit/actor_qat_recorder_tests.log`：4 项 recorder 回归通过；测试主动模拟 W&B 离线错误。
- 既有 two-term codec/VJP CPU 测试通过；原生 GPU 全宽计算由下述 smoke 覆盖。
- `/home/caiyuliang/brc_v1_audit/actor_qat_export_smoke_20260909_v2/final_report.json`：真实四任务环境数据、batch1024、critic4096；主阶段/对齐阶段各 16 次更新，loss、状态和梯度有限。FP8/BF16 恢复逐值一致，物理状态下一步最大差异分别 0 与 1.14e-12，沿用 V1 rtol=1e-6/atol=1e-7。首次 smoke 因过严的 FP32 逐值断言停止，保留在 v1 目录。
- 同目录 `supplement.json`：两阶段保存状态上的 Q-only actor 梯度均非零，包括输入、主体及 mean/log_std 头；确定性采样规则验证通过。
- 四任务真实观测 batch64、batch1：mean、raw log_std、变换后 log_std/std、确定性/随机动作、pre-tanh 与 log-prob 全部逐值一致。固定验证阈值仍为 rtol=1e-5、atol=1e-6（log-prob atol=1e-4），没有放宽。
- `actor_smoke_package/deployment_eval.json`：每任务 1 episode 的工程配对检查，四任务 return 差均为 0。这不是 500k 正式结果，不据此评价学习效果。

Smoke 模型共有 215808 个 Dense 权重：FP32 参考 863232 B；FP8 code
216064 B（含 padding 256 B），scale 27008 B，辅助参数 10032 B。
参数总计 253104 B，weights.npz 实际 259396 B；Dense 存储压缩 3.5513 倍。
这不是整个训练显存或整个部署目录的压缩比。Smoke learner 的 JAX 活跃峰值
5721703936 B，pool 峰值 6444548096 B，二者分别记录。部署 batch1 同步测量
受共享 GPU 与 Python 调度影响，两个测量的中位数约 1.12 ms / 4.64 ms；
不宣称速度提升。完整原始统计含编译时间、数组清单及运行内存。

正式配置见 `configs/actor_qat_fp8_export_v1.yaml`，沿用 V1 的四任务 seed42、
500k env steps、5000 warmup、batch1024、每步 2 updates、critic4096、
FP8 CARRY Adam、LAG target、reward_mean normalizer、每 5k 评估 10 episodes。
只启动一条新的正式训练；不因中途 return 波动改阈值或学习率。

启动命令（绝对 Python 路径在脚本中固定）：

```bash
cd /home/caiyuliang/BRC_FP8_actor_qat_export
bash scripts/run_actor_qat_fp8_export.sh
```

正式 run 使用独立 tmux 会话，GPU/PID/W&B/run 路径记录在
`/home/caiyuliang/brc_v1_audit/actor_qat_fp8_export_launch.json`。
预期最终目录为 `<run_dir>/export/actor_fp8_500k/`，运行期状态见
`<run_dir>/actor_export_status.json`（完成训练后开始创建）。完整 500k 学习曲线、
450k/500k 与最后三次评估、最终部署包和每任务 10 episodes 配对评估属于
尚需正式运行产生的结果，不能用 smoke 代替。

正式启动核验：2026-09-09 16:44:53（Asia/Shanghai）启动，PID 26127，
tmux session `brc_actor_qat_fp8_gpu1_20260909_164453`，socket
`/home/caiyuliang/brc_v1_audit/tmux.sock`。运行目录为
`/home/caiyuliang/brc_v1_runs/DMC_DOGS/actor_qat_fp8_export_s42_gpu1_20260909_164453`。
执行源码提交为 `bc2e80dabfe0fe2b0f27e0540cce1b36b7015ff9`，启动快照已核对。
[W&B 正式 run](https://wandb.ai/cai200661-sun-yat/FP8%20RL/runs/actor-qat-fp8-s42-gpu1-20260909_164453)
已在线同步。核验时 env_step=5000、update_step=3，actor loss=26.9681、
critic loss=23.2745，均为有限值；处于 carry_main。500k 正式训练、
最终导出及其部署评估仍在等待运行完成。此追加仅记录启动事实，不改变训练源码。
