# V1：online residual 双项 MXFP8 计算（2026-09-08）

从 `codex/blackwell-fp8-direct` 的 `9904a05` 和本地未提交生产修正建立独立 worktree。
原工作区不变；完整 diff、新文件备份及 SHA256 位于仓库外
`/home/caiyuliang/brc_v1_audit/initial`。归因工作区只有独立续训适配器，本轮未迁入其调度器。

四个 residual Dense 使用 `s*(B(X,C)+B(X,R)/16)+b`；输入梯度为
`s*(B(G,C.T)+B(G,R.T)/16)`；逻辑权重梯度仅 `B(X.T,G)` 一次。
`two_term_dense.py` 的 VJP 返回一个 FP32 权重梯度，C/R/scale 不产生独立梯度。
Actor、input/head、FP8 CARRY Adam block128/gain16 和 LAG target 的数学路径保持基线；
新路径保留历史 delayed-scale metadata，不把零梯度覆盖为新 scale。

后端为 CUTLASS `147295a3d4b75f3aeff247c25b8927cea9a7006a`，
`MainloopSm120TmaWarpSpecializedBlockScaled`，E4M3/E8M0-32，FP32 输出及累加。
每个方向重新 RTN 打包，scale 按 reduction 分块并 swizzle；小 batch 内部补零后裁切。
使用 JAX FFI 接收设备 buffer、CUDA stream 与 XLA scratch；ensemble 通过 FFI sequential vmap。
构建缓存以 SM120、源码、构建配置、CUTLASS、CUDA/JAX hash 隔离。

验证：恒等量化 VJP 1 项通过；原生测试 2 项通过，覆盖小 batch、真实
1024×4096×4096 及 4096×1024×4096 shape，量化操作数 oracle relative L2 最大
`4.03e-8`。Nsight trace 确认一次完整 VJP 发射 5 次实际 SM120 block-scaled kernel。
48 项原 FP8/optimizer CPU 回归通过。
完整 learner smoke：16 updates、四任务各一 episode、非零有限 dQ/da、保存恢复下一步
逐状态完全一致（max error=0）。Adam residual moments 520 MiB，持久编码未改变；
GPU allocator peak 约 6.45 GiB，初始化及多种首次编译分别记录，不视作稳态。
证据位于 `/home/caiyuliang/brc_v1_audit/mxfp8_smoke_v2/report.json` 和相邻 trace/log。

原 shell 注入 CUDA12.4 LD_LIBRARY_PATH 导致 FP32 oracle 编译崩溃；已清理环境重验。
早期 cuBLASLt 探针未通过，生产仅使用经过执行验证的 CUTLASS。
三组统一独立 evaluation 环境，并在评估后恢复 learner/NumPy RNG。
正式入口固定 seed42、500k、UTD2、batch1024、width4096、5k eval，每条从头初始化；
NaN/Inf 会失败退出，不自动重启。正式运行不使用 smoke checkpoint。
返回值、长期稳定性、三组速度比较尚无结果，启动后不继续监控。
