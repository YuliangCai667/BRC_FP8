#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
export CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 BRC_CUDA_ROOT=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=${BRC_GPU_UUID:-GPU-280b96e2-67fc-577e-ba63-1ccd9493c231}
# The installed XLA/cuBLAS combination can autotune an invalid large FP8
# algorithm on SM120. This setting passed the matched legacy/hybrid checks.
export XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_ENTITY=cai200661-sun-yat WANDB_PROJECT='FP8 RL'
unset LD_LIBRARY_PATH
run_id=${BRC_RUN_ID:-actor_hybrid_qat_fp8_export_s42_$(date +%Y%m%d_%H%M%S)}
run_root=${BRC_RUN_ROOT:-/home/caiyuliang/brc_v1_runs}
# Additional arguments support recovery with --resume_from; the formal protocol
# is fixed below. No run or GPU process is stopped by this launcher.
exec /home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
 --env_names=DMC_DOGS --seed=42 --max_steps=500000 \
 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 \
 --critic_precision=fp8_resident --fp8_resident_carry=true --critic_optimizer_state=fp8_carry \
 --fp8_resident_canonicalization=false --fp8_all_dense_kernels=false --fp8_input_dense_kernel=false --fp8_output_dense_kernel=false \
 --target_critic_precision=fp8_lag --fp8_amax_history_length=1024 \
 --critic_residual_compute_format=hybrid --critic_residual_compute_terms=main_plus_carry \
 --critic_residual_compute_rounding=rtn --critic_residual_compute_rht=false \
 --actor_training_recipe=carry_body_bf16_edge_qat --actor_body_compute=hybrid_main_plus_carry \
 --actor_edge_storage=bf16 --actor_weight_qat=true --actor_export_codec=e4m3fn_block32_fp32scale_rtn_v1 \
 --actor_export_align_start=450000 --actor_export_on_finish=true \
 --paper_alignment=true --return_bootstrap=reward_mean \
 --eval_seed_offset=0 --eval_episodes=10 --eval_interval=5000 --offline_evaluation=true --render=false \
 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 \
 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true \
 --log_to_wandb=true --wandb_name="$run_id" --run_root="$run_root" --run_id="$run_id" \
 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 \
 --profile_interval=25000 --profile_window=10 "$@"
