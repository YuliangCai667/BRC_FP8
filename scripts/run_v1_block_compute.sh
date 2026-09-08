#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then echo 'usage: GPU_INDEX FORMAT SEED MAX_ENV_STEPS' >&2; exit 2; fi
gpu_index=$1
format=$2
seed=$3
max_steps=$4
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
python_path=/home/caiyuliang/anaconda3/envs/brc/bin/python
export CUDA_ROOT=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 BRC_CUDA_ROOT=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$(nvidia-smi -i "$gpu_index" --query-gpu=uuid --format=csv,noheader)
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export WANDB_MODE=offline MUJOCO_GL=egl PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
run_id=${BRC_RUN_ID:-brc_v1_${format}_twoterm_s${seed}_${max_steps}_$(date +%Y%m%d_%H%M%S)}
run_root=${BRC_RUN_ROOT:-/home/caiyuliang/brc_v1_runs}
exec "$python_path" train.py \
 --env_names=DMC_DOGS --seed="$seed" --max_steps="$max_steps" \
 --start_training=5000 --replay_buffer_size=1000000 --batch_size=1024 --updates_per_step=2 --width_critic=4096 \
 --critic_precision=fp8_resident --fp8_resident_carry=true --critic_optimizer_state=fp8_carry \
 --fp8_resident_canonicalization=false --fp8_all_dense_kernels=false --fp8_input_dense_kernel=false --fp8_output_dense_kernel=false \
 --target_critic_precision=fp8_lag --fp8_amax_history_length=1024 \
 --critic_residual_compute_format="$format" --critic_residual_compute_terms=main_plus_carry \
 --critic_residual_compute_rounding=rtn --critic_residual_compute_rht=false \
 --paper_alignment=true --return_bootstrap=reward_mean \
 --eval_seed_offset=0 --eval_episodes=10 --eval_interval=5000 --offline_evaluation=true --render=false \
 --tensor_stats_interval=25000 --analysis_checkpoint_interval=50000 --recovery_checkpoint_interval=100000 \
 --keep_last_analysis_checkpoints=2 --keep_last_recovery_checkpoints=1 --save_replay_buffer=true \
 --log_to_wandb=false --wandb_name="$run_id" --run_root="$run_root" --run_id="$run_id" \
 --metrics_interval=1000 --metrics_flush_interval=1000 --system_metrics_interval_sec=10 --profile_interval=25000 --profile_window=10
