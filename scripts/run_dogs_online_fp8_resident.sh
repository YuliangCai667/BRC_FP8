#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID SEED" >&2
  exit 2
fi

export CUDA_ROOT=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

gpu_id=$1
seed=$2
run_id="brc_dmc_dogs_online_fp8_resident_s${seed}-$(date +%Y%m%d-%H%M%S)"

env -u LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES="$gpu_id" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
  --env_names=DMC_DOGS \
  --seed="$seed" \
  --eval_seed_offset=0 \
  --max_steps=500000 \
  --start_training=5000 \
  --replay_buffer_size=1000000 \
  --batch_size=1024 \
  --updates_per_step=2 \
  --width_critic=4096 \
  --critic_precision=fp8_resident \
  --target_critic_precision=fp8_direct \
  --fp8_amax_history_length=1024 \
  --paper_alignment=true \
  --return_bootstrap=reward_mean \
  --eval_interval=25000 \
  --eval_episodes=10 \
  --offline_evaluation=true \
  --render=false \
  --log_to_wandb=true \
  --wandb_name="$run_id" \
  --run_root=runs \
  --run_id="$run_id" \
  --metrics_interval=1000 \
  --metrics_flush_interval=1000 \
  --system_metrics_interval_sec=10 \
  --profile_interval=25000 \
  --profile_window=10 \
  --tensor_stats_interval=25000 \
  --analysis_checkpoint_interval=50000 \
  --recovery_checkpoint_interval=100000 \
  --keep_last_analysis_checkpoints=2 \
  --keep_last_recovery_checkpoints=1 \
  --save_replay_buffer=true
