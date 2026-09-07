#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 GPU_ID SEED [MAX_STEPS] [RESUME_FROM]" >&2
  exit 2
fi

export CUDA_ROOT=/usr/local/cuda-12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH

gpu_id=$1
seed=$2
max_steps=${3:-500000}
resume_from=${4:-}
log_to_wandb=${BRC_LOG_TO_WANDB:-true}
target_critic_precision=${BRC_TARGET_CRITIC_PRECISION:-fp8_direct}
optimizer_state=${BRC_CRITIC_OPTIMIZER_STATE:-fp32}
run_tag=${BRC_RUN_TAG:-}
tag_segment=
if [[ "$target_critic_precision" != "fp8_direct" && "$target_critic_precision" != "fp8_lag" ]]; then
  echo "BRC_TARGET_CRITIC_PRECISION must be fp8_direct or fp8_lag" >&2
  exit 2
fi
if [[ -n "$run_tag" ]]; then
  if [[ ! "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "BRC_RUN_TAG must contain only letters, digits, underscores, or hyphens" >&2
    exit 2
  fi
  tag_segment="_${run_tag}"
fi
resume_args=()
if [[ -n "$resume_from" ]]; then
  resume_args+=(--resume_from="$resume_from")
fi

if [[ -z "$resume_from" && "$max_steps" -le 5001 ]]; then
  run_id="brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_smoke${tag_segment}_s${seed}-$(date +%Y%m%d-%H%M%S)"
  tensor_stats_interval=${BRC_TENSOR_STATS_INTERVAL:-5000}
  analysis_checkpoint_interval=${BRC_ANALYSIS_CHECKPOINT_INTERVAL:-5000}
  recovery_checkpoint_interval=${BRC_RECOVERY_CHECKPOINT_INTERVAL:-5000}
else
  run_id="brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16${tag_segment}_s${seed}"
  tensor_stats_interval=${BRC_TENSOR_STATS_INTERVAL:-25000}
  analysis_checkpoint_interval=${BRC_ANALYSIS_CHECKPOINT_INTERVAL:-50000}
  recovery_checkpoint_interval=${BRC_RECOVERY_CHECKPOINT_INTERVAL:-100000}
fi

env -u LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES="$gpu_id" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/caiyuliang/anaconda3/envs/brc/bin/python train.py \
  --env_names=DMC_DOGS \
  --seed="$seed" \
  --eval_seed_offset=0 \
  --max_steps="$max_steps" \
  --start_training=5000 \
  --replay_buffer_size=1000000 \
  --batch_size=1024 \
  --updates_per_step=2 \
  --width_critic=4096 \
  --critic_precision=fp8_resident \
  --fp8_resident_carry=true \
  --critic_optimizer_state="$optimizer_state" \
  --fp8_resident_canonicalization=false \
  --target_critic_precision="$target_critic_precision" \
  --fp8_amax_history_length=1024 \
  --paper_alignment=true \
  --return_bootstrap=reward_mean \
  --eval_interval=25000 \
  --eval_episodes=10 \
  --offline_evaluation=true \
  --render=false \
  --log_to_wandb="$log_to_wandb" \
  --wandb_name="$run_id" \
  --run_root=runs \
  --run_id="$run_id" \
  --metrics_interval=1000 \
  --metrics_flush_interval=1000 \
  --system_metrics_interval_sec=10 \
  --profile_interval=25000 \
  --profile_window=10 \
  --tensor_stats_interval="$tensor_stats_interval" \
  --analysis_checkpoint_interval="$analysis_checkpoint_interval" \
  --recovery_checkpoint_interval="$recovery_checkpoint_interval" \
  --keep_last_analysis_checkpoints=2 \
  --keep_last_recovery_checkpoints=1 \
  --save_replay_buffer=true \
  "${resume_args[@]}"
