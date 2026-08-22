#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID SEED {legacy_frozen|legacy_recreate|l1|bootstrap|entropy|paper_recreate}" >&2
  exit 2
fi

gpu_id=$1
seed=$2
variant=$3
reset_mode=recreate
alignment_flags=(--paper_alignment=false)

case "$variant" in
  legacy_frozen)
    reset_mode=frozen
    ;;
  legacy_recreate)
    ;;
  l1)
    alignment_flags+=(--task_embedding_norm=l1)
    ;;
  bootstrap)
    alignment_flags+=(--return_bootstrap=critic)
    ;;
  entropy)
    alignment_flags+=(--entropy_correction=empirical_per_task)
    ;;
  paper_recreate)
    alignment_flags=(--paper_alignment=true)
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

run_id="${variant}-seed${seed}-$(date +%Y%m%d-%H%M%S)"

CUDA_VISIBLE_DEVICES="$gpu_id" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python3 train.py \
  --env_names=METAWORLD_ALL \
  --seed="$seed" \
  --max_steps=500000 \
  --start_training=5000 \
  --replay_buffer_size=1000000 \
  --batch_size=1024 \
  --updates_per_step=2 \
  --width_critic=4096 \
  --eval_interval=25000 \
  --eval_episodes=10 \
  --offline_evaluation=true \
  --render=false \
  --log_to_wandb=true \
  --run_id="$run_id" \
  --wandb_name="$run_id" \
  --metaworld_reset_mode="$reset_mode" \
  "${alignment_flags[@]}"
