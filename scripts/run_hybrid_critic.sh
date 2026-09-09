#!/usr/bin/env bash
# Reproduce the FP32-actor baseline or the isolated true-CARRY critic variant.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
compute=${1:-legacy}
if (($#)); then shift; fi
case "$compute" in
  legacy) terms=main_only ;;
  hybrid) terms=main_plus_carry ;;
  *) echo 'Expected legacy or hybrid' >&2; exit 2 ;;
esac
export BRC_RUN_ID=${BRC_RUN_ID:-critic_${compute}_carry_s42_$(date +%Y%m%d_%H%M%S)}
exec bash "$root/scripts/run_actor_qat_fp8_export.sh" \
  --actor_training_recipe=fp32 --actor_export_on_finish=false --actor_weight_qat=false \
  --critic_residual_compute_format="$compute" --critic_residual_compute_terms="$terms" "$@"
