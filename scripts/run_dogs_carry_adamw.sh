#!/usr/bin/env bash
# Frozen blue-line protocol: residual CARRY weights + lag target, moments only.
set -euo pipefail
export BRC_TARGET_CRITIC_PRECISION=fp8_lag
export BRC_FP8_ALL_DENSE_KERNELS=false
export BRC_FP8_INPUT_DENSE_KERNEL=false
export BRC_FP8_OUTPUT_DENSE_KERNEL=false
export BRC_CRITIC_OPTIMIZER_STATE=${BRC_CRITIC_OPTIMIZER_STATE:-fp8_carry}
export BRC_RUN_TAG=${BRC_RUN_TAG:-target_lag_adam_${BRC_CRITIC_OPTIMIZER_STATE}_v1}
exec bash "$(dirname "$0")/run_dogs_online_fp8_resident_carry.sh" "$@"
