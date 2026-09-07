#!/usr/bin/env bash
set -euo pipefail

# Same Dogs/CARRY/LAG/moment protocol, with an independent run identity.
export BRC_CRITIC_OPTIMIZER_STATE=${BRC_CRITIC_OPTIMIZER_STATE:-fp8_carry}
export BRC_RUN_TAG=${BRC_RUN_TAG:-pure_training_adam_${BRC_CRITIC_OPTIMIZER_STATE}_v1}
exec bash "$(dirname "$0")/run_dogs_carry_adamw.sh" "$@"
