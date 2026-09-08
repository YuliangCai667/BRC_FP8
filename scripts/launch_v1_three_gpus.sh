#!/usr/bin/env bash
# Requires the specified smoke reports to have passed; each launcher exits into training.
set -euo pipefail
audit=/home/caiyuliang/brc_v1_audit
socket=$audit/tmux.sock
shared_arg=
if [[ ${BRC_ALLOW_SHARED_GPU:-false} == true ]]; then shared_arg=--allow-shared; fi
formats=(${BRC_LAUNCH_FORMATS:-mxfp8 nvfp4 mxfp4})
for format in "${formats[@]}"; do
  case "$format" in
    mxfp8) gpu=0; root=/home/caiyuliang/BRC_FP8_v1_mxfp8;;
    nvfp4) gpu=1; root=/home/caiyuliang/BRC_FP8_v1_nvfp4_run;;
    mxfp4) gpu=2; root=/home/caiyuliang/BRC_FP8_v1_mxfp4_run;;
    *) exit 2;;
  esac
  run_id=brc_v1_${format}_twoterm_s42_500k_$(date +%Y%m%d_%H%M%S)
  smoke=$audit/${format}_smoke_passed.json
  test -f "$smoke"
  session=brc_v1_${format}_gpu${gpu}
  tmux -S "$socket" new-session -d -s "$session" -c "$root" \
    "/home/caiyuliang/anaconda3/envs/brc/bin/python scripts/start_v1_when_idle.py --gpu=$gpu --format=$format --run-id=$run_id --smoke-report=$smoke --status=$audit/${format}_launch.json $shared_arg > $audit/${format}_formal.log 2>&1"
done
