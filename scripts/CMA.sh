#!/usr/bin/env bash
# SAC on the CMA benchmark. Append options to override, e.g. bash scripts/CMA.sh --seed 42
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$repo_root/analog_sizing/train.py" --algo SAC --env_name CMA --seed 50 --name test \
    --lr 0.003 --gamma 0.8 --learning_starts 120 --alpha 0.0 --lookup_style tanh --tt_threshold -1.0 \
    --total_timesteps 24000 --eval_freq 24000 --num_eval 100 \
    --eval_spec_path ngspice_specs_gen_cma_gf180_tran_test_200 "$@"
