#!/usr/bin/env bash
# SAC on the COMP benchmark. Append options to override, e.g. bash scripts/COMP.sh --seed 42
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$repo_root/analog_sizing/train.py" --algo SAC --env_name COMP --seed 50 --name test \
    --lr 0.003 --gamma 0.8 --learning_starts 120 --alpha 0.0 --lookup_style tanh --tt_threshold -1.0 \
    --total_timesteps 8000 --eval_freq 8000 --num_eval 100 \
    --eval_spec_path ngspice_specs_gen_comp_gf180_test_100 "$@"
