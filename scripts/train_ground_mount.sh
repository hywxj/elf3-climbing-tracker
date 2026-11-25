#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
runs="${ELF3_RUNTIME_ROOT:-$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1}/runs"
output="${ELF3_MOUNT_OUTPUT:-$runs/ground_mount_070_$(date +%Y%m%d_%H%M%S)}"

cd "$project_dir"
printf '训练输出 %s\n' "$output"
exec conda run --no-capture-output -n isaac_sim_env python scripts/train_mount.py train \
    --headless --device cuda:0 \
    --entry-checkpoint "$runs/knee_070_contact_transfer_20261122_204705/checkpoint_02300.pt" \
    --ground-trajectory "$runs/mount_070_ground_to_steady_rise_20261123/trajectory.npz" \
    --num-envs 1024 --iterations 3000 --rollout-steps 48 --eval-interval 100 \
    --learning-rate 0.00005 --record --output "$output" "$@"
