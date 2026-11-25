#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src:$project_dir/third_party/rsl_rl:${PYTHONPATH:-}"
assets="$project_dir/../output/contact_tracker"
examples="$project_dir/../output/contact_examples"
runs="${ELF3_RUNTIME_ROOT:-$HOME/humanoid_datasheet/datasets/elf3_climbing/runtime_v1}/runs"
out="${ELF3_CONTACT_OUTPUT:-$runs/contact_070_$(date +%Y%m%d_%H%M%S)}"
weights="${ELF3_CONTACT_WEIGHTS:-$assets/student_acp_gait40_transfer.pt}"
# Keep the completion check aligned with argparse when the caller overrides --output.
arguments=("$@")
for ((i = 0; i < ${#arguments[@]}; i++)); do
  case "${arguments[i]}" in
    --output) out="${arguments[i+1]:?--output needs a directory}" ;;
    --output=*) out="${arguments[i]#--output=}" ;;
  esac
done
printf '训练输出 %s\n' "$out"
conda run --no-capture-output -n isaac_sim_env python -u -m elf3_climbing.contact_runner \
  --headless --device cuda:0 --weights "$weights" --num-envs 1024 \
  --knee-diagnostic "$examples/knee2300_success_v1.npz" \
  --examples "$examples/ground_walk_v1.npz" "$examples/gait_student_verified_far_v1.npz" \
    "$examples/gait_anchor_walking_starts_v1.npz" "$examples/ground_brace_v1.npz" \
    "$examples/bridge_focused_c/isaac_ground_bridge_loaded_knee.npz" \
    "$examples/bridge_swing_d/isaac_ground_bridge_loaded_knee.npz" \
    "$examples/knee2300_success_v1.npz" \
    "$examples/knee2300_varied_success_v1.npz" \
    "$examples/bridge_bilateral_f/isaac_ground_bridge_knee.npz" \
    "$examples/bridge_verified_g/isaac_ground_bridge_knee.npz" \
    "$examples/bridge_dagger_points_h/dagger_full_bridge.npz" \
  --iterations 3000 --rollout-steps 32 --eval-interval 100 --learning-rate 0.000008 --noise 0.025 \
  --knee-hold-seconds 0.50 --demo-updates 8 \
  --output "$out" "$@"
test -f "$out/completion.json" && test ! -f "$out/error.json"
