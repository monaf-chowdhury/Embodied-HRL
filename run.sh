#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run.sh --offline_algo {bc|bc_iql|td3bc|awr|bet} [extra train.py args]

Examples:
  bash run.sh --offline_algo bc
  bash run.sh --offline_algo bc_iql
  bash run.sh --offline_algo td3bc --td3bc_alpha 2.0
  bash run.sh --offline_algo awr --awr_temperature 0.5
  bash run.sh --offline_algo bet --bet_num_bins 128
  bash run.sh --offline_algo bc_iql --rebuild_demo_cache
  bash run.sh --offline_algo bc_iql --chain_eval_episodes 100 --no_video

Notes:
  - Common default: DINOv2 + action chunk 4 + all four tasks.
  - Extra args are passed directly to train.py.
  - Rebuild cache only when encoder, action chunk, tasks, image size, or reward weights change.
EOF
}

OFFLINE_ALGO=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline_algo)
      OFFLINE_ALGO="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OFFLINE_ALGO" ]]; then
  echo "ERROR: --offline_algo is required."
  usage
  exit 1
fi

case "$OFFLINE_ALGO" in
  bc|bc_iql|iql|td3bc|td3_bc|awr|bet|behavior_transformer|sequence_bc)
    ;;
  *)
    echo "ERROR: unknown --offline_algo '$OFFLINE_ALGO'"
    usage
    exit 1
    ;;
esac

SAFE_ALGO="${OFFLINE_ALGO//-/_}"
SAFE_ALGO="${SAFE_ALGO// /_}"
LOG_DIR="logs/ablations/${SAFE_ALGO}_dinov2_chunk4"

COMMON_ARGS=(
  --offline_algo "$OFFLINE_ALGO"
  --seed 42
  --device cuda
  --encoder dinov2
  --action_chunk 4
  --hidden_dim 256
  --n_layers 3
  --bc_steps 30000
  --offline_rl_steps 100000
  --batch_size 256
  --single_task_eval_episodes 20
  --chain_eval_episodes 20
  --log_interval 500
  --log_dir "$LOG_DIR"
  --demo_datasets franka-complete franka-mixed franka-partial
  --tasks microwave kettle "light switch" "slide cabinet"
)

ALGO_ARGS=()
case "$OFFLINE_ALGO" in
  td3bc|td3_bc)
    ALGO_ARGS+=(--td3bc_alpha 2.5)
    ;;
  awr)
    ALGO_ARGS+=(--awr_temperature 1.0 --awr_max_weight 20)
    ;;
  bet|behavior_transformer|sequence_bc)
    ALGO_ARGS+=(--bet_steps 80000 --bet_num_bins 64 --bet_offset_weight 5.0)
    ;;
esac

echo "Running offline algorithm: $OFFLINE_ALGO"
echo "Log dir: $LOG_DIR"
python train.py "${COMMON_ARGS[@]}" "${ALGO_ARGS[@]}" "${EXTRA_ARGS[@]}"

