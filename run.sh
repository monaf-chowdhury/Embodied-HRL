#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run.sh --offline_algo {flow_bc|qc_fql} [extra train.py args]

Algorithms:
  flow_bc  Flow-matching behavior cloning only (no critic). Staged experiment 1:
           "does an expressive policy class alone beat the old Gaussian BC?"
  qc_fql   Q-chunking + Flow Q-Learning: chunked twin critic + flow BC +
           one-step Q-maximizing actor. Staged experiment 2.

Examples:
  bash run.sh --offline_algo flow_bc --demo_datasets franka-partial
  bash run.sh --offline_algo qc_fql  --demo_datasets franka-partial
  bash run.sh --offline_algo qc_fql  --fql_alpha 3.0 --best_of_n 4
  bash run.sh --offline_algo qc_fql  --online_finetune --online_steps 200000
  bash run.sh --offline_algo qc_fql  --rebuild_demo_cache
  bash run.sh --offline_algo qc_fql  --chain_eval_episodes 100 --no_video

Notes:
  - Defaults: DINOv3 + action chunk 4 + all four tasks.
  - Extra args are passed straight through to train.py.
  - Rebuild the cache (--rebuild_demo_cache) only when the encoder, action chunk,
    tasks, image size, or reward weights (progress/completion/action_cost/sigma)
    change. The offline algorithm choice alone never requires a rebuild.
  - Validate on partial/mixed data: on near-expert "complete" data QC-FQL ~ flow
    BC ~ BC, so the headroom only shows up where the data is suboptimal.
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
  flow_bc|qc_fql)
    ;;
  *)
    echo "ERROR: unknown --offline_algo '$OFFLINE_ALGO' (use flow_bc or qc_fql)"
    usage
    exit 1
    ;;
esac

LOG_DIR="logs/${OFFLINE_ALGO}_dinov3_chunk4"

COMMON_ARGS=(
  --offline_algo "$OFFLINE_ALGO"
  --seed 42
  --device cuda
  --encoder dinov3
  --action_chunk 4
  --hidden_dim 256
  --n_layers 3
  --flow_bc_steps 30000
  --offline_rl_steps 100000
  --batch_size 256
  --single_task_eval_episodes 100
  --chain_eval_episodes 100
  --log_interval 500
  --log_dir "$LOG_DIR"
  --demo_datasets franka-complete franka-mixed franka-partial
  --tasks microwave kettle "light switch" "slide cabinet"
)

echo "Running offline algorithm: $OFFLINE_ALGO"
echo "Log dir: $LOG_DIR"
python train.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
