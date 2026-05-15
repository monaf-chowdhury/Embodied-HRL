# Embodied-HRL: Lean Skill Learning Branch

This branch intentionally drops the learned hierarchy and teacher/student
curriculum. The current research path is simpler:

```text
FrankaKitchen demos -> replay labels -> rendered images -> per-task BC/IQL skills -> scripted chaining eval
```

## Active Files

```text
config.py        Dataclass config
demo_dataset.py  D4RL/Minari loading, replay labels, render/encode cache
encoder.py       Frozen R3M or DINOv2 image encoder
env_wrapper.py   FrankaKitchen image/state wrapper
networks.py      Minimal MLP helper
specialist.py    One visual BC/IQL policy per task
train.py         Main training/evaluation entrypoint
plots.py         TensorBoard-to-PNG diagnostics
utils.py         task indices, goals, eval helpers, video saving
```

## What Is Trained

For each configured task, the code trains a separate visual policy.

Input:

```text
image latent z + normalized proprio + task goal/current/delta/mask
```

Output:

```text
9-D Franka action, or H x 9 if action_chunk > 1
```

Training:

```text
1. BC on replay-labelled demo segments
2. Optional IQL on the same demo transitions
3. Deterministic single-task evaluation
4. Deterministic scripted-chain evaluation
```

There is no learned manager in this branch.

## Main Command

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder r3m \
  --bc_steps 30000 \
  --iql_steps 100000 \
  --batch_size 256 \
  --single_task_eval_episodes 20 \
  --chain_eval_episodes 15 \
  --log_dir logs/lean_skills_all_four \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --tasks microwave kettle "light switch" "slide cabinet"
```

## Prefix-State Diagnostic

```bash
python train.py \
  --seed 42 \
  --device cuda \
  --encoder r3m \
  --bc_steps 30000 \
  --iql_steps 100000 \
  --prefix_eval_only \
  --prefix_target_task "light switch" \
  --prefix_condition_tasks microwave kettle \
  --prefix_eval_states 20 \
  --log_dir logs/prefix_light_after_mw_kettle \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --tasks microwave kettle "light switch" "slide cabinet"
```

## What To Watch

Primary metrics:

```text
single_task/<task>_success_rate
eval/full_task_success_rate
eval/mean_tasks_completed
prefix/success_rate
```

If microwave/kettle work but light switch/cabinet fail, the bottleneck is
skill competence, not hierarchy.

## Research Direction

First make each individual skill competent from images. Then test scripted
composition. Only after that should online fine-tuning or a learned manager be
added back.
