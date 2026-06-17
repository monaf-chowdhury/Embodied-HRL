# Shared Skill-Conditioned QC-FQL

This branch pivots from isolated per-task specialists to **one shared,
task-conditioned action-chunk policy and critic**.

The research thesis is:

> Learn reusable skill-conditioned generative policies from shared offline data,
> then compose and repair them for long-horizon manipulation.

## Architecture

One shared model is used for every configured task:

- Frozen visual encoder: DINOv2, DINOv3, or R3M.
- Learned actor task-ID embedding.
- Learned critic task-ID embedding.
- One `FlowActor`: flow-matching behavior prior plus one-step FQL actor.
- One `TwinQ`: chunk critic over `Q(x_task, action_chunk)`.
- One target critic plus target critic task embedding.

The policy/critic conditioning vector is:

```text
x = [
  z_image,
  proprio_norm,
  task_embedding,
  task_goal_padded,
  task_current_padded,
  (task_goal_padded - task_current_padded) * task_mask,
  task_mask
]
```

Actions are chunks:

```text
action_chunk in R^(H * env_action_dim)
H = 4
env_action_dim = 9
```

## Data

The cache stores raw chunk transitions once:

```text
z_t, state_t, action_chunk, z_next, state_next, nstep,
env_done, reward_vec[n_tasks], task_done_vec[n_tasks],
task_complete_vec[n_tasks]
```

At sample time, the dataset samples:

1. A raw chunk uniformly.
2. A task ID uniformly.
3. A task-conditioned view of that same transition.

Rows where the sampled task was already complete at the chunk start are rejected,
so the shared critic does not learn post-terminal behavior for completed skills.

The main sampler is `sample_shared_relabel_batch`, used by shared QC-FQL. The
positive skill-row sampler is retained only for the diagnostic
`shared_flow_bc_positive` baseline.

## Offline Objective

Critic:

```text
R_k = sum_i gamma^i r_k(t+i)

target =
  R_k + gamma^nstep * (1 - done_k)
        * min_j Q_target_j(x_next_k, mu_omega(x_next_k, noise))

L_critic =
  MSE(Q1(x_k, a_chunk), target)
  + MSE(Q2(x_k, a_chunk), target)
```

Actor:

```text
L_actor =
  L_flow
  + alpha * || mu_omega(x_k, noise) - flow_ode_theta(x_k, noise).detach() ||^2
  - normalized_Q(x_k, mu_omega(x_k, noise))
```

`best_of_n` defaults to `8` for evaluation. Flow/FQL policies are latent-sampled
policies, so evaluating only zero noise is not a valid deterministic mean.

## Composition

The first version deliberately uses a fixed predicate planner:

1. Read benchmark task completion bits.
2. Pick the first incomplete task in the configured sequence.
3. Condition the shared policy on that task.
4. Execute action chunks until task completion, option budget, or environment termination.

There is no learned manager in v1. This keeps the contribution focused on the
shared reusable skill-conditioned policy.

## Algorithms

Use:

```bash
python train.py --offline_algo shared_qc_fql --encoder dinov3 \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --best_of_n 8 --log_dir logs/shared_qcfql_seed0 --seed 0 --no_video
```

Diagnostic BC baseline:

```bash
python train.py --offline_algo shared_flow_bc_positive --encoder dinov3 \
  --demo_datasets franka-complete franka-mixed franka-partial \
  --log_dir logs/shared_flowbc_positive_seed0 --seed 0 --no_video
```

Legacy aliases are accepted:

- `qc_fql` -> `shared_qc_fql`
- `flow_bc` -> `shared_flow_bc_positive`
