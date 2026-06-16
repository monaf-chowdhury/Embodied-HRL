# Skill-Conditioned QC-FQL

This branch replaces the Gaussian-actor + IQL/LQL stack with **per-skill QC-FQL**:
an expressive **flow-matching** policy class (Flow Q-Learning) trained with a
**chunked, unbiased n-step critic** (Q-chunking). The skill-transfer vision is
unchanged — still one goal-conditioned expert per task, chained by a scripted
controller, fine-tuned online.

References:
- Flow Q-Learning (FQL) — Park, Li, Levine, **ICML 2025**, arXiv:2502.02538
- Reinforcement Learning with Action Chunking (QC) — Li, Zhou, Levine, **NeurIPS 2025**, arXiv:2507.07969

## Why the switch

The Gaussian actor + IQL advantage-weighting collapsed to BC on near-expert,
low-within-state-diversity demos (`exp(beta*adv) ≈ 1`). QC-FQL fixes the
*extraction* problem at its root:

- The **flow policy** models the full (multimodal) behavior action distribution
  instead of averaging modes into mush.
- The **one-step actor** is improved by gradient-ascending Q through a
  differentiable policy (DDPG-style), regularized toward the flow — this needs
  only a usable critic gradient, **not** within-state action diversity in the data.
- **Best-of-N** (optional, eval-time) samples N one-step candidates and lets the
  critic pick — manufacturing candidate diversity the data lacks.
- The **chunked n-step critic** propagates the sparse completion reward over the
  whole chunk in one backup (what the LQL hinges were hacking at — now principled).

---

## Architecture

Per skill `k`: `{ FlowActor, TwinQ critic, TwinQ critic_target }`. No value net,
no Gaussian actor. All networks are MLPs (`LayerNorm + ReLU`, optional dropout).

**Conditioning feature (state input to every net), unchanged:**
```
x = [ z(DINOv3) ‖ proprio_norm ‖ task_target ‖ task_cur ‖ (task_target-task_cur)*mask ‖ task_mask ]
```

**Chunk action:** `a ∈ R^{H*env_action_dim}` (H=4, env_action_dim=9 → 36-dim).

| Net | Input | Output |
|---|---|---|
| velocity field `v_θ(s, x_t, t)` | `[x ‖ x_t(36) ‖ t(1)]` | velocity `(36)` |
| one-step actor `μ_ω(s, z)` | `[x ‖ z(36)]` | action chunk `(36)` |
| twin critic `Q(s, a)` | `[x ‖ a(36)]` | `(q1, q2)` scalars |

### Equations

**BC flow (rectified flow / conditional flow matching):**
```
x0 ~ N(0, I),  x1 = a,  x_t = (1-t) x0 + t x1,  t ~ U[0,1]
L_flow = E || v_θ(s, x_t, t) - (x1 - x0) ||^2
```
Action by Euler-integrating `dx/dt = v_θ(s, x, t)` from `t=0` to `t=1` (`flow_steps`).

**Critic (Q-chunking, unbiased n-step backup):**
```
a'      = μ_ω(s', z'),  z' ~ N(0, I)
R_disc  = sum_{i=0}^{nstep-1} gamma^i r_{t+i}        (stored at data-build time)
target  = R_disc + gamma^{nstep} * (1 - done) * min_i Q_target,i(s', a')
L_critic = MSE(Q1, target) + MSE(Q2, target)
```
`nstep` is the actual env steps in the chunk (= H, or < H for a truncated /
skill-completing chunk, which is terminal so the bootstrap drops out).
**gamma is per env step (`gamma^nstep`), not per chunk** — the bug Codex flagged.

**Actor (Flow Q-Learning, single joint update):**
```
L_actor = L_flow
        + alpha * || μ_ω(s, z) - flow_ode(s, z).detach() ||^2     (distillation / behavior constraint)
        - Q(s, μ_ω(s, z))                                          (Q-maximization; normalized if fql_normalize_q)
```
`alpha` is the behavior-constraint dial: large → stay near the flow (BC-like);
small → aggressive Q-maximization. The Q term is divided by `|Q|.mean()` when
`fql_normalize_q=True` so `alpha` is the dominant, scale-invariant knob.

**Target update:** Polyak `tau = target_tau` on the critic after each step.

**Action selection (`get_worker_chunk`):**
A flow / one-step policy *is* a map from a latent `z ~ N(0, I)` to an action —
there is no closed-form "mean", so deployment **samples a latent** (this is FQL's
`sample_actions`; zeroing the latent would evaluate one arbitrary, never-targeted
slice of the policy). Evaluation reproducibility comes from seeding the RNG around
each eval (see `rng_isolated`), not from zeroing the latent.
- `flow_bc` mode → integrate the BC flow ODE from a sampled latent.
- `qc_fql` mode → one-step actor on a sampled latent; if `best_of_n > 1`, sample
  N latents and take `argmax_a min Q(s, a)`.

---

## Data pipeline changes (cache `v9_qcfql_discounted_chunk_return`)

- Chunk reward is the **discounted return** `sum_i gamma^i r_{t+i}` (was a plain
  sum), and each transition stores **`nstep`** (env steps in the chunk).
- The visual `z_next` chunk-alignment fix (v6) and skill-completion termination
  (v7) are retained.
- The dead `w_seg_prog` field was dropped (it was stored and cached but never
  read into a batch). This format change is the v8 → v9 cache bump.
- **A cache rebuild is required** (it happens automatically on first run; ~2h
  per encoder). Old caches are ignored by the version bump.
- The data **build** (render → encode → label rewards) is unchanged in spirit
  but the consumed surface is a single uniform per-skill sampler
  (`sample_worker_task_batch`).

### Per-skill labeling is success-segment based (design choice, not a bug)

`_label_episode_from_replay` labels the demo steps **leading up to each
configured-task completion** with that task's id; transitions that never reach a
configured completion stay unlabeled (`-1`) and are dropped. So each skill trains
on the "approach that achieved task k" segments — this is the skill-transfer
definition of a demonstration, and it is deliberately kept simple. The cost is
weak *within-state action contrast* (the same thing that collapsed IQL); QC-FQL
sidesteps that via the one-step Q-ascent + best-of-N, which need a Q **gradient**,
not in-data action diversity. If experiment 2 shows `qc_fql ≈ flow_bc`, the next
deliberate experiment is **full per-skill offline RL**: relabel *every*
partial/mixed transition with the per-skill shaped reward (drop the success-only
filter, mark `done` only on actual completion) to give the critic real
suboptimal contrast. That is a different experiment and is intentionally not the
default.

---

## What was removed (dead in the QC-FQL branch)

- IQL / LQL / AWR / TD3+BC / BeT algorithms and all their config + CLI.
- Gaussian `SkillActor`, `ValueNet`, the value-expectile machinery, AWAC online.
- The LQL chunk-chain code (`ensure_chain_links`, `sample_lql_segments`, …).
- The complex weighted/focus/global worker samplers and **all manager data**
  (`add_manager`, `m_*`, `sample_manager_batch`) — there is no learned manager.
- `_parse_tb.py` (one-off LQL log parser).
- `eval_replay.py` (standalone repeated-eval probe) — its purpose is now built
  into `train.py` (`final_eval_repeats`: repeated disjoint-seed final eval,
  reported as mean ± std).
- The `BufferConfig` dataclass and the `w_seg_prog` dataset field (both unread).

`plots.py` now plots QC-FQL tags (`03_qc_diagnostics.png`, `09_online_qcfql.png`).

---

## Correctness fixes from code review

These hardened the parts that silently invalidate experiments (eval / data):

1. **Latent sampling at deployment.** `get_worker_chunk` previously used a
   **zero** latent for "deterministic" eval, which evaluates one arbitrary slice
   of a flow/one-step policy rather than the learned distribution. It now samples
   `z ~ N(0, I)` (FQL `sample_actions`); reproducibility comes from RNG seeding.
2. **Eval RNG isolation.** Every evaluator (`evaluate_*`, `_prefix_actor_eval`)
   is wrapped with `@rng_isolated`: it seeds the global RNG internally (12345,
   for a reproducible/comparable rollout) but restores the surrounding stream on
   exit, so an eval can no longer reset the online/training RNG to the eval seed.
   `_prefix_actor_eval` now seeds too, so model selection is comparable across
   steps and does not perturb the offline noise stream.
3. **Online chunk storage.** Truncated online chunks (skill completed / env
   terminated mid-chunk) store only the **executed** actions, padded to `H` by
   repeating the last executed step — matching the offline `_chunk_actions`
   padding. The proposed-but-never-stepped tail no longer enters `Q(s, a_chunk)`.
4. **Plotting.** `plot_prefix_validation` took `sw` without it being a parameter
   (a `NameError` exactly when prefix data existed); fixed and the stale
   `plot_lql_diagnostics` / `plot_online_awac` names were renamed.

---


## Config knobs (`SpecialistConfig`)

| Knob | Default | Meaning |
|---|---|---|
| `offline_algo` | `qc_fql` | `flow_bc` (BC only) or `qc_fql` |
| `n_flow_bc_steps` | 30000 | flow-BC steps (also total in `flow_bc` mode) |
| `n_offline_rl_steps` | 100000 | QC-FQL joint steps |
| `flow_steps` | 10 | Euler steps for the BC flow ODE |
| `fql_alpha` | 10.0 | distillation / behavior-constraint coefficient |
| `fql_normalize_q` | True | scale-invariant Q term |
| `best_of_n` | 1 | >1 → best-of-N action selection at eval |
| `target_tau` | 0.005 | critic target soft-update |
| `use_layernorm` / `dropout` | True / 0.0 | net regularization |
| `eval_interval` / `eval_prefix_states` | 10000 / 100 | prefix-val model selection |

Online (`OnlineConfig`): `updates_per_env_step`, `demo_fraction` (demo share of
each batch), `exploration_noise` (one-step actor noise scale), `eval_interval_steps`,
`rollback_drop_tolerance`.

---

## Staged experiments (the questions this answers, in order)

Validate on **partial / mixed** (suboptimal data) — not complete (near-expert),
where any extraction method ≈ BC.

**0. Pre-flight (seconds, no GPU/env):**
```bash
python test.py
```

**1. Does an expressive policy class alone beat the Gaussian?  (flow BC)**
```bash
python train.py --encoder dinov3 --offline_algo flow_bc --seed 0 \
  --demo_datasets franka-partial --log_dir logs/flowbc_partial_seed0 --no_video
```
If light-switch prefix-val rises vs the old Gaussian BC → the policy class was a
real bottleneck.

**2. Does QC-FQL improve over flow BC?  (Q-guidance + chunked critic)**
```bash
python train.py --encoder dinov3 --offline_algo qc_fql --seed 0 \
  --demo_datasets franka-partial --log_dir logs/qcfql_partial_seed0 --no_video
```
Watch `03_qc_diagnostics.png`: `qc_q_mean` should settle near the reward scale
(a few × completion_bonus), **not** blow up. If QC-FQL > flow BC, the critic
ranking is useful; if not, the critic lacks a useful ranking and the bottleneck
is data coverage/recovery.

**3. Online QC-FQL fine-tuning (after offline is confirmed):**
```bash
python train.py --encoder dinov3 --offline_algo qc_fql --seed 0 \
  --online_finetune --log_dir logs/qcfql_online_seed0
```

Tuning order if needed: `fql_alpha` (start 10, lower to loosen the behavior
constraint), then `best_of_n` (e.g. 4–8 at eval), then `flow_steps`.

---

## Risks / limitations (watch these)

- **`fql_alpha` matters.** Too low → critic errors push the one-step actor
  off-support; too high → pure BC. Start conservative (10), lower gradually.
- **Q-inflation watch.** Even with the chunked backup, an over-eager Q term can
  inflate `qc_q_mean`; the normalized Q term + behavior constraint guard against
  it. Monitor the diagnostics plot.
- **Near-expert ceiling.** On expert-only data (complete) QC-FQL ≈ flow BC ≈ BC.
  Gains require suboptimal/multimodal data (partial/mixed, online).
- **Fixed chunk length** `H=4` is a hyperparameter; too long hurts reactivity in
  contact-rich phases (not expected to bite in kitchen).
- **The online loop is new and not yet validated end-to-end.** Confirm offline
  (experiments 1–2) before enabling `--online_finetune`.
