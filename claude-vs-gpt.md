# Claude vs GPT Codebase Comparison
## Visual HRL for FrankaKitchen-v1

---

# Part 1: agent.py, buffers.py, config.py, utils.py, plots.py, encoder.py, env_wrapper.py

---

## agent.py

### What each tries to achieve
Both implement the `VisualHRLAgent` class — the central controller for the hierarchical RL system. It handles subgoal selection, worker action, reward computation, network updates, and threshold calibration.

### Key Differences

**Task identity as explicit signal (GPT's biggest structural difference)**
GPT introduces a `task_id` (integer) and `task_onehot` vector that flows through every component — manager Q-network, worker actor, worker critic, and both buffers. The manager selects not just a landmark in latent space but a landmark-with-task-identity. The worker receives an explicit one-hot encoding of which task to work on. This directly addresses your instruction: *"The worker needs to know which task to work on, not just which latent point to reach."*

Claude instead uses `compute_task_progress_focused()` — it infers which task to focus on by finding the task whose proprio-goal is closest to the agent's *current* proprio state, then computes progress only for that task. This is implicit task selection derived from state, not explicit conditioning.

**Worker reward structure**
GPT computes three task-progress terms:
- `selected_task_progress_weight * delta[task_id]` — progress on the explicitly selected task
- `any_task_progress_weight * sum(max(delta, 0))` — progress on any task
- `completion_bonus * task_completed_delta` — step-level completion bonus

Claude computes only one focused task-progress term based on nearest task from proprio distance, plus sparse reward and a latent navigation hint.

GPT's approach is richer and more explicit. It gives the worker a gradient on both its assigned task and any incidental progress — this is reasonable since completing adjacent tasks is still useful.

**Manager reward structure**
Both follow the same hierarchy (completion dominant → progress secondary → env reward → latent nav), but GPT's manager reward also has `selected_task_progress_weight` and `any_task_progress_weight` separated cleanly. GPT's `completion_bonus=25.0` vs Claude's `10.0` — GPT is more aggressive about the dominant signal.

**calibrate_success_threshold**
This is a significant difference. Claude redesigns calibration for L2-normalised space: `threshold = mean_step_dist * (K/4)`, bounded to `[2*mean_step, 0.5]`. This is principled for the unit hypersphere. GPT uses percentile of nearest-neighbor landmark distances: `p10 * 0.35`, bounded by `[1.5*mean_step, p50*0.75]`. **GPT's approach has a problem**: it computes nearest-neighbor distances in raw (un-normalised) R3M space since GPT's encoder doesn't L2-normalise. In 2048-d raw R3M, these pairwise distances will be large and the calibration will produce a large threshold. Claude's approach is more correct given L2-normalisation.

**Memory efficiency**
GPT defaults `z_storage_dtype=float16` in buffers. At 2048-d this halves memory from ~8MB to ~4MB per 1000 transitions. A practical improvement Claude doesn't have.

**`choose_task_from_state`**
GPT has a helper that picks the lowest-progress task (most urgent) as the current task, and can filter to incomplete tasks only. This is used during training to select which task the worker should focus on at episode start. Claude has no equivalent — task focus is always inferred moment-to-moment from state.

**Missing in GPT: encoder diagnosis**
Claude's `calibrate_success_threshold` prints step distance stats and warns if distances are out of expected range. GPT also prints step distance stats but doesn't have the encoder's `diagnose_distances()` method. Both check if step mean is in the requested 1–20 range.

**Winner: GPT** for explicit task conditioning (directly addresses your instruction), richer reward structure, and memory efficiency. Claude's threshold calibration is more principled for L2-normalised space.

---

## buffers.py

### What each tries to achieve
Both implement `HighLevelBuffer` (manager transitions) and `LowLevelBuffer` (worker transitions) as ring-buffer replay memories.

### Key Differences

**Float16 storage (GPT)**
GPT stores z-vectors as `float16` by default. With 2048-d vectors at 300k capacity, this saves ~1.2GB RAM. Claude stores everything as `float32`. This is a meaningful practical difference.

**task_id in every transition (GPT)**
Both buffers in GPT store `task_id` per transition. This supports the explicit task conditioning throughout the system.

**task_deltas stored per transition (GPT)**
GPT's `LowLevelBuffer` stores `task_deltas` (shape: `[n_tasks]`) and `task_completed_delta` per step. This allows landmark updates to preferentially sample transitions where any task made progress.

Claude stores a single scalar `task_delta` (for the focused task only). GPT stores the full vector — more information preserved for landmark biasing.

**`get_landmark_data()` method (GPT)**
GPT exposes `get_landmark_data()` which returns z, proprio, task_deltas, and task_completed_delta for all stored transitions. This is a cleaner interface for the landmark update pipeline than Claude's `get_all_z()` + `get_task_biased_z()` pair.

Claude's `get_task_biased_z()` is a more complex method that computes the biased candidate pool inside the buffer. GPT pushes that logic into `landmarks.py` which has access to task_deltas. Cleaner separation of concerns in GPT.

**Welford algorithm correctness**
Both implement Welford's online algorithm for proprio normalisation. GPT's is slightly more correct — it uses `delta * delta2` (the standard two-pass formula) while Claude uses `delta * (x - mean)` which is mathematically equivalent but written differently. Both are correct.

**Capacity differences**
GPT: `capacity=300_000`, `high_capacity=50_000`, `batch_size=512`
Claude: `capacity=1_000_000`, `high_buffer=200_000`, `batch_size=256`

GPT's lower capacity is reasonable given float16 storage — same effective memory footprint. Claude's larger capacity with float32 would require significantly more RAM. GPT's `batch_size=512` will give more stable gradient estimates per update.

**Winner: GPT** — float16 storage, per-transition task_delta vector (full, not focused), cleaner data interface, better batch size.

---

## config.py

### What each tries to achieve
Both define all hyperparameters as dataclasses.

### Key Differences

**Manager reward weights**
GPT: `completion_bonus=25.0`, `selected_task_progress_weight=8.0`, `any_task_progress_weight=3.0`, `env_reward_weight=5.0`
Claude: `task_completion_bonus=10.0`, `task_progress_bonus=3.0`, `env_reward_weight=1.0`, `nav_bonus=0.5`

GPT is much more aggressive on completion bonus (25 vs 10) and env reward passthrough (5 vs 1). GPT's `env_reward_weight=5.0` is arguably too high — this could make the manager chase shaped reward rather than task completions. Claude's `env_reward_weight=1.0` is safer.

**subgoal_horizon and max_high_level_steps**
GPT: `subgoal_horizon=40`, `max_high_level_steps=8`
Claude: `subgoal_horizon=20`, `max_high_steps=14`

GPT gives the worker 40 steps per subgoal attempt but only allows 8 attempts per episode (8×40=320 steps/ep). Claude gives 20 steps with 14 attempts (14×20=280 steps/ep). Both land near the 280-step episode limit. GPT's longer horizon gives the worker more time per subgoal but fewer manager decisions per episode, meaning the manager learns slower.

**epsilon_decay_steps**
GPT: `400_000`, Claude: `800_000`. GPT decays exploration faster — risky if the manager hasn't learned anything useful yet. Claude's slower decay is more conservative.

**total_timesteps**
GPT: `1_000_000`, Claude: `2_000_000`. GPT is optimistic about sample efficiency; Claude allocates more budget. Given the difficulty of the task, Claude's choice is safer.

**n_warmup**
GPT: `120`, Claude: `200`. Claude collects more random data before training starts, giving better landmark initialization.

**LandmarkConfig differences**
GPT: `n_landmarks=128`, `demo_fraction=0.25` (explicit fraction for demo states in FPS pool)
Claude: `n_landmarks=200`, no explicit demo_fraction

More landmarks in Claude gives better coverage of the state space.

**ReachabilityConfig**
Claude keeps it explicitly with `reject_threshold=0.0`. GPT drops the config entirely — cleaner since it's disabled.

**Worker hidden_dim**
GPT: `hidden_dim=384`, Claude: `hidden_dim=512`. Claude's larger network for the worker is appropriate given the 2×2048 input.

**Winner: Draw with edge to Claude** — Claude's more conservative hyperparameters (slower epsilon decay, more timesteps, more warmup, more landmarks) are safer for a hard task. GPT's `env_reward_weight=5.0` is questionable. But GPT is cleaner for dropping unused reachability config.

---

## encoder.py

### What each tries to achieve
Both load a frozen R3M backbone and encode images to latent vectors.

### Critical Difference: L2 Normalisation

**Claude L2-normalises** the output onto the unit hypersphere (`F.normalize(features, dim=-1)`). All distances live in [0, 2]. This is consistent throughout — the threshold calibration, distance checks, and step-distance diagnostics all assume this range.

**GPT does NOT L2-normalise**. It returns raw R3M features. Raw R3M (ResNet-50 fc output) has values roughly in the range of a few hundred to a few thousand in L2 norm. Step-to-step distances in raw space are in the 1–50 range that you requested verification for. GPT's calibration method (percentile of NN-landmark distances) is designed for this raw space.

Both approaches are internally consistent, but they create different expectations:
- Claude: distances in [0, 2], threshold ~0.1–0.4
- GPT: distances in [1, 50], threshold calibrated from NN distances

The key question is: **which is better for this task?** Raw R3M has more discriminative power because L2-normalisation discards magnitude information. However, L2-normalisation makes distances more geometrically uniform and the calibration more predictable. For FPS landmark selection, L2-normalisation means all landmarks are on a sphere — FPS will spread them uniformly on that sphere, which is desirable. For raw R3M, the manifold is irregular.

**Claude adds `diagnose_distances()`** — a useful debugging method that runs 200 random steps and prints distance statistics. GPT has no equivalent.

**GPT's encoder has a preprocessing note**: it defines `self.preprocess` with ImageNet normalisation, but then for R3M it doesn't use it (R3M uses `x * 255.0` directly, which is correct). The preprocess transform is only used for DINOv2. This is slightly confusing but not a bug for R3M.

**Winner: Claude** — L2 normalisation is more principled for the unit hypersphere geometry, makes FPS more meaningful, and the distance diagnostic is very useful.

---

## env_wrapper.py

Both versions are **identical** — same camera parameters, same gymnasium-robotics API, same info fields, same step/reset logic. Both keep the dead `HierarchicalKitchenWrapper` class that's never called in training. No meaningful difference here.

---

## utils.py

Both are **identical** — same `save_image` and `save_video` implementations. No difference.

---

## plots.py

### What each tries to achieve
Both generate diagnostic plots from TensorBoard logs.

### Key Differences

Claude's `plots.py` is significantly more complete:
- 7 named plot functions covering all metrics
- Overview dashboard (6 panels on one page)
- Multi-run comparison mode (`--compare` flag)
- Proper axis formatting (k/M suffixes)
- Reachability accuracy with 50% random baseline reference line
- Tag-consistent naming throughout

GPT's `plots.py` is minimal — a single 2×3 dashboard with no standalone plot functions, no comparison mode, no axis formatting. It's functional but thin.

**Tag mismatch in Claude**: plots.py looks for `eval/success_rate` but `train.py` logs `eval/task_completion_rate`. This will show "No data" on the dashboard's primary panel.

**GPT's tags**: `eval/any_task_success_rate` and `eval/full_task_success_rate` — two separate metrics. This is better than Claude's single metric since it tracks both "at least 1 task" and "all 4 tasks" completions.

**Winner: Claude** for completeness of plotting, but GPT has better eval metric design (`any_task` vs `full_task`).

---

## Overall Summary (Part 1 — these 7 files)

| File | Winner | Key Reason |
|------|--------|------------|
| agent.py | GPT | Explicit task conditioning, richer reward, memory efficiency |
| buffers.py | GPT | float16 storage, full task_delta vector, cleaner interface |
| config.py | Claude (edge) | Safer hyperparameters, more budget |
| encoder.py | Claude | L2 normalisation + distance diagnostics |
| env_wrapper.py | Tie | Identical |
| utils.py | Tie | Identical |
| plots.py | Claude | More complete, but with tag mismatch |

The most architecturally significant difference across these 7 files is **GPT's explicit task conditioning** (task_id/task_onehot flowing through everything) vs **Claude's implicit task focus** (inferred from proprio distance). GPT's approach more directly addresses your instruction that the worker needs to know which task to work on.

---
---

# Part 2: train.py, landmarks.py, networks.py — and Final Verdict

---

## train.py

### What each tries to achieve
Both implement the two-phase training loop: Phase 1 is random warmup to fill buffers and initialize landmarks, Phase 2 is the hierarchical training loop where manager and worker learn simultaneously.

### Key Differences

**Warmup phase — task_id assignment**
GPT computes `per_task_progress` at every warmup step and assigns a `task_id` to each transition via `choose_task_from_state(ep_props[future_idx])` — which picks the task with lowest current progress (most urgent). This task_id flows into the buffer and is used for reward computation immediately during warmup. Claude's warmup uses `compute_worker_reward()` with no task_id — it infers focus from proprio distance at reward compute time. GPT's warmup is richer and consistent with how the full training loop works.

**Recalibration during training (GPT's critical advantage)**
GPT calls `agent.calibrate_success_threshold()` every `update_freq` episodes during Phase 2, in addition to once after warmup. This means the threshold adapts as the landmark distribution improves. Claude calibrates **only once** after warmup, then never again. This is a significant gap — as hindsight states and task-progress-biased landmarks replace random-walk landmarks, the nearest-neighbor distances change and the threshold should update accordingly.

**Worker update frequency**
GPT updates the worker every 2 steps (`total_steps % 2 == 0`). Claude updates every 4 steps (`total_steps % 4 == 0`). GPT runs twice as many gradient updates, which accelerates worker learning but risks overfitting to early, noisy transitions. With `batch_size=512`, GPT's updates are more stable per step.

**Eval metric — three separate flags**
GPT's `evaluate()` returns `any_task_success_rate` (≥1 task), `full_task_success_rate` (all 4 tasks), and `env_reward_success_rate` (ep_reward > 0 — the old misleading metric, kept for comparison). This directly addresses your instruction about misleading eval. Claude returns only `task_completion_rate` (≥1 task). GPT's three-metric eval is more diagnostic — you can see if the agent gets any task vs all tasks vs just shaped reward.

**Task progress computation in inner loop**
GPT computes `per_task_progress(proprio, agent.tasks)` before and after every step — full vector for all 4 tasks. Claude computes `compute_task_progress_focused()` which only returns the nearest task's progress. GPT's approach stores the complete picture; Claude discards 3/4 of the information at compute time.

**`_latent_dists` collection**
GPT collects step distances during both warmup AND training (inside the inner loop). Claude collects only during warmup. GPT's ongoing collection means `_latent_dists` reflects the actual training distribution, making recalibration more accurate.

**Missing in GPT: encoder diagnosis**
Claude calls `agent.encoder.diagnose_distances(env, n_steps=200)` after warmup, printing a full distance report. GPT has no equivalent diagnostic call. You'd need to infer encoder health from threshold calibration output alone.

**Missing in GPT: hindsight pool size logging**
Claude logs `train/hindsight_pool_size` to TensorBoard. GPT doesn't. Minor but useful for debugging.

**Episode termination condition**
Both terminate on `env_done` or `max_high_steps`. Neither uses SSE. Identical behavior.

**Winner: GPT** — ongoing threshold recalibration is critical and Claude lacks it entirely. GPT's three-metric eval is more informative. GPT's task_id propagation through warmup is consistent. The only advantage Claude has here is the encoder diagnosis print.

---

## landmarks.py

### What each tries to achieve
Both maintain a fixed-size set of landmark latent vectors via Farthest Point Sampling (FPS), with mechanisms to bias the candidate pool toward task-relevant states.

### Key Differences

**Task-ID per landmark (GPT's core structural addition)**
Every landmark in GPT has an associated `task_id`. This is the most important difference. When the manager selects a landmark, it knows not just where to navigate but which task is associated with that destination. The worker receives this task_id explicitly. Claude has no concept of task identity per landmark — landmarks are anonymous points in latent space.

**Candidate pool construction — how task bias is applied**
Claude's approach (`get_task_biased_z()` in buffers): select top-30% by scalar `task_delta` + all positive `task_delta` transitions. This is done in the buffer before landmarks sees any data.

GPT's approach (inside `landmarks.update()`): it receives the full `replay` dict with `task_deltas` (N×4 matrix), `task_completed_delta`, and `proprio`. It then:
1. Computes `max_delta = td_recent.max(axis=1)` — best task progress per transition
2. Uses 85th percentile threshold OR `completed_delta > 0` to create `priority_mask`
3. Separately samples `priority_fraction=0.5` from high-priority states and `0.5` from general states
4. Assigns task_id to each candidate: uses `argmax(task_deltas)` if any progress, else infers from proprio via `infer_task_id_from_proprio()`

This is significantly more sophisticated. The landmark update knows which task each candidate state is relevant to, producing a landmark set where each point has meaningful task identity.

**Demo landmark task inference**
Claude: demo landmarks are stored without task_id — they're anonymous in the candidate pool.
GPT: infers task_ids for demo landmarks by finding which priority state each demo frame is closest to in latent space, then inheriting that task_id. This is a clever heuristic — demo frames near task-progress states get labeled with the corresponding task.

**`infer_task_id_from_proprio()`**
GPT has a standalone function that assigns task_id by finding which task's goal state is nearest in normalised proprio space. This is used as fallback when task_deltas are near zero. Claude has no equivalent — if there's no task delta, the state has no task identity.

**`select_explore()` — exploration with task weighting**
Claude: `argmin(visit_counts)` — pure least-visited.
GPT: `argmin(visit_counts - 0.25 * success_counts)` — slightly prefers landmarks that have never succeeded, discouraging repeated visits to easy-to-reach but task-irrelevant landmarks. A small but thoughtful difference.

**FPS identical**: both implement the same greedy FPS algorithm. No difference.

**Hindsight pool with task_id**
Claude: `add_success_state(z)` — no task identity on hindsight states.
GPT: `add_success_state(z, task_id)` — hindsight states carry which task was just completed. When selected as landmarks, the worker knows exactly which task was completed at that state. This is much more useful.

**Winner: GPT by a significant margin** — task-aware landmarks are the whole point. Demo inference, hindsight with task_id, and priority sampling with per-task delta vectors all make GPT's landmark system substantially richer.

---

## networks.py

### What each tries to achieve
Both define the manager Q-network, worker SAC actor, and worker SAC critic as PyTorch modules.

### Key Differences

**Task one-hot in every network (GPT)**
GPT adds a dedicated `task_stream` (small 2-layer MLP: n_tasks → 32) in both actor and critic. The task one-hot is processed separately and concatenated into the merged representation before the trunk. This means the network explicitly learns how task identity modulates behavior — the gradient for "task A requires moving toward the microwave" flows through a dedicated pathway.

Claude has no task stream. Its networks take only `[z_current, z_subgoal, proprio]`. Task is invisible to the networks.

**Manager Q-network input**
GPT: `[z_current (2048), z_landmark (2048), task_onehot (4)]` = 4100-d input directly into a single MLP.
Claude: `[z_current (2048), z_landmark (2048)]` = 4096-d, but first compressed by a dedicated `input_compress` layer (Linear → LayerNorm → ReLU) before the main MLP trunk.

Claude's two-stage compression (compress first, then MLP) is more principled for very high-dimensional inputs. GPT feeds 4100-d directly into the first Linear layer — for 2048-d inputs this means the first layer has ~2M parameters just for that layer alone, which is heavy. Claude's compression to `hidden_dim=512` before the trunk is more parameter-efficient.

**Worker network: hidden_dim**
GPT: `hidden_dim=384` for worker networks.
Claude: `hidden_dim=512` for worker networks.

Given that both networks take 2×2048-d inputs, the first compression layer is the bottleneck. Claude's 512-d hidden is more appropriate for 4096-d inputs. GPT's 384-d is slightly lean but probably fine.

**ReachabilityPredictor**
Claude keeps it as dead code. GPT removes it entirely. GPT is cleaner.

**`build_mlp` — LayerNorm placement**
Both use LayerNorm after every hidden layer. Identical pattern.

**Parameter count estimate**
GPT's Manager: first Linear is `4100 → 512` = ~2.1M params just for that layer. Claude's first compression is `4096 → 512` = ~2.1M similarly, but then the MLP trunk starts at 512, keeping subsequent layers small. The difference is marginal in practice.

GPT's Worker Actor: img_stream `4096 → 384 → 384`, prop_stream `59 → 192 → 192`, task_stream `4 → 32 → 32`, trunk `608 → 384 → 384`. Total ~4.5M.
Claude's Worker Actor: img_stream `4096 → 512 → 256`, prop_stream `59 → 128 → 64`, trunk `320 → 512 → 512`. Total ~5.5M.

Claude's networks are slightly larger but the difference is negligible on GPU.

**Winner: GPT** for having the task stream (architecturally critical for the task-conditioning approach), and for cleanliness (no dead reachability code). Claude's input compression is more principled, but it's a minor point.

---

## Final Verdict

### Head-to-head by your 8 stated goals

| Your Goal | Claude | GPT | Winner |
|-----------|--------|-----|--------|
| 1. Remove SSE | ✅ Done | ✅ Done | Tie |
| 2. Drop projection MLPs + recalibrate | ✅ L2-normed, principled calibration | ✅ Raw features, adapts calibration during training | GPT (recalibration) |
| 3. Disable reachability filter | ✅ Disabled | ✅ Removed entirely | GPT (cleaner) |
| 4. Break bootstrapping deadlock | ✅ Redesigned reward hierarchy | ✅ Same + task_id makes reward more targeted | GPT |
| 5. Task-progress-biased landmarks + calibration review | ✅ Biased pool, one-time calibration | ✅ Biased pool + per-task identity + recalibration during training | GPT |
| 6. Worker knows which task to work on | ❌ Implicit only (proprio nearest) | ✅ Explicit task_id flows everywhere | GPT |
| 7. Unambiguous eval metric | ✅ task_completion_rate (≥1) | ✅ Three metrics: any/full/env_reward | GPT |
| 8. Config correctness | ✅ Conservative, safe | ⚠️ env_reward_weight=5.0 too high, 1M steps too low | Claude |

**Overall winner: GPT**, but not by a clean margin.

---

### Why GPT wins

The single most important architectural decision across all your 8 goals is **Goal 6 — giving the worker explicit task identity**. This is the linchpin. Without it, even if you fix the manager reward, fix the landmarks, and fix the calibration, the worker still can't know which object to move toward. GPT solves this end-to-end: every landmark has a task_id, every worker action conditions on a task one-hot, every reward includes a selected-task-progress term, and the manager learns which task-landmark pairs lead to progress.

GPT also wins on ongoing threshold recalibration (critical as landmarks improve) and on eval metric design (three metrics vs one).

---

### Why you shouldn't use GPT as-is

Three concrete problems that need fixing before you run it:

**1. `env_reward_weight=5.0` in ManagerConfig is too high.** The sparse env reward in FrankaKitchen is already captured by `completion_bonus=25.0`. Adding `5.0 × cumulative_env_reward` on top means the manager chases shaped reward incidentally, not just task completions. Change this to `1.0` or `0.5`.

**2. `total_timesteps=1_000_000` is likely insufficient.** FrankaKitchen 4-task completion is a hard exploration problem. Claude's `2_000_000` is safer. You can always stop early if it converges, but you can't extend a run that's capped too low.

**3. GPT's encoder does not L2-normalise.** The threshold calibration (percentile of NN distances in raw R3M space) is internally consistent, but raw R3M distances are in the 1–50 range and very noisy across dimensions. L2-normalisation from Claude's encoder is more principled and makes FPS more geometrically meaningful. This is worth taking from Claude into GPT.

---

### My recommendation

**Use GPT's architecture but patch in these three things from Claude:**

1. **L2 normalisation** from `encoder.py` — add `features = F.normalize(features, dim=-1)` before returning from `encode_raw()`.
2. **`diagnose_distances()`** from Claude's encoder — useful one-time health check after warmup.
3. **Config fixes**: set `env_reward_weight=1.0`, `total_timesteps=2_000_000`, `n_warmup=200`, `epsilon_decay_steps=600_000`, `n_landmarks=200`.

The explicit task conditioning in GPT's architecture (task_id per landmark → task_onehot in worker networks → task-aware manager reward → task-aware hindsight) is the right solution to your core problem. That design philosophy runs consistently through every file in GPT's codebase, and it's what's missing from Claude's more implicit approach.