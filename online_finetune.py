"""Online QC-FQL fine-tuning (lean).

Continue the *same* QC-FQL update used offline on a growing mixed replay:
  1. roll out chained episodes (scripted next-incomplete controller) with the
     stochastic one-step actor for exploration,
  2. store per-chunk transitions (discounted h-step return + nstep + done) into
     a per-skill online buffer,
  3. run qc_fql_step on batches mixing demo data and online data,
  4. periodically evaluate (RNG-isolated, with a confirmation eval) and keep the
     best checkpoint; roll back on a confirmed drop.

This is the paper-style recipe: "keep running the offline objective online" on a
growing mixed replay (no separate repair/freeze/anchor machinery).

NOTE: this online loop is new for the QC-FQL branch and has not been validated
end-to-end in this environment. Confirm offline QC-FQL first (the staged
experiments in QCFQL.md), then enable --online_finetune.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from specialist import SkillAgent
from utils import preserve_rng_state

import torch


_BATCH_KEYS = (
    "z", "proprio", "task_target", "task_cur", "task_mask", "task_id",
    "action", "reward", "done", "nstep", "z_next", "proprio_next", "task_cur_next",
)


class OnlineReplay:
    """Shared raw chunk replay, relabeled by sampled task id at batch time."""

    def __init__(self, n_tasks: int, capacity_per_skill: int):
        self.n_tasks = int(n_tasks)
        self.capacity = int(capacity_per_skill) * self.n_tasks
        self.buf: List[Dict[str, np.ndarray]] = []

    def add(self, tr: Dict[str, np.ndarray]):
        self.buf.append(tr)
        if len(self.buf) > self.capacity:
            self.buf.pop(0)

    def __len__(self) -> int:
        return int(len(self.buf))

    def sample(self, agent: SkillAgent, n: int) -> Dict[str, np.ndarray]:
        if len(self.buf) == 0:
            raise RuntimeError("Cannot sample an empty online replay.")
        valid = [
            i for i, row in enumerate(self.buf)
            if np.any(np.asarray(row["task_complete_vec"], dtype=np.float32) < 0.5)
        ]
        if valid:
            idx = np.asarray(valid, dtype=np.int64)[
                np.random.randint(0, len(valid), size=int(n))]
        else:
            idx = np.random.randint(0, len(self.buf), size=int(n))
        rows = [self.buf[i] for i in idx]
        complete = np.stack([r["task_complete_vec"] for r in rows], axis=0).astype(np.float32)
        task_id = self._sample_incomplete_task_ids(complete)
        p_raw = np.stack([r["proprio_raw"] for r in rows], axis=0).astype(np.float32)
        p_next_raw = np.stack([r["proprio_next_raw"] for r in rows], axis=0).astype(np.float32)
        tt = np.zeros((len(rows), agent.max_goal_dim), dtype=np.float32)
        tc = np.zeros_like(tt)
        tm = np.zeros_like(tt)
        tc_next = np.zeros_like(tt)
        for k in np.unique(task_id):
            mask = task_id == int(k)
            idx_k = agent.spec.indices(int(k))
            tt[mask] = agent.spec.goal_vec_padded[int(k)]
            tm[mask] = agent.spec.goal_mask_padded[int(k)]
            cur = p_raw[mask][:, idx_k]
            nxt = p_next_raw[mask][:, idx_k]
            tc[mask, :cur.shape[1]] = cur
            tc_next[mask, :nxt.shape[1]] = nxt
        p = np.stack([agent.normalize_proprio(row) for row in p_raw], axis=0)
        p_next = np.stack([agent.normalize_proprio(row) for row in p_next_raw], axis=0)
        reward = np.asarray([rows[i]["reward_vec"][task_id[i]] for i in range(len(rows))], dtype=np.float32)
        task_done = np.asarray([rows[i]["task_done_vec"][task_id[i]] for i in range(len(rows))], dtype=np.float32)
        env_done = np.asarray([rows[i]["env_done"] for i in range(len(rows))], dtype=np.float32)
        return {
            "z": np.stack([r["z"] for r in rows], axis=0).astype(np.float32),
            "proprio": p.astype(np.float32),
            "task_target": tt,
            "task_cur": tc,
            "task_mask": tm,
            "task_id": task_id,
            "action": np.stack([r["action"] for r in rows], axis=0).astype(np.float32),
            "reward": reward,
            "done": np.maximum(env_done, task_done).astype(np.float32),
            "nstep": np.asarray([r["nstep"] for r in rows], dtype=np.float32),
            "z_next": np.stack([r["z_next"] for r in rows], axis=0).astype(np.float32),
            "proprio_next": p_next.astype(np.float32),
            "task_cur_next": tc_next.astype(np.float32),
        }

    def _sample_incomplete_task_ids(self, complete: np.ndarray) -> np.ndarray:
        task_id = np.random.randint(0, self.n_tasks, size=complete.shape[0]).astype(np.int64)
        bad = complete[np.arange(complete.shape[0]), task_id] > 0.5
        for _ in range(8):
            if not np.any(bad):
                break
            task_id[bad] = np.random.randint(0, self.n_tasks, size=int(np.sum(bad)))
            bad = complete[np.arange(complete.shape[0]), task_id] > 0.5
        if np.any(bad):
            for row in np.where(bad)[0]:
                avail = np.where(complete[row] < 0.5)[0]
                task_id[row] = int(np.random.choice(avail)) if len(avail) else int(np.random.randint(0, self.n_tasks))
        return task_id.astype(np.int64)


def _mixed_batch(agent: SkillAgent, ds, replay: OnlineReplay,
                 batch_size: int, demo_fraction: float) -> Dict[str, np.ndarray]:
    """Concatenate a demo sub-batch and an online sub-batch into one QC-FQL batch."""
    if len(replay) > 0:
        n_demo = int(round(batch_size * float(demo_fraction)))
    else:
        n_demo = batch_size
    n_demo = int(np.clip(n_demo, 0, batch_size))
    n_online = batch_size - n_demo

    demo = (ds.sample_shared_relabel_batch(n_demo, spec=agent.spec, proprio_normalizer=agent.normalize_proprio)
            if n_demo > 0 else None)
    onl = replay.sample(agent, n_online) if n_online > 0 else None

    out: Dict[str, np.ndarray] = {}
    for k in _BATCH_KEYS:
        parts = []
        if demo is not None:
            parts.append(np.asarray(demo[k], dtype=np.float32))
        if onl is not None:
            parts.append(np.asarray(onl[k], dtype=np.float32))
        out[k] = np.concatenate(parts, axis=0)
    return out


def _onestep_chunk(agent: SkillAgent, z: np.ndarray, state: np.ndarray,
                   task_id: int, noise_scale: float) -> np.ndarray:
    """One-step actor chunk with exploration noise; returns (H, env_action_dim)."""
    x = agent._input_from_state(z, state, task_id)
    with torch.no_grad():
        noise = float(noise_scale) * torch.randn(1, agent.action_dim, device=agent.device)
        a = agent.flow.onestep_action(x, noise).clamp(-1.0, 1.0)
    return a.cpu().numpy().reshape(agent.H_chunk, agent.env_action_dim)


def collect_chain_episode(agent: SkillAgent, config: Config,
                          replay: OnlineReplay, episode_seed: int) -> Dict[str, float]:
    """Roll out one scripted-chain episode, storing per-chunk transitions."""
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=True,
    )
    gamma = float(config.worker.gamma)
    noise_scale = float(config.online.exploration_noise)
    order = list(range(agent.n_tasks))
    env_steps = 0
    try:
        img, state = env.reset(seed=episode_seed)
        z = agent.encoder.encode_numpy(img).squeeze()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)
        done = False
        n_opts = 0
        while (not done and completion.sum() < agent.n_tasks
               and n_opts < config.manager.max_high_level_steps):
            remaining = [k for k in order if completion[k] < 0.5]
            task_id = int(remaining[0]) if remaining else int(order[0])
            n_opts += 1
            steps_in_opt = 0
            chosen_completed = False
            while steps_in_opt < config.manager.subgoal_horizon and not done and not chosen_completed:
                chunk = _onestep_chunk(agent, z, state, task_id, noise_scale)
                z_t = np.asarray(z, dtype=np.float32).copy()
                state_t = np.asarray(state, dtype=np.float64).copy()
                completion_t = completion.astype(np.float32).copy()
                R_vec = np.zeros(agent.n_tasks, dtype=np.float32)
                task_done_vec = np.zeros(agent.n_tasks, dtype=np.float32)
                nstep, skill_done, env_done = 0, False, False
                for h in range(agent.H_chunk):
                    if steps_in_opt >= config.manager.subgoal_horizon:
                        break
                    a_step = chunk[h]
                    next_img, _env_reward, done_env, info = env.step(a_step)
                    next_state = np.asarray(info["state"], dtype=np.float64)
                    names = info.get("tasks_completed_names", [])
                    raw_completion = agent.spec.completion_mask_from_names(names)
                    newly_vec = np.maximum(raw_completion - completion, 0.0)
                    for k in range(agent.n_tasks):
                        err_before = agent.spec.task_error(state, k)
                        err_after = agent.spec.task_error(next_state, k)
                        R_vec[k] += (gamma ** nstep) * agent._worker_step_reward(
                            err_before, err_after, a_step, bool(newly_vec[k] > 0.5), k)
                    task_done_vec = np.maximum(task_done_vec, newly_vec.astype(np.float32))
                    nstep += 1
                    completion = np.maximum(completion, raw_completion)
                    state = next_state
                    z = agent.encoder.encode_numpy(next_img).squeeze()
                    env_steps += 1
                    steps_in_opt += 1
                    agent.total_env_steps += 1
                    if completion[task_id] > 0.5:
                        skill_done = True
                        chosen_completed = True
                    if bool(done_env):
                        env_done = True
                        done = True
                    if skill_done or env_done:
                        break
                # Store only the actions actually executed, padded to H by
                # repeating the last executed step. This matches the offline
                # _chunk_actions padding so Q(s, a_chunk) sees the same action
                # layout on truncated (skill-completing / terminal) chunks; the
                # proposed-but-never-stepped tail must not enter the critic.
                n_exec = max(1, nstep)
                exec_chunk = chunk[:n_exec]
                if n_exec < agent.H_chunk:
                    pad = np.repeat(exec_chunk[-1:], agent.H_chunk - n_exec, axis=0)
                    exec_chunk = np.concatenate([exec_chunk, pad], axis=0)
                replay.add({
                    "z": z_t,
                    "proprio_raw": state_t.astype(np.float32),
                    "action": exec_chunk.reshape(-1).astype(np.float32),
                    "reward_vec": R_vec.astype(np.float32),
                    "task_done_vec": task_done_vec.astype(np.float32),
                    "task_complete_vec": completion_t.astype(np.float32),
                    "env_done": np.float32(1.0 if env_done else 0.0),
                    "nstep": np.float32(max(1, nstep)),
                    "z_next": np.asarray(z, dtype=np.float32).copy(),
                    "proprio_next_raw": np.asarray(state, dtype=np.float32).copy(),
                })
    finally:
        env.close()

    done_count = int(completion.sum())
    return {
        "env_steps": float(env_steps),
        "tasks_done": float(done_count),
        "full_success": float(done_count >= agent.n_tasks),
    }


def run_online_finetuning(agent: SkillAgent,
                          config: Config,
                          writer=None,
                          evaluate_fn: Optional[Callable] = None,
                          initial_eval: Optional[Dict[str, float]] = None,
                          verbose: bool = True) -> Dict[str, float]:
    if agent.demo_dataset is None:
        raise RuntimeError("Online fine-tuning requires agent.demo_dataset for mixed replay.")
    ds = agent.demo_dataset
    replay = OnlineReplay(agent.n_tasks, config.online.online_buffer_capacity_per_skill)

    total_steps, episode, n_updates = 0, 0, 0
    update_credit = 0.0
    last_eval_step = 0
    rollback_count = 0
    batch_size = int(config.online.batch_size)
    demo_fraction = float(config.online.demo_fraction)
    tol = float(config.online.rollback_drop_tolerance)
    baseline_full = float(initial_eval.get("eval/full_task_success_rate", 0.0)) if initial_eval else 0.0
    best_full = baseline_full
    ckpt_dir = os.path.join(config.training.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "checkpoint_online_best.pt")
    agent.save(best_path)

    if verbose:
        print("=" * 76)
        print("  STAGE B  --  Online QC-FQL fine-tuning (mixed demo + online replay)")
        print("-" * 76)
        print(f"  Baseline full SR : {baseline_full * 100:.1f}%")
        print(f"  updates/env_step={config.online.updates_per_env_step}  demo_fraction={demo_fraction}  "
              f"exploration_noise={config.online.exploration_noise}")

    recent: List[Dict[str, float]] = []
    while total_steps < int(config.online.total_env_steps):
        ep = collect_chain_episode(agent, config, replay,
                                   episode_seed=config.training.seed + 50_000 + episode)
        episode += 1
        total_steps += int(ep["env_steps"])
        recent.append(ep)
        if len(recent) > max(1, int(config.online.log_interval_episodes)):
            recent.pop(0)

        update_credit += float(config.online.updates_per_env_step) * ep["env_steps"]
        last_metrics: Dict[str, float] = {}
        while update_credit >= 1.0:
            update_credit -= 1.0
            if len(replay) <= 0:
                break
            batch = _mixed_batch(agent, ds, replay, batch_size, demo_fraction)
            critic_batch = _mixed_batch(agent, ds, replay, batch_size, demo_fraction)
            last_metrics = agent.qc_fql_step(batch, critic_batch)
            n_updates += 1

        if writer is not None:
            writer.add_scalar("online/tasks_done", float(ep["tasks_done"]), total_steps)
            writer.add_scalar("online/episode_full_success", float(ep["full_success"]), total_steps)
            writer.add_scalar("online/replay_total", float(len(replay)), total_steps)
            for key, value in last_metrics.items():
                writer.add_scalar(f"online/{key}", float(value), total_steps)
        if verbose and episode % max(1, int(config.online.log_interval_episodes)) == 0:
            mean_done = float(np.mean([r["tasks_done"] for r in recent]))
            print(f"  step {total_steps:,}/{config.online.total_env_steps}  ep={episode}  "
                  f"updates={n_updates:,}  replay={len(replay):,}  recent_tasks_done={mean_done:.2f}/{agent.n_tasks}")

        if (evaluate_fn is not None and int(config.online.eval_interval_steps) > 0
                and (total_steps - last_eval_step) >= int(config.online.eval_interval_steps)):
            last_eval_step = total_steps
            with preserve_rng_state():
                eval_stats = evaluate_fn(agent, config, config.eval.n_eval_episodes, record_dir=None)
                full = float(eval_stats["eval/full_task_success_rate"])
                if full > best_full or full < best_full - tol:
                    confirm = evaluate_fn(agent, config, config.eval.n_eval_episodes, record_dir=None)
                    full = 0.5 * (full + float(confirm["eval/full_task_success_rate"]))
            if writer is not None:
                writer.add_scalar("online_eval/eval/full_task_success_rate", full, total_steps)
                writer.add_scalar("online_eval/eval/mean_tasks_completed",
                                  float(eval_stats["eval/mean_tasks_completed"]), total_steps)
            if verbose:
                print("=" * 76)
                print(f"  ONLINE EVAL step={total_steps:,}: full={full*100:5.1f}%  "
                      f"tasks={eval_stats['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
                print("=" * 76)
            if full > best_full:
                best_full = full
                agent.save(best_path)
            elif full < best_full - tol:
                rollback_count += 1
                if verbose:
                    print(f"  [Rollback] confirmed full SR {full*100:.1f}% vs best "
                          f"{best_full*100:.1f}%; restoring best checkpoint.")
                agent.load(best_path)
                agent.reset_optimizers()

    final_path = os.path.join(ckpt_dir, "checkpoint_online_final.pt")
    agent.save(final_path)
    if os.path.isfile(best_path):
        agent.load(best_path)
        if verbose:
            print(f"  [Stage B] Restoring best checkpoint (best_full={best_full*100:.1f}%).")

    return {
        "online/env_steps": float(total_steps),
        "online/episodes": float(episode),
        "online/updates": float(n_updates),
        "online/replay_total": float(len(replay)),
        "online/best_full_success_rate": float(max(best_full, 0.0)),
        "online/rollback_count": float(rollback_count),
    }
