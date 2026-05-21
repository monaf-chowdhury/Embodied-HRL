"""Conservative chain-context online fine-tuning.

This module keeps the high-level controller scripted and only fine-tunes the
per-skill workers. Online data is collected under the chained state
distribution induced by earlier skills, then mixed with demo replay for
AWAC-style conservative updates.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from specialist import SkillAgent


class OnlineSkillReplay:
    def __init__(self, n_tasks: int, capacity_per_skill: int):
        self.n_tasks = int(n_tasks)
        self.capacity = int(capacity_per_skill)
        self.storage: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(self.n_tasks)]

    def __len__(self) -> int:
        return int(sum(len(buf) for buf in self.storage))

    def task_size(self, task_id: int) -> int:
        return len(self.storage[int(task_id)])

    def add(self,
            task_id: int,
            z: np.ndarray,
            proprio: np.ndarray,
            task_target: np.ndarray,
            task_cur: np.ndarray,
            task_mask: np.ndarray,
            action: np.ndarray,
            reward: float,
            z_next: np.ndarray,
            proprio_next: np.ndarray,
            task_cur_next: np.ndarray,
            done: float):
        task_id = int(task_id)
        item = {
            "z": np.asarray(z, dtype=np.float16),
            "proprio": np.asarray(proprio, dtype=np.float32),
            "task_target": np.asarray(task_target, dtype=np.float32),
            "task_cur": np.asarray(task_cur, dtype=np.float32),
            "task_mask": np.asarray(task_mask, dtype=np.float32),
            "task_id": np.asarray(task_id, dtype=np.int64),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.asarray(float(reward), dtype=np.float32),
            "z_next": np.asarray(z_next, dtype=np.float16),
            "proprio_next": np.asarray(proprio_next, dtype=np.float32),
            "task_cur_next": np.asarray(task_cur_next, dtype=np.float32),
            "done": np.asarray(float(done), dtype=np.float32),
        }
        buf = self.storage[task_id]
        if len(buf) >= self.capacity:
            del buf[0]
        buf.append(item)

    def sample_task_batch(self, task_id: int, batch_size: int, proprio_normalizer=None) -> Dict[str, np.ndarray]:
        task_id = int(task_id)
        buf = self.storage[task_id]
        if not buf:
            raise RuntimeError(f"No online samples for task id={task_id}.")
        idx = np.random.choice(len(buf), size=int(batch_size), replace=True)
        items = [buf[int(i)] for i in idx]
        out = {}
        for key in items[0].keys():
            out[key] = np.stack([item[key] for item in items], axis=0)
        out["z"] = out["z"].astype(np.float32)
        out["z_next"] = out["z_next"].astype(np.float32)
        out["task_id"] = out["task_id"].astype(np.int64)
        if proprio_normalizer is not None:
            out["proprio"] = np.stack([proprio_normalizer(row) for row in out["proprio"]], axis=0).astype(np.float32)
            out["proprio_next"] = np.stack(
                [proprio_normalizer(row) for row in out["proprio_next"]], axis=0
            ).astype(np.float32)
        return out


def _concat_batches(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not a:
        return b
    if not b:
        return a
    return {key: np.concatenate([a[key], b[key]], axis=0) for key in a.keys()}


def _mixed_task_batch(agent: SkillAgent,
                      online_replay: OnlineSkillReplay,
                      task_id: int,
                      batch_size: int,
                      demo_fraction: float) -> Dict[str, np.ndarray]:
    online_n = online_replay.task_size(task_id)
    if online_n <= 0:
        return agent.demo_dataset.sample_worker_task_batch(
            task_id, batch_size, proprio_normalizer=agent.normalize_proprio
        )
    n_demo = int(round(batch_size * float(demo_fraction)))
    n_demo = min(max(0, n_demo), batch_size)
    n_online = batch_size - n_demo
    if n_online <= 0:
        n_online = 1
        n_demo = batch_size - 1
    demo_batch = {}
    if n_demo > 0:
        demo_batch = agent.demo_dataset.sample_worker_task_batch(
            task_id, n_demo, proprio_normalizer=agent.normalize_proprio
        )
    online_batch = online_replay.sample_task_batch(
        task_id, n_online, proprio_normalizer=agent.normalize_proprio
    )
    return _concat_batches(demo_batch, online_batch)


def _demo_fraction(config: Config, env_steps: int) -> float:
    start = float(config.online.demo_fraction_start)
    end = float(config.online.demo_fraction_end)
    decay = max(1, int(config.online.demo_fraction_decay_steps))
    frac = min(max(float(env_steps) / decay, 0.0), 1.0)
    return start + frac * (end - start)


def _task_order(agent: SkillAgent, config: Config) -> List[int]:
    if config.training.controller_order_mode == "stage_a_rank":
        return list(agent.curriculum_task_order)
    return list(range(agent.n_tasks))


def _choose_update_task(agent: SkillAgent,
                        online_replay: OnlineSkillReplay,
                        failure_ema: np.ndarray) -> Optional[int]:
    available = np.array(
        [k for k in range(agent.n_tasks) if online_replay.task_size(k) > 0],
        dtype=np.int64,
    )
    if available.size == 0:
        return None
    weights = 1.0 + float(agent.config.online.failure_priority) * failure_ema[available]
    weights = weights.astype(np.float64)
    weights /= np.sum(weights)
    return int(np.random.choice(available, p=weights))


def collect_chain_episode(agent: SkillAgent,
                          config: Config,
                          online_replay: OnlineSkillReplay,
                          episode_seed: int) -> Dict[str, object]:
    order = _task_order(agent, config)
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
        terminate_on_tasks_completed=True,
    )
    task_attempted = np.zeros(agent.n_tasks, dtype=np.float32)
    task_completed = np.zeros(agent.n_tasks, dtype=np.float32)
    terminations: Dict[str, int] = {}
    try:
        img, state = env.reset(seed=episode_seed)
        z = agent.encoder.encode_numpy(img).squeeze()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)
        done = False
        n_opts = 0
        env_steps = 0
        ep_reward = 0.0
        chosen_successes = 0

        while not done and completion.sum() < agent.n_tasks and n_opts < config.manager.max_high_level_steps:
            remaining = [k for k in order if completion[k] < 0.5]
            task_id = int(remaining[0]) if remaining else int(order[0])
            task_attempted[task_id] = 1.0
            chosen_name = agent.spec.name(task_id)
            option_steps = 0
            option_done = False
            option_reason = "budget"

            while option_steps < config.manager.subgoal_horizon and not done and not option_done:
                chunk_start_z = z.copy()
                chunk_start_state = state.copy()
                chunk_start_completion = completion.copy()
                task_target = agent.spec.padded_goal_for(task_id)
                task_cur = agent.spec.padded_state_slice_for(chunk_start_state, task_id)
                task_mask = agent.spec.padded_mask_for(task_id)

                chunk = agent.get_worker_chunk(
                    z, state, state, task_id, deterministic=True
                ).astype(np.float32)
                if config.online.exploration_noise > 0:
                    chunk = chunk + np.random.normal(
                        0.0,
                        float(config.online.exploration_noise),
                        size=chunk.shape,
                    ).astype(np.float32)
                    chunk = np.clip(chunk, -1.0, 1.0)

                executed = chunk.copy()
                chunk_reward = 0.0
                chunk_steps = 0
                chunk_completed = False
                for h in range(agent.H_chunk):
                    if option_steps >= config.manager.subgoal_horizon:
                        break
                    action_step = executed[h]
                    next_img, env_reward, done_env, info = env.step(action_step)
                    next_state = np.asarray(info["state"], dtype=np.float64)
                    next_z = agent.encoder.encode_numpy(next_img).squeeze()
                    raw_completion = agent.spec.completion_mask_from_names(info.get("tasks_completed_names", []))
                    completion_next = np.maximum(completion, raw_completion)
                    just_completed = (completion_next > 0.5) & (completion < 0.5)
                    chosen_completed = bool(just_completed[task_id] > 0.5)

                    err_before = agent.spec.task_error(state, task_id)
                    err_after = agent.spec.task_error(next_state, task_id)
                    step_reward = agent._worker_step_reward(
                        err_before,
                        err_after,
                        action_step,
                        completion_bit_flipped=chosen_completed,
                    )
                    chunk_reward += float(step_reward)
                    ep_reward += float(env_reward)

                    state = next_state
                    z = next_z
                    completion = completion_next
                    img = next_img
                    done = bool(done_env)
                    env_steps += 1
                    option_steps += 1
                    chunk_steps += 1

                    if chosen_completed:
                        chunk_completed = True
                        option_done = True
                        option_reason = "completed"
                        chosen_successes += 1
                        task_completed[task_id] = 1.0
                        break
                    if done:
                        option_done = True
                        option_reason = "env_done"
                        break

                if chunk_steps <= 0:
                    break
                if chunk_steps < agent.H_chunk:
                    executed[chunk_steps:] = executed[chunk_steps - 1]
                transition_done = float(done or chunk_completed)
                online_replay.add(
                    task_id=task_id,
                    z=chunk_start_z,
                    proprio=chunk_start_state,
                    task_target=task_target,
                    task_cur=task_cur,
                    task_mask=task_mask,
                    action=executed.reshape(-1),
                    reward=chunk_reward,
                    z_next=z,
                    proprio_next=state,
                    task_cur_next=agent.spec.padded_state_slice_for(state, task_id),
                    done=transition_done,
                )

            n_opts += 1
            terminations[option_reason] = terminations.get(option_reason, 0) + 1

        done_count = int(completion.sum())
        return {
            "env_steps": env_steps,
            "options": n_opts,
            "tasks_done": done_count,
            "any_success": float(done_count >= 1),
            "full_success": float(done_count >= agent.n_tasks),
            "chosen_sr": float(chosen_successes / max(n_opts, 1)),
            "env_reward": float(ep_reward),
            "task_attempted": task_attempted,
            "task_completed": task_completed,
            "terminations": terminations,
        }
    finally:
        env.close()


def run_online_finetuning(agent: SkillAgent,
                          config: Config,
                          writer=None,
                          evaluate_fn: Optional[Callable] = None,
                          verbose: bool = True) -> Dict[str, float]:
    if agent.demo_dataset is None:
        raise RuntimeError("Online fine-tuning requires agent.demo_dataset for demo replay.")

    replay = OnlineSkillReplay(
        n_tasks=agent.n_tasks,
        capacity_per_skill=config.online.online_buffer_capacity_per_skill,
    )
    failure_ema = np.clip(1.0 - np.asarray(agent.stage_a_task_success, dtype=np.float32), 0.0, 1.0)
    total_steps = 0
    episode = 0
    update_credit = 0.0
    n_updates = 0
    best_full = -1.0
    recent: List[Dict[str, object]] = []
    last_eval_step = 0
    ckpt_dir = os.path.join(config.training.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if verbose:
        print("=" * 76)
        print("  STAGE B  --  Conservative Chain-Context Online AWAC")
        print("-" * 76)

    while total_steps < int(config.online.total_env_steps):
        ep_stats = collect_chain_episode(
            agent,
            config,
            replay,
            episode_seed=config.training.seed + 50_000 + episode,
        )
        episode += 1
        ep_steps = int(ep_stats["env_steps"])
        total_steps += ep_steps
        recent.append(ep_stats)
        if len(recent) > max(1, int(config.online.log_interval_episodes)):
            recent.pop(0)

        attempted = ep_stats["task_attempted"]
        completed = ep_stats["task_completed"]
        for k in range(agent.n_tasks):
            if attempted[k] > 0.5:
                failure = 1.0 - float(completed[k])
                failure_ema[k] = 0.95 * failure_ema[k] + 0.05 * failure
        if writer is not None:
            writer.add_scalar("train/ep_tasks_completed", float(ep_stats["tasks_done"]), total_steps)
            writer.add_scalar("train/ep_env_reward", float(ep_stats["env_reward"]), total_steps)
            writer.add_scalar("train/ep_options", float(ep_stats["options"]), total_steps)
            writer.add_scalar("train/worker_buffer_size", float(len(replay)), total_steps)
            writer.add_scalar("online/episode_full_success", float(ep_stats["full_success"]), total_steps)
            writer.add_scalar("online/episode_any_success", float(ep_stats["any_success"]), total_steps)
            writer.add_scalar("online/episode_chosen_sr", float(ep_stats["chosen_sr"]), total_steps)

        update_credit += ep_steps * float(config.online.updates_per_env_step)
        while update_credit >= 1.0:
            task_id = _choose_update_task(agent, replay, failure_ema)
            if task_id is None:
                break
            demo_frac = _demo_fraction(config, total_steps)
            batch = _mixed_task_batch(
                agent,
                replay,
                task_id,
                int(config.online.batch_size),
                demo_fraction=demo_frac,
            )
            demo_anchor = agent.demo_dataset.sample_worker_task_batch(
                task_id,
                int(config.online.batch_size),
                proprio_normalizer=agent.normalize_proprio,
            )
            metrics = agent.online_awac_step(task_id, batch, demo_anchor)
            n_updates += 1
            update_credit -= 1.0
            if writer is not None and n_updates % max(1, int(config.specialist.log_interval)) == 0:
                safe = agent.tasks[task_id].replace(" ", "_")
                for key, value in metrics.items():
                    writer.add_scalar(f"online/skill/{safe}/{key}", float(value), total_steps)
                writer.add_scalar("online/demo_fraction", float(demo_frac), total_steps)
                writer.add_scalar("online/replay_total", float(len(replay)), total_steps)
                for k, name in enumerate(agent.tasks):
                    writer.add_scalar(f"online/replay/{name.replace(' ', '_')}_size", replay.task_size(k), total_steps)
                    writer.add_scalar(f"online/failure_ema/{name.replace(' ', '_')}", failure_ema[k], total_steps)

        if verbose and episode % max(1, int(config.online.log_interval_episodes)) == 0:
            mean_tasks = np.mean([float(s["tasks_done"]) for s in recent])
            full_sr = np.mean([float(s["full_success"]) for s in recent])
            any_sr = np.mean([float(s["any_success"]) for s in recent])
            mean_opts = np.mean([float(s["options"]) for s in recent])
            print("-" * 76)
            print(f"  Online step {total_steps:,} / {config.online.total_env_steps:,}  "
                  f"episode={episode}  updates={n_updates:,}  replay={len(replay):,}")
            print(f"  Recent: any={any_sr*100:5.1f}%  full={full_sr*100:5.1f}%  "
                  f"tasks={mean_tasks:.2f}/{agent.n_tasks}  options={mean_opts:.1f}")
            print(f"  Demo fraction={_demo_fraction(config, total_steps):.2f}  "
                  f"failure_ema={np.round(failure_ema, 3).tolist()}")

        should_eval = (
            evaluate_fn is not None
            and int(config.online.eval_interval_steps) > 0
            and (total_steps - last_eval_step) >= int(config.online.eval_interval_steps)
        )
        if should_eval:
            last_eval_step = total_steps
            eval_stats = evaluate_fn(agent, config, config.eval.n_eval_episodes, record_dir=None)
            full = float(eval_stats["eval/full_task_success_rate"])
            if writer is not None:
                for key, value in eval_stats.items():
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        writer.add_scalar(f"online_eval/{key}", float(value), total_steps)
            if verbose:
                print("=" * 76)
                print(f"  ONLINE EVAL step={total_steps:,}: full={full*100:5.1f}%  "
                      f"any={eval_stats['eval/any_task_success_rate']*100:5.1f}%  "
                      f"tasks={eval_stats['eval/mean_tasks_completed']:.2f}/{agent.n_tasks}")
                print("=" * 76)
            if full > best_full:
                best_full = full
                agent.save(os.path.join(ckpt_dir, "checkpoint_online_best.pt"))

    agent.save(os.path.join(ckpt_dir, "checkpoint_online_final.pt"))
    return {
        "online/env_steps": float(total_steps),
        "online/episodes": float(episode),
        "online/updates": float(n_updates),
        "online/replay_total": float(len(replay)),
        "online/best_full_success_rate": float(max(best_full, 0.0)),
    }
