"""Conservative online fine-tuning for per-skill policies.

The default online mode is reliability-gated skill repair:
  1. evaluate the scripted chain,
  2. freeze skills whose chain completion is already high,
  3. collect online data only for unreliable skills from prefix-induced states,
  4. train critics on all attempts, but train actors only on demos plus
     successful online attempts.

The older full-chain AWAC collector is still available with
``--online_mode chain`` for ablations.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

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

    def task_size(self,
                  task_id: int,
                  success_only: bool = False,
                  include_high_return_failures: bool = False) -> int:
        buf = self.storage[int(task_id)]
        if not success_only:
            return len(buf)
        return len(self._quality_items(buf, include_high_return_failures=include_high_return_failures))

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
            done: float,
            episode_success: float = 0.0,
            episode_return: float = 0.0):
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
            "episode_success": np.asarray(float(episode_success), dtype=np.float32),
            "episode_return": np.asarray(float(episode_return), dtype=np.float32),
        }
        buf = self.storage[task_id]
        if len(buf) >= self.capacity:
            del buf[0]
        buf.append(item)

    def extend_attempt(self, task_id: int, transitions: Sequence[Dict[str, np.ndarray]], success: bool, ret: float):
        for tr in transitions:
            self.add(task_id=task_id, episode_success=float(success), episode_return=float(ret), **tr)

    def sample_task_batch(self,
                          task_id: int,
                          batch_size: int,
                          proprio_normalizer=None,
                          success_only: bool = False,
                          include_high_return_failures: bool = False) -> Dict[str, np.ndarray]:
        task_id = int(task_id)
        buf = self.storage[task_id]
        if success_only:
            buf = self._quality_items(buf, include_high_return_failures=include_high_return_failures)
        if not buf:
            kind = "successful/high-return online" if success_only else "online"
            raise RuntimeError(f"No {kind} samples for task id={task_id}.")
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

    @staticmethod
    def _quality_items(buf: Sequence[Dict[str, np.ndarray]],
                       include_high_return_failures: bool = False) -> List[Dict[str, np.ndarray]]:
        if not buf:
            return []
        successes = [item for item in buf if float(item.get("episode_success", 0.0)) > 0.5]
        if not include_high_return_failures:
            return successes
        returns = np.asarray([float(item.get("episode_return", 0.0)) for item in buf], dtype=np.float32)
        threshold = float(np.quantile(returns, 0.75)) if returns.size else 0.0
        high_return = [
            item for item in buf
            if float(item.get("episode_return", 0.0)) > 0.0
            and float(item.get("episode_return", 0.0)) >= threshold
        ]
        success_ids = {id(item) for item in successes}
        selected = successes + [item for item in high_return if id(item) not in success_ids]
        return selected


def _concat_batches(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not a:
        return b
    if not b:
        return a
    keys = [k for k in a.keys() if k in b]
    return {key: np.concatenate([a[key], b[key]], axis=0) for key in keys}


def _demo_fraction(config: Config, env_steps: int) -> float:
    start = float(config.online.demo_fraction_start)
    end = float(config.online.demo_fraction_end)
    decay = max(1, int(config.online.demo_fraction_decay_steps))
    frac = min(max(float(env_steps) / decay, 0.0), 1.0)
    return start + frac * (end - start)


def _bc_anchor_weight(config: Config, env_steps: int) -> float:
    start = float(config.online.bc_anchor_weight)
    end = float(config.online.bc_anchor_weight_end)
    decay = max(1, int(config.online.bc_anchor_decay_steps))
    frac = min(max(float(env_steps) / decay, 0.0), 1.0)
    return start + frac * (end - start)


def _task_order(agent: SkillAgent, config: Config) -> List[int]:
    if config.training.controller_order_mode == "stage_a_rank":
        return list(agent.curriculum_task_order)
    return list(range(agent.n_tasks))


def _demo_batch(agent: SkillAgent, task_id: int, batch_size: int) -> Dict[str, np.ndarray]:
    return agent.demo_dataset.sample_worker_task_batch(
        task_id, int(batch_size), proprio_normalizer=agent.normalize_proprio
    )


def _mixed_task_batch(agent: SkillAgent,
                      online_replay: OnlineSkillReplay,
                      task_id: int,
                      batch_size: int,
                      demo_fraction: float,
                      success_only: bool = False,
                      include_high_return_failures: bool = False) -> Dict[str, np.ndarray]:
    online_n = online_replay.task_size(
        task_id,
        success_only=success_only,
        include_high_return_failures=include_high_return_failures,
    )
    if online_n <= 0:
        return _demo_batch(agent, task_id, batch_size)
    n_demo = int(round(batch_size * float(demo_fraction)))
    n_demo = min(max(0, n_demo), batch_size)
    n_online = batch_size - n_demo
    if n_online <= 0:
        n_online = 1
        n_demo = batch_size - 1
    demo_batch = _demo_batch(agent, task_id, n_demo) if n_demo > 0 else {}
    online_batch = online_replay.sample_task_batch(
        task_id,
        n_online,
        proprio_normalizer=agent.normalize_proprio,
        success_only=success_only,
        include_high_return_failures=include_high_return_failures,
    )
    return _concat_batches(demo_batch, online_batch)


def _choose_update_task(agent: SkillAgent,
                        online_replay: OnlineSkillReplay,
                        trainable_task_ids: Sequence[int],
                        failure_ema: np.ndarray) -> Optional[int]:
    available = np.array(
        [k for k in trainable_task_ids if online_replay.task_size(k) > 0],
        dtype=np.int64,
    )
    if available.size == 0:
        return None
    weights = 1.0 + float(agent.config.online.failure_priority) * failure_ema[available]
    weights = weights.astype(np.float64)
    weights /= np.sum(weights)
    return int(np.random.choice(available, p=weights))


def _chain_completion_rates(agent: SkillAgent, initial_eval: Optional[Dict[str, float]]) -> np.ndarray:
    rates = np.asarray(agent.stage_a_task_success, dtype=np.float32).copy()
    if initial_eval is None:
        return rates
    for k, name in enumerate(agent.tasks):
        safe = name.replace(" ", "_")
        key = f"eval/task/{safe}_completion_rate"
        if key in initial_eval:
            rates[k] = float(initial_eval[key])
    return rates


def _repair_task_sets(agent: SkillAgent,
                      config: Config,
                      initial_eval: Optional[Dict[str, float]]) -> tuple[List[int], List[int], np.ndarray]:
    rates = _chain_completion_rates(agent, initial_eval)
    threshold = float(config.online.freeze_success_threshold)
    frozen = [k for k, rate in enumerate(rates) if rate >= threshold]
    trainable = [k for k in range(agent.n_tasks) if k not in frozen]
    if not trainable:
        # Keep the best-performing skill frozen and allow the weakest one to update
        # so that an online run remains well-defined on nearly solved tasks.
        weakest = int(np.argmin(rates))
        trainable = [weakest]
        frozen = [k for k in range(agent.n_tasks) if k != weakest]
    return frozen, trainable, rates


def _frontier_task_ids(agent: SkillAgent,
                       config: Config,
                       frozen_task_ids: Sequence[int],
                       trainable_task_ids: Sequence[int]) -> List[int]:
    """Return trainable skills whose scripted prefix is already reliable."""
    order = _task_order(agent, config)
    frozen = set(int(k) for k in frozen_task_ids)
    trainable = set(int(k) for k in trainable_task_ids)
    frontier: List[int] = []
    prefix_reliable = True
    for task_id in order:
        task_id = int(task_id)
        if task_id in trainable and prefix_reliable:
            frontier.append(task_id)
        if task_id not in frozen:
            prefix_reliable = False
    if frontier:
        return frontier
    return [int(k) for k in trainable_task_ids]


def _next_trainable_task(order: Sequence[int],
                         trainable_task_ids: Sequence[int],
                         task_id: int) -> Optional[int]:
    trainable = set(int(k) for k in trainable_task_ids)
    try:
        start = list(order).index(int(task_id)) + 1
    except ValueError:
        return None
    for nxt in list(order)[start:]:
        nxt = int(nxt)
        if nxt in trainable:
            return nxt
    return None


def _active_repair_task_ids(agent: SkillAgent,
                            config: Config,
                            frozen_task_ids: Sequence[int],
                            trainable_task_ids: Sequence[int],
                            failure_ema: np.ndarray,
                            online_replay: Optional[OnlineSkillReplay] = None) -> List[int]:
    """Return skills allowed to receive online gradient updates.

    Collection still starts from the frontier skill, but once a frontier skill
    has non-trivial recent success we also update the next trainable skill from
    successful-prefix attempts. This avoids starving the next skill of data.
    """
    order = _task_order(agent, config)
    active = list(_frontier_task_ids(agent, config, frozen_task_ids, trainable_task_ids))
    threshold = float(config.online.next_skill_collection_threshold)
    for task_id in list(active):
        recent_success = 1.0 - float(failure_ema[int(task_id)])
        if recent_success < threshold:
            if online_replay is None:
                continue
            nxt_probe = _next_trainable_task(order, trainable_task_ids, int(task_id))
            has_next_data = (
                nxt_probe is not None
                and online_replay.task_size(int(nxt_probe)) > 0
            )
            if not has_next_data:
                continue
        nxt = _next_trainable_task(order, trainable_task_ids, int(task_id))
        if nxt is not None and nxt not in active:
            active.append(int(nxt))
    if online_replay is not None:
        for task_id in trainable_task_ids:
            task_id = int(task_id)
            if task_id not in active and online_replay.task_size(task_id) > 0:
                active.append(task_id)
    return active


def _make_transition(agent: SkillAgent,
                     task_id: int,
                     z: np.ndarray,
                     state: np.ndarray,
                     action_chunk: np.ndarray,
                     reward: float,
                     z_next: np.ndarray,
                     state_next: np.ndarray,
                     done: float) -> Dict[str, np.ndarray]:
    return {
        "z": z,
        "proprio": state,
        "task_target": agent.spec.padded_goal_for(task_id),
        "task_cur": agent.spec.padded_state_slice_for(state, task_id),
        "task_mask": agent.spec.padded_mask_for(task_id),
        "action": action_chunk.reshape(-1),
        "reward": float(reward),
        "z_next": z_next,
        "proprio_next": state_next,
        "task_cur_next": agent.spec.padded_state_slice_for(state_next, task_id),
        "done": float(done),
    }


def _execute_option_for_collection(agent: SkillAgent,
                                   env: FrankaKitchenImageWrapper,
                                   task_id: int,
                                   img: np.ndarray,
                                   state: np.ndarray,
                                   z: np.ndarray,
                                   completion: np.ndarray,
                                   collect_transitions: bool,
                                   exploration_noise: float) -> Dict[str, object]:
    transitions: List[Dict[str, np.ndarray]] = []
    done = False
    option_steps = 0
    option_return = 0.0
    env_reward_sum = 0.0
    chosen_success = False
    termination = "budget"

    while option_steps < agent.config.manager.subgoal_horizon and not done and not chosen_success:
        start_z = z.copy()
        start_state = state.copy()
        chunk = agent.get_worker_chunk(z, state, state, task_id, deterministic=True).astype(np.float32)
        if exploration_noise > 0:
            chunk = chunk + np.random.normal(0.0, float(exploration_noise), size=chunk.shape).astype(np.float32)
            chunk = np.clip(chunk, -1.0, 1.0)
        executed = chunk.copy()
        chunk_reward = 0.0
        chunk_steps = 0
        chunk_success = False

        for h in range(agent.H_chunk):
            if option_steps >= agent.config.manager.subgoal_horizon:
                break
            action_step = executed[h]
            next_img, env_reward, done_env, info = env.step(action_step)
            next_state = np.asarray(info["state"], dtype=np.float64)
            next_z = agent.encoder.encode_numpy(next_img).squeeze()
            raw_completion = agent.spec.completion_mask_from_names(info.get("tasks_completed_names", []))
            completion_next = np.maximum(completion, raw_completion)
            just_completed = (completion_next > 0.5) & (completion < 0.5)
            task_completed = bool(just_completed[task_id] > 0.5)

            err_before = agent.spec.task_error(state, task_id)
            err_after = agent.spec.task_error(next_state, task_id)
            step_reward = agent._worker_step_reward(
                err_before,
                err_after,
                action_step,
                completion_bit_flipped=task_completed,
                task_id=task_id,
            )
            chunk_reward += float(step_reward)
            option_return += float(step_reward)
            env_reward_sum += float(env_reward)

            img = next_img
            state = next_state
            z = next_z
            completion = completion_next
            done = bool(done_env)
            option_steps += 1
            chunk_steps += 1

            if task_completed:
                chosen_success = True
                chunk_success = True
                termination = "completed"
                break
            if done:
                termination = "env_done"
                break

        if chunk_steps <= 0:
            break
        if chunk_steps < agent.H_chunk:
            executed[chunk_steps:] = executed[chunk_steps - 1]
        if collect_transitions:
            transitions.append(
                _make_transition(
                    agent=agent,
                    task_id=task_id,
                    z=start_z,
                    state=start_state,
                    action_chunk=executed,
                    reward=chunk_reward,
                    z_next=z,
                    state_next=state,
                    done=float(done or chunk_success),
                )
            )

    return {
        "img": img,
        "state": state,
        "z": z,
        "completion": completion,
        "done": done,
        "steps": option_steps,
        "success": bool(chosen_success),
        "return": float(option_return),
        "env_reward": float(env_reward_sum),
        "termination": termination,
        "transitions": transitions,
    }


def collect_repair_episode(agent: SkillAgent,
                           config: Config,
                           online_replay: OnlineSkillReplay,
                           target_task_id: int,
                           episode_seed: int,
                           trainable_task_ids: Optional[Sequence[int]] = None,
                           collect_next_on_success: bool = False) -> Dict[str, object]:
    order = _task_order(agent, config)
    trainable = set(range(agent.n_tasks) if trainable_task_ids is None else [int(k) for k in trainable_task_ids])
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
        env_steps = 0
        n_opts = 0
        ep_reward = 0.0
        prefix_ok = True
        collected_task_ids: List[int] = []

        target_pos = order.index(int(target_task_id)) if int(target_task_id) in order else len(order)
        for pos, task_id in enumerate(order):
            if done or n_opts >= config.manager.max_high_level_steps:
                break
            task_id = int(task_id)
            if completion[task_id] > 0.5:
                continue
            collect = task_id == int(target_task_id)
            if not collect and pos > target_pos:
                break
            task_attempted[task_id] = 1.0
            target_transitions: List[Dict[str, np.ndarray]] = []
            target_return = 0.0
            target_success = False
            while (
                not done
                and n_opts < config.manager.max_high_level_steps
                and completion[task_id] < 0.5
            ):
                result = _execute_option_for_collection(
                    agent=agent,
                    env=env,
                    task_id=task_id,
                    img=img,
                    state=state,
                    z=z,
                    completion=completion,
                    collect_transitions=collect,
                    exploration_noise=float(config.online.exploration_noise) if collect else 0.0,
                )
                img = result["img"]
                state = result["state"]
                z = result["z"]
                completion = result["completion"]
                done = bool(result["done"])
                env_steps += int(result["steps"])
                ep_reward += float(result["env_reward"])
                n_opts += 1
                terminations[str(result["termination"])] = terminations.get(str(result["termination"]), 0) + 1
                if collect:
                    target_transitions.extend(result["transitions"])
                    target_return += float(result["return"])
                if bool(result["success"]):
                    task_completed[task_id] = 1.0
                    target_success = True
                    break
                if int(result["steps"]) <= 0:
                    break

            if completion[task_id] < 0.5 and not collect:
                prefix_ok = False
                break

            if collect:
                online_replay.extend_attempt(
                    task_id=target_task_id,
                    transitions=target_transitions,
                    success=target_success,
                    ret=target_return,
                )
                collected_task_ids.append(int(target_task_id))
                if (
                    target_success
                    and collect_next_on_success
                    and not done
                    and n_opts < config.manager.max_high_level_steps
                ):
                    next_task_id = _next_trainable_task(order, trainable, task_id)
                    if next_task_id is not None and completion[next_task_id] < 0.5:
                        task_attempted[next_task_id] = 1.0
                        next_transitions: List[Dict[str, np.ndarray]] = []
                        next_return = 0.0
                        next_success = False
                        while (
                            not done
                            and n_opts < config.manager.max_high_level_steps
                            and completion[next_task_id] < 0.5
                        ):
                            result = _execute_option_for_collection(
                                agent=agent,
                                env=env,
                                task_id=next_task_id,
                                img=img,
                                state=state,
                                z=z,
                                completion=completion,
                                collect_transitions=True,
                                exploration_noise=float(config.online.exploration_noise),
                            )
                            img = result["img"]
                            state = result["state"]
                            z = result["z"]
                            completion = result["completion"]
                            done = bool(result["done"])
                            env_steps += int(result["steps"])
                            ep_reward += float(result["env_reward"])
                            n_opts += 1
                            terminations[str(result["termination"])] = (
                                terminations.get(str(result["termination"]), 0) + 1
                            )
                            next_transitions.extend(result["transitions"])
                            next_return += float(result["return"])
                            if bool(result["success"]):
                                task_completed[next_task_id] = 1.0
                                next_success = True
                                break
                            if int(result["steps"]) <= 0:
                                break
                        online_replay.extend_attempt(
                            task_id=next_task_id,
                            transitions=next_transitions,
                            success=next_success,
                            ret=next_return,
                        )
                        collected_task_ids.append(int(next_task_id))
                break

        done_count = int(completion.sum())
        target_success = bool(task_completed[int(target_task_id)] > 0.5)
        return {
            "env_steps": env_steps,
            "options": n_opts,
            "tasks_done": done_count,
            "any_success": float(done_count >= 1),
            "full_success": float(done_count >= agent.n_tasks),
            "chosen_sr": float(target_success),
            "env_reward": float(ep_reward),
            "target_task": int(target_task_id),
            "target_success": float(target_success),
            "prefix_ok": float(prefix_ok),
            "collected_task_ids": collected_task_ids,
            "task_attempted": task_attempted,
            "task_completed": task_completed,
            "terminations": terminations,
        }
    finally:
        env.close()


def collect_chain_episode(agent: SkillAgent,
                          config: Config,
                          online_replay: OnlineSkillReplay,
                          episode_seed: int,
                          trainable_task_ids: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Legacy full-chain collector. It only stores trainable task attempts."""
    order = _task_order(agent, config)
    trainable = set(range(agent.n_tasks) if trainable_task_ids is None else trainable_task_ids)
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
            collect = task_id in trainable
            result = _execute_option_for_collection(
                agent=agent,
                env=env,
                task_id=task_id,
                img=img,
                state=state,
                z=z,
                completion=completion,
                collect_transitions=collect,
                exploration_noise=float(config.online.exploration_noise) if collect else 0.0,
            )
            img = result["img"]
            state = result["state"]
            z = result["z"]
            completion = result["completion"]
            done = bool(result["done"])
            env_steps += int(result["steps"])
            ep_reward += float(result["env_reward"])
            n_opts += 1
            terminations[str(result["termination"])] = terminations.get(str(result["termination"]), 0) + 1
            if bool(result["success"]):
                chosen_successes += 1
                task_completed[task_id] = 1.0
            if collect:
                online_replay.extend_attempt(
                    task_id=task_id,
                    transitions=result["transitions"],
                    success=bool(result["success"]),
                    ret=float(result["return"]),
                )

        done_count = int(completion.sum())
        return {
            "env_steps": env_steps,
            "options": n_opts,
            "tasks_done": done_count,
            "any_success": float(done_count >= 1),
            "full_success": float(done_count >= agent.n_tasks),
            "chosen_sr": float(chosen_successes / max(n_opts, 1)),
            "env_reward": float(ep_reward),
            "target_task": -1,
            "target_success": 0.0,
            "prefix_ok": 1.0,
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
                          initial_eval: Optional[Dict[str, float]] = None,
                          verbose: bool = True) -> Dict[str, float]:
    if agent.demo_dataset is None:
        raise RuntimeError("Online fine-tuning requires agent.demo_dataset for demo replay.")

    replay = OnlineSkillReplay(
        n_tasks=agent.n_tasks,
        capacity_per_skill=config.online.online_buffer_capacity_per_skill,
    )
    frozen_task_ids, trainable_task_ids, chain_rates = _repair_task_sets(agent, config, initial_eval)
    failure_ema = np.clip(1.0 - chain_rates.astype(np.float32), 0.0, 1.0)
    collect_task_ids = _frontier_task_ids(agent, config, frozen_task_ids, trainable_task_ids)
    active_task_ids = _active_repair_task_ids(
        agent, config, frozen_task_ids, trainable_task_ids, failure_ema, replay
    )
    total_steps = 0
    episode = 0
    update_credit = 0.0
    n_updates = 0
    baseline_full = float(initial_eval.get("eval/full_task_success_rate", 0.0)) if initial_eval else 0.0
    best_full = baseline_full
    recent: List[Dict[str, object]] = []
    last_eval_step = 0
    ckpt_dir = os.path.join(config.training.log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "checkpoint_online_best.pt")
    agent.save(best_path)
    rollback_count = 0

    if verbose:
        print("=" * 76)
        stage_name = "Reliability-Gated Skill Repair" if config.online.mode == "skill_repair" else "Chain-Context AWAC"
        print(f"  STAGE B  --  Conservative Online AWAC ({stage_name})")
        print("-" * 76)
        print(f"  Frozen skills    : {[agent.tasks[k] for k in frozen_task_ids]}")
        print(f"  Trainable skills : {[agent.tasks[k] for k in trainable_task_ids]}")
        print(f"  Collect frontier : {[agent.tasks[k] for k in collect_task_ids]}")
        print(f"  Active updates   : {[agent.tasks[k] for k in active_task_ids]}")
        print(f"  Chain rates      : {np.round(chain_rates, 3).tolist()}")
        print(f"  Baseline full SR : {baseline_full * 100:.1f}%")

    while total_steps < int(config.online.total_env_steps):
        if config.online.mode == "skill_repair":
            weights = 1.0 + float(config.online.failure_priority) * failure_ema[collect_task_ids]
            weights = weights.astype(np.float64)
            weights /= np.sum(weights)
            target_task = int(np.random.choice(np.asarray(collect_task_ids, dtype=np.int64), p=weights))
            collect_next = (
                bool(config.online.collect_next_on_success)
                or (
                    1.0 - float(failure_ema[target_task])
                    >= float(config.online.next_skill_collection_threshold)
                )
            )
            ep_stats = collect_repair_episode(
                agent,
                config,
                replay,
                target_task_id=target_task,
                episode_seed=config.training.seed + 50_000 + episode,
                trainable_task_ids=trainable_task_ids,
                collect_next_on_success=collect_next,
            )
        else:
            ep_stats = collect_chain_episode(
                agent,
                config,
                replay,
                episode_seed=config.training.seed + 50_000 + episode,
                trainable_task_ids=trainable_task_ids,
            )

        episode += 1
        ep_steps = int(ep_stats["env_steps"])
        total_steps += ep_steps
        recent.append(ep_stats)
        if len(recent) > max(1, int(config.online.log_interval_episodes)):
            recent.pop(0)

        attempted = ep_stats["task_attempted"]
        completed = ep_stats["task_completed"]
        for k in trainable_task_ids:
            if attempted[k] > 0.5:
                failure = 1.0 - float(completed[k])
                failure_ema[k] = 0.95 * failure_ema[k] + 0.05 * failure
        active_task_ids = _active_repair_task_ids(
            agent, config, frozen_task_ids, trainable_task_ids, failure_ema, replay
        )
        if writer is not None:
            writer.add_scalar("train/ep_tasks_completed", float(ep_stats["tasks_done"]), total_steps)
            writer.add_scalar("train/ep_env_reward", float(ep_stats["env_reward"]), total_steps)
            writer.add_scalar("train/ep_options", float(ep_stats["options"]), total_steps)
            writer.add_scalar("train/worker_buffer_size", float(len(replay)), total_steps)
            writer.add_scalar("online/episode_full_success", float(ep_stats["full_success"]), total_steps)
            writer.add_scalar("online/episode_any_success", float(ep_stats["any_success"]), total_steps)
            writer.add_scalar("online/episode_chosen_sr", float(ep_stats["chosen_sr"]), total_steps)
            writer.add_scalar("online/episode_prefix_ok", float(ep_stats.get("prefix_ok", 1.0)), total_steps)
            target = int(ep_stats.get("target_task", -1))
            if target >= 0:
                writer.add_scalar(f"online/target_attempt/{agent.tasks[target].replace(' ', '_')}", 1.0, total_steps)
            for collected_id in ep_stats.get("collected_task_ids", []):
                safe_collected = agent.tasks[int(collected_id)].replace(" ", "_")
                writer.add_scalar(f"online/collected_attempt/{safe_collected}", 1.0, total_steps)

        update_credit += ep_steps * float(config.online.updates_per_env_step)
        while update_credit >= 1.0:
            task_id = _choose_update_task(agent, replay, active_task_ids, failure_ema)
            if task_id is None:
                break
            demo_frac = _demo_fraction(config, total_steps)
            bc_anchor = _bc_anchor_weight(config, total_steps)
            critic_batch = _mixed_task_batch(
                agent,
                replay,
                task_id,
                int(config.online.batch_size),
                demo_fraction=demo_frac,
                success_only=False,
            )
            if config.online.actor_success_only:
                success_count = replay.task_size(
                    task_id,
                    success_only=True,
                    include_high_return_failures=bool(config.online.actor_include_high_return_failures),
                )
                if success_count >= int(config.online.min_actor_success_samples):
                    actor_batch = _mixed_task_batch(
                        agent,
                        replay,
                        task_id,
                        int(config.online.batch_size),
                        demo_fraction=demo_frac,
                        success_only=True,
                        include_high_return_failures=bool(config.online.actor_include_high_return_failures),
                    )
                else:
                    actor_batch = _demo_batch(agent, task_id, int(config.online.batch_size))
            else:
                actor_batch = critic_batch
            demo_anchor = _demo_batch(agent, task_id, int(config.online.batch_size))
            metrics = agent.online_awac_step(
                task_id,
                critic_batch,
                demo_anchor,
                actor_batch=actor_batch,
                bc_anchor_weight=bc_anchor,
            )
            n_updates += 1
            update_credit -= 1.0
            if writer is not None and n_updates % max(1, int(config.specialist.log_interval)) == 0:
                safe = agent.tasks[task_id].replace(" ", "_")
                for key, value in metrics.items():
                    writer.add_scalar(f"online/skill/{safe}/{key}", float(value), total_steps)
                writer.add_scalar("online/demo_fraction", float(demo_frac), total_steps)
                writer.add_scalar("online/bc_anchor_weight", float(bc_anchor), total_steps)
                writer.add_scalar("online/replay_total", float(len(replay)), total_steps)
                writer.add_scalar(
                    f"online/replay/{safe}_quality_size",
                    replay.task_size(task_id, success_only=True),
                    total_steps,
                )
                writer.add_scalar(
                    f"online/replay/{safe}_actor_quality_size",
                    replay.task_size(
                        task_id,
                        success_only=True,
                        include_high_return_failures=bool(config.online.actor_include_high_return_failures),
                    ),
                    total_steps,
                )
                for k, name in enumerate(agent.tasks):
                    writer.add_scalar(f"online/replay/{name.replace(' ', '_')}_size", replay.task_size(k), total_steps)
                    writer.add_scalar(f"online/failure_ema/{name.replace(' ', '_')}", failure_ema[k], total_steps)

        if verbose and episode % max(1, int(config.online.log_interval_episodes)) == 0:
            mean_tasks = np.mean([float(s["tasks_done"]) for s in recent])
            full_sr = np.mean([float(s["full_success"]) for s in recent])
            any_sr = np.mean([float(s["any_success"]) for s in recent])
            mean_opts = np.mean([float(s["options"]) for s in recent])
            prefix_ok = np.mean([float(s.get("prefix_ok", 1.0)) for s in recent])
            print("-" * 76)
            print(f"  Online step {total_steps:,} / {config.online.total_env_steps:,}  "
                  f"episode={episode}  updates={n_updates:,}  replay={len(replay):,}")
            print(f"  Recent: any={any_sr*100:5.1f}%  full={full_sr*100:5.1f}%  "
                  f"tasks={mean_tasks:.2f}/{agent.n_tasks}  options={mean_opts:.1f}  "
                  f"prefix_ok={prefix_ok*100:5.1f}%")
            print(f"  Demo fraction={_demo_fraction(config, total_steps):.2f}  "
                  f"BC anchor={_bc_anchor_weight(config, total_steps):.2f}  "
                  f"failure_ema={np.round(failure_ema, 3).tolist()}")
            print(f"  Collect frontier={[agent.tasks[k] for k in collect_task_ids]}  "
                  f"Active updates={[agent.tasks[k] for k in active_task_ids]}")
            print(f"  Replay quality sizes={[replay.task_size(k, success_only=True) for k in range(agent.n_tasks)]}")

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
                agent.save(best_path)
            elif full < best_full - float(config.online.rollback_drop_tolerance):
                rollback_count += 1
                if verbose:
                    print(f"  [Rollback] eval full SR dropped to {full*100:.1f}% "
                          f"from best {best_full*100:.1f}%; restoring best checkpoint.")
                agent.load(best_path)
            frozen_task_ids, trainable_task_ids, chain_rates = _repair_task_sets(agent, config, eval_stats)
            failure_ema = np.maximum(failure_ema, np.clip(1.0 - chain_rates.astype(np.float32), 0.0, 1.0))
            collect_task_ids = _frontier_task_ids(agent, config, frozen_task_ids, trainable_task_ids)
            active_task_ids = _active_repair_task_ids(
                agent, config, frozen_task_ids, trainable_task_ids, failure_ema
            )
            if verbose:
                print(f"  [Repair] frozen={[agent.tasks[k] for k in frozen_task_ids]}  "
                      f"collect={[agent.tasks[k] for k in collect_task_ids]}  "
                      f"active={[agent.tasks[k] for k in active_task_ids]}")

    final_path = os.path.join(ckpt_dir, "checkpoint_online_final.pt")
    agent.save(final_path)

    # Restore the best checkpoint so the caller evaluates the best model,
    # not the final (potentially degraded) model after 200k steps.
    if os.path.isfile(best_path):
        agent.load(best_path)
        if verbose:
            print(f"  [Stage B] Restoring best checkpoint "
                  f"(best_full={best_full*100:.1f}%) for final evaluation.")

    return {
        "online/env_steps": float(total_steps),
        "online/episodes": float(episode),
        "online/updates": float(n_updates),
        "online/replay_total": float(len(replay)),
        "online/best_full_success_rate": float(max(best_full, 0.0)),
        "online/rollback_count": float(rollback_count),
        "online/n_frozen_skills": float(len(frozen_task_ids)),
        "online/n_trainable_skills": float(len(trainable_task_ids)),
    }
