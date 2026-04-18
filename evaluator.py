"""
evaluator.py — Evaluation routines for SMGW.

Metrics produced (per evaluation run):
  * any_task_success_rate     — fraction of eps where ≥1 task completed
  * full_task_success_rate    — fraction of eps where ALL tasks completed  (MAIN metric)
  * mean_tasks_completed      — average number of tasks completed per ep
  * mean_options_used         — average number of options (manager decisions)
  * mean_chosen_task_success  — how often the chosen task actually got its bit flipped
  * mean_env_reward           — raw env reward per ep (for comparison with baselines)

Chained success is what we really care about: full_task_success_rate.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Optional

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import SMGWAgent
from utils import save_video


def evaluate(agent: SMGWAgent,
             config: Config,
             n_episodes: Optional[int] = None,
             deterministic: bool = True,
             record_dir: Optional[str] = None,
             n_videos: int = 0) -> Dict[str, float]:
    """
    Run `n_episodes` deterministic evaluation episodes.

    If `record_dir` is set, save the first `n_videos` episodes as MP4s.
    """
    n_eps = n_episodes or config.eval.n_eval_episodes
    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    any_success = []
    full_success = []
    tasks_completed = []
    options_used = []
    chosen_task_successes = []
    env_rewards = []
    termination_reasons: Dict[str, int] = {}

    for ep_i in range(n_eps):
        collect_frames = (record_dir is not None and ep_i < n_videos)
        frames_accum = []

        img, state = env.reset()
        z = agent.encoder.encode_numpy(img).squeeze()
        proprio = state.copy()
        completion = np.zeros(agent.n_tasks, dtype=np.float32)

        if collect_frames:
            frames_accum.append(img.copy())

        done = False
        n_options = 0
        chosen_success_count = 0
        ep_env_reward = 0.0

        while (not done
               and n_options < config.manager.max_high_level_steps
               and completion.sum() < agent.n_tasks):

            task_id = agent.select_task(z, proprio, state, completion,
                                        deterministic=deterministic)

            result = agent.execute_option(
                env=env, task_id=task_id,
                start_img=img, start_state=state, start_z=z,
                completion=completion,
                deterministic_worker=deterministic,
                collect_frames=collect_frames,
                train_worker_online=False,
            )

            # Advance episode state from option result
            state = result.proprio_end
            proprio = state
            z = result.z_end
            completion = result.completion_end
            ep_env_reward += result.env_reward_sum
            done = result.env_done
            n_options += 1
            if result.chosen_task_completed:
                chosen_success_count += 1

            reason = result.termination_reason
            termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

            if collect_frames:
                # append frames from this option (skip the first frame to
                # avoid duplicates — it's the same as the previous last frame)
                if result.frames:
                    frames_accum.extend(result.frames[1:])

            # The env doesn't give us an updated "img" field back from
            # execute_option because we pushed frames out separately. To
            # keep the outer loop in sync, re-render:
            img = env.render_image()

        n_done = int(completion.sum())
        any_success.append(n_done >= 1)
        full_success.append(n_done >= agent.n_tasks)
        tasks_completed.append(n_done)
        options_used.append(n_options)
        chosen_task_successes.append(chosen_success_count / max(n_options, 1))
        env_rewards.append(ep_env_reward)

        if collect_frames and frames_accum:
            out_path = os.path.join(record_dir, f"ep_{ep_i:03d}.mp4")
            save_video(frames_accum, out_path, fps=config.training.video_fps)

    env.close()

    return {
        'any_task_success_rate': float(np.mean(any_success)),
        'full_task_success_rate': float(np.mean(full_success)),
        'mean_tasks_completed': float(np.mean(tasks_completed)),
        'mean_options_used': float(np.mean(options_used)),
        'mean_chosen_task_success': float(np.mean(chosen_task_successes)),
        'mean_env_reward': float(np.mean(env_rewards)),
        'std_env_reward': float(np.std(env_rewards)),
        'n_episodes': n_eps,
        'termination_reasons': termination_reasons,
    }
