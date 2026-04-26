"""
single_task_eval.py — per-task evaluation that bypasses the manager.

This is THE diagnostic for Step 2. After BC pretraining, we want to know:
  "If I tell the worker 'do task K' from a fresh env reset, can it do it?"
That is a pure worker-capability question, and it has to work before the
hierarchy can work.

Protocol:
  For each task K in tasks_to_complete:
    Repeat N_eps times:
      1. Reset env.
      2. Fix the manager's choice = K (bypass the manager entirely).
      3. Run the worker deterministically for up to max_steps env steps.
      4. Record whether K's completion bit flipped.
    Report success rate per task.

We also optionally record a video of the first episode per task so you
can watch what the worker actually does.

Expected numbers after BC alone (Step 2 target):
  - microwave      : 40-70% success
  - slide cabinet  : 40-70%
  - light switch   : 20-50%
  - kettle         : 10-40%  (hardest - 7-D pose)

If all four are 0% after BC, something is wrong with the data pipeline.
If microwave is >40% but kettle is 0%, that's normal and expected.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from typing import Dict, List, Optional

from config import Config
from env_wrapper import FrankaKitchenImageWrapper
from agent import SMGWAgent
from utils import save_video


def run_single_task_eval(agent: SMGWAgent,
                         config: Config,
                         n_episodes_per_task: int = 20,
                         max_steps: int = 280,
                         record_videos: bool = True,
                         video_dir: Optional[str] = None,
                         verbose: bool = True) -> Dict[str, Dict]:
    """
    Args:
      n_episodes_per_task : how many fresh resets per task
      max_steps           : hard cap per episode (env defaults to 280)
      record_videos       : save the first episode per task as MP4
      video_dir           : where to save; defaults to <log_dir>/single_task_videos
    Returns:
      {
        'microwave':       {'success': 0.65, 'mean_err_final': 0.13, ...},
        'kettle':          {...},
        ...
        '_overall_success': 0.42,
      }
    """
    if video_dir is None:
        video_dir = os.path.join(config.training.log_dir, "single_task_videos")
    if record_videos:
        os.makedirs(video_dir, exist_ok=True)

    env = FrankaKitchenImageWrapper(
        tasks_to_complete=config.training.tasks_to_complete,
        img_size=config.encoder.img_size,
    )

    results: Dict[str, Dict] = {}
    all_success = []

    for k, task_name in enumerate(agent.tasks):
        successes = 0
        final_errs: List[float] = []
        first_completion_step: List[int] = []

        for ep_i in range(n_episodes_per_task):
            record_this = record_videos and ep_i == 0
            frames = []

            img, state = env.reset()
            z = agent.encoder.encode_numpy(img).squeeze()
            proprio = state.copy()

            err0 = agent.spec.task_error(state, k)
            if record_this:
                frames.append(img.copy())

            completed = False
            completion_step = -1

            H = agent.H_chunk
            step = 0
            while step < max_steps:
                # Ask the worker for a chunk. We pass the chosen task k.
                chunk = agent.get_worker_chunk(z, proprio, state, k,
                                               deterministic=True)
                # Execute the chunk step-by-step
                for h in range(H):
                    if step >= max_steps:
                        break
                    next_img, env_r, done, info = env.step(chunk[h])
                    next_state = np.asarray(info['state'], dtype=np.float64)
                    next_z = agent.encoder.encode_numpy(next_img).squeeze()

                    # Check if task k got its completion bit flipped
                    completed_names = info.get('tasks_completed_names', [])
                    if not completed and (task_name in completed_names):
                        completed = True
                        completion_step = step + 1

                    if record_this:
                        frames.append(next_img)

                    state = next_state
                    proprio = next_state
                    z = next_z
                    step += 1

                    if done:
                        break
                if done or completed:
                    break

            err_final = agent.spec.task_error(state, k)
            final_errs.append(err_final)
            if completed:
                successes += 1
                first_completion_step.append(completion_step)

            if record_this and frames:
                safe_name = task_name.replace(' ', '_')
                out_path = os.path.join(video_dir,
                                        f"{safe_name}_ep0.mp4")
                save_video(frames, out_path,
                           fps=config.training.video_fps)

        sr = successes / max(n_episodes_per_task, 1)
        all_success.extend([1.0] * successes
                           + [0.0] * (n_episodes_per_task - successes))

        results[task_name] = {
            'success_rate': float(sr),
            'n_episodes': n_episodes_per_task,
            'n_successes': successes,
            'mean_err_final': float(np.mean(final_errs)) if final_errs else float('nan'),
            'median_err_final': float(np.median(final_errs)) if final_errs else float('nan'),
            'mean_steps_to_success': float(np.mean(first_completion_step))
                if first_completion_step else float('nan'),
        }

        if verbose:
            print(f"    {task_name:>16s}  SR={sr*100:5.1f}%  "
                  f"({successes}/{n_episodes_per_task})  "
                  f"final_err_mean={np.mean(final_errs):.3f}  "
                  f"median={np.median(final_errs):.3f}")

    env.close()
    overall = float(np.mean(all_success)) if all_success else 0.0
    results['_overall_success'] = overall
    if verbose:
        print(f"    {'OVERALL':>16s}  {overall*100:5.1f}%")
    return results


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run single-task evaluation on a trained / BC-pretrained agent"
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to a .pt checkpoint saved by agent.save()')
    parser.add_argument('--n_eps', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=280)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--video_dir', default=None)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = Config()
    config.training.device = args.device
    config.__post_init__()

    agent = SMGWAgent(config)
    agent.load(args.checkpoint)
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Running single-task eval: {args.n_eps} eps x {len(agent.tasks)} tasks\n")

    results = run_single_task_eval(
        agent, config,
        n_episodes_per_task=args.n_eps,
        max_steps=args.max_steps,
        record_videos=not args.no_video,
        video_dir=args.video_dir,
    )
    print("\nResults:")
    for k, v in results.items():
        if k.startswith('_'):
            continue
        print(f"  {k}: {v}")
    print(f"\nOverall success rate: {results['_overall_success']*100:.1f}%")