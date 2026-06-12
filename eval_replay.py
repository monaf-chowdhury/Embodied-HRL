"""
eval_replay.py — Direct replay test for the best-vs-final eval gap.

Loads a saved checkpoint and runs the scripted chain eval several times in a
fresh process. This discriminates the three hypotheses for why an online-eval
number (e.g. 70%) does not reproduce after "restoring best checkpoint" (56%):

  Outcome 1: repeats VARY (std > 0)
      -> Evals of an identical policy are non-deterministic (renderer/GPU noise
         compounding chaotically over long rollouts). The online "best" is a
         max over noisy draws and is inflated by selection bias. No restore bug.

  Outcome 2: repeats IDENTICAL and match the logged online-eval number
      -> Eval is deterministic and the checkpoint is faithful; the in-run final
        
         eval itself was perturbed (e.g. video recording side effects).

  Outcome 3: repeats IDENTICAL but do NOT match the logged number
      -> The state evaluated during training differed from the state saved:
         a genuine save/load or in-training-eval contamination bug.

Usage (on the training server):
    python eval_replay.py --checkpoint logs/full_lql_seed0/checkpoints/checkpoint_online_best.pt \
        --encoder dinov3 --repeats 3
"""
from __future__ import annotations

import argparse

import numpy as np

from config import Config
from specialist import SkillAgent
from train import evaluate_scripted_chain


def main():
    parser = argparse.ArgumentParser(description="Replay-eval a saved checkpoint N times.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="dinov3",
                        choices=["r3m", "dinov2", "dinov3"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.encoder.name = args.encoder
    cfg.refresh_encoder_dim()
    cfg.training.device = args.device
    cfg.training.record_video = False

    agent = SkillAgent(cfg)
    agent.load(args.checkpoint)
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Running {args.repeats} x {args.episodes}-episode scripted chain evals "
          f"(identical eval seeds each repeat)\n")

    full_rates = []
    for r in range(args.repeats):
        stats = evaluate_scripted_chain(agent, cfg, args.episodes, record_dir=None)
        full = float(stats["eval/full_task_success_rate"])
        full_rates.append(full)
        per_task = "  ".join(
            f"{name.split()[0]}={stats[f'eval/task/{name.replace(chr(32), chr(95))}_completion_rate']*100:.0f}%"
            for name in cfg.training.tasks_to_complete
        )
        print(f"  repeat {r + 1}: full={full*100:5.1f}%  "
              f"tasks={stats['eval/mean_tasks_completed']:.2f}/4  [{per_task}]")

    arr = np.asarray(full_rates)
    print(f"\n  mean={arr.mean()*100:.1f}%  std={arr.std()*100:.2f}pp  "
          f"range=[{arr.min()*100:.1f}%, {arr.max()*100:.1f}%]")
    if arr.std() > 1e-9:
        print("\n  VERDICT: evals are NON-DETERMINISTIC for an identical policy.")
        print("  Online 'best' numbers are max-of-noisy-draws (selection bias);")
        print("  report mean +/- std of repeated evals of the restored checkpoint instead.")
    else:
        print("\n  VERDICT: evals are deterministic in this process.")
        print("  Compare the number above with the logged ONLINE EVAL for this checkpoint:")
        print("  match    -> in-run final eval was perturbed (check video-recording path);")
        print("  mismatch -> save/load or in-training eval-state bug; investigate agent.save/load.")


if __name__ == "__main__":
    main()
