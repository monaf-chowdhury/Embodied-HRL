"""
test_action_convention.py — Verify that Minari D4RL kitchen actions replay
correctly in gymnasium-robotics FrankaKitchen-v1.

Run:
    python test_action_convention.py

What it does:
  1. Loads the first episode from D4RL/kitchen/complete-v2.
  2. Sets the initial MuJoCo state in FrankaKitchen-v1 from the demo's stored
     qpos/qvel (assuming gymnasium-robotics obs layout).
  3. Replays the demo actions step by step.
  4. Compares the resulting observations to the stored next_observations.
  5. Reports per-region RMSE so you can see WHERE the states diverge.

Expected outcome if everything is fine:
  Mean RMSE across all 59 dims should be < 0.05 for at least the first 20 steps.

If RMSE is large (> 0.5):
  - Action convention mismatch (D4RL vs gymnasium-robotics scaling).
  - OR obs layout mismatch (different dim ordering in Minari vs gym-robotics).

The test also prints the task-relevant dims (object qpos, 18-38) separately,
because that is what matters for task-success.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import mujoco

gym.register_envs(gymnasium_robotics)

TASKS = ['microwave', 'kettle', 'light switch', 'slide cabinet']


# ---------------------------------------------------------------------------
# State setter (mirrors demo_loader._render_trajectory_images logic)
# ---------------------------------------------------------------------------

def set_kitchen_state(raw_env, obs_59d: np.ndarray):
    """
    Set the FrankaKitchen-v1 MuJoCo state from a 59-d observation vector.

    Assumed obs layout (gymnasium-robotics):
        [robot_qpos(9), robot_qvel(9), obj_qpos(21), obj_qvel(20)] = 59

    n_qpos = 30 (9 robot + 21 object)
    n_qvel = 30 (9 robot + ~21 object, but Kitchen model may have 30 or fewer)
    """
    model = raw_env.model
    data = raw_env.data
    n_qpos = model.nq   # should be 30
    n_qvel = model.nv   # should be 30

    robot_qpos = obs_59d[0:9]
    robot_qvel = obs_59d[9:18]
    obj_qpos   = obs_59d[18:39]            # 21 dims
    obj_qvel   = obs_59d[39:39 + (n_qvel - 9)]  # up to 20 dims

    qpos = np.zeros(n_qpos, dtype=np.float64)
    qvel = np.zeros(n_qvel, dtype=np.float64)
    qpos[:9]             = robot_qpos
    qpos[9:9 + min(21, n_qpos - 9)] = obj_qpos[:n_qpos - 9]
    qvel[:9]             = robot_qvel
    qvel[9:9 + len(obj_qvel)] = obj_qvel

    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Action convention diagnostic")
    print("=" * 60)

    # ---- Load Minari episode ----
    try:
        import minari
    except ImportError:
        print("ERROR: minari not installed. Run: pip install 'minari[all]'")
        return

    print("\nLoading D4RL/kitchen/complete-v2 ...")
    try:
        ds = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    ep = next(ds.iterate_episodes())
    obs_raw = ep.observations
    if isinstance(obs_raw, dict):
        obs_arr = np.array(obs_raw['observation'], dtype=np.float64)
    else:
        obs_arr = np.array(obs_raw, dtype=np.float64)

    actions = np.array(ep.actions, dtype=np.float32)
    T = len(actions)

    print(f"  Episode length : {T} steps")
    print(f"  Obs shape      : {obs_arr.shape}  (expect (T+1, 59))")
    print(f"  Action shape   : {actions.shape}  (expect (T, 9))")
    print(f"  Action range   : [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Demo obs[0] robot_qpos   (0-8)  : {obs_arr[0, 0:9].round(3)}")
    print(f"  Demo obs[0] robot_qvel   (9-17) : {obs_arr[0, 9:18].round(3)}")
    print(f"  Demo obs[0] obj_qpos    (18-38) : {obs_arr[0, 18:39].round(3)}")

    # ---- Create env ----
    print("\nCreating FrankaKitchen-v1 ...")
    env = gym.make(
        'FrankaKitchen-v1',
        tasks_to_complete=TASKS,
        render_mode=None,
    )
    env.reset()
    raw_env = env.unwrapped

    # ---- Check model dims ----
    n_qpos = raw_env.model.nq
    n_qvel = raw_env.model.nv
    print(f"  MuJoCo model nq={n_qpos}, nv={n_qvel}")

    # ---- Replay loop ----
    N_TEST = min(150, T)
    errors_all = []
    errors_robot_qpos  = []
    errors_robot_qvel  = []
    errors_obj_qpos    = []
    errors_obj_qvel    = []

    print(f"\nReplaying first {N_TEST} steps ...")

    # Set initial state from demo
    try:
        set_kitchen_state(raw_env, obs_arr[0])
    except Exception as e:
        print(f"  WARNING: could not set initial state ({e}). "
              f"Results may be unreliable.")

    for t in range(N_TEST):
        obs_dict, _r, _term, _trunc, _info = env.step(actions[t])
        result_obs = np.array(obs_dict['observation'], dtype=np.float64)
        expected_obs = obs_arr[t + 1]

        err_all  = np.abs(result_obs - expected_obs)
        errors_all.append(err_all)
        errors_robot_qpos.append(err_all[0:9])
        errors_robot_qvel.append(err_all[9:18])
        errors_obj_qpos.append(err_all[18:39])
        errors_obj_qvel.append(err_all[39:])

        if t < 5:
            print(f"  step {t:3d}: RMSE={np.sqrt(np.mean(err_all**2)):.4f}  "
                  f"obj_qpos[13]={result_obs[31]:.4f} vs {expected_obs[31]:.4f}")

        # Stop if completely blown up
        if np.sqrt(np.mean(err_all ** 2)) > 5.0:
            print(f"  ABORT: state exploded at step {t}")
            N_TEST = t + 1
            break

    env.close()

    # ---- Summary ----
    def rmse(lst):
        return float(np.sqrt(np.mean(np.concatenate(lst) ** 2)))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Steps tested       : {N_TEST}")
    print(f"RMSE (all 59 dims) : {rmse(errors_all):.5f}")
    print(f"RMSE robot_qpos    : {rmse(errors_robot_qpos):.5f}")
    print(f"RMSE robot_qvel    : {rmse(errors_robot_qvel):.5f}")
    print(f"RMSE obj_qpos      : {rmse(errors_obj_qpos):.5f}   ← task-relevant")
    print(f"RMSE obj_qvel      : {rmse(errors_obj_qvel):.5f}")

    rmse_total = rmse(errors_all)
    rmse_obj   = rmse(errors_obj_qpos)

    print()
    if rmse_total < 0.05:
        print("PASS ✓  Actions replay faithfully. Action convention is compatible.")
    elif rmse_total < 0.5:
        print("WARN ⚠  Minor state divergence — possible physics noise or small "
              "scaling difference. BC may still learn, but IQL is recommended.")
    else:
        print("FAIL ✗  Large state divergence!")
        print("  The demo actions do NOT reproduce the expected state transitions")
        print("  in FrankaKitchen-v1. This will make any form of BC/IQL fail.")
        print()
        if rmse_obj > 0.5 and rmse(errors_robot_qpos) < 0.1:
            print("  Likely cause: action-convention mismatch (D4RL/adept vs")
            print("  gymnasium-robotics uses different control gain or action scale).")
        elif rmse_obj > 0.5 and rmse(errors_robot_qpos) > 0.5:
            print("  Likely cause: observation layout mismatch between Minari and")
            print("  gymnasium-robotics (qpos/qvel ordering differs).")
        print()
        print("  Recommended fix: use IQL with dense task-space rewards instead")
        print("  of replaying raw actions. IQL does not need the actions to re-")
        print("  produce the correct state trajectory — it only needs the (s, a, r, s')")
        print("  transitions to be internally consistent within the demo dataset.")

    # Per-dim breakdown of obj_qpos errors (task-relevant dims)
    print()
    mean_obj_err = np.mean(np.stack(errors_obj_qpos), axis=0)
    print("Mean |err| per obj_qpos dim (18-38):")
    labels = {0: 'btm_burn0', 1: 'btm_burn1', 2:'?', 3:'?', 4:'?', 5:'?',
              6: 'top_burn0', 7: 'top_burn1', 8: 'light0', 9: 'light1',
              10: 'slide', 11: 'hinge0', 12: 'hinge1', 13: 'microwave',
              14: 'kettle_x', 15: 'kettle_y', 16: 'kettle_z',
              17: 'kettle_q0', 18: 'kettle_q1', 19: 'kettle_q2', 20: 'kettle_q3'}
    for i, e in enumerate(mean_obj_err):
        tag = labels.get(i, '?')
        print(f"  obj_qpos[{i:2d}] (dim {18+i:2d}) [{tag:12s}]: {e:.5f}")


if __name__ == "__main__":
    main()
