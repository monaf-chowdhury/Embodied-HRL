import gym
import d4rl
import numpy as np
import cv2

def get_goal_image_and_encoding(encoder, img_size=224, device='cuda'):
    """
    Render the kitchen in its fully-completed goal state and encode it.
    Uses the hardcoded OBS_ELEMENT_GOALS from D4RL's kitchen_envs.py.
    Returns z_goal: (proj_dim,) numpy array
    """
    # Goal joint positions for all 4 required tasks (from D4RL source)
    # The 60-d obs vector indices and their goal values:
    GOAL_INDICES = {
        'microwave':     (np.array([22]),           np.array([-0.75])),
        'kettle':        (np.array([23,24,25,26,27,28,29]), np.array([-0.23,0.75,1.62,0.99,0.,0.,-0.06])),
        'light switch':  (np.array([17, 18]),       np.array([-0.69, -0.05])),
        'slide cabinet': (np.array([19]),           np.array([0.37])),
    }

    env = gym.make('kitchen-complete-v0')
    env.reset()

    # Access the underlying sim and set goal joint positions
    sim = env.unwrapped.sim
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()

    for element, (indices, goal_vals) in GOAL_INDICES.items():
        # The obs indices map to qpos with a small offset
        # Kitchen obs = qpos[9:] for the object joints (first 9 are robot)
        for i, idx in enumerate(indices):
            qpos[idx + 9] = goal_vals[i]  # +9 to skip robot joints

    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.forward()  # propagate the state

    # Render
    img = sim.render(width=img_size, height=img_size, camera_id=0)
    img = np.array(img, dtype=np.uint8)
    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size))

    env.close()

    # Encode
    z_goal = encoder.encode_numpy(img).squeeze()
    print(f"Goal image rendered. z_goal shape: {z_goal.shape}, norm: {np.linalg.norm(z_goal):.4f}")
    return z_goal, img