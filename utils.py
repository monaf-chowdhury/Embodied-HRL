"""
utils.py — Utility functions for Visual HRL on FrankaKitchen-v1.

Goal image rendering has been REMOVED.  The agent no longer uses a
global goal encoding.  This file now contains only small helpers that
are shared across train.py and other modules.
"""

import os
import numpy as np
import cv2


def save_image(img: np.ndarray, path: str):
    """Save an RGB uint8 image as PNG."""
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(path, img[:, :, ::-1])   # RGB -> BGR for OpenCV


def save_video(frames: list, path: str, fps: int = 15):
    """
    Save a list of (H, W, 3) uint8 RGB frames as an MP4 video.

    Uses cv2.VideoWriter with the mp4v codec.  Falls back to writing a
    GIF via PIL if cv2 video writing is unavailable.

    Args:
        frames : list of (H, W, 3) uint8 RGB numpy arrays
        path   : output file path, e.g. 'logs/videos/ep_000.mp4'
        fps    : frames per second
    """
    if not frames:
        return

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    H, W = frames[0].shape[:2]

    # Try mp4v first
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    if writer.isOpened():
        for frame in frames:
            writer.write(frame[:, :, ::-1])   # RGB -> BGR
        writer.release()
        return

    # Fallback: save as GIF
    writer.release()
    gif_path = os.path.splitext(path)[0] + '.gif'
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(1000 / fps),
        )
        print(f"  [Video] Saved GIF fallback: {gif_path}")
    except Exception as e:
        print(f"  [Video] WARNING: could not save video or GIF: {e}")