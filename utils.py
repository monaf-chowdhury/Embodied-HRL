"""
utils.py — Small shared helpers (image/video saving).

The encoder/landmark helpers from the old codebase are deleted; SMGW
does not use goal images or landmark GIFs.
"""
import os
import numpy as np
import cv2


def save_image(img: np.ndarray, path: str):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    cv2.imwrite(path, img[:, :, ::-1])   # RGB -> BGR for OpenCV


def save_video(frames: list, path: str, fps: int = 15):
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    H, W = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))

    if writer.isOpened():
        for frame in frames:
            writer.write(frame[:, :, ::-1])
        writer.release()
        return

    writer.release()
    gif_path = os.path.splitext(path)[0] + '.gif'
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            loop=0, duration=int(1000 / fps),
        )
        print(f"  [Video] Saved GIF fallback: {gif_path}")
    except Exception as e:
        print(f"  [Video] WARNING: could not save video or GIF: {e}")


def format_time(seconds: float) -> str:
    import datetime
    return str(datetime.timedelta(seconds=int(seconds)))


def format_steps(s: int) -> str:
    if s >= 1_000_000:
        return f"{s/1e6:.2f}M"
    if s >= 1_000:
        return f"{s/1e3:.1f}k"
    return str(s)
