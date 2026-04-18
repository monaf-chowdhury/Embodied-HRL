"""
Compatibility shim.

The new codebase removes latent landmark subgoals entirely.
This file remains only so old imports do not silently fail.
"""

from utils import build_demo_task_prototypes


class LandmarkBuffer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'LandmarkBuffer is intentionally removed in the task-grounded rewrite. '
            'The new hierarchy uses explicit task selection at the manager level '
            'and task-space targets at the worker level.'
        )


__all__ = ['build_demo_task_prototypes', 'LandmarkBuffer']
