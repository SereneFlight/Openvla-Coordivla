"""
RoboTwin action processing helpers.

The model-side gripper convention follows OpenVLA/Open-X:
0 = closed, 1 = open. Arm joints are represented as deltas, while the
gripper remains an absolute continuous opening state.
"""

import numpy as np


GRIPPER_CLOSED_VALUE = 0.0
GRIPPER_OPEN_VALUE = 1.0

def standardize_gripper_actions(
    gripper: np.ndarray,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Convert RoboTwin gripper trajectories to continuous OpenVLA-style absolute states."""
    gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
    gripper = np.clip(gripper, 0.0, 1.0)
    if invert:
        gripper = 1.0 - gripper
    return gripper.astype(np.float32)


def build_joint_delta_gripper_cont_abs(arm: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Build action labels: joint deltas plus continuous absolute gripper opening."""
    arm = np.asarray(arm, dtype=np.float32)
    gripper = standardize_gripper_actions(gripper)
    if arm.shape[0] != gripper.shape[0]:
        raise ValueError(f"arm length {arm.shape[0]} does not match gripper length {gripper.shape[0]}")
    if arm.shape[0] < 2:
        return np.zeros((0, arm.shape[-1] + 1), dtype=np.float32)

    arm_delta = arm[1:] - arm[:-1]
    return np.concatenate([arm_delta, gripper[1:, None]], axis=1).astype(np.float32)


def clip_gripper_qpos(
    action: np.ndarray,
    gripper_dims=(6, 13),
    closed_value: float = GRIPPER_CLOSED_VALUE,
    open_value: float = GRIPPER_OPEN_VALUE,
) -> np.ndarray:
    """Clip decoded continuous gripper qpos values before execution."""
    action = np.asarray(action, dtype=np.float32).copy()
    for dim in gripper_dims:
        if dim < action.shape[-1]:
            action[..., dim] = np.clip(action[..., dim], closed_value, open_value)
    return action.astype(np.float32)
