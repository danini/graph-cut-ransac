import numpy as np

def calculate_error(gt_pose, est_pose):
    """Calculate the translation and rotation error between two poses."""
    R2R1 = np.dot(gt_pose[:, 0:3].T, est_pose[:, 0:3])
    cos_angle = max(-1.0, min(1.0, 0.5 * (R2R1.trace() - 1.0)))
    err_R = np.arccos(cos_angle) * 180.0 / np.pi
    err_t = np.linalg.norm(gt_pose[:, 3] - est_pose[:, 3])
    return err_R, err_t
