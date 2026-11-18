import cv2
import numpy as np
import pygcransac


def reprojection_cost(correspondences, pose):
    R, t = pose[:, :3], pose[:, 3]
    Z = correspondences[:, 2:] @ R.T + t
    residuals = Z[:, :2] / Z[:, 2:] - correspondences[:, :2]
    return np.sum(residuals ** 2)


def test_find6DPose_minimizes_reprojection_error():
    # Noisy 2D-3D correspondences in normalized image coordinates, no outliers.
    rng = np.random.default_rng(0)
    n = 100
    points = rng.uniform(-1, 1, (n, 3)) + [0, 0, 5]
    R, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))
    t = np.array([0.1, -0.1, 0.5])
    Z = points @ R.T + t
    image_points = Z[:, :2] / Z[:, 2:] + rng.normal(0, 0.002, (n, 2))
    correspondences = np.ascontiguousarray(np.hstack([image_points, points]))

    pose, mask = pygcransac.find6DPose(
        correspondences, probabilities=[], threshold=0.02, sampler=0,
        spatial_coherence_weight=0.0, min_iters=100, max_iters=1000)
    assert pose is not None
    assert mask.sum() == n

    # The final bundle adjustment of find6DPose minimizes the reprojection
    # error over the inliers: compare with OpenCV's Levenberg-Marquardt.
    _, rvec, tvec = cv2.solvePnP(points, image_points, np.eye(3), None,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
    reference = np.hstack([cv2.Rodrigues(rvec)[0], tvec])
    assert reprojection_cost(correspondences, pose) <= \
        reprojection_cost(correspondences, reference) * (1 + 1e-6)
