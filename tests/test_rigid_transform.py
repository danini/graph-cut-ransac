import os
import pygcransac
import numpy as np
import pytest
from test_utils import calculate_error

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

CORRESPONDENCES_PATH = os.path.join(THIS_PATH, '..', 'build/data/rigid_pose_example/rigid_pose_example_points.txt')
POSE_PATH = os.path.join(THIS_PATH, '..', 'build/data/rigid_pose_example/rigid_pose_example_gt.txt')

CORRESPONDENCES = np.loadtxt(CORRESPONDENCES_PATH)
GT_POSE = np.loadtxt(POSE_PATH)
GROUND_TRUTH_T = GT_POSE[:4, :]
THRESHOLD = 0.03

# Translating the points so there are no negative coordinates.
# This is only important if the space partitioning technique is used to
# accelerate the robust estimation, or when the spatial coherence term is >0.
MIN_COORDINATES = np.min(CORRESPONDENCES, axis=0)
T1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [-MIN_COORDINATES[0], -MIN_COORDINATES[1], -MIN_COORDINATES[2], 1]])
T2INV = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [MIN_COORDINATES[3], MIN_COORDINATES[4], MIN_COORDINATES[5], 1]])
TRANSFORMED_CORRESPONDENCES = CORRESPONDENCES - MIN_COORDINATES


def verify_pygcransac(corrs, threshold, min_iters=1000, max_iters=5000, use_sprt = False):    
    n = len(corrs)
    
    pose, mask = pygcransac.findRigidTransform(
        np.ascontiguousarray(corrs), 
        probabilities = [],
        threshold = threshold, 
        neighborhood_size = 4,
        sampler = 1,
        min_iters = min_iters,
        max_iters = max_iters,
        spatial_coherence_weight = 0.0,
        use_space_partitioning = not use_sprt,
        neighborhood = 0,
        conf = 0.999,
        use_sprt = use_sprt)    
    return pose, mask


def tranform_points(corrs, T):
    n = len(corrs)
    points1 = np.float32([corrs[i][0:3] for i in np.arange(n)]).reshape(-1,3)
    points2 = np.float32([corrs[i][3:6] for i in np.arange(n)]).reshape(-1,3)
    
    transformed_corrs = np.zeros((corrs.shape[0], 6))

    for i in range(n):
        p1 = np.append(correspondences[i][:3], 1)
        p2 = p1.dot(T)
        transformed_corrs[i][:3] = p2[:3]
        transformed_corrs[i][3:] = corrs[i][3:]
    return transformed_corrs


@pytest.mark.parametrize("use_sprt", [True, False])
def test_gc_ransac(use_sprt):
    """GC-RANSAC without SPRT test."""
    gc_t, gc_mask = verify_pygcransac(TRANSFORMED_CORRESPONDENCES, THRESHOLD, 5000, 5000, use_sprt=use_sprt)
    if gc_t is None:
        gc_t = np.eye(4)
    else:
        gc_t = T1 @ gc_t @ T2INV
        gc_t = gc_t.T

    err_r, err_t = calculate_error(GROUND_TRUTH_T, gc_t)
    assert err_r < 4  # rotation less than 4 degrees
    assert err_t < 0.2  # translation less than 0.2 cm

