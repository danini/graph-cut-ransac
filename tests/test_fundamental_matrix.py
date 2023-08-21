import os
import numpy as np
import cv2
import pygcransac
from time import time
import pytest

skip_windows = pytest.mark.skipif(os.name == 'nt', reason='Test fails on Windows')

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
KYOTO_PATH = os.path.join(THIS_PATH, '..', "build/data/Kyoto")
img1 = cv2.cvtColor(
    cv2.imread(os.path.join(KYOTO_PATH, "Kyoto1.jpg")), cv2.COLOR_BGR2RGB
)
img2 = cv2.cvtColor(
    cv2.imread(os.path.join(KYOTO_PATH, "Kyoto2.jpg")), cv2.COLOR_BGR2RGB
)


def get_probabilities(tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))
    return probabilities


def verify_pygcransac_fundam(kps1, kps2, tentatives, h1, w1, h2, w2, sampler_id):
    correspondences = np.float32(
        [(kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives]
    ).reshape(-1, 4)
    inlier_probabilities = []

    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks.
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(tentatives)

    H, mask = pygcransac.findFundamentalMatrix(
        np.ascontiguousarray(correspondences),
        h1,
        w1,
        h2,
        w2,
        threshold=0.75,
        sampler=sampler_id,
        max_iters=5000,
        min_iters=50,
        probabilities=inlier_probabilities,
    )
    return H, mask


@skip_windows
def test_find_fundamental_matrix():
    # We will detect ORB features and match them with cross-check test
    det = cv2.SIFT_create(8000)
    kps1, descs1 = det.detectAndCompute(img1, None)
    kps2, descs2 = det.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    SNN_threshold = 0.8
    matches = bf.knnMatch(descs1, descs2, k=2)

    # Apply ratio test
    snn_ratios = []
    tentatives = []
    for m, n in matches:
        if m.distance < SNN_threshold * n.distance:
            tentatives.append(m)
            snn_ratios.append(m.distance / n.distance)

    sorted_indices = np.argsort(snn_ratios)
    tentatives = list(np.array(tentatives)[sorted_indices])

    gc_F, gc_mask = verify_pygcransac_fundam(
        kps1,
        kps2,
        tentatives,
        img1.shape[0],
        img1.shape[1],
        img2.shape[0],
        img2.shape[1],
        4,
    )

    assert gc_mask.sum() > 380
