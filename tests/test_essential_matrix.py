import os
import pytest
import numpy as np
import cv2
import pygcransac
from time import time

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
FOUNTAIN_PATH = os.path.join(THIS_PATH, "../build/data/fountain/")
IMG1 = cv2.cvtColor(
    cv2.imread(os.path.join(FOUNTAIN_PATH, "fountain1.jpg")),
    cv2.COLOR_BGR2RGB,
)
FOUNTAIN2_PATH = os.path.join(THIS_PATH, "../build/data/fountain/fountain1.jpg")
IMG2 = cv2.cvtColor(
    cv2.imread(os.path.join(FOUNTAIN_PATH, "fountain2.jpg")),
    cv2.COLOR_BGR2RGB,
)
K1 = np.loadtxt(os.path.join(FOUNTAIN_PATH, "fountain1.K"))
K2 = np.loadtxt(os.path.join(FOUNTAIN_PATH, "fountain2.K"))


def get_probabilities(tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))
    return probabilities


def verify_pygcransac_ess(kps1, kps2, tentatives, K1, K2, h1, w1, h2, w2, sampler_id):
    correspondences = np.float32(
        [(kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives]
    ).reshape(-1, 4)
    inlier_probabilities = []

    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks.
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(tentatives)

    E, mask = pygcransac.findEssentialMatrix(
        np.ascontiguousarray(
            correspondences
        ),  # Point correspondences in the two images
        K1,  # Intrinsic camera parameters of the source image
        K2,  # Intrinsic camera parameters of the destination image
        h1,
        w1,
        h2,
        w2,  # The sizes of the images
        probabilities=inlier_probabilities,  # Inlier probabilities. This is not used if the sampler is not 3 (NG-RANSAC) or 4 (AR-Sampler)
        threshold=0.75,  # Inlier-outlier threshold
        conf=0.99,  # RANSAC confidence
        min_iters=50,  # The minimum iteration number in RANSAC. If time does not matter, I suggest setting it to, e.g., 1000
        max_iters=5000,  # The maximum iteration number in RANSAC
        sampler=sampler_id,
    )  # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
    return E, mask


def test_find_essential_matrix():
    det = cv2.SIFT_create(8000)
    kps1, descs1 = det.detectAndCompute(IMG1, None)
    kps2, descs2 = det.detectAndCompute(IMG2, None)

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

    gc_E, gc_E_mask = verify_pygcransac_ess(
        kps1,
        kps2,
        tentatives,
        K1,
        K2,
        IMG1.shape[0],
        IMG1.shape[1],
        IMG2.shape[0],
        IMG2.shape[1],
        4,
    )

    assert gc_E_mask.sum() > 450
