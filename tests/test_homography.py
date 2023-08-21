import os
import numpy as np
import cv2
import pygcransac
from time import time


THIS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(THIS_PATH, "..", "examples", "img")

IMG1 = cv2.cvtColor(cv2.imread(os.path.join(DATA_PATH, "grafA.png")), cv2.COLOR_BGR2RGB)
IMG2 = cv2.cvtColor(cv2.imread(os.path.join(DATA_PATH, "grafB.png")), cv2.COLOR_BGR2RGB)
H_gt = np.linalg.inv(np.loadtxt(os.path.join(DATA_PATH, "graf_model.txt")))


def get_probabilities(tentatives):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    for i in range(len(tentatives)):
        probabilities.append(1.0 - i / len(tentatives))
    return probabilities


def verify_pygcransac(kps1, kps2, tentatives, h1, w1, h2, w2, sampler_id):
    correspondences = np.float32(
        [(kps1[m.queryIdx].pt + kps2[m.trainIdx].pt) for m in tentatives]
    ).reshape(-1, 4)
    inlier_probabilities = []

    # NG-RANSAC and AR-Sampler require an inlier probability to be provided for each point.
    # Since deep learning-based prediction is not applied here, we calculate the probabilities
    # from the SNN ratio ranks.
    if sampler_id == 3 or sampler_id == 4:
        inlier_probabilities = get_probabilities(tentatives)

    H, mask = pygcransac.findHomography(
        np.ascontiguousarray(correspondences),
        h1,
        w1,
        h2,
        w2,
        use_sprt=False,
        spatial_coherence_weight=0.1,
        neighborhood_size=3,
        probabilities=inlier_probabilities,
        sampler=sampler_id,
        use_space_partitioning=True,
    )
    return H, mask


def test_homography():
    #We will detect ORB features and match them with cross-check test
    det = cv2.SIFT_create(8000)    
    kps1, descs1 = det.detectAndCompute(IMG1,None)
    kps2, descs2 = det.detectAndCompute(IMG2,None)

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

    gc_H, gc_mask = verify_pygcransac(
        kps1,
        kps2,
        tentatives,
        IMG1.shape[1],
        IMG1.shape[0],
        IMG2.shape[1],
        IMG2.shape[0],
        2,
    )
    assert gc_mask.sum() > 250
