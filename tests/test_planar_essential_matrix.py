import os
import numpy as np
import cv2
import pygcransac


K = np.array([[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]])

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(THIS_PATH, "test_data")
img1 = cv2.cvtColor(
    cv2.imread(os.path.join(TEST_DATA_PATH, "004520.png")), cv2.COLOR_BGR2RGB
)
img2 = cv2.cvtColor(
    cv2.imread(os.path.join(TEST_DATA_PATH, "004523.png")), cv2.COLOR_BGR2RGB
)


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

    E, mask = pygcransac.findPlanarEssentialMatrix(
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
        spatial_coherence_weight=0.0,  # The spatial coherence weight for GC-RANSAC
        sampler=4,
    )  # Sampler index (0 - Uniform, 1 - PROSAC, 2 - P-NAPSAC, 3 - NG-RANSAC, 4 - AR-Sampler)
    return E, mask


def test_planar_essential_matrix():
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

    gc_E, gc_E_mask = verify_pygcransac_ess(
        kps1,
        kps2,
        tentatives,
        K,
        K,
        img1.shape[0],
        img1.shape[1],
        img2.shape[0],
        img2.shape[1],
        4,
    )

    assert gc_E_mask.sum() > 600
