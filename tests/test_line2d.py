import numpy as np
import pygcransac


def test_line2d_minimal_fit():
    # Noise-free points on the line y = 2x + 1: every sample of 2 points gives the true line, so
    # ten iterations find every point. With the wrong normal of the 2-point fit, a single call
    # still found every point in about a third of the calls, so the test makes 20 calls.
    x = np.linspace(-100.0, 100.0, 50)
    points = np.ascontiguousarray(np.c_[x, 2 * x + 1])
    for _ in range(20):
        line, mask = pygcransac.findLine2D(points, 640, 480, np.ones(len(points)), threshold=0.5,
                                           min_iters=10, max_iters=10)
        assert line is not None
        assert int(np.sum(mask)) == len(points)
        a, b, c = np.ravel(line)[:3]
        assert np.allclose(a * x + b * (2 * x + 1) + c, 0, atol=1e-6 * np.hypot(a, b))
