// Test of MAGSACScoringFunction::getScore: the score of a single point is its
// MAGSAC++ weight divided by the weight of a point with zero residual. The
// MAGSAC++ paper gives it in closed form (degrees of freedom 4, so the upper
// incomplete gamma function of order 3/2):
//   w(r) / w(0) = (G(r^2 / (2 sigma^2)) - G(k^2 / 2)) / (G(0) - G(k^2 / 2)),
//   G(x) = sqrt(x) exp(-x) + sqrt(pi) / 2 erfc(sqrt(x)),  sigma = threshold / k.
// Exits with 0 if getScore matches it at every tested residual.
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "estimators/estimator.h"
#include "estimators/solver_linear_model.h"
#include "estimators/linear_model_estimator.h"
#include "scoring_function.h"

using namespace gcransac;
using LineSolver = estimator::solver::LinearModelSolver<2>;
using LineEstimator = estimator::LinearModelEstimator<LineSolver, LineSolver, 2>;

static double upperGamma32(double x)
{
	return std::sqrt(x) * std::exp(-x) + std::sqrt(M_PI) / 2.0 * std::erfc(std::sqrt(x));
}

int main()
{
	LineEstimator estimator;
	MAGSACScoringFunction<LineEstimator> scoring;
	Model model; // The line y = 0
	model.descriptor.resize(3, 1);
	model.descriptor << 0, 1, 0;

	const double k = 3.64; // The value of k used by getScore
	int failures = 0;
	for (double threshold : {0.5, 2.0, 10.0})
	{
		const double sigma = threshold / k;
		const double gamma_k = upperGamma32(k * k / 2.0);
		for (double fraction : {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995})
		{
			const double residual = fraction * threshold;
			const cv::Mat point = (cv::Mat_<double>(1, 2) << 0.0, residual);
			std::vector<size_t> inliers;
			const Score score = scoring.getScore(point, model, estimator, threshold, inliers);
			const double expected = (upperGamma32(residual * residual / (2.0 * sigma * sigma)) - gamma_k) /
				(upperGamma32(0.0) - gamma_k);
			// getScore reads the gamma values from a lookup table: allow for its step
			const bool ok = std::abs(score.value - expected) < 1e-4;
			failures += !ok;
			std::printf("%s threshold %g, residual %g: score %.7f, expected %.7f\n",
				ok ? "ok  " : "FAIL", threshold, residual, score.value, expected);
		}
	}
	return failures == 0 ? 0 : 1;
}
