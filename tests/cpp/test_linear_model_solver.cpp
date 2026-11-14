// Regression test of #57: LinearModelSolver::estimateModel with a null sample
// (meaning "the first sampleNumber_ points") and exactly sampleSize() points,
// which LinearModelEstimator::estimateModelNonminimal passes after normalizing
// the points. Without the fix, the minimal fast path dereferences the null sample
// and this program crashes. Exits with 0 if the fits are exact.
#include <cmath>
#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include "estimators/estimator.h"
#include "estimators/solver_linear_model.h"
#include "estimators/linear_model_estimator.h"

using namespace gcransac;

template <size_t _Dimension>
static int check(const cv::Mat &points, const char *name)
{
	using Solver = estimator::solver::LinearModelSolver<_Dimension>;
	estimator::LinearModelEstimator<Solver, Solver, _Dimension> estimator;
	std::vector<size_t> sample(points.rows);
	for (size_t i = 0; i < sample.size(); ++i)
		sample[i] = i;
	std::vector<Model> models;
	if (!estimator.estimateModelNonminimal(points, sample.data(), points.rows, &models) || models.empty())
	{
		std::printf("FAIL %s: no model\n", name);
		return 1;
	}
	for (int i = 0; i < points.rows; ++i)
	{
		const double r = estimator.residual(points.row(i), models[0]);
		if (!(std::abs(r) < 1e-9))
		{
			std::printf("FAIL %s: residual of point %d is %g\n", name, i, r);
			return 1;
		}
	}
	std::printf("ok %s\n", name);
	return 0;
}

int main()
{
	// 2 points of the line y = 2 x + 1, and 3 points of the plane z = 0.1 x + 0.2 y + 1
	const cv::Mat line = (cv::Mat_<double>(2, 2) << 0, 1, 1, 3);
	const cv::Mat plane = (cv::Mat_<double>(3, 3) << 0, 0, 1, 1, 0, 1.1, 0, 1, 1.2);
	return check<2>(line, "line from 2 points") + check<3>(plane, "plane from 3 points");
}
