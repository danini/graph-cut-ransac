// Regression test: the Jacobian accumulators of the bundle adjustments must read
// the weight of the k-th point of the sample at weights[sample[k]], as the solvers
// that call them do (one weight per data point), not at weights[k].
// Inliers have weight 1, outliers weight 0. Two checks:
// 1. refine_relpose, given every point in reverse order, fits the inliers exactly;
// 2. the cost and the normal equations of the relative-pose, fundamental-matrix and
//    homography accumulators don't depend on the order of the points in the sample.
// Reading weights[k] gives the outliers weight 1 and some inliers weight 0, which
// breaks both. Exits with 0 if every check passes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "relative_pose/bundle.h"
#include "relative_pose/jacobian_impl.h"
#include "relative_pose/robust_loss.h"

namespace
{
	constexpr size_t inlier_number = 30, outlier_number = 10, point_number = inlier_number + outlier_number;

	std::mt19937 generator(42);
	double uniform(double a, double b) { return a + (b - a) * (generator() / 4294967296.0); }

	// Every point, in increasing or in reverse order
	std::vector<size_t> sample(bool reversed)
	{
		std::vector<size_t> indices(point_number);
		for (size_t k = 0; k < point_number; ++k)
			indices[k] = reversed ? point_number - 1 - k : k;
		return indices;
	}

	// One weight per point: 1 for the inliers (the first points), 0 for the outliers
	std::vector<double> weightsByPoint()
	{
		std::vector<double> weights(point_number, 0.0);
		std::fill(weights.begin(), weights.begin() + inlier_number, 1.0);
		return weights;
	}

	bool check(const char *name, double error, double tolerance)
	{
		const bool ok = error < tolerance;
		std::printf("%-60s %.3g (%s)\n", name, error, ok ? "ok" : "FAILED");
		return ok;
	}

	// Two views of random 3D points, in normalized image coordinates, with outliers in the second view
	cv::Mat twoViews(const pose_lib::CameraPose &pose)
	{
		cv::Mat points(point_number, 4, CV_64F);
		for (size_t i = 0; i < point_number; ++i)
		{
			Eigen::Vector3d X;
			for (int j = 0; j < 3; ++j)
				X(j) = j < 2 ? uniform(-1, 1) : uniform(3, 6);
			const Eigen::Vector3d Y = pose.apply(X);
			points.at<double>(i, 0) = X(0) / X(2);
			points.at<double>(i, 1) = X(1) / X(2);
			points.at<double>(i, 2) = i < inlier_number ? Y(0) / Y(2) : uniform(-0.5, 0.5);
			points.at<double>(i, 3) = i < inlier_number ? Y(1) / Y(2) : uniform(-0.5, 0.5);
		}
		return points;
	}

	// Largest Sampson distance of the inliers to the epipolar geometry E
	double maxInlierSampson(const cv::Mat &points, const Eigen::Matrix3d &E)
	{
		double worst = 0;
		for (size_t i = 0; i < inlier_number; ++i)
		{
			const Eigen::Vector3d x1(points.at<double>(i, 0), points.at<double>(i, 1), 1.0);
			const Eigen::Vector3d x2(points.at<double>(i, 2), points.at<double>(i, 3), 1.0);
			const Eigen::Vector3d Ex1 = E * x1, Etx2 = E.transpose() * x2;
			worst = std::max(worst, std::abs(x2.dot(Ex1)) / std::sqrt(Ex1.head<2>().squaredNorm() + Etx2.head<2>().squaredNorm()));
		}
		return worst;
	}

	// The cost and the normal equations (N parameters) of 'Accumulator' at 'model', with the
	// sample in increasing and in reverse order, must be the same.
	template <typename Accumulator, int N, typename Model, typename... Extra>
	bool testOrder(const char *name, const cv::Mat &points, const Model &model, Extra &...extra)
	{
		const pose_lib::TrivialLoss loss;
		const std::vector<double> weights = weightsByPoint();
		double cost[2];
		Eigen::Matrix<double, N, N> JtJ[2];
		Eigen::Matrix<double, N, 1> Jtr[2];
		for (int reversed = 0; reversed < 2; ++reversed)
		{
			const std::vector<size_t> indices = sample(reversed);
			const Accumulator accumulator(points, indices.data(), indices.size(), loss, weights.data());
			cost[reversed] = accumulator.residual(model);
			JtJ[reversed].setZero();
			Jtr[reversed].setZero();
			accumulator.accumulate(model, JtJ[reversed], Jtr[reversed], extra...);
		}
		// Only the lower half of JtJ is computed
		const Eigen::Matrix<double, N, N> dJtJ = (JtJ[1] - JtJ[0]).template triangularView<Eigen::Lower>();
		const Eigen::Matrix<double, N, N> JtJ0 = JtJ[0].template triangularView<Eigen::Lower>();
		const double error = std::max({std::abs(cost[1] - cost[0]) / cost[0],
			dJtJ.norm() / JtJ0.norm(), (Jtr[1] - Jtr[0]).norm() / Jtr[0].norm()});
		return check(name, error, 1e-12);
	}
}

int main()
{
	const pose_lib::CameraPose true_pose(
		Eigen::Matrix3d(Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.3, 1.0, 0.1).normalized())),
		Eigen::Vector3d(1.0, 0.1, 0.2).normalized());
	const cv::Mat points = twoViews(true_pose);

	// A pose near the true one
	const pose_lib::CameraPose initial_pose(
		Eigen::Matrix3d(Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix() * true_pose.R()),
		(true_pose.t + Eigen::Vector3d(0.0, 0.02, -0.02)).normalized());

	// 1. Bundle adjustment of the relative pose, from initial_pose
	const std::vector<size_t> reversed = sample(true);
	const std::vector<double> weights = weightsByPoint();
	pose_lib::BundleOptions options;
	options.loss_type = pose_lib::BundleOptions::LossType::TRIVIAL; // only the weights reject the outliers
	pose_lib::CameraPose pose = initial_pose;
	pose_lib::refine_relpose(points, reversed.data(), reversed.size(), &pose, options, weights.data());
	Eigen::Matrix3d E;
	pose_lib::essential_from_motion(pose, &E);
	bool ok = check("refine_relpose: largest Sampson distance of the inliers", maxInlierSampson(points, E), 1e-6);

	// 2. The accumulators, at models that don't fit the inliers exactly
	Eigen::Matrix<double, 3, 2> tangent_basis;
	ok &= testOrder<pose_lib::RelativePoseJacobianAccumulator<pose_lib::TrivialLoss>, 5>(
		"RelativePoseJacobianAccumulator: change with the sample order", points, initial_pose, tangent_basis);

	Eigen::Matrix3d F;
	pose_lib::essential_from_motion(initial_pose, &F);
	ok &= testOrder<pose_lib::FundamentalJacobianAccumulator<pose_lib::TrivialLoss>, 7>(
		"FundamentalJacobianAccumulator: change with the sample order", points, pose_lib::FactorizedFundamentalMatrix(F));

	Eigen::Matrix3d H;
	H << 1.1, 0.05, 0.2, -0.03, 0.95, -0.1, 0.02, -0.01, 1.0;
	ok &= testOrder<pose_lib::HomographyJacobianAccumulator<pose_lib::TrivialLoss>, 8>(
		"HomographyJacobianAccumulator: change with the sample order", points, H);

	return ok ? 0 : 1;
}
