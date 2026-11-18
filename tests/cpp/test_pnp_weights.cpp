// Regression test: CameraJacobianAccumulator, the accumulator of the PnP bundle adjustment,
// must read the weight of the k-th point of the sample at weights[sample[k]] (one weight per
// data point, as MAGSAC++ passes them), not at weights[k].
// Inliers have weight 1, outliers weight 0, and the sample lists every point in reverse order.
// Three checks:
// 1. refine_pnp, from a pose near the true one, fits the inliers exactly;
// 2. PnPBundleAdjustment (EPnP, then refine_pnp) fits the inliers exactly;
// 3. the cost and the normal equations of the accumulator don't depend on the order of the sample.
// Reading weights[k] gives the outliers weight 1 and some inliers weight 0, which breaks all three.
// Exits with 0 if every check passes.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp> // solvePnP, used by solver_pnp_bundle_adjustment.h
#include <opencv2/core/core.hpp>
#include "model.h"
#include "relative_pose/bundle.h"
#include "relative_pose/jacobian_impl.h"
#include "relative_pose/robust_loss.h"
#include "estimators/solver_pnp_bundle_adjustment.h"

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

	// 2D-3D correspondences (u, v, x, y, z) in normalized image coordinates; the outliers' image points are random
	cv::Mat correspondences(const pose_lib::CameraPose &pose)
	{
		cv::Mat points(point_number, 5, CV_64F);
		for (size_t i = 0; i < point_number; ++i)
		{
			const Eigen::Vector3d X(uniform(-1, 1), uniform(-1, 1), uniform(4, 6));
			const Eigen::Vector3d Z = pose.apply(X);
			points.at<double>(i, 0) = i < inlier_number ? Z(0) / Z(2) : uniform(-0.3, 0.3);
			points.at<double>(i, 1) = i < inlier_number ? Z(1) / Z(2) : uniform(-0.3, 0.3);
			for (int j = 0; j < 3; ++j)
				points.at<double>(i, 2 + j) = X(j);
		}
		return points;
	}

	// Largest reprojection error of the inliers
	double maxInlierError(const cv::Mat &points, const pose_lib::CameraPose &pose)
	{
		double worst = 0;
		for (size_t i = 0; i < inlier_number; ++i)
		{
			const Eigen::Vector3d X(points.at<double>(i, 2), points.at<double>(i, 3), points.at<double>(i, 4));
			const Eigen::Vector2d x(points.at<double>(i, 0), points.at<double>(i, 1));
			worst = std::max(worst, (pose.apply(X).hnormalized() - x).norm());
		}
		return worst;
	}
}

int main()
{
	const pose_lib::CameraPose true_pose(
		Eigen::Matrix3d(Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.3, 1.0, 0.1).normalized())),
		Eigen::Vector3d(0.1, -0.1, 0.5));
	const cv::Mat points = correspondences(true_pose);
	const std::vector<size_t> reversed = sample(true);
	const std::vector<double> weights = weightsByPoint();

	// A pose near the true one
	const pose_lib::CameraPose initial_pose(
		Eigen::Matrix3d(Eigen::AngleAxisd(0.01, Eigen::Vector3d::UnitZ()).toRotationMatrix() * true_pose.R()),
		true_pose.t + Eigen::Vector3d(0.0, 0.02, -0.02));

	// 1. refine_pnp, from initial_pose
	pose_lib::BundleOptions options;
	options.loss_type = pose_lib::BundleOptions::LossType::TRIVIAL; // only the weights reject the outliers
	pose_lib::CameraPose pose = initial_pose;
	pose_lib::refine_pnp(points, reversed.data(), reversed.size(), &pose, options, weights.data());
	bool ok = check("refine_pnp: largest reprojection error of the inliers", maxInlierError(points, pose), 1e-8);

	// 2. PnPBundleAdjustment, as the non-minimal solver of the PnP estimator
	gcransac::estimator::solver::PnPBundleAdjustment solver(pose_lib::BundleOptions::LossType::TRIVIAL);
	std::vector<gcransac::Model> models;
	solver.estimateModel(points, reversed.data(), reversed.size(), models, weights.data());
	double error = models.size() == 1 ? 0 : 1;
	for (const auto &model : models)
		error = std::max(error, maxInlierError(points, pose_lib::CameraPose(
			Eigen::Matrix3d(model.descriptor.block<3, 3>(0, 0)), Eigen::Vector3d(model.descriptor.col(3)))));
	ok &= check("PnPBundleAdjustment: largest reprojection error of the inliers", error, 1e-8);

	// 3. The accumulator at initial_pose, with the sample in increasing and in reverse order
	const pose_lib::TrivialLoss loss;
	pose_lib::Camera camera;
	camera.model_id = pose_lib::NullCameraModel::model_id; // as refine_pnp without a camera
	double cost[2];
	Eigen::Matrix<double, 6, 6> JtJ[2];
	Eigen::Matrix<double, 6, 1> Jtr[2];
	for (int order = 0; order < 2; ++order)
	{
		const std::vector<size_t> indices = sample(order == 1);
		const pose_lib::CameraJacobianAccumulator<pose_lib::NullCameraModel, pose_lib::TrivialLoss> accumulator(
			points, indices.data(), indices.size(), camera, loss, weights.data());
		cost[order] = accumulator.residual(initial_pose);
		JtJ[order].setZero();
		Jtr[order].setZero();
		accumulator.accumulate(initial_pose, JtJ[order], Jtr[order]);
	}
	// Only the lower half of JtJ is computed
	const Eigen::Matrix<double, 6, 6> dJtJ = (JtJ[1] - JtJ[0]).triangularView<Eigen::Lower>();
	const Eigen::Matrix<double, 6, 6> JtJ0 = JtJ[0].triangularView<Eigen::Lower>();
	error = std::max({std::abs(cost[1] - cost[0]) / cost[0], dJtJ.norm() / JtJ0.norm(), (Jtr[1] - Jtr[0]).norm() / Jtr[0].norm()});
	ok &= check("CameraJacobianAccumulator: change with the sample order", error, 1e-12);

	return ok ? 0 : 1;
}
