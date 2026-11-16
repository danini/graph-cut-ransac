// Regression test: the final iterated least squares of GC-RANSAC must give each inlier its own
// Tukey weight. The homography, fundamental-matrix and rigid-transformation estimators read the
// weights of their non-minimal fit by position in the sample (weights[k] is the weight of point
// sample[k]; MAGSAC passes them that way), while GC-RANSAC stored them by point (weights[sample[k]]).
// For each estimator, on noisy inliers listed after the outliers, the first weighted fit of
// iteratedLeastSquaresFitting, started from the true model, must be the fit of the inliers with
// their own weights. Exits with 0 if every check passes.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "GCRANSAC.h"
#include "types.h"
#include "neighborhood/grid_neighborhood_graph.h"

namespace
{
	constexpr size_t outlier_number = 20, inlier_number = 60, point_number = outlier_number + inlier_number;
	constexpr double threshold = 3.0;

	std::mt19937 generator(42);
	double uniform(double a, double b) { return std::uniform_real_distribution<double>(a, b)(generator); }
	double noise() { return std::normal_distribution<double>(0.0, 1.0)(generator); }

	// The estimator, which also keeps the model of the first weighted non-minimal fit
	template <class _Estimator>
	class Recorder : public _Estimator
	{
	public:
		using _Estimator::_Estimator;
		mutable std::vector<gcransac::Model> first_weighted_fit;

		bool estimateModelNonminimal(const cv::Mat &data_, const size_t *sample_, const size_t &sample_number_,
			std::vector<gcransac::Model> *models_, const double *weights_ = nullptr) const override
		{
			const bool ok = _Estimator::estimateModelNonminimal(data_, sample_, sample_number_, models_, weights_);
			if (weights_ != nullptr && first_weighted_fit.empty())
				first_weighted_fit = *models_;
			return ok;
		}
	};

	// Gives access to the final iterated least squares
	template <class _Estimator>
	class TestGCRANSAC : public gcransac::GCRANSAC<_Estimator, gcransac::neighborhood::GridNeighborhoodGraph<4>>
	{
	public:
		bool fit(const cv::Mat &points_, const _Estimator &estimator_, std::vector<size_t> &inliers_, gcransac::Model &model_)
		{
			this->squared_truncated_threshold = 9.0 / 4.0 * threshold * threshold; // as in run()
			this->scoring_function->initialize(this->squared_truncated_threshold, points_.rows);
			return this->iteratedLeastSquaresFitting(points_, estimator_, threshold, inliers_, model_);
		}
	};

	// Checks that the first weighted fit of the iterated least squares, from 'true_model', is the fit of
	// its inliers with their Tukey weights, passed by position.
	template <class _Estimator>
	bool check(const char *name, const cv::Mat &points, const gcransac::Model &true_model)
	{
		const Recorder<_Estimator> estimator;
		std::vector<size_t> inliers;
		for (size_t i = 0; i < point_number; ++i)
			if (estimator.residual(points.row(i), true_model) < threshold)
				inliers.push_back(i);

		// The expected fit
		const double squared_truncated_threshold = 9.0 / 4.0 * threshold * threshold;
		std::vector<double> weights;
		for (const size_t i : inliers)
		{
			const double w = std::max(0.0, 1.0 - estimator.squaredResidual(points.row(i), true_model) / squared_truncated_threshold);
			weights.push_back(w * w);
		}
		std::vector<gcransac::Model> expected;
		estimator._Estimator::estimateModelNonminimal(points, inliers.data(), inliers.size(), &expected, weights.data());

		gcransac::Model model = true_model;
		TestGCRANSAC<Recorder<_Estimator>> gcransac;
		gcransac.fit(points, estimator, inliers, model);

		double error = 1.0;
		if (expected.size() == 1 && estimator.first_weighted_fit.size() == 1)
		{
			const Eigen::MatrixXd a = expected[0].descriptor / expected[0].descriptor.norm();
			Eigen::MatrixXd b = estimator.first_weighted_fit[0].descriptor / estimator.first_weighted_fit[0].descriptor.norm();
			if ((a - b).norm() > (a + b).norm())
				b = -b;
			error = (a - b).norm();
		}
		const bool ok = error < 1e-9;
		std::printf("%-50s %.3g (%s)\n", name, error, ok ? "ok" : "FAILED");
		return ok;
	}
}

int main()
{
	bool ok = true;

	{ // Homography, points in pixels, the outliers first
		Eigen::Matrix3d H;
		H << 1.1, 0.05, 20.0, -0.03, 0.95, -10.0, 1e-4, -2e-4, 1.0;
		cv::Mat points(point_number, 4, CV_64F);
		for (size_t i = 0; i < point_number; ++i)
		{
			const Eigen::Vector3d x(uniform(0, 640), uniform(0, 480), 1.0);
			const Eigen::Vector2d y = (H * x).hnormalized();
			const bool outlier = i < outlier_number;
			points.at<double>(i, 0) = x(0);
			points.at<double>(i, 1) = x(1);
			points.at<double>(i, 2) = outlier ? uniform(0, 640) : y(0) + noise();
			points.at<double>(i, 3) = outlier ? uniform(0, 480) : y(1) + noise();
		}
		gcransac::Model model;
		model.descriptor = H;
		ok &= check<gcransac::utils::DefaultHomographyEstimator>("homography", points, model);
	}

	{ // Fundamental matrix of two views of 3D points, the outliers first
		Eigen::Matrix3d K;
		K << 500, 0, 320, 0, 500, 240, 0, 0, 1;
		const Eigen::Matrix3d R = Eigen::AngleAxisd(0.2, Eigen::Vector3d(0.3, 1.0, 0.1).normalized()).toRotationMatrix();
		const Eigen::Vector3d t(1.0, 0.1, 0.2);
		cv::Mat points(point_number, 4, CV_64F);
		for (size_t i = 0; i < point_number; ++i)
		{
			const Eigen::Vector3d X(uniform(-2, 2), uniform(-2, 2), uniform(4, 8));
			const Eigen::Vector2d x = (K * X).hnormalized(), y = (K * (R * X + t)).hnormalized();
			const bool outlier = i < outlier_number;
			points.at<double>(i, 0) = x(0) + noise();
			points.at<double>(i, 1) = x(1) + noise();
			points.at<double>(i, 2) = outlier ? uniform(0, 640) : y(0) + noise();
			points.at<double>(i, 3) = outlier ? uniform(0, 480) : y(1) + noise();
		}
		Eigen::Matrix3d tx;
		tx << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;
		gcransac::Model model;
		model.descriptor = K.inverse().transpose() * tx * R * K.inverse();
		ok &= check<gcransac::utils::DefaultFundamentalMatrixEstimator>("fundamental matrix", points, model);
	}

	{ // Rigid transformation of 3D points, the outliers first
		const Eigen::Matrix3d R = Eigen::AngleAxisd(0.3, Eigen::Vector3d(1.0, -0.5, 0.2).normalized()).toRotationMatrix();
		const Eigen::Vector3d t(5.0, -2.0, 3.0);
		cv::Mat points(point_number, 6, CV_64F);
		for (size_t i = 0; i < point_number; ++i)
		{
			const Eigen::Vector3d X(uniform(-50, 50), uniform(-50, 50), uniform(-50, 50));
			const Eigen::Vector3d Y = R * X + t;
			const bool outlier = i < outlier_number;
			for (int j = 0; j < 3; ++j)
			{
				points.at<double>(i, j) = X(j);
				points.at<double>(i, 3 + j) = outlier ? uniform(-50, 50) : Y(j) + noise();
			}
		}
		// The model maps (x, y, z, 1) to the second point by its transpose
		gcransac::Model model;
		model.descriptor.resize(4, 3);
		model.descriptor << R.transpose(), t.transpose();
		ok &= check<gcransac::utils::DefaultRigidTransformationEstimator>("rigid transformation", points, model);
	}

	return ok ? 0 : 1;
}
