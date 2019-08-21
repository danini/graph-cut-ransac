#pragma once

#include <math.h>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
template<class _MinimalSolverEngine, class _NonMinimalSolverEngine>
class EssentialMatrixEstimator : public theia::Estimator < cv::Mat, Model >
{
protected:
	const std::shared_ptr<const _MinimalSolverEngine> minimal_solver;
	const std::shared_ptr<const _NonMinimalSolverEngine> non_minimal_solver;

	const Eigen::Matrix3d intrinsics_src, // The intrinsic parameters of the source camera
		intrinsics_dst; // The intrinsic parameters of the destination camera

public:
	EssentialMatrixEstimator(Eigen::Matrix3d intrinsics_src_, // The intrinsic parameters of the source camera
		Eigen::Matrix3d intrinsics_dst_) : // The intrinsic parameters of the destination camera
		intrinsics_src(intrinsics_src_),
		intrinsics_dst(intrinsics_dst_),
		minimal_solver(std::make_shared<const _MinimalSolverEngine>()),
		non_minimal_solver(std::make_shared<const _NonMinimalSolverEngine>())
	{}
	~EssentialMatrixEstimator() {}

	// The size of a minimal sample required for the estimation
	constexpr size_t sampleSize() const {
		return _MinimalSolverEngine::sampleSize();
	}

	// The size of a sample when doing inner RANSAC on a non-minimal sample
	inline size_t inlierLimit() const {
		return 7 * sampleSize();
	}

	// Estimating the essential matrix from a minimal sample
	inline bool estimateModel(const cv::Mat& data, // The data points
		const size_t *sample, // The selected sample which will be used for estimation
		std::vector<Model>* models) const // The estimated model parameters
	{
		constexpr size_t sample_size = 5; // The size of a minimal sample

		// Estimating the model parameters by the solver engine
		if (!minimal_solver->estimateModel(data, // The data points
			sample, // The selected sample which will be used for estimation
			sample_size, // The size of a minimal sample required for the estimation
			*models)) // The estimated model parameters
			return false;

		/* Orientation constraint check */
		for (short model_idx = models->size() - 1; model_idx >= 0; --model_idx)
			if (!all_ori_valid(models->at(model_idx).descriptor,
				data, // The data points
				sample, // The selected sample which will be used for estimation
				sample_size)) // The size of a minimal sample required for the estimation
				models->erase(models->begin() + model_idx); // Delete the model if the orientation constraint does not hold

		// Return true, if at least one model is kept
		return models->size() > 0;
	}

	// The squared sampson distance between a point correspondence and an essential matrix
	inline double squaredSampsonDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		const double residual = sampsonDistance(point, descriptor);
		return residual * residual;
	}

	// The sampson distance between a point correspondence and an essential matrix
	inline double sampsonDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		const double* s = reinterpret_cast<double *>(point.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double e11 = descriptor(0, 0);
		const double e12 = descriptor(0, 1);
		const double e13 = descriptor(0, 2);
		const double e21 = descriptor(1, 0);
		const double e22 = descriptor(1, 1);
		const double e23 = descriptor(1, 2);
		const double e31 = descriptor(2, 0);
		const double e32 = descriptor(2, 1);
		const double e33 = descriptor(2, 2);

		double rxc = e11 * x2 + e21 * y2 + e31;
		double ryc = e12 * x2 + e22 * y2 + e32;
		double rwc = e13 * x2 + e23 * y2 + e33;
		double r = (x1 * rxc + y1 * ryc + rwc);
		double rx = e11 * x1 + e12 * y1 + e13;
		double ry = e21 * x1 + e22 * y1 + e23;

		return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry);
	}

	// The symmetric epipolar distance between a point correspondence and an essential matrix
	inline double symmetricEpipolarDistance(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		const double* s = reinterpret_cast<double *>(point.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double e11 = descriptor(0, 0);
		const double e12 = descriptor(0, 1);
		const double e13 = descriptor(0, 2);
		const double e21 = descriptor(1, 0);
		const double e22 = descriptor(1, 1);
		const double e23 = descriptor(1, 2);
		const double e31 = descriptor(2, 0);
		const double e32 = descriptor(2, 1);
		const double e33 = descriptor(2, 2);

		const double rxc = e11 * x2 + e21 * y2 + e31;
		const double ryc = e12 * x2 + e22 * y2 + e32;
		const double rwc = e13 * x2 + e23 * y2 + e33;
		const double r = (x1 * rxc + y1 * ryc + rwc);
		const double rx = e11 * x1 + e12 * y1 + e13;
		const double ry = e21 * x1 + e22 * y1 + e23;
		const double a = rxc * rxc + ryc * ryc;
		const double b = rx * rx + ry * ry;

		return r * r * (a + b) / (a * b);
	}

	// The squared residual function used for deciding which points are inliers
	inline double squaredResidual(const cv::Mat& point,
		const Model& model) const
	{
		return squaredSampsonDistance(point, model.descriptor);
	}

	// The squared residual function used for deciding which points are inliers
	inline double squaredResidual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		return squaredSampsonDistance(point, descriptor);
	}

	// The residual function used for deciding which points are inliers
	inline double residual(const cv::Mat& point,
		const Model& model) const
	{
		return residual(point, model.descriptor);
	}

	// The residual function used for deciding which points are inliers
	inline double residual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		return sampsonDistance(point, descriptor);
	}

	// Validate the model by checking the number of inlier with symmetric epipolar distance
	// instead of Sampson distance. In general, Sampson distance is more accurate but less
	// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
	// every so-far-the-best model is checked if it has enough inlier with symmetric
	// epipolar distance as well. 
	bool isValidModel(const Model& model,
		const cv::Mat& data,
		const std::vector<size_t> &inliers,
		const double threshold) const
	{
		size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
		const Eigen::Matrix3d &descriptor = model.descriptor; // The decriptor of the current model
		static const size_t M = sampleSize(); // Size of a minimal sample

		// Iterate through the inliers determined by Sampson distance
		for (const auto &idx : inliers)
			// Calculate the residual using symmetric epipolar distance and check if
			// it is smaller than the threshold.
			if (symmetricEpipolarDistance(data.row(idx), descriptor) < threshold)
				// Increase the inlier number and terminate if enough inliers have been found.
				if (++inlier_number > M)
					return true;
		// If the algorithm has not terminated earlier, there are not enough inliers.
		return false;
	}

	inline bool estimateModelNonminimal(
		const cv::Mat& data,
		const size_t *sample,
		size_t sample_number,
		std::vector<Model>* models) const
	{
		// model calculation 
		const size_t M = sample_number;

		if (M < 5)
			return false;

		cv::Mat normalized_points(M, data.cols, data.type()); // The normalized point coordinates
		Eigen::Matrix3d T1, T2; // The normalizing transformations in the 1st and 2nd images

		// Normalize the point coordinates to achieve numerical stability when
		// applying the least-squares model fitting.
		if (!normalizePoints(data, // The data points
			sample, // The points to which the model will be fit
			M, // The number of points
			normalized_points, // The normalized point coordinates
			T1, // The normalizing transformation in the first image
			T2)) // The normalizing transformation in the second image
			return false;

		// The eight point fundamental matrix fitting algorithm
		if (!non_minimal_solver->estimateModel(normalized_points,
			nullptr,
			M,
			*models))
			return false;
		
		/* Orientation constraint check */
		for (short model_idx = models->size() - 1; model_idx >= 0; --model_idx)
			if (!all_ori_valid(models->at(model_idx).descriptor,
				data,
				sample,
				M))
				models->erase(models->begin() + model_idx);

		// Denormalizing the estimated fundamental matrices
		const Eigen::Matrix3d T2_transpose = T2.transpose();
		for (auto &model : *models)
			model.descriptor = T2_transpose * model.descriptor * T1;
		return true;
	}

	inline bool normalizePoints(
		const cv::Mat& data, // The data points
		const size_t *sample, // The points to which the model will be fit
		size_t sample_number,// The number of points
		cv::Mat &normalized_points, // The normalized point coordinates
		Eigen::Matrix3d &T1, // The normalizing transformation in the first image
		Eigen::Matrix3d &T2) const // The normalizing transformation in the second image
	{
		const size_t cols = data.cols;
		double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points.data);
		const double *points_ptr = reinterpret_cast<double *>(data.data);

		double mass_point_src[2], // Mass point in the first image
			mass_point_dst[2]; // Mass point in the second image

		// Initializing the mass point coordinates
		mass_point_src[0] =
			mass_point_src[1] =
			mass_point_dst[0] =
			mass_point_dst[1] =
			0.0;

		// Calculating the mass points in both images
		for (size_t i = 0; i < sample_number; ++i)
		{
			// Get pointer of the current point
			const double *d_idx = points_ptr + cols * sample[i];

			// Add the coordinates to that of the mass points
			mass_point_src[0] += d_idx[0];
			mass_point_src[1] += d_idx[1];
			mass_point_dst[0] += d_idx[2];
			mass_point_dst[1] += d_idx[3];
		}

		// Get the average
		mass_point_src[0] /= sample_number;
		mass_point_src[1] /= sample_number;
		mass_point_dst[0] /= sample_number;
		mass_point_dst[1] /= sample_number;

		// Get the mean distance from the mass points
		double average_distance_src = 0.0,
			average_distance_dst = 0.0;
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = d_idx[0];
			const double y1 = d_idx[1];
			const double x2 = d_idx[2];
			const double y2 = d_idx[3];

			const double dx1 = mass_point_src[0] - x1;
			const double dy1 = mass_point_src[1] - y1;
			const double dx2 = mass_point_dst[0] - x2;
			const double dy2 = mass_point_dst[1] - y2;

			average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
			average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		average_distance_src /= sample_number;
		average_distance_dst /= sample_number;

		// Calculate the sqrt(2) / MeanDistance ratios
		static const double sqrt_2 = sqrt(2);
		const double ratio_src = sqrt_2 / average_distance_src;
		const double ratio_dst = sqrt_2 / average_distance_dst;

		// Compute the normalized coordinates
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = d_idx[0];
			const double y1 = d_idx[1];
			const double x2 = d_idx[2];
			const double y2 = d_idx[3];

			*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
			*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
			*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
			*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;
		}

		// Creating the normalizing transformations
		T1 << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1;

		T2 << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1;
		return true;
	}

	/************** oriented epipolar constraints ******************/
	inline void epipole(Eigen::Vector3d &ec,
		const Eigen::Matrix3d &F) const
	{
		ec = F.row(0).cross(F.row(2));

		for (auto i = 0; i < 3; i++)
			if ((ec(i) > 1.9984e-15) ||
				(ec(i) < -1.9984e-15))
				return;
		ec = F.row(1).cross(F.row(2));
	}

	inline double getorisig(const Eigen::Matrix3d &F,
		const Eigen::Vector3d &ec,
		const cv::Mat &u) const
	{
		double s1, s2;
		s1 = F(0, 0) * u.at<double>(2) + F(1, 0) * u.at<double>(3) + F(2, 0);
		s2 = ec(1) - ec(2) * u.at<double>(1);
		return s1 * s2;
	}

	inline int all_ori_valid(const Eigen::Matrix3d &F,
		const cv::Mat &data,
		const size_t *sample,
		int N) const
	{
		Eigen::Vector3d ec;
		double sig, sig1;
		int i;
		epipole(ec, F);

		if (sample == nullptr)
		{
			sig1 = getorisig(F, ec, data.row(0));
			for (i = 1; i < N; i++)
			{
				sig = getorisig(F, ec, data.row(i));
				if (sig1 * sig < 0) return 0;
			}
		}
		else
		{
			sig1 = getorisig(F, ec, data.row(sample[0]));
			for (i = 1; i < N; i++)
			{
				sig = getorisig(F, ec, data.row(sample[i]));
				if (sig1 * sig < 0) return 0;
			}
		}
		return 1;
	}
};