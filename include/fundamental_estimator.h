#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
	class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
class FundamentalMatrixEstimator : public theia::Estimator < cv::Mat, Model >
{
protected:
	// Minimal solver engine used for estimating a model from a minimal sample
	const std::shared_ptr<const _MinimalSolverEngine> minimal_solver; 

	// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
	const std::shared_ptr<const _NonMinimalSolverEngine> non_minimal_solver; 

	// The lower bound of the inlier ratio which is required to pass the validity test.
	// The validity test measures what proportion of the inlier (by Sampson distance) is inlier
	// when using symmetric epipolar distance. 
	const double minimum_inlier_ratio_in_validity_check; 

public:
	FundamentalMatrixEstimator(const double minimum_inlier_ratio_in_validity_check_ = 0.5) :
		// Minimal solver engine used for estimating a model from a minimal sample
		minimal_solver(std::make_shared<const _MinimalSolverEngine>()), 
		// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
		non_minimal_solver(std::make_shared<const _NonMinimalSolverEngine>()), 
		// The lower bound of the inlier ratio which is required to pass the validity test.
		// It is clamped to be in interval [0, 1].
		minimum_inlier_ratio_in_validity_check(std::clamp(minimum_inlier_ratio_in_validity_check_, 0.0, 1.0)) 
	{}

	~FundamentalMatrixEstimator() {}

	// The size of a non-minimal sample required for the estimation
	static constexpr size_t nonMinimalSampleSize() {
		return _NonMinimalSolverEngine::sampleSize();
	}

	// The size of a minimal sample required for the estimation
	static constexpr size_t sampleSize() {
		return _MinimalSolverEngine::sampleSize();
	}

	// The size of a sample when doing inner RANSAC on a non-minimal sample
	inline size_t inlierLimit() const {
		return 7 * sampleSize();
	}

	inline bool estimateModel(const cv::Mat& data,
		const size_t *sample,
		std::vector<Model>* models) const
	{
		// Model calculation by the seven point algorithm
		constexpr size_t sample_size = 7;

		// Estimate the model parameters by the minimal solver
		minimal_solver->estimateModel(data, 
			sample, 
			sample_size, 
			*models);

		// Orientation constraint check 
		for (short model_idx = models->size() - 1; model_idx >= 0; --model_idx)
			if (!isOrientationValid(models->at(model_idx).descriptor,
				data,
				sample,
				sample_size))
				models->erase(models->begin() + model_idx);

		// The estimation was successfull if at least one model is kept
		return models->size() > 0;
	}

	inline double squaredSampsonDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		const double residual = sampsonDistance(point, descriptor);
		return residual * residual;
	}

	inline double sampsonDistance(const cv::Mat& point_,
		const Eigen::Matrix3d& descriptor_) const
	{
		const double* s = reinterpret_cast<double *>(point_.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double f11 = descriptor_(0, 0);
		const double f12 = descriptor_(0, 1);
		const double f13 = descriptor_(0, 2);
		const double f21 = descriptor_(1, 0);
		const double f22 = descriptor_(1, 1);
		const double f23 = descriptor_(1, 2);
		const double f31 = descriptor_(2, 0);
		const double f32 = descriptor_(2, 1);
		const double f33 = descriptor_(2, 2);

		double rxc = f11 * x2 + f21 * y2 + f31;
		double ryc = f12 * x2 + f22 * y2 + f32;
		double rwc = f13 * x2 + f23 * y2 + f33;
		double r = (x1 * rxc + y1 * ryc + rwc);
		double rx = f11 * x1 + f12 * y1 + f13;
		double ry = f21 * x1 + f22 * y1 + f23;

		return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry);
	}

	inline double symmetricEpipolarDistance(const cv::Mat& point_,
		const Eigen::MatrixXd& descriptor_) const
	{
		const double* s = reinterpret_cast<double *>(point_.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double f11 = descriptor_(0, 0);
		const double f12 = descriptor_(0, 1);
		const double f13 = descriptor_(0, 2);
		const double f21 = descriptor_(1, 0);
		const double f22 = descriptor_(1, 1);
		const double f23 = descriptor_(1, 2);
		const double f31 = descriptor_(2, 0);
		const double f32 = descriptor_(2, 1);
		const double f33 = descriptor_(2, 2);

		const double rxc = f11 * x2 + f21 * y2 + f31;
		const double ryc = f12 * x2 + f22 * y2 + f32;
		const double rwc = f13 * x2 + f23 * y2 + f33;
		const double r = (x1 * rxc + y1 * ryc + rwc);
		const double rx = f11 * x1 + f12 * y1 + f13;
		const double ry = f21 * x1 + f22 * y1 + f23;
		const double a = rxc * rxc + ryc * ryc;
		const double b = rx * rx + ry * ry;

		return r * r * (a + b) / (a * b);
	}

	inline double squaredResidual(const cv::Mat& point_,
		const Model& model_) const
	{
		return squaredSampsonDistance(point_, model_.descriptor);
	}

	inline double squaredResidual(const cv::Mat& point_,
		const Eigen::MatrixXd& descriptor_) const
	{
		return squaredSampsonDistance(point_, descriptor_);
	}

	inline double residual(const cv::Mat& point_,
		const Model& model_) const
	{
		return residual(point_, model_.descriptor);
	}

	inline double residual(const cv::Mat& point_,
		const Eigen::MatrixXd& descriptor_) const
	{
		return sampsonDistance(point_, descriptor_);
	}

	// Validate the model by checking the number of inlier with symmetric epipolar distance
	// instead of Sampson distance. In general, Sampson distance is more accurate but less
	// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
	// every so-far-the-best model is checked if it has enough inlier with symmetric
	// epipolar distance as well. 
	inline bool isValidModel(const Model& model_,
		const cv::Mat& data_,
		const std::vector<size_t> &inliers_,
		const double threshold_) const
	{
		size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
		const Eigen::Matrix3d &descriptor = model_.descriptor; // The decriptor of the current model
		constexpr size_t sample_size = sampleSize(); // Size of a minimal sample
		// Minimum number of inliers which should be inlier as well when using symmetric epipolar distance instead of Sampson distance
		const size_t minimum_inlier_number = 
			MAX(sample_size, inliers_.size() * minimum_inlier_ratio_in_validity_check);
		
		// Iterate through the inliers determined by Sampson distance
		for (const auto &idx : inliers_)
			// Calculate the residual using symmetric epipolar distance and check if
			// it is smaller than the threshold_.
			if (symmetricEpipolarDistance(data_.row(idx), descriptor) < threshold_)
				// Increase the inlier number and terminate if enough inliers have been found.
				if (++inlier_number > minimum_inlier_number)
					return true;
		// If the algorithm has not terminated earlier, there are not enough inliers.
		return false;
	}

	inline bool estimateModelNonminimal(
		const cv::Mat& data_,
		const size_t *sample_,
		const size_t &sample_number_,
		std::vector<Model>* models_) const
	{
		if (sample_number_ < nonMinimalSampleSize())
			return false;

		cv::Mat normalized_points(sample_number_, data_.cols, data_.type()); // The normalized point coordinates
		Eigen::Matrix3d normalizing_transform_source, // The normalizing transformations in the source image
			normalizing_transform_destination; // The normalizing transformations in the destination image

		// Normalize the point coordinates to achieve numerical stability when
		// applying the least-squares model fitting.
		if (!normalizePoints(data_, // The data points
			sample_, // The points to which the model will be fit
			sample_number_, // The number of points
			normalized_points, // The normalized point coordinates
			normalizing_transform_source, // The normalizing transformation in the first image
			normalizing_transform_destination)) // The normalizing transformation in the second image
			return false;

		// The eight point fundamental matrix fitting algorithm
		non_minimal_solver->estimateModel(normalized_points,
			nullptr,
			sample_number_,
			*models_);

		// Denormalizing the estimated fundamental matrices
		const Eigen::Matrix3d normalizing_transform_destination_transpose = normalizing_transform_destination.transpose();
		for (auto &model : *models_)
		{
			// Transform the estimated fundamental matrix back to the not normalized space
			model.descriptor = normalizing_transform_destination_transpose * model.descriptor * normalizing_transform_source;
			// Enforce the rank-two constraint on the estimated fundamental matrix
			enforceRankTwoConstraint(model); 

			// Normalizing the fundamental matrix elements
			model.descriptor.normalize();
			if (model.descriptor(2, 2) < 0)
				model.descriptor = -model.descriptor;
		}
		return true;
	}

	inline void enforceRankTwoConstraint(Model &model_) const
	{
		// Applying SVD decomposition to the estimated fundamental matrix
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(
			model_.descriptor,
			Eigen::ComputeThinU | Eigen::ComputeThinV);

		// Making the last singular value zero
		Eigen::Matrix3d diagonal = svd.singularValues().asDiagonal();
		diagonal(2, 2) = 0.0;

		// Getting back the fundamental matrix from the SVD decomposition
		// using the new singular values
		model_.descriptor = 
			svd.matrixU() * diagonal * svd.matrixV().transpose();
	}

	inline bool normalizePoints(
		const cv::Mat& data_, // The data points
		const size_t *sample_, // The points to which the model will be fit
		const size_t &sample_number_,// The number of points
		cv::Mat &normalized_points_, // The normalized point coordinates
		Eigen::Matrix3d &normalizing_transform_source_, // The normalizing transformation in the first image
		Eigen::Matrix3d &normalizing_transform_destination_) const // The normalizing transformation in the second image
	{
		const size_t cols = data_.cols;
		double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points_.data);
		const double *points_ptr = reinterpret_cast<double *>(data_.data);

		double mass_point_src[2], // Mass point in the first image
			mass_point_dst[2]; // Mass point in the second image

		// Initializing the mass point coordinates
		mass_point_src[0] =
			mass_point_src[1] =
			mass_point_dst[0] =
			mass_point_dst[1] =
			0.0;

		// Calculating the mass points in both images
		for (size_t i = 0; i < sample_number_; ++i)
		{
			// Get pointer of the current point
			const double *d_idx = points_ptr + cols * sample_[i];

			// Add the coordinates to that of the mass points
			mass_point_src[0] += *(d_idx);
			mass_point_src[1] += *(d_idx + 1);
			mass_point_dst[0] += *(d_idx + 2);
			mass_point_dst[1] += *(d_idx + 3);
		}

		// Get the average
		mass_point_src[0] /= sample_number_;
		mass_point_src[1] /= sample_number_;
		mass_point_dst[0] /= sample_number_;
		mass_point_dst[1] /= sample_number_;

		// Get the mean distance from the mass points
		double average_distance_src = 0.0,
			average_distance_dst = 0.0;
		for (size_t i = 0; i < sample_number_; ++i)
		{
			const double *d_idx = points_ptr + cols * sample_[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			const double dx1 = mass_point_src[0] - x1;
			const double dy1 = mass_point_src[1] - y1;
			const double dx2 = mass_point_dst[0] - x2;
			const double dy2 = mass_point_dst[1] - y2;

			average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1);
			average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		average_distance_src /= sample_number_;
		average_distance_dst /= sample_number_;

		// Calculate the sqrt(2) / MeanDistance ratios
		const double ratio_src = M_SQRT2 / average_distance_src;
		const double ratio_dst = M_SQRT2 / average_distance_dst;

		// Compute the normalized coordinates
		for (size_t i = 0; i < sample_number_; ++i)
		{
			const double *d_idx = points_ptr + cols * sample_[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
			*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
			*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
			*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;
		}

		// Creating the normalizing transformations
		normalizing_transform_source_ << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1;

		normalizing_transform_destination_ << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1;
		return true;
	}

	/************** Oriented epipolar constraints ******************/
	inline void getEpipole(
		Eigen::Vector3d &epipole_, // The epipole 
		const Eigen::Matrix3d &fundamental_matrix_) const
	{
		constexpr double epsilon = 1.9984e-15;
		epipole_ = fundamental_matrix_.row(0).cross(fundamental_matrix_.row(2));

		for (auto i = 0; i < 3; i++)
			if ((epipole_(i) > epsilon) ||
				(epipole_(i) < -epsilon))
				return;
		epipole_ = fundamental_matrix_.row(1).cross(fundamental_matrix_.row(2));
	}

	inline double getOrientationSignum(
		const Eigen::Matrix3d &fundamental_matrix_,
		const Eigen::Vector3d &epipole_,
		const cv::Mat &point_) const
	{
		double signum1 = fundamental_matrix_(0, 0) * point_.at<double>(2) + fundamental_matrix_(1, 0) * point_.at<double>(3) + fundamental_matrix_(2, 0),
			signum2 = epipole_(1) - epipole_(2) * point_.at<double>(1);
		return signum1 * signum2;
	}

	inline int isOrientationValid(
		const Eigen::Matrix3d &fundamental_matrix_, // The fundamental matrix
		const cv::Mat &data_, // The data points
		const size_t *sample_, // The sample used for the estimation
		size_t sample_size_) const // The size of the sample
	{
		Eigen::Vector3d epipole; // The epipole in the second image
		getEpipole(epipole, fundamental_matrix_);

		double signum1, signum2;

		// The sample is null pointer, the method is applied to normalized data_
		if (sample_ == nullptr)
		{
			// Get the sign of orientation of the first point_ in the sample
			signum2 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(0));
			for (size_t i = 1; i < sample_size_; i++)
			{
				// Get the sign of orientation of the i-th point_ in the sample
				signum1 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(i));
				// The signs should be equal, otherwise, the fundamental matrix is invalid
				if (signum2 * signum1 < 0)
					return false;
			}
		}
		else
		{
			// Get the sign of orientation of the first point_ in the sample
			signum2 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(sample_[0]));
			for (size_t i = 1; i < sample_size_; i++)
			{
				// Get the sign of orientation of the i-th point_ in the sample
				signum1 = getOrientationSignum(fundamental_matrix_, epipole, data_.row(sample_[i]));
				// The signs should be equal, otherwise, the fundamental matrix is invalid
				if (signum2 * signum1 < 0)
					return false;
			}
		}
		return true;
	}
};