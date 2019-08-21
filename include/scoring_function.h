#pragma once

#include <math.h>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

/* RANSAC Scoring */
struct Score {

	/* Number of inliers, rectangular gain function */
	size_t I;

	/* Score */
	double J;

	Score() :
		I(0),
		J(0.0)
	{

	}

	inline bool operator<(const Score& s2_)
	{
		return J < s2_.J &&
			I <= s2_.I;
	}

	inline bool operator>(const Score& s2_)
	{
		return *this > s2_;
	}
};

template<class _ModelEstimator>
class ScoringFunction
{
public:
	ScoringFunction()
	{

	}

	virtual ~ScoringFunction()
	{

	}

	virtual inline Score getScore(const cv::Mat &points_, // The input data points
		const Model &model_, // The current model parameters
		const _ModelEstimator &estimator_, // The model estimator
		const double threshold_, // The inlier-outlier threshold
		std::vector<size_t> &inliers_, // The selected inliers
		const Score &best_score_ = Score(), // The score of the current so-far-the-best model
		const bool store_inliers_ = true) const = 0;

	virtual void initialize(const double threshold_,
		const size_t point_number_) = 0;

};

template<class _ModelEstimator>
class MSACScoringFunction : public ScoringFunction<_ModelEstimator>
{
protected:
	double squared_truncated_threshold; // Squared truncated threshold
	size_t point_number; // Number of points

public:
	MSACScoringFunction()
	{

	}

	~MSACScoringFunction()
	{

	}

	void initialize(const double squared_truncated_threshold_,
		const size_t point_number_)
	{
		squared_truncated_threshold = squared_truncated_threshold_;
		point_number = point_number_;
	}

	// Return the score of a model w.r.t. the data points and the threshold
	inline Score getScore(const cv::Mat &points_, // The input data points
		const Model &model_, // The current model parameters
		const _ModelEstimator &estimator_, // The model estimator
		const double threshold_, // The inlier-outlier threshold
		std::vector<size_t> &inliers_, // The selected inliers
		const Score &best_score_ = Score(), // The score of the current so-far-the-best model
		const bool store_inliers_ = true) const
	{
		Score score; // The current score
		if (store_inliers_) // If the inlier should be stored, clear the variables
			inliers_.clear();
		double squared_residual; // The point-to-model residual

		// Iterate through all points, calculate the squared_residuals and store the points as inliers if needed.
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			// Calculate the point-to-model residual
			squared_residual = estimator_.squaredResidual(points_.row(point_idx), model_.descriptor);

			// If the residual is smaller than the threshold, store it as an inlier and
			// increase the score.
			if (squared_residual < squared_truncated_threshold)
			{
				if (store_inliers_) // Store the point as an inlier if needed.
					inliers_.emplace_back(point_idx);

				// Increase the inlier number
				++(score.I);
				// Increase the score. The original truncated quadratic loss is as follows: 
				// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
				score.J += squared_residual; // Truncated quadratic cost
			}

			// Interrupt if there is no chance of being better than the best model
			if (point_number - point_idx + score.I < best_score_.I)
				return Score();
		}

		// Return the final score
		return score;
	}
};