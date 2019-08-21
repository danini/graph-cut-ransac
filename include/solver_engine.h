#pragma once

#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>
#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "model.h"

namespace solver
{
	// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
	class SolverEngine
	{
	public:
		SolverEngine()
		{
		}

		~SolverEngine()
		{
		}

		// The minimum number of points required for the estimation
		static constexpr size_t sampleSize()
		{
			return 0;
		}

		// Estimate the model parameters from the given point sample
		virtual inline bool estimateModel(
			const cv::Mat& data_,
			const size_t *sample_,
			size_t sample_number_,
			std::vector<Model> &models_) const = 0;
	};
}