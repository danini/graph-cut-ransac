// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
#pragma once

#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace solver
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		class FundamentalMatrixEightPointSolver : public SolverEngine
		{
		public:
			FundamentalMatrixEightPointSolver()
			{
			}

			~FundamentalMatrixEightPointSolver()
			{
			}

			static constexpr size_t sampleSize()
			{
				return 8;
			}

			inline bool estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_) const;
		};

		inline bool FundamentalMatrixEightPointSolver::estimateModel(
			const cv::Mat& data_,
			const size_t *sample_,
			size_t sample_number_,
			std::vector<Model> &models_) const
		{
			if (sample_ == nullptr)
				sample_number_ = data_.rows;

			Eigen::MatrixXd coefficients(sample_number_, 9);
			const double *data_ptr = reinterpret_cast<double *>(data_.data);
			const int cols = data_.cols;

			// form a linear system: i-th row of A(=a) represents
			// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
			double x0, y0, x1, y1;
			for (size_t i = 0; i < sample_number_; i++)
			{
				int offset;
				if (sample_ == nullptr)
					offset = cols * i;
				else
					offset = cols * sample_[i];

				x0 = data_ptr[offset];
				y0 = data_ptr[offset + 1];
				x1 = data_ptr[offset + 2];
				y1 = data_ptr[offset + 3];

				coefficients(i, 0) = x1 * x0;
				coefficients(i, 1) = x1 * y0;
				coefficients(i, 2) = x1;
				coefficients(i, 3) = y1 * x0;
				coefficients(i, 4) = y1 * y0;
				coefficients(i, 5) = y1;
				coefficients(i, 6) = x0;
				coefficients(i, 7) = y0;
				coefficients(i, 8) = 1;
			}

			// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
			// the solution is linear subspace of dimensionality 2.
			// => use the last two singular std::vectors as a basis of the space
			// (according to SVD properties)
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(
				// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
				// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
				// to apply SVD to a smaller matrix.
				coefficients.transpose() * coefficients,
				Eigen::ComputeFullV);

			const Eigen::Matrix<double, 9, 1> &null_space =
				svd.matrixV().rightCols<1>();

			FundamentalMatrix model;
			model.descriptor << null_space(0), null_space(1), null_space(2),
				null_space(3), null_space(4), null_space(5),
				null_space(6), null_space(7), null_space(8);
			models_.push_back(model);
			return true;
		}
	}
}