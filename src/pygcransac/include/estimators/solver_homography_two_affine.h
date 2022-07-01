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
#include "homography_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyTwoAffineSolver : public SolverEngine
			{
			public:
				HomographyTwoAffineSolver()
				{
				}

				~HomographyTwoAffineSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}
				
				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sampleNumber_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool HomographyTwoAffineSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sampleNumber_ < sampleSize())
				{
					fprintf(stderr, "There were not enough affine correspondences provided for the solver (%d < 2).\n", sampleNumber_);
					return false;
				}

				// The number of equations per correspondences
				constexpr size_t kEquationNumber = 6;
				// The row number of the coefficient matrix
				const size_t kRowNumber =
					sampleNumber_ == 2 ? // If a minimal sample is used
					8 : // Use the constraints only from 1AC and 1PC.
					kEquationNumber * sampleNumber_; // Otherwise, use all constraints.
				// The coefficient matrix
				Eigen::MatrixXd coefficients(kRowNumber, 9);

				const size_t kColumns = data_.cols;
				const double *kDataPtr = reinterpret_cast<double *>(data_.data);
				size_t rowIdx = 0;
				double weight = 1.0;

				for (size_t i = 0; i < sampleNumber_; ++i)
				{
					const size_t kIdx = 
						sample_ == nullptr ? i : sample_[i];

					const double *kPointPtr = 
						kDataPtr + kIdx * kColumns;

					const double
						&x1 = kPointPtr[0],
						&y1 = kPointPtr[1],
						&x2 = kPointPtr[2],
						&y2 = kPointPtr[3],
						&a11 = kPointPtr[4],
						&a12 = kPointPtr[5],
						&a21 = kPointPtr[6],
						&a22 = kPointPtr[7];

					if (weights_ != nullptr)
						weight = weights_[kIdx];

					const double
						kMinusWeightTimesX1 = -weight * x1,
						kMinusWeightTimesY1 = -weight * y1,
						kWeightTimesX2 = weight * x2,
						kWeightTimesY2 = weight * y2;

					coefficients(rowIdx, 0) = kMinusWeightTimesX1;
					coefficients(rowIdx, 1) = kMinusWeightTimesY1;
					coefficients(rowIdx, 2) = -weight;
					coefficients(rowIdx, 3) = 0;
					coefficients(rowIdx, 4) = 0;
					coefficients(rowIdx, 5) = 0;
					coefficients(rowIdx, 6) = kWeightTimesX2 * x1;
					coefficients(rowIdx, 7) = kWeightTimesX2 * y1;
					coefficients(rowIdx, 8) = kWeightTimesX2;
					++rowIdx;

					coefficients(rowIdx, 0) = 0;
					coefficients(rowIdx, 1) = 0;
					coefficients(rowIdx, 2) = 0;
					coefficients(rowIdx, 3) = kMinusWeightTimesX1;
					coefficients(rowIdx, 4) = kMinusWeightTimesY1;
					coefficients(rowIdx, 5) = -weight;
					coefficients(rowIdx, 6) = kWeightTimesY2 * x1;
					coefficients(rowIdx, 7) = kWeightTimesY2 * y1;
					coefficients(rowIdx, 8) = kWeightTimesY2;
					++rowIdx;

					// If the minimal case is considered, we 
					// do not need all constraints to estimate 
					// the homography.
					if (sampleNumber_ == 2 && 
						i == 1)
						break;

					coefficients(rowIdx, 0) = -1;
					coefficients(rowIdx, 1) = 0;
					coefficients(rowIdx, 2) = 0;
					coefficients(rowIdx, 3) = 0;
					coefficients(rowIdx, 4) = 0;
					coefficients(rowIdx, 5) = 0;
					coefficients(rowIdx, 6) = x2 + a11 * x1;
					coefficients(rowIdx, 7) = a11 * y1;
					coefficients(rowIdx, 8) = a11;
					++rowIdx;

					coefficients(rowIdx, 0) = 0;
					coefficients(rowIdx, 1) = -1;
					coefficients(rowIdx, 2) = 0;
					coefficients(rowIdx, 3) = 0;
					coefficients(rowIdx, 4) = 0;
					coefficients(rowIdx, 5) = 0;
					coefficients(rowIdx, 6) = a12 * x1;
					coefficients(rowIdx, 7) = x2 + a12 * y1;
					coefficients(rowIdx, 8) = a12;
					++rowIdx;

					coefficients(rowIdx, 0) = 0;
					coefficients(rowIdx, 1) = 0;
					coefficients(rowIdx, 2) = 0;
					coefficients(rowIdx, 3) = -1;
					coefficients(rowIdx, 4) = 0;
					coefficients(rowIdx, 5) = 0;
					coefficients(rowIdx, 6) = y2 + a21 * x1;
					coefficients(rowIdx, 7) = a21 * y1;
					coefficients(rowIdx, 8) = a21;
					++rowIdx;

					coefficients(rowIdx, 0) = 0;
					coefficients(rowIdx, 1) = 0;
					coefficients(rowIdx, 2) = 0;
					coefficients(rowIdx, 3) = 0;
					coefficients(rowIdx, 4) = -1;
					coefficients(rowIdx, 5) = 0;
					coefficients(rowIdx, 6) = a22 * x1;
					coefficients(rowIdx, 7) = y2 + a22 * y1;
					coefficients(rowIdx, 8) = a22;
					++rowIdx;
				}

				// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
				// the solution is linear subspace of dimensionality 2.
				// => use the last two singular std::vectors as a basis of the space
				// (according to SVD properties)
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					// Theoretically, it would be faster to apply SVD only to matrix coefficients, but
					// multiplication is faster than SVD in the Eigen library. Therefore, it is faster
					// to apply SVD to a smaller matrix.
					coefficients,
					Eigen::ComputeThinV);

				const Eigen::Matrix<double, 9, 1> &h =
					svd.matrixV().rightCols<1>();

				Homography model;
				model.descriptor << h(0), h(1), h(2),
					h(3), h(4), h(5),
					h(6), h(7), h(8);
				model.descriptor /= model.descriptor(2,2);

				models_.emplace_back(model);
				return true;
			}
		}
	}
}