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

#include <iostream>

#include "utils.h"
#include "solver_engine.h"
#include "fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating an essential matrix between two images
			// assuming that the camera moves on a plane, e.g., it is attached to a moving vehicle.
			// This it the line-intersection-based solver from
			// Choi, Sunglok, and Jong-Hwan Kim. "Fast and reliable minimal relative pose estimation under planar motion." 
			// Image and Vision Computing 69 (2018): 103-112.
			class EssentialMatrixTwoPointsPlanar : public SolverEngine
			{
			public:
				EssentialMatrixTwoPointsPlanar()
				{
				}

				~EssentialMatrixTwoPointsPlanar()
				{
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation.
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}
				
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool EssentialMatrixTwoPointsPlanar::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Check if the sample size is correct, i.e.,
				// this solver can't solve the over-determined case. 
				constexpr size_t minimalSampleSize = sampleSize();
				if (sample_number_ != minimalSampleSize)
				{
					fprintf(stderr, "Method '%s' is used with incorrect sample size (%d instead of %d).\n",
						"EssentialMatrixTwoPointsPlanar", sample_number_, minimalSampleSize);
					return false;
				}

				// The pointer to the data
				const double * dataPointer = reinterpret_cast<double *>(data_.data);
				// The number of columns in the data matrix
				const int cols = data_.cols;
				// Building the coefficient matrices
				Eigen::Matrix<double, minimalSampleSize, 2> A1, A2;

				// Filling the coefficient matrix from each point correspondence
				const double *pointPointer = NULL;
				for (size_t pointIdx = 0; pointIdx < minimalSampleSize; ++pointIdx)
				{
					pointPointer = dataPointer + sample_[pointIdx] * cols;
					double x1 = *pointPointer++,
						y1 = *pointPointer++,
						x2 = *pointPointer++,
						y2 = *pointPointer;

					A1(pointIdx, 0) = y1;
					A1(pointIdx, 1) = -y1 * x2;
					A2(pointIdx, 0) = -y2;
					A2(pointIdx, 1) = x1 * y2;
				}

				Eigen::Matrix2d pseudoInverseA2 = (A2.transpose() * A2).inverse() * A2.transpose();
				Eigen::Matrix2d C = -pseudoInverseA2 * A1;
				Eigen::Matrix2d C2 = C.transpose() * C;

				double a = C2(0, 0);
				double b = C2(1, 0);
				double c = C2(1, 1);

				double CF1 = 4 * b*b + (a - c)*(a - c);
				double CF2 = -2 * (a - c)*(2 - a - c);
				double CF3 = (2 - a - c)*(2 - a - c) - 4 * b*b;
				
				double cos2alpha1 = (-CF2 + sqrt(CF2*CF2 - 4 * CF1*CF3)) / (2 * CF1);
				double cos2alpha2 = (-CF2 - sqrt(CF2*CF2 - 4 * CF1*CF3)) / (2 * CF1);

				std::vector<double> alphas = 
					{ acos(cos2alpha1) / 2, acos(cos2alpha2) / 2, -acos(cos2alpha1) / 2, -acos(cos2alpha2) / 2 };

				double alpha;
				Eigen::Vector2d v1, v2;

				for (size_t pose_idx = 0; pose_idx < 4; ++pose_idx)
				{
					alpha = alphas[pose_idx];

					v1(0) = cos(alpha);
					v1(1) = sin(alpha);
					
					v2 = C * v1;
					v2 = v2 / v2.norm();

					double alpha = atan2(v1(1), v1(0));
					double alphaminusbeta = atan2(v2(1), v2(0));
					double beta = alpha - alphaminusbeta;
					
					Model model;
					model.descriptor.resize(3, 3);
					model.descriptor << 0, -sin(alpha), 0,
						sin(alphaminusbeta), 0, -cos(alphaminusbeta),
						0, cos(alpha), 0;
					models_.push_back(model);
				}
				return true;
			}
		}
	}
}