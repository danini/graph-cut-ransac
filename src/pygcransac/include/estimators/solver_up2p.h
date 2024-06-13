// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "solver_engine.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class UP2PSolver : public SolverEngine
			{
			public:
				UP2PSolver()
				{
				}

				~UP2PSolver()
				{
				}


				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 2;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 2;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
				int solve_quadratic_real(double a, double b, double c, double roots[2]) const
				{
					double b2m4ac = b * b - 4 * a * c;
					if (b2m4ac < 0)
						return 0;

					double sq = std::sqrt(b2m4ac);

					// Choose sign to avoid cancellations
					roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
					roots[1] = c / (a * roots[0]);

					return 2;
				}
			};
			
			OLGA_INLINE bool UP2PSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				const double * data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t columns = data_.cols;

				Eigen::Vector2d x[2];
				Eigen::Vector3d X[2];
				
				for ( int k = 0; k < 2; k++ )
				{
					double * data_ptr;
					if (sample_ == nullptr)
						data_ptr = reinterpret_cast<double *>(data_.row(k).data);
					else
						data_ptr = reinterpret_cast<double *>(data_.row(sample_[k]).data);

					x[k] = Eigen::Vector2d(data_ptr[0], data_ptr[1]);
					X[k] = Eigen::Vector3d(data_ptr[2], data_ptr[3], data_ptr[4]);
				}

				Eigen::Matrix<double, 4, 4> A;
				Eigen::Matrix<double, 4, 2> b;

				A << -x[0](2), 0, x[0](0), X[0](0) * x[0](2) - X[0](2) * x[0](0), 0, -x[0](2), x[0](1),
					-X[0](1) * x[0](2) - X[0](2) * x[0](1), -x[1](2), 0, x[1](0), X[1](0) * x[1](2) - X[1](2) * x[1](0), 0,
					-x[1](2), x[1](1), -X[1](1) * x[1](2) - X[1](2) * x[1](1);
				b << -2 * X[0](0) * x[0](0) - 2 * X[0](2) * x[0](2), X[0](2) * x[0](0) - X[0](0) * x[0](2), -2 * X[0](0) * x[0](1),
					X[0](2) * x[0](1) - X[0](1) * x[0](2), -2 * X[1](0) * x[1](0) - 2 * X[1](2) * x[1](2),
					X[1](2) * x[1](0) - X[1](0) * x[1](2), -2 * X[1](0) * x[1](1), X[1](2) * x[1](1) - X[1](1) * x[1](2);

				// b = A.partialPivLu().solve(b);
				b = A.inverse() * b;

				const double c2 = b(3, 0);
				const double c3 = b(3, 1);

				double qq[2];
				const int sols = solve_quadratic_real(1.0, c2, c3, qq);

				models_.clear();
				for (int i = 0; i < sols; ++i) {
					const double q = qq[i];
					const double q2 = q * q;
					const double inv_norm = 1.0 / (1 + q2);
					const double cq = (1 - q2) * inv_norm;
					const double sq = 2 * q * inv_norm;

					Eigen::Matrix3d R;
					R.setIdentity();
					R(0, 0) = cq;
					R(0, 2) = sq;
					R(2, 0) = -sq;
					R(2, 2) = cq;

					Eigen::Vector3d t;
					t = b.block<3, 1>(0, 0) * q + b.block<3, 1>(0, 1);
					t *= -inv_norm;

					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << R, t;
					models_.emplace_back(model);
				}
				
				return models_.size();
			}
		}
	}
}