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

#include "estimators/solver_engine.h"
#include "estimators/fundamental_estimator.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class FundamentalMatrixFourSIFTSolver : public SolverEngine
			{
			public:
				FundamentalMatrixFourSIFTSolver()
				{
				}

				~FundamentalMatrixFourSIFTSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				static constexpr const char *name()
				{
					return "4SIFT";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 3;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sampleNumber_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};

			OLGA_INLINE bool FundamentalMatrixFourSIFTSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{				
				Eigen::MatrixXd coefficients(7, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;
				double c[4];
				double t0, t1, t2;
				int i, n;

				// Form a linear system: i-th row of A(=a) represents
				// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
				size_t row = 0;
				for (i = 0; i < sampleNumber_; i++)
				{
					const int sample_idx = sample_[i];
					const int offset = cols * sample_idx;

					const double
						u1 = data_ptr[offset],
						v1 = data_ptr[offset + 1],
						u2 = data_ptr[offset + 2],
						v2 = data_ptr[offset + 3],
						q1 = data_ptr[offset + 4],
						q2 = data_ptr[offset + 5],
						o1 = data_ptr[offset + 6],
						o2 = data_ptr[offset + 7];
    
					const double 
						s1 = sin(o1),
						c1 = cos(o1),
						s2 = sin(o2),
						c2 = cos(o2),
						q = q2 / q1;
						
					coefficients(row, 0) = u2 * u1;
					coefficients(row, 1) = u2 * v1;
					coefficients(row, 2) = u2;
					coefficients(row, 3) = v2 * u1;
					coefficients(row, 4) = v2 * v1;
					coefficients(row, 5) = v2;
					coefficients(row, 6) = u1;
					coefficients(row, 7) = v1;
					coefficients(row, 8) = 1;
					++row;

					if (i == 3)
						break;

					coefficients(row, 0) = s2*s2 * q2 * u1 - c1 * c2 * q1 * u2 - q2*u1;
					coefficients(row, 1) = s2*s2 * q2 * v1 - s1 * c2 * q1 * u2 - q2*v1;
					coefficients(row, 2) = s2*s2 * q2 -q2;
					coefficients(row, 3) = -c2 * s2 * q2 * u1 - c1*c2*q1*v2;
					coefficients(row, 4) = -c2 * s2 * q2 * v1 - s1*c2*q1*v2;
					coefficients(row, 5) = -c2*s2*q2;
					coefficients(row, 6) = -c1*c2*q1;
					coefficients(row, 7) = -s1*c2*q1;
					coefficients(row, 8) = 0;
					++row;
				}

				Eigen::Matrix<double, 9, 1> f1, f2;

				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
				if (lu.dimensionOfKernel() != 2) 
					return false;

				const Eigen::Matrix<double, 9, 2> null_space = 
					lu.kernel();

				f1 = null_space.col(0);
				f2 = null_space.col(1);

				// f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
				// as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
				// so f ~ lambda*f1 + (1 - lambda)*f2.
				// use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
				// it will be a cubic equation.
				// find c - polynomial coefficients.
				for (i = 0; i < 9; i++)
					f1[i] -= f2[i];

				t0 = f2[4] * f2[8] - f2[5] * f2[7];
				t1 = f2[3] * f2[8] - f2[5] * f2[6];
				t2 = f2[3] * f2[7] - f2[4] * f2[6];

				c[0] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2;

				c[1] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -
					f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
					f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
					f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
					f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
					f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
					f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);

				t0 = f1[4] * f1[8] - f1[5] * f1[7];
				t1 = f1[3] * f1[8] - f1[5] * f1[6];
				t2 = f1[3] * f1[7] - f1[4] * f1[6];

				c[2] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
					f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
					f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
					f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
					f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
					f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
					f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);

				c[3] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
				
				// Check if the sum of the polynomical coefficients is close to zero. 
				// In this case "psolve.realRoots(real_roots)" gets into an infinite loop.
				if (fabs(c[0]+c[1]+c[2]+c[3]) < 1e-9) 
					return false;

				// solve the cubic equation; there can be 1 to 3 roots ...
				Eigen::Matrix<double, 4, 1> polynomial;
				for (auto i = 0; i < 4; ++i)
					polynomial(i) = c[i];
				Eigen::PolynomialSolver<double, 3> psolve(polynomial);

				std::vector<double> real_roots;
				psolve.realRoots(real_roots);

				n = real_roots.size();
				if (n < 1 || n > 3)
					return false;

				double f[8];
				for (const double &root : real_roots)
				{
					// for each root form the fundamental matrix
					double lambda = root, 
						mu = 1.;
					double s = f1[8] * root + f2[8];

					// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
					if (fabs(s) > std::numeric_limits<double>::epsilon())
					{
						mu = 1.0f / s;
						lambda *= mu;

						for (auto i = 0; i < 8; ++i)
							f[i] = f1[i] * lambda + f2[i] * mu;

						FundamentalMatrix model;
						model.descriptor << f[0], f[1], f[2],
							f[3], f[4], f[5],
							f[6], f[7], 1.0;

						if (model.descriptor.hasNaN())
							continue;
						models_.push_back(model);
					}
				}
				return true;
			}
		}
	}
}