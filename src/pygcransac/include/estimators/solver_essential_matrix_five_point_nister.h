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
#include "../maths/sturm_polynomial_solver.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialMatrixFivePointNisterSolver : public SolverEngine
			{
			public:
				EssentialMatrixFivePointNisterSolver()
				{
				}

				~EssentialMatrixFivePointNisterSolver()
				{
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation.
				static constexpr bool needsGravity()
				{
					return false;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 5;
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
					return 10;
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
				OLGA_INLINE void computeTraceConstraints(
					const Eigen::Matrix<double, 4, 9> &N, 
					Eigen::Matrix<double, 10, 20> &coeffs) const;

				OLGA_INLINE void o1(const double a[4], const double b[4], double c[10]) const;
				OLGA_INLINE void o1p(const double a[4], const double b[4], double c[10]) const;
				OLGA_INLINE void o1m(const double a[4], const double b[4], double c[10]) const;
				OLGA_INLINE void o2(const double a[10], const double b[4], double c[20]) const;
				OLGA_INLINE void o2p(const double a[10], const double b[4], double c[20]) const;
			};

			// a, b are first order polys [x,y,z,1]
			// c is degree 2 poly with order
			// [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::o1(const double a[4], const double b[4], double c[10]) const
			{
				c[0] = a[0] * b[0];
				c[1] = a[0] * b[1] + a[1] * b[0];
				c[2] = a[0] * b[2] + a[2] * b[0];
				c[3] = a[0] * b[3] + a[3] * b[0];
				c[4] = a[1] * b[1];
				c[5] = a[1] * b[2] + a[2] * b[1];
				c[6] = a[1] * b[3] + a[3] * b[1];
				c[7] = a[2] * b[2];
				c[8] = a[2] * b[3] + a[3] * b[2];
				c[9] = a[3] * b[3];
			}

			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::o1p(const double a[4], const double b[4], double c[10]) const
			{
				c[0] += a[0] * b[0];
				c[1] += a[0] * b[1] + a[1] * b[0];
				c[2] += a[0] * b[2] + a[2] * b[0];
				c[3] += a[0] * b[3] + a[3] * b[0];
				c[4] += a[1] * b[1];
				c[5] += a[1] * b[2] + a[2] * b[1];
				c[6] += a[1] * b[3] + a[3] * b[1];
				c[7] += a[2] * b[2];
				c[8] += a[2] * b[3] + a[3] * b[2];
				c[9] += a[3] * b[3];
			}

			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::o1m(const double a[4], const double b[4], double c[10]) const
			{
				c[0] -= a[0] * b[0];
				c[1] -= a[0] * b[1] + a[1] * b[0];
				c[2] -= a[0] * b[2] + a[2] * b[0];
				c[3] -= a[0] * b[3] + a[3] * b[0];
				c[4] -= a[1] * b[1];
				c[5] -= a[1] * b[2] + a[2] * b[1];
				c[6] -= a[1] * b[3] + a[3] * b[1];
				c[7] -= a[2] * b[2];
				c[8] -= a[2] * b[3] + a[3] * b[2];
				c[9] -= a[3] * b[3];
			}

			// a is second degree poly with order
			// [ x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]
			// b is first degree with order
			// [x y z 1]
			// c is third degree with order (same as nister's paper)
			// [ x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1]
			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::o2(const double a[10], const double b[4], double c[20]) const
			{
				c[0] = a[0] * b[0];
				c[1] = a[4] * b[1];
				c[2] = a[0] * b[1] + a[1] * b[0];
				c[3] = a[1] * b[1] + a[4] * b[0];
				c[4] = a[0] * b[2] + a[2] * b[0];
				c[5] = a[0] * b[3] + a[3] * b[0];
				c[6] = a[4] * b[2] + a[5] * b[1];
				c[7] = a[4] * b[3] + a[6] * b[1];
				c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
				c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
				c[10] = a[2] * b[2] + a[7] * b[0];
				c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
				c[12] = a[3] * b[3] + a[9] * b[0];
				c[13] = a[5] * b[2] + a[7] * b[1];
				c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
				c[15] = a[6] * b[3] + a[9] * b[1];
				c[16] = a[7] * b[2];
				c[17] = a[7] * b[3] + a[8] * b[2];
				c[18] = a[8] * b[3] + a[9] * b[2];
				c[19] = a[9] * b[3];
			}

			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::o2p(const double a[10], const double b[4], double c[20]) const
			{
				c[0] += a[0] * b[0];
				c[1] += a[4] * b[1];
				c[2] += a[0] * b[1] + a[1] * b[0];
				c[3] += a[1] * b[1] + a[4] * b[0];
				c[4] += a[0] * b[2] + a[2] * b[0];
				c[5] += a[0] * b[3] + a[3] * b[0];
				c[6] += a[4] * b[2] + a[5] * b[1];
				c[7] += a[4] * b[3] + a[6] * b[1];
				c[8] += a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
				c[9] += a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
				c[10] += a[2] * b[2] + a[7] * b[0];
				c[11] += a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
				c[12] += a[3] * b[3] + a[9] * b[0];
				c[13] += a[5] * b[2] + a[7] * b[1];
				c[14] += a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
				c[15] += a[6] * b[3] + a[9] * b[1];
				c[16] += a[7] * b[2];
				c[17] += a[7] * b[3] + a[8] * b[2];
				c[18] += a[8] * b[3] + a[9] * b[2];
				c[19] += a[9] * b[3];
			}

			OLGA_INLINE void EssentialMatrixFivePointNisterSolver::computeTraceConstraints(
				const Eigen::Matrix<double, 4, 9> &N, 
				Eigen::Matrix<double, 10, 20> &coeffs) const 
			{
				double const *N_ptr = N.data();
				#define EE(i, j) N_ptr + 4 * (3 * j + i)

				double d[60];

				// Determinant constraint
				Eigen::Matrix<double, 1, 20> row;
				double *c_data = row.data();

				o1(EE(0, 1), EE(1, 2), d);
				o1m(EE(0, 2), EE(1, 1), d);
				o2(d, EE(2, 0), c_data);

				o1(EE(0, 2), EE(1, 0), d);
				o1m(EE(0, 0), EE(1, 2), d);
				o2p(d, EE(2, 1), c_data);

				o1(EE(0, 0), EE(1, 1), d);
				o1m(EE(0, 1), EE(1, 0), d);
				o2p(d, EE(2, 2), c_data);

				coeffs.row(9) = row;

				double *EET[3][3] = {{d, d + 10, d + 20}, {d + 10, d + 40, d + 30}, {d + 20, d + 30, d + 50}};

				// Compute EE^T (equation 20 in paper)
				for (int i = 0; i < 3; ++i) {
					for (int j = i; j < 3; ++j) {
						o1(EE(i, 0), EE(j, 0), EET[i][j]);
						o1p(EE(i, 1), EE(j, 1), EET[i][j]);
						o1p(EE(i, 2), EE(j, 2), EET[i][j]);
					}
				}

				// Subtract trace (equation 22 in paper)
				for (int i = 0; i < 10; ++i) {
					double t = 0.5 * (EET[0][0][i] + EET[1][1][i] + EET[2][2][i]);
					EET[0][0][i] -= t;
					EET[1][1][i] -= t;
					EET[2][2][i] -= t;
				}

				int cnt = 0;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						o2(EET[i][0], EE(0, j), c_data);
						o2p(EET[i][1], EE(1, j), c_data);
						o2p(EET[i][2], EE(2, j), c_data);
						coeffs.row(cnt++) = row;
					}
				}

			#undef EE
			}

			OLGA_INLINE bool EssentialMatrixFivePointNisterSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				Eigen::MatrixXd coefficients(sample_number_, 9);
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const int cols = data_.cols;

				// Step 1. Create the nx9 matrix containing epipolar constraints.
				//   Essential matrix is a linear combination of the 4 vectors spanning the null space of this
				//   matrix.
				int offset;
				double x0, y0, x1, y1, weight = 1.0;
				for (size_t i = 0; i < sample_number_; i++)
				{
					if (sample_ == nullptr)
					{
						offset = cols * i;
						if (weights_ != nullptr)
							weight = weights_[i];
					}
					else
					{
						offset = cols * sample_[i];
						if (weights_ != nullptr)
							weight = weights_[sample_[i]];
					}

					x0 = data_ptr[offset];
					y0 = data_ptr[offset + 1];
					x1 = data_ptr[offset + 2];
					y1 = data_ptr[offset + 3];
					
					// Precalculate these values to avoid calculating them multiple times
					const double
						weight_times_x0 = weight * x0,
						weight_times_x1 = weight * x1,
						weight_times_y0 = weight * y0,
						weight_times_y1 = weight * y1;

					coefficients.row(i) <<
						weight_times_x0 * x1,
						weight_times_x0 * y1,
						weight_times_x0,
						weight_times_y0 * x1,
						weight_times_y0 * y1,
						weight_times_y0,
						weight_times_x1,
						weight_times_y1,
						weight;
				}

				// Extract the null space from a minimal sampling (using LU) or non-minimal sampling (using SVD).
				Eigen::Matrix<double, 4, 9> nullSpace;

				if (sample_number_ == 5) {
					const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
					if (lu.dimensionOfKernel() != 4) {
						return false;
					}
					nullSpace = lu.kernel().transpose();
				}
				else {

					const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
						coefficients.transpose() * coefficients);
					const Eigen::ArrayXd &singularVals = eigensolver.eigenvalues().real();

					double smallestValues[3];
					smallestValues[0] = 
						smallestValues[1] = 
						smallestValues[2] = 
						1e10;
					size_t smallestIndices[3];

					for (size_t rowIdx = 0; rowIdx < singularVals.size(); ++rowIdx)
					{
						if (singularVals[rowIdx] < smallestValues[0])
						{
							smallestValues[2] = smallestValues[1];
							smallestValues[1] = smallestValues[0];
							smallestValues[0] = singularVals[rowIdx];
							
							smallestIndices[2] = smallestIndices[1];
							smallestIndices[1] = smallestIndices[0];
							smallestIndices[0] = rowIdx;
						} else if (singularVals[rowIdx] < smallestValues[1])
						{
							smallestValues[2] = smallestValues[1];
							smallestValues[1] = singularVals[rowIdx];
							
							smallestIndices[2] = smallestIndices[1];
							smallestIndices[1] = rowIdx;
						} else if (singularVals[rowIdx] < smallestValues[2])
						{
							smallestValues[2] = singularVals[rowIdx];							
							smallestIndices[2] = rowIdx;
						}
					}

					nullSpace << 
						eigensolver.eigenvectors().col(smallestIndices[0]).transpose(),
						eigensolver.eigenvectors().col(smallestIndices[1]).transpose(),
						eigensolver.eigenvectors().col(smallestIndices[2]).transpose();
				}

				// Compute equation coefficients for the trace constraints + determinant
				Eigen::Matrix<double, 10, 20> coeffs;
				computeTraceConstraints(nullSpace, coeffs);
				coeffs.block<10, 10>(0, 10) = coeffs.block<10, 10>(0, 0).partialPivLu().solve(coeffs.block<10, 10>(0, 10));

				// Perform eliminations using the 6 bottom rows
				Eigen::Matrix<double, 3, 13> A;
				for (int i = 0; i < 3; ++i) {
					A(i, 0) = 0.0;
					A.block<1, 3>(i, 1) = coeffs.block<1, 3>(4 + 2 * i, 10);
					A.block<1, 3>(i, 0) -= coeffs.block<1, 3>(5 + 2 * i, 10);

					A(i, 4) = 0.0;
					A.block<1, 3>(i, 5) = coeffs.block<1, 3>(4 + 2 * i, 13);
					A.block<1, 3>(i, 4) -= coeffs.block<1, 3>(5 + 2 * i, 13);

					A(i, 8) = 0.0;
					A.block<1, 4>(i, 9) = coeffs.block<1, 4>(4 + 2 * i, 16);
					A.block<1, 4>(i, 8) -= coeffs.block<1, 4>(5 + 2 * i, 16);
				}

				// Compute degree 10 poly representing determinant (equation 14 in the paper)
				double c[11];
				c[0] = A(0, 12) * A(1, 3) * A(2, 7) - A(0, 12) * A(1, 7) * A(2, 3) - A(0, 3) * A(2, 7) * A(1, 12) +
					A(0, 7) * A(2, 3) * A(1, 12) + A(0, 3) * A(1, 7) * A(2, 12) - A(0, 7) * A(1, 3) * A(2, 12);
				c[1] = A(0, 11) * A(1, 3) * A(2, 7) - A(0, 11) * A(1, 7) * A(2, 3) + A(0, 12) * A(1, 2) * A(2, 7) +
					A(0, 12) * A(1, 3) * A(2, 6) - A(0, 12) * A(1, 6) * A(2, 3) - A(0, 12) * A(1, 7) * A(2, 2) -
					A(0, 2) * A(2, 7) * A(1, 12) - A(0, 3) * A(2, 6) * A(1, 12) - A(0, 3) * A(2, 7) * A(1, 11) +
					A(0, 6) * A(2, 3) * A(1, 12) + A(0, 7) * A(2, 2) * A(1, 12) + A(0, 7) * A(2, 3) * A(1, 11) +
					A(0, 2) * A(1, 7) * A(2, 12) + A(0, 3) * A(1, 6) * A(2, 12) + A(0, 3) * A(1, 7) * A(2, 11) -
					A(0, 6) * A(1, 3) * A(2, 12) - A(0, 7) * A(1, 2) * A(2, 12) - A(0, 7) * A(1, 3) * A(2, 11);
				c[2] = A(0, 10) * A(1, 3) * A(2, 7) - A(0, 10) * A(1, 7) * A(2, 3) + A(0, 11) * A(1, 2) * A(2, 7) +
					A(0, 11) * A(1, 3) * A(2, 6) - A(0, 11) * A(1, 6) * A(2, 3) - A(0, 11) * A(1, 7) * A(2, 2) +
					A(1, 1) * A(0, 12) * A(2, 7) + A(0, 12) * A(1, 2) * A(2, 6) + A(0, 12) * A(1, 3) * A(2, 5) -
					A(0, 12) * A(1, 5) * A(2, 3) - A(0, 12) * A(1, 6) * A(2, 2) - A(0, 12) * A(1, 7) * A(2, 1) -
					A(0, 1) * A(2, 7) * A(1, 12) - A(0, 2) * A(2, 6) * A(1, 12) - A(0, 2) * A(2, 7) * A(1, 11) -
					A(0, 3) * A(2, 5) * A(1, 12) - A(0, 3) * A(2, 6) * A(1, 11) - A(0, 3) * A(2, 7) * A(1, 10) +
					A(0, 5) * A(2, 3) * A(1, 12) + A(0, 6) * A(2, 2) * A(1, 12) + A(0, 6) * A(2, 3) * A(1, 11) +
					A(0, 7) * A(2, 1) * A(1, 12) + A(0, 7) * A(2, 2) * A(1, 11) + A(0, 7) * A(2, 3) * A(1, 10) +
					A(0, 1) * A(1, 7) * A(2, 12) + A(0, 2) * A(1, 6) * A(2, 12) + A(0, 2) * A(1, 7) * A(2, 11) +
					A(0, 3) * A(1, 5) * A(2, 12) + A(0, 3) * A(1, 6) * A(2, 11) + A(0, 3) * A(1, 7) * A(2, 10) -
					A(0, 5) * A(1, 3) * A(2, 12) - A(0, 6) * A(1, 2) * A(2, 12) - A(0, 6) * A(1, 3) * A(2, 11) -
					A(0, 7) * A(1, 1) * A(2, 12) - A(0, 7) * A(1, 2) * A(2, 11) - A(0, 7) * A(1, 3) * A(2, 10);
				c[3] = A(0, 3) * A(1, 7) * A(2, 9) - A(0, 3) * A(1, 9) * A(2, 7) - A(0, 7) * A(1, 3) * A(2, 9) +
					A(0, 7) * A(1, 9) * A(2, 3) + A(0, 9) * A(1, 3) * A(2, 7) - A(0, 9) * A(1, 7) * A(2, 3) +
					A(0, 10) * A(1, 2) * A(2, 7) + A(0, 10) * A(1, 3) * A(2, 6) - A(0, 10) * A(1, 6) * A(2, 3) -
					A(0, 10) * A(1, 7) * A(2, 2) + A(1, 0) * A(0, 12) * A(2, 7) + A(0, 11) * A(1, 1) * A(2, 7) +
					A(0, 11) * A(1, 2) * A(2, 6) + A(0, 11) * A(1, 3) * A(2, 5) - A(0, 11) * A(1, 5) * A(2, 3) -
					A(0, 11) * A(1, 6) * A(2, 2) - A(0, 11) * A(1, 7) * A(2, 1) + A(1, 1) * A(0, 12) * A(2, 6) +
					A(0, 12) * A(1, 2) * A(2, 5) + A(0, 12) * A(1, 3) * A(2, 4) - A(0, 12) * A(1, 4) * A(2, 3) -
					A(0, 12) * A(1, 5) * A(2, 2) - A(0, 12) * A(1, 6) * A(2, 1) - A(0, 12) * A(1, 7) * A(2, 0) -
					A(0, 0) * A(2, 7) * A(1, 12) - A(0, 1) * A(2, 6) * A(1, 12) - A(0, 1) * A(2, 7) * A(1, 11) -
					A(0, 2) * A(2, 5) * A(1, 12) - A(0, 2) * A(2, 6) * A(1, 11) - A(0, 2) * A(2, 7) * A(1, 10) -
					A(0, 3) * A(2, 4) * A(1, 12) - A(0, 3) * A(2, 5) * A(1, 11) - A(0, 3) * A(2, 6) * A(1, 10) +
					A(0, 4) * A(2, 3) * A(1, 12) + A(0, 5) * A(2, 2) * A(1, 12) + A(0, 5) * A(2, 3) * A(1, 11) +
					A(0, 6) * A(2, 1) * A(1, 12) + A(0, 6) * A(2, 2) * A(1, 11) + A(0, 6) * A(2, 3) * A(1, 10) +
					A(0, 7) * A(2, 0) * A(1, 12) + A(0, 7) * A(2, 1) * A(1, 11) + A(0, 7) * A(2, 2) * A(1, 10) +
					A(0, 0) * A(1, 7) * A(2, 12) + A(0, 1) * A(1, 6) * A(2, 12) + A(0, 1) * A(1, 7) * A(2, 11) +
					A(0, 2) * A(1, 5) * A(2, 12) + A(0, 2) * A(1, 6) * A(2, 11) + A(0, 2) * A(1, 7) * A(2, 10) +
					A(0, 3) * A(1, 4) * A(2, 12) + A(0, 3) * A(1, 5) * A(2, 11) + A(0, 3) * A(1, 6) * A(2, 10) -
					A(0, 4) * A(1, 3) * A(2, 12) - A(0, 5) * A(1, 2) * A(2, 12) - A(0, 5) * A(1, 3) * A(2, 11) -
					A(0, 6) * A(1, 1) * A(2, 12) - A(0, 6) * A(1, 2) * A(2, 11) - A(0, 6) * A(1, 3) * A(2, 10) -
					A(0, 7) * A(1, 0) * A(2, 12) - A(0, 7) * A(1, 1) * A(2, 11) - A(0, 7) * A(1, 2) * A(2, 10);
				c[4] = A(0, 2) * A(1, 7) * A(2, 9) - A(0, 2) * A(1, 9) * A(2, 7) + A(0, 3) * A(1, 6) * A(2, 9) +
					A(0, 3) * A(1, 7) * A(2, 8) - A(0, 3) * A(1, 8) * A(2, 7) - A(0, 3) * A(1, 9) * A(2, 6) -
					A(0, 6) * A(1, 3) * A(2, 9) + A(0, 6) * A(1, 9) * A(2, 3) - A(0, 7) * A(1, 2) * A(2, 9) -
					A(0, 7) * A(1, 3) * A(2, 8) + A(0, 7) * A(1, 8) * A(2, 3) + A(0, 7) * A(1, 9) * A(2, 2) +
					A(0, 8) * A(1, 3) * A(2, 7) - A(0, 8) * A(1, 7) * A(2, 3) + A(0, 9) * A(1, 2) * A(2, 7) +
					A(0, 9) * A(1, 3) * A(2, 6) - A(0, 9) * A(1, 6) * A(2, 3) - A(0, 9) * A(1, 7) * A(2, 2) +
					A(0, 10) * A(1, 1) * A(2, 7) + A(0, 10) * A(1, 2) * A(2, 6) + A(0, 10) * A(1, 3) * A(2, 5) -
					A(0, 10) * A(1, 5) * A(2, 3) - A(0, 10) * A(1, 6) * A(2, 2) - A(0, 10) * A(1, 7) * A(2, 1) +
					A(1, 0) * A(0, 11) * A(2, 7) + A(1, 0) * A(0, 12) * A(2, 6) + A(0, 11) * A(1, 1) * A(2, 6) +
					A(0, 11) * A(1, 2) * A(2, 5) + A(0, 11) * A(1, 3) * A(2, 4) - A(0, 11) * A(1, 4) * A(2, 3) -
					A(0, 11) * A(1, 5) * A(2, 2) - A(0, 11) * A(1, 6) * A(2, 1) - A(0, 11) * A(1, 7) * A(2, 0) +
					A(1, 1) * A(0, 12) * A(2, 5) + A(0, 12) * A(1, 2) * A(2, 4) - A(0, 12) * A(1, 4) * A(2, 2) -
					A(0, 12) * A(1, 5) * A(2, 1) - A(0, 12) * A(1, 6) * A(2, 0) - A(0, 0) * A(2, 6) * A(1, 12) -
					A(0, 0) * A(2, 7) * A(1, 11) - A(0, 1) * A(2, 5) * A(1, 12) - A(0, 1) * A(2, 6) * A(1, 11) -
					A(0, 1) * A(2, 7) * A(1, 10) - A(0, 2) * A(2, 4) * A(1, 12) - A(0, 2) * A(2, 5) * A(1, 11) -
					A(0, 2) * A(2, 6) * A(1, 10) - A(0, 3) * A(2, 4) * A(1, 11) - A(0, 3) * A(2, 5) * A(1, 10) +
					A(0, 4) * A(2, 2) * A(1, 12) + A(0, 4) * A(2, 3) * A(1, 11) + A(0, 5) * A(2, 1) * A(1, 12) +
					A(0, 5) * A(2, 2) * A(1, 11) + A(0, 5) * A(2, 3) * A(1, 10) + A(0, 6) * A(2, 0) * A(1, 12) +
					A(0, 6) * A(2, 1) * A(1, 11) + A(0, 6) * A(2, 2) * A(1, 10) + A(0, 7) * A(2, 0) * A(1, 11) +
					A(0, 7) * A(2, 1) * A(1, 10) + A(0, 0) * A(1, 6) * A(2, 12) + A(0, 0) * A(1, 7) * A(2, 11) +
					A(0, 1) * A(1, 5) * A(2, 12) + A(0, 1) * A(1, 6) * A(2, 11) + A(0, 1) * A(1, 7) * A(2, 10) +
					A(0, 2) * A(1, 4) * A(2, 12) + A(0, 2) * A(1, 5) * A(2, 11) + A(0, 2) * A(1, 6) * A(2, 10) +
					A(0, 3) * A(1, 4) * A(2, 11) + A(0, 3) * A(1, 5) * A(2, 10) - A(0, 4) * A(1, 2) * A(2, 12) -
					A(0, 4) * A(1, 3) * A(2, 11) - A(0, 5) * A(1, 1) * A(2, 12) - A(0, 5) * A(1, 2) * A(2, 11) -
					A(0, 5) * A(1, 3) * A(2, 10) - A(0, 6) * A(1, 0) * A(2, 12) - A(0, 6) * A(1, 1) * A(2, 11) -
					A(0, 6) * A(1, 2) * A(2, 10) - A(0, 7) * A(1, 0) * A(2, 11) - A(0, 7) * A(1, 1) * A(2, 10);
				c[5] = A(0, 1) * A(1, 7) * A(2, 9) - A(0, 1) * A(1, 9) * A(2, 7) + A(0, 2) * A(1, 6) * A(2, 9) +
					A(0, 2) * A(1, 7) * A(2, 8) - A(0, 2) * A(1, 8) * A(2, 7) - A(0, 2) * A(1, 9) * A(2, 6) +
					A(0, 3) * A(1, 5) * A(2, 9) + A(0, 3) * A(1, 6) * A(2, 8) - A(0, 3) * A(1, 8) * A(2, 6) -
					A(0, 3) * A(1, 9) * A(2, 5) - A(0, 5) * A(1, 3) * A(2, 9) + A(0, 5) * A(1, 9) * A(2, 3) -
					A(0, 6) * A(1, 2) * A(2, 9) - A(0, 6) * A(1, 3) * A(2, 8) + A(0, 6) * A(1, 8) * A(2, 3) +
					A(0, 6) * A(1, 9) * A(2, 2) - A(0, 7) * A(1, 1) * A(2, 9) - A(0, 7) * A(1, 2) * A(2, 8) +
					A(0, 7) * A(1, 8) * A(2, 2) + A(0, 7) * A(1, 9) * A(2, 1) + A(0, 8) * A(1, 2) * A(2, 7) +
					A(0, 8) * A(1, 3) * A(2, 6) - A(0, 8) * A(1, 6) * A(2, 3) - A(0, 8) * A(1, 7) * A(2, 2) +
					A(0, 9) * A(1, 1) * A(2, 7) + A(0, 9) * A(1, 2) * A(2, 6) + A(0, 9) * A(1, 3) * A(2, 5) -
					A(0, 9) * A(1, 5) * A(2, 3) - A(0, 9) * A(1, 6) * A(2, 2) - A(0, 9) * A(1, 7) * A(2, 1) +
					A(0, 10) * A(1, 0) * A(2, 7) + A(0, 10) * A(1, 1) * A(2, 6) + A(0, 10) * A(1, 2) * A(2, 5) +
					A(0, 10) * A(1, 3) * A(2, 4) - A(0, 10) * A(1, 4) * A(2, 3) - A(0, 10) * A(1, 5) * A(2, 2) -
					A(0, 10) * A(1, 6) * A(2, 1) - A(0, 10) * A(1, 7) * A(2, 0) + A(1, 0) * A(0, 11) * A(2, 6) +
					A(1, 0) * A(0, 12) * A(2, 5) + A(0, 11) * A(1, 1) * A(2, 5) + A(0, 11) * A(1, 2) * A(2, 4) -
					A(0, 11) * A(1, 4) * A(2, 2) - A(0, 11) * A(1, 5) * A(2, 1) - A(0, 11) * A(1, 6) * A(2, 0) +
					A(1, 1) * A(0, 12) * A(2, 4) - A(0, 12) * A(1, 4) * A(2, 1) - A(0, 12) * A(1, 5) * A(2, 0) -
					A(0, 0) * A(2, 5) * A(1, 12) - A(0, 0) * A(2, 6) * A(1, 11) - A(0, 0) * A(2, 7) * A(1, 10) -
					A(0, 1) * A(2, 4) * A(1, 12) - A(0, 1) * A(2, 5) * A(1, 11) - A(0, 1) * A(2, 6) * A(1, 10) -
					A(0, 2) * A(2, 4) * A(1, 11) - A(0, 2) * A(2, 5) * A(1, 10) - A(0, 3) * A(2, 4) * A(1, 10) +
					A(0, 4) * A(2, 1) * A(1, 12) + A(0, 4) * A(2, 2) * A(1, 11) + A(0, 4) * A(2, 3) * A(1, 10) +
					A(0, 5) * A(2, 0) * A(1, 12) + A(0, 5) * A(2, 1) * A(1, 11) + A(0, 5) * A(2, 2) * A(1, 10) +
					A(0, 6) * A(2, 0) * A(1, 11) + A(0, 6) * A(2, 1) * A(1, 10) + A(0, 7) * A(2, 0) * A(1, 10) +
					A(0, 0) * A(1, 5) * A(2, 12) + A(0, 0) * A(1, 6) * A(2, 11) + A(0, 0) * A(1, 7) * A(2, 10) +
					A(0, 1) * A(1, 4) * A(2, 12) + A(0, 1) * A(1, 5) * A(2, 11) + A(0, 1) * A(1, 6) * A(2, 10) +
					A(0, 2) * A(1, 4) * A(2, 11) + A(0, 2) * A(1, 5) * A(2, 10) + A(0, 3) * A(1, 4) * A(2, 10) -
					A(0, 4) * A(1, 1) * A(2, 12) - A(0, 4) * A(1, 2) * A(2, 11) - A(0, 4) * A(1, 3) * A(2, 10) -
					A(0, 5) * A(1, 0) * A(2, 12) - A(0, 5) * A(1, 1) * A(2, 11) - A(0, 5) * A(1, 2) * A(2, 10) -
					A(0, 6) * A(1, 0) * A(2, 11) - A(0, 6) * A(1, 1) * A(2, 10) - A(0, 7) * A(1, 0) * A(2, 10);
				c[6] = A(0, 0) * A(1, 7) * A(2, 9) - A(0, 0) * A(1, 9) * A(2, 7) + A(0, 1) * A(1, 6) * A(2, 9) +
					A(0, 1) * A(1, 7) * A(2, 8) - A(0, 1) * A(1, 8) * A(2, 7) - A(0, 1) * A(1, 9) * A(2, 6) +
					A(0, 2) * A(1, 5) * A(2, 9) + A(0, 2) * A(1, 6) * A(2, 8) - A(0, 2) * A(1, 8) * A(2, 6) -
					A(0, 2) * A(1, 9) * A(2, 5) + A(0, 3) * A(1, 4) * A(2, 9) + A(0, 3) * A(1, 5) * A(2, 8) -
					A(0, 3) * A(1, 8) * A(2, 5) - A(0, 3) * A(1, 9) * A(2, 4) - A(0, 4) * A(1, 3) * A(2, 9) +
					A(0, 4) * A(1, 9) * A(2, 3) - A(0, 5) * A(1, 2) * A(2, 9) - A(0, 5) * A(1, 3) * A(2, 8) +
					A(0, 5) * A(1, 8) * A(2, 3) + A(0, 5) * A(1, 9) * A(2, 2) - A(0, 6) * A(1, 1) * A(2, 9) -
					A(0, 6) * A(1, 2) * A(2, 8) + A(0, 6) * A(1, 8) * A(2, 2) + A(0, 6) * A(1, 9) * A(2, 1) -
					A(0, 7) * A(1, 0) * A(2, 9) - A(0, 7) * A(1, 1) * A(2, 8) + A(0, 7) * A(1, 8) * A(2, 1) +
					A(0, 7) * A(1, 9) * A(2, 0) + A(0, 8) * A(1, 1) * A(2, 7) + A(0, 8) * A(1, 2) * A(2, 6) +
					A(0, 8) * A(1, 3) * A(2, 5) - A(0, 8) * A(1, 5) * A(2, 3) - A(0, 8) * A(1, 6) * A(2, 2) -
					A(0, 8) * A(1, 7) * A(2, 1) + A(0, 9) * A(1, 0) * A(2, 7) + A(0, 9) * A(1, 1) * A(2, 6) +
					A(0, 9) * A(1, 2) * A(2, 5) + A(0, 9) * A(1, 3) * A(2, 4) - A(0, 9) * A(1, 4) * A(2, 3) -
					A(0, 9) * A(1, 5) * A(2, 2) - A(0, 9) * A(1, 6) * A(2, 1) - A(0, 9) * A(1, 7) * A(2, 0) +
					A(0, 10) * A(1, 0) * A(2, 6) + A(0, 10) * A(1, 1) * A(2, 5) + A(0, 10) * A(1, 2) * A(2, 4) -
					A(0, 10) * A(1, 4) * A(2, 2) - A(0, 10) * A(1, 5) * A(2, 1) - A(0, 10) * A(1, 6) * A(2, 0) +
					A(1, 0) * A(0, 11) * A(2, 5) + A(1, 0) * A(0, 12) * A(2, 4) + A(0, 11) * A(1, 1) * A(2, 4) -
					A(0, 11) * A(1, 4) * A(2, 1) - A(0, 11) * A(1, 5) * A(2, 0) - A(0, 12) * A(1, 4) * A(2, 0) -
					A(0, 0) * A(2, 4) * A(1, 12) - A(0, 0) * A(2, 5) * A(1, 11) - A(0, 0) * A(2, 6) * A(1, 10) -
					A(0, 1) * A(2, 4) * A(1, 11) - A(0, 1) * A(2, 5) * A(1, 10) - A(0, 2) * A(2, 4) * A(1, 10) +
					A(0, 4) * A(2, 0) * A(1, 12) + A(0, 4) * A(2, 1) * A(1, 11) + A(0, 4) * A(2, 2) * A(1, 10) +
					A(0, 5) * A(2, 0) * A(1, 11) + A(0, 5) * A(2, 1) * A(1, 10) + A(0, 6) * A(2, 0) * A(1, 10) +
					A(0, 0) * A(1, 4) * A(2, 12) + A(0, 0) * A(1, 5) * A(2, 11) + A(0, 0) * A(1, 6) * A(2, 10) +
					A(0, 1) * A(1, 4) * A(2, 11) + A(0, 1) * A(1, 5) * A(2, 10) + A(0, 2) * A(1, 4) * A(2, 10) -
					A(0, 4) * A(1, 0) * A(2, 12) - A(0, 4) * A(1, 1) * A(2, 11) - A(0, 4) * A(1, 2) * A(2, 10) -
					A(0, 5) * A(1, 0) * A(2, 11) - A(0, 5) * A(1, 1) * A(2, 10) - A(0, 6) * A(1, 0) * A(2, 10);
				c[7] = A(0, 0) * A(1, 6) * A(2, 9) + A(0, 0) * A(1, 7) * A(2, 8) - A(0, 0) * A(1, 8) * A(2, 7) -
					A(0, 0) * A(1, 9) * A(2, 6) + A(0, 1) * A(1, 5) * A(2, 9) + A(0, 1) * A(1, 6) * A(2, 8) -
					A(0, 1) * A(1, 8) * A(2, 6) - A(0, 1) * A(1, 9) * A(2, 5) + A(0, 2) * A(1, 4) * A(2, 9) +
					A(0, 2) * A(1, 5) * A(2, 8) - A(0, 2) * A(1, 8) * A(2, 5) - A(0, 2) * A(1, 9) * A(2, 4) +
					A(0, 3) * A(1, 4) * A(2, 8) - A(0, 3) * A(1, 8) * A(2, 4) - A(0, 4) * A(1, 2) * A(2, 9) -
					A(0, 4) * A(1, 3) * A(2, 8) + A(0, 4) * A(1, 8) * A(2, 3) + A(0, 4) * A(1, 9) * A(2, 2) -
					A(0, 5) * A(1, 1) * A(2, 9) - A(0, 5) * A(1, 2) * A(2, 8) + A(0, 5) * A(1, 8) * A(2, 2) +
					A(0, 5) * A(1, 9) * A(2, 1) - A(0, 6) * A(1, 0) * A(2, 9) - A(0, 6) * A(1, 1) * A(2, 8) +
					A(0, 6) * A(1, 8) * A(2, 1) + A(0, 6) * A(1, 9) * A(2, 0) - A(0, 7) * A(1, 0) * A(2, 8) +
					A(0, 7) * A(1, 8) * A(2, 0) + A(0, 8) * A(1, 0) * A(2, 7) + A(0, 8) * A(1, 1) * A(2, 6) +
					A(0, 8) * A(1, 2) * A(2, 5) + A(0, 8) * A(1, 3) * A(2, 4) - A(0, 8) * A(1, 4) * A(2, 3) -
					A(0, 8) * A(1, 5) * A(2, 2) - A(0, 8) * A(1, 6) * A(2, 1) - A(0, 8) * A(1, 7) * A(2, 0) +
					A(0, 9) * A(1, 0) * A(2, 6) + A(0, 9) * A(1, 1) * A(2, 5) + A(0, 9) * A(1, 2) * A(2, 4) -
					A(0, 9) * A(1, 4) * A(2, 2) - A(0, 9) * A(1, 5) * A(2, 1) - A(0, 9) * A(1, 6) * A(2, 0) +
					A(0, 10) * A(1, 0) * A(2, 5) + A(0, 10) * A(1, 1) * A(2, 4) - A(0, 10) * A(1, 4) * A(2, 1) -
					A(0, 10) * A(1, 5) * A(2, 0) + A(1, 0) * A(0, 11) * A(2, 4) - A(0, 11) * A(1, 4) * A(2, 0) -
					A(0, 0) * A(2, 4) * A(1, 11) - A(0, 0) * A(2, 5) * A(1, 10) - A(0, 1) * A(2, 4) * A(1, 10) +
					A(0, 4) * A(2, 0) * A(1, 11) + A(0, 4) * A(2, 1) * A(1, 10) + A(0, 5) * A(2, 0) * A(1, 10) +
					A(0, 0) * A(1, 4) * A(2, 11) + A(0, 0) * A(1, 5) * A(2, 10) + A(0, 1) * A(1, 4) * A(2, 10) -
					A(0, 4) * A(1, 0) * A(2, 11) - A(0, 4) * A(1, 1) * A(2, 10) - A(0, 5) * A(1, 0) * A(2, 10);
				c[8] = A(0, 0) * A(1, 5) * A(2, 9) + A(0, 0) * A(1, 6) * A(2, 8) - A(0, 0) * A(1, 8) * A(2, 6) -
					A(0, 0) * A(1, 9) * A(2, 5) + A(0, 1) * A(1, 4) * A(2, 9) + A(0, 1) * A(1, 5) * A(2, 8) -
					A(0, 1) * A(1, 8) * A(2, 5) - A(0, 1) * A(1, 9) * A(2, 4) + A(0, 2) * A(1, 4) * A(2, 8) -
					A(0, 2) * A(1, 8) * A(2, 4) - A(0, 4) * A(1, 1) * A(2, 9) - A(0, 4) * A(1, 2) * A(2, 8) +
					A(0, 4) * A(1, 8) * A(2, 2) + A(0, 4) * A(1, 9) * A(2, 1) - A(0, 5) * A(1, 0) * A(2, 9) -
					A(0, 5) * A(1, 1) * A(2, 8) + A(0, 5) * A(1, 8) * A(2, 1) + A(0, 5) * A(1, 9) * A(2, 0) -
					A(0, 6) * A(1, 0) * A(2, 8) + A(0, 6) * A(1, 8) * A(2, 0) + A(0, 8) * A(1, 0) * A(2, 6) +
					A(0, 8) * A(1, 1) * A(2, 5) + A(0, 8) * A(1, 2) * A(2, 4) - A(0, 8) * A(1, 4) * A(2, 2) -
					A(0, 8) * A(1, 5) * A(2, 1) - A(0, 8) * A(1, 6) * A(2, 0) + A(0, 9) * A(1, 0) * A(2, 5) +
					A(0, 9) * A(1, 1) * A(2, 4) - A(0, 9) * A(1, 4) * A(2, 1) - A(0, 9) * A(1, 5) * A(2, 0) +
					A(0, 10) * A(1, 0) * A(2, 4) - A(0, 10) * A(1, 4) * A(2, 0) - A(0, 0) * A(2, 4) * A(1, 10) +
					A(0, 4) * A(2, 0) * A(1, 10) + A(0, 0) * A(1, 4) * A(2, 10) - A(0, 4) * A(1, 0) * A(2, 10);
				c[9] = A(0, 0) * A(1, 4) * A(2, 9) + A(0, 0) * A(1, 5) * A(2, 8) - A(0, 0) * A(1, 8) * A(2, 5) -
					A(0, 0) * A(1, 9) * A(2, 4) + A(0, 1) * A(1, 4) * A(2, 8) - A(0, 1) * A(1, 8) * A(2, 4) -
					A(0, 4) * A(1, 0) * A(2, 9) - A(0, 4) * A(1, 1) * A(2, 8) + A(0, 4) * A(1, 8) * A(2, 1) +
					A(0, 4) * A(1, 9) * A(2, 0) - A(0, 5) * A(1, 0) * A(2, 8) + A(0, 5) * A(1, 8) * A(2, 0) +
					A(0, 8) * A(1, 0) * A(2, 5) + A(0, 8) * A(1, 1) * A(2, 4) - A(0, 8) * A(1, 4) * A(2, 1) -
					A(0, 8) * A(1, 5) * A(2, 0) + A(0, 9) * A(1, 0) * A(2, 4) - A(0, 9) * A(1, 4) * A(2, 0);
				c[10] = A(0, 0) * A(1, 4) * A(2, 8) - A(0, 0) * A(1, 8) * A(2, 4) - A(0, 4) * A(1, 0) * A(2, 8) +
						A(0, 4) * A(1, 8) * A(2, 0) + A(0, 8) * A(1, 0) * A(2, 4) - A(0, 8) * A(1, 4) * A(2, 0);

				// Solve for the roots using sturm bracketing
				double roots[10];
				int n_sols = poselib::sturm::bisect_sturm<10>(c, roots);

				// Back substitution to recover essential matrices
				Eigen::Matrix<double, 3, 2> B;
				Eigen::Matrix<double, 3, 1> b;
				Eigen::Matrix<double, 2, 1> xz;
				Eigen::Matrix<double, 3, 3> E;
				Eigen::Map<Eigen::Matrix<double, 1, 9>> e(E.data());
				models_.reserve(n_sols);
				for (int i = 0; i < n_sols; ++i) 
				{
					const double z = roots[i];
					const double z2 = z * z;
					const double z3 = z2 * z;
					const double z4 = z2 * z2;

					B.col(0) = A.block<3, 1>(0, 0) * z3 + A.block<3, 1>(0, 1) * z2 + A.block<3, 1>(0, 2) * z + A.block<3, 1>(0, 3);
					B.col(1) = A.block<3, 1>(0, 4) * z3 + A.block<3, 1>(0, 5) * z2 + A.block<3, 1>(0, 6) * z + A.block<3, 1>(0, 7);
					b = A.block<3, 1>(0, 8) * z4 + A.block<3, 1>(0, 9) * z3 + A.block<3, 1>(0, 10) * z2 + A.block<3, 1>(0, 11) * z +
						A.block<3, 1>(0, 12);

					// We try to solve using top two rows
					xz = B.block<2, 2>(0, 0).inverse() * b.block<2, 1>(0, 0);

					// If this fails we revert to more expensive QR solver using all three rows
					if (std::abs(B.row(2) * xz - b(2)) > 1e-6) {
						xz = B.colPivHouseholderQr().solve(b);
					}

					const double x = -xz(0), y = -xz(1);
					e = nullSpace.row(0) * x + nullSpace.row(1) * y + nullSpace.row(2) * z + nullSpace.row(3);

					// Since the rows of N are orthogonal unit vectors, we can normalize the coefficients instead
					const double inv_norm = 1.0 / std::sqrt(x * x + y * y + z * z + 1.0);
					e *= inv_norm;

					EssentialMatrix model;
					model.descriptor = E;
					models_.push_back(model);
				}

				return models_.size();
			}
		}
	}
}