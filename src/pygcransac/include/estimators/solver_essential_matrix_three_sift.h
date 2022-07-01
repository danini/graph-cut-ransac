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
			class EssentialMatrixThreeSIFTSolver : public SolverEngine
			{
			public:
				EssentialMatrixThreeSIFTSolver() : minimumTraceValue(5.0)
				{
				}

				~EssentialMatrixThreeSIFTSolver()
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
					return "3SIFT";
				}

				// The maximum number of solutions that this algorithm returns
				static constexpr size_t maximumSolutions()
				{
					return 3;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sampleNumber_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				const double minimumTraceValue;

				inline Eigen::Matrix<double, 1, 10> multiplyDegOnePoly(
					const Eigen::RowVector4d& a,
					const Eigen::RowVector4d& b) const;

				inline Eigen::Matrix<double, 1, 20> multiplyDegTwoDegOnePoly(
					const Eigen::Matrix<double, 1, 10>& a,
					const Eigen::RowVector4d& b) const;

				inline Eigen::Matrix<double, 10, 20> buildConstraintMatrix(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

				inline Eigen::Matrix<double, 9, 20> getTraceConstraint(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

				inline Eigen::Matrix<double, 1, 10>
					computeEETranspose(const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j) const;

				inline Eigen::Matrix<double, 1, 20> getDeterminantConstraint(
					const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;
			};

			OLGA_INLINE bool EssentialMatrixThreeSIFTSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sampleNumber_,
				std::vector<Model> &models_,
				const double *weights_) const
			{				
				Eigen::MatrixXd coefficients(6, 9);
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

					const double
						u21 = c1 * q1,
						v21 = s1 * q1,
						u22 = c2 * q2,
						v22 = s2 * q2;
						
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
					
				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
				if (lu.dimensionOfKernel() != 3) 
					return false;

				// The null-space of the problem
				const Eigen::Matrix<double, 9, 3> &kNullSpace =
					lu.kernel(); 

				// The three null-vectors
				const Eigen::Matrix<double, 9, 1> &kX = kNullSpace.col(0);
				const Eigen::Matrix<double, 9, 1> &kY = kNullSpace.col(1);
				const Eigen::Matrix<double, 9, 1> &kZ = kNullSpace.col(2);

				const double &x1 = kX(0);
				const double &x2 = kX(1);
				const double &x3 = kX(2);
				const double &x4 = kX(3);
				const double &x5 = kX(4);
				const double &x6 = kX(5);
				const double &x7 = kX(6);
				const double &x8 = kX(7);
				const double &x9 = kX(8);

				const double &y1 = kY(0);
				const double &y2 = kY(1);
				const double &y3 = kY(2);
				const double &y4 = kY(3);
				const double &y5 = kY(4);
				const double &y6 = kY(5);
				const double &y7 = kY(6);
				const double &y8 = kY(7);
				const double &y9 = kY(8);

				const double &z1 = kZ(0);
				const double &z2 = kZ(1);
				const double &z3 = kZ(2);
				const double &z4 = kZ(3);
				const double &z5 = kZ(4);
				const double &z6 = kZ(5);
				const double &z7 = kZ(6);
				const double &z8 = kZ(7);
				const double &z9 = kZ(8);

				// Coefficients of : a ^ 3, b ^ 3, a ^ 2 b, a b ^ 2, a ^ 2, b ^ 2, a b, a, b, 1
				Eigen::Matrix<double, 10, 9> C;
				Eigen::Matrix<double, 10, 1> b;

				const double
					x1sqr = x1 * x1,
					x2sqr = x2 * x2,
					x3sqr = x3 * x3,
					x4sqr = x4 * x4,
					x5sqr = x5 * x5,
					x6sqr = x6 * x6,
					x7sqr = x7 * x7,
					x8sqr = x8 * x8,
					x9sqr = x9 * x9,
					y1sqr = y1 * y1,
					y2sqr = y2 * y2,
					y3sqr = y3 * y3,
					y4sqr = y4 * y4,
					y5sqr = y5 * y5,
					y6sqr = y6 * y6,
					y7sqr = y7 * y7,
					y8sqr = y8 * y8,
					y9sqr = y9 * y9,
					z1sqr = z1 * z1,
					z2sqr = z2 * z2,
					z3sqr = z3 * z3,
					z4sqr = z4 * z4,
					z5sqr = z5 * z5,
					z6sqr = z6 * z6,
					z7sqr = z7 * z7,
					z8sqr = z8 * z8,
					z9sqr = z9 * z9,
					x1x2 = x1 * x2,
					x1x3 = x1 * x3,
					x1x4 = x1 * x4,
					x1x5 = x1 * x5,
					x1x6 = x1 * x6,
					x1x7 = x1 * x7,
					x1x8 = x1 * x8,
					x1x9 = x1 * x9,
					y1y2 = y1 * y2,
					y1y3 = y1 * y3,
					y1y4 = y1 * y4,
					y1y5 = y1 * y5,
					y1y6 = y1 * y6,
					y1y7 = y1 * y7,
					y1y8 = y1 * y8,
					y1y9 = y1 * y9,
					x2x3 = x2 * x3,
					x2x4 = x2 * x4,
					x2x5 = x2 * x5,
					x2x6 = x2 * x6,
					x2x7 = x2 * x7,
					x2x8 = x2 * x8,
					x2x9 = x2 * x9,
					y2y3 = y2 * y3,
					y2y4 = y2 * y4,
					y2y5 = y2 * y5,
					y2y6 = y2 * y6,
					y2y7 = y2 * y7,
					y2y8 = y2 * y8,
					y2y9 = y2 * y9,
					x3x4 = x3 * x4,
					x3x5 = x3 * x5,
					x3x6 = x3 * x6,
					x3x7 = x3 * x7,
					x3x8 = x3 * x8,
					x3x9 = x3 * x9,
					y3y4 = y3 * y4,
					y3y5 = y3 * y5,
					y3y6 = y3 * y6,
					y3y7 = y3 * y7,
					y3y8 = y3 * y8,
					y3y9 = y3 * y9,
					x4x5 = x4 * x5,
					x4x6 = x4 * x6,
					x4x7 = x4 * x7,
					x4x8 = x4 * x8,
					x4x9 = x4 * x9,
					y4y5 = y4 * y5,
					y4y6 = y4 * y6,
					y4y7 = y4 * y7,
					y4y8 = y4 * y8,
					y4y9 = y4 * y9,
					x5x6 = x5 * x6,
					x5x7 = x5 * x7,
					x5x8 = x5 * x8,
					x5x9 = x5 * x9,
					y5y6 = y5 * y6,
					y5y7 = y5 * y7,
					y5y8 = y5 * y8,
					y5y9 = y5 * y9,
					x6x7 = x6 * x7,
					x6x8 = x6 * x8,
					x6x9 = x6 * x9,
					y6y7 = y6 * y7,
					y6y8 = y6 * y8,
					y6y9 = y6 * y9,
					x7x8 = x7 * x8,
					x7x9 = x7 * x9,
					y7y8 = y7 * y8,
					y7y9 = y7 * y9,
					x8x9 = x8 * x9,
					y8y9 = y8 * y9;

				C(0, 0) = (-x1 * x9sqr + 2 * x3*x7x9 - x1 * x8sqr + 2 * x2*x7x8 + x1 * x7sqr - x1 * x6sqr + 2 * x3x4*x6 - x1 * x5sqr + 2 * x2x4*x5 + x1 * x4sqr + x1 * x3sqr + x1 * x2sqr + x1 * x1sqr);
				C(0, 1) = (-y1 * y9sqr + 2 * y3*y7y9 - y1 * y8sqr + 2 * y2*y7y8 + y1 * y7sqr - y1 * y6sqr + 2 * y3y4*y6 - y1 * y5sqr + 2 * y2y4*y5 + y1 * y4sqr + y1 * y3sqr + y1 * y2sqr + y1 * y1sqr);
				C(0, 2) = (2 * x3x7 - 2 * x1*x9)*y9 + (2 * x2x7 - 2 * x1x8)*y8 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y7 + (2 * x3x4 - 2 * x1x6)*y6 + (2 * x2x4 - 2 * x1x5)*y5 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y4 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y3 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y2 + (-x9 * x9 - x8 * x8 + x7 * x7 - x6 * x6 - x5 * x5 + x4 * x4 + x3 * x3 + x2 * x2 + 3 * x1sqr)*y1;
				C(0, 3) = -x1 * y9 *y9 + (2 * x3*y7 + 2 * x7*y3 - 2 * x9*y1)*y9 - x1 * y8sqr + (2 * x2*y7 + 2 * x7*y2 - 2 * x8*y1)*y8 + x1 * y7sqr + (2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*y7 - x1 * y6sqr + (2 * x3*y4 + 2 * x4*y3 - 2 * x6*y1)*y6 - x1 * y5sqr + (2 * x2*y4 + 2 * x4*y2 - 2 * x5*y1)*y5 + x1 * y4sqr + (2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*y4 + x1 * y3sqr + 2 * x3*y1y3 + x1 * y2sqr + 2 * x2*y1y2 + 3 * x1*y1sqr;
				C(0, 4) = (2 * x3x7 - 2 * x1*x9)*z9 + (2 * x2x7 - 2 * x1x8)*z8 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z7 + (2 * x3x4 - 2 * x1x6)*z6 + (2 * x2x4 - 2 * x1x5)*z5 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z4 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z3 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z2 + (-x9 * x9 - x8 * x8 + x7 * x7 - x6 * x6 - x5 * x5 + x4 * x4 + x3 * x3 + x2 * x2 + 3 * x1sqr)*z1;
				C(0, 5) = (2 * y3y7 - 2 * y1*y9)*z9 + (2 * y2y7 - 2 * y1y8)*z8 + (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z7 + (2 * y3y4 - 2 * y1y6)*z6 + (2 * y2y4 - 2 * y1y5)*z5 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z4 + (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z3 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z2 + (-y9 * y9 - y8 * y8 + y7 * y7 - y6 * y6 - y5 * y5 + y4 * y4 + y3 * y3 + y2 * y2 + 3 * y1sqr)*z1;
				C(0, 6) = (-2 * x1*y9 + 2 * x3*y7 + 2 * x7*y3 - 2 * x9*y1)*z9 + (-2 * x1*y8 + 2 * x2*y7 + 2 * x7*y2 - 2 * x8*y1)*z8 + (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z7 + (-2 * x1*y6 + 2 * x3*y4 + 2 * x4*y3 - 2 * x6*y1)*z6 + (-2 * x1*y5 + 2 * x2*y4 + 2 * x4*y2 - 2 * x5*y1)*z5 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z4 + (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z3 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z2 + (-2 * x9*y9 - 2 * x8*y8 + 2 * x7*y7 - 2 * x6*y6 - 2 * x5*y5 + 2 * x4*y4 + 2 * x3*y3 + 2 * x2*y2 + 6 * x1*y1)*z1;
				C(0, 7) = -x1 * z9sqr + (2 * x3*z7 + 2 * x7*z3 - 2 * x9*z1)*z9 - x1 * z8sqr + (2 * x2*z7 + 2 * x7*z2 - 2 * x8*z1)*z8 + x1 * z7sqr + (2 * x9*z3 + 2 * x8*z2 + 2 * x7*z1)*z7 - x1 * z6sqr + (2 * x3*z4 + 2 * x4*z3 - 2 * x6*z1)*z6 - x1 * z5sqr + (2 * x2*z4 + 2 * x4*z2 - 2 * x5*z1)*z5 + x1 * z4sqr + (2 * x6*z3 + 2 * x5*z2 + 2 * x4*z1)*z4 + x1 * z3sqr + 2 * x3*z1*z3 + x1 * z2sqr + 2 * x2*z1*z2 + 3 * x1*z1sqr;
				C(0, 8) = -y1 * z9sqr + (2 * y3*z7 + 2 * y7*z3 - 2 * y9*z1)*z9 - y1 * z8sqr + (2 * y2*z7 + 2 * y7*z2 - 2 * y8*z1)*z8 + y1 * z7sqr + (2 * y9*z3 + 2 * y8*z2 + 2 * y7*z1)*z7 - y1 * z6sqr + (2 * y3*z4 + 2 * y4*z3 - 2 * y6*z1)*z6 - y1 * z5sqr + (2 * y2*z4 + 2 * y4*z2 - 2 * y5*z1)*z5 + y1 * z4sqr + (2 * y6*z3 + 2 * y5*z2 + 2 * y4*z1)*z4 + y1 * z3sqr + 2 * y3*z1*z3 + y1 * z2sqr + 2 * y2*z1*z2 + 3 * y1*z1sqr;
				b(0) = -(-z1 * z9sqr + 2 * z3*z7*z9 - z1 * z8sqr + 2 * z2*z7*z8 + z1 * z7sqr - z1 * z6sqr + 2 * z3*z4*z6 - z1 * z5sqr + 2 * z2*z4*z5 + z1 * z4sqr + z1 * z3sqr + z1 * z2sqr + z1 * z1sqr);

				C(1, 0) = (-x2 * x9sqr + 2 * x3x8*x9 + x2 * x8sqr + 2 * x1*x7x8 - x2 * x7sqr - x2 * x6sqr + 2 * x3x5*x6 + x2 * x5sqr + 2 * x1x4*x5 - x2 * x4sqr + x2 * x3sqr + x2 * x2sqr + x1 * x1 * x2);
				C(1, 1) = (-y2 * y9sqr + 2 * y3y8*y9 + y2 * y8sqr + 2 * y1*y7y8 - y2 * y7sqr - y2 * y6sqr + 2 * y3y5*y6 + y2 * y5sqr + 2 * y1y4*y5 - y2 * y4sqr + y2 * y3sqr + y2 * y2sqr + y1 * y1 * y2);
				C(1, 2) = (2 * x3x8 - 2 * x2x9)*y9 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y8 + (2 * x1x8 - 2 * x2x7)*y7 + (2 * x3x5 - 2 * x2x6)*y6 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y5 + (2 * x1x5 - 2 * x2x4)*y4 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y3 + (-x9 * x9 + x8 * x8 - x7 * x7 - x6 * x6 + x5 * x5 - x4 * x4 + x3 * x3 + 3 * x2sqr + x1 * x1)*y2 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y1;
				C(1, 3) = -x2 * y9sqr + (2 * x3*y8 + 2 * x8*y3 - 2 * x9*y2)*y9 + x2 * y8sqr + (2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*y8 - x2 * y7sqr + (2 * x8*y1 - 2 * x7*y2)*y7 - x2 * y6sqr + (2 * x3*y5 + 2 * x5*y3 - 2 * x6*y2)*y6 + x2 * y5sqr + (2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*y5 - x2 * y4sqr + (2 * x5*y1 - 2 * x4*y2)*y4 + x2 * y3sqr + 2 * x3*y2y3 + 3 * x2*y2sqr + 2 * x1*y1y2 + x2 * y1sqr;
				C(1, 4) = (2 * x3x8 - 2 * x2x9)*z9 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z8 + (2 * x1x8 - 2 * x2x7)*z7 + (2 * x3x5 - 2 * x2x6)*z6 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z5 + (2 * x1x5 - 2 * x2x4)*z4 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z3 + (-x9 * x9 + x8 * x8 - x7 * x7 - x6 * x6 + x5 * x5 - x4 * x4 + x3 * x3 + 3 * x2sqr + x1 * x1)*z2 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z1;
				C(1, 5) = (2 * y3y8 - 2 * y2y9)*z9 + (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z8 + (2 * y1y8 - 2 * y2y7)*z7 + (2 * y3y5 - 2 * y2y6)*z6 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z5 + (2 * y1y5 - 2 * y2y4)*z4 + (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z3 + (-y9 * y9 + y8 * y8 - y7 * y7 - y6 * y6 + y5 * y5 - y4 * y4 + y3 * y3 + 3 * y2sqr + y1 * y1)*z2 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z1;
				C(1, 6) = (-2 * x2*y9 + 2 * x3*y8 + 2 * x8*y3 - 2 * x9*y2)*z9 + (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z8 + (2 * x1*y8 - 2 * x2*y7 - 2 * x7*y2 + 2 * x8*y1)*z7 + (-2 * x2*y6 + 2 * x3*y5 + 2 * x5*y3 - 2 * x6*y2)*z6 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z5 + (2 * x1*y5 - 2 * x2*y4 - 2 * x4*y2 + 2 * x5*y1)*z4 + (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z3 + (-2 * x9*y9 + 2 * x8*y8 - 2 * x7*y7 - 2 * x6*y6 + 2 * x5*y5 - 2 * x4*y4 + 2 * x3*y3 + 6 * x2*y2 + 2 * x1*y1)*z2 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z1;
				C(1, 7) = -x2 * z9sqr + (2 * x3*z8 + 2 * x8*z3 - 2 * x9*z2)*z9 + x2 * z8sqr + (2 * x1*z7 + 2 * x9*z3 + 2 * x8*z2 + 2 * x7*z1)*z8 - x2 * z7sqr + (2 * x8*z1 - 2 * x7*z2)*z7 - x2 * z6sqr + (2 * x3*z5 + 2 * x5*z3 - 2 * x6*z2)*z6 + x2 * z5sqr + (2 * x1*z4 + 2 * x6*z3 + 2 * x5*z2 + 2 * x4*z1)*z5 - x2 * z4sqr + (2 * x5*z1 - 2 * x4*z2)*z4 + x2 * z3sqr + 2 * x3*z2*z3 + 3 * x2*z2sqr + 2 * x1*z1*z2 + x2 * z1sqr;
				C(1, 8) = -y2 * z9sqr + (2 * y3*z8 + 2 * y8*z3 - 2 * y9*z2)*z9 + y2 * z8sqr + (2 * y1*z7 + 2 * y9*z3 + 2 * y8*z2 + 2 * y7*z1)*z8 - y2 * z7sqr + (2 * y8*z1 - 2 * y7*z2)*z7 - y2 * z6sqr + (2 * y3*z5 + 2 * y5*z3 - 2 * y6*z2)*z6 + y2 * z5sqr + (2 * y1*z4 + 2 * y6*z3 + 2 * y5*z2 + 2 * y4*z1)*z5 - y2 * z4sqr + (2 * y5*z1 - 2 * y4*z2)*z4 + y2 * z3sqr + 2 * y3*z2*z3 + 3 * y2*z2sqr + 2 * y1*z1*z2 + y2 * z1sqr;
				b(1) = -(-z2 * z9sqr + 2 * z3*z8*z9 + z2 * z8sqr + 2 * z1*z7*z8 - z2 * z7sqr - z2 * z6sqr + 2 * z3*z5*z6 + z2 * z5sqr + 2 * z1*z4*z5 - z2 * z4sqr + z2 * z3sqr + z2 * z2sqr + z1 * z1 * z2);

				C(2, 0) = (x3*x9sqr + (2 * x2x8 + 2 * x1x7)*x9 - x3 * x8sqr - x3 * x7sqr + x3 * x6sqr + (2 * x2x5 + 2 * x1x4)*x6 - x3 * x5sqr - x3 * x4sqr + x3 * x3sqr + (x2sqr + x1 * x1)*x3);
				C(2, 1) = (y3*y9sqr + (2 * y2y8 + 2 * y1y7)*y9 - y3 * y8sqr - y3 * y7sqr + y3 * y6sqr + (2 * y2y5 + 2 * y1y4)*y6 - y3 * y5sqr - y3 * y4sqr + y3 * y3sqr + (y2sqr + y1 * y1)*y3);
				C(2, 2) = (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y9 + (2 * x2x9 - 2 * x3x8)*y8 + (2 * x1*x9 - 2 * x3x7)*y7 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y6 + (2 * x2x6 - 2 * x3x5)*y5 + (2 * x1x6 - 2 * x3x4)*y4 + (x9sqr - x8 * x8 - x7 * x7 + x6 * x6 - x5 * x5 - x4 * x4 + 3 * x3sqr + x2 * x2 + x1 * x1)*y3 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y2 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y1;
				C(2, 3) = x3 * y9sqr + (2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*y9 - x3 * y8sqr + (2 * x9*y2 - 2 * x8*y3)*y8 - x3 * y7sqr + (2 * x9*y1 - 2 * x7*y3)*y7 + x3 * y6sqr + (2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*y6 - x3 * y5sqr + (2 * x6*y2 - 2 * x5*y3)*y5 - x3 * y4sqr + (2 * x6*y1 - 2 * x4*y3)*y4 + 3 * x3*y3sqr + (2 * x2*y2 + 2 * x1*y1)*y3 + x3 * y2sqr + x3 * y1sqr;
				C(2, 4) = (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z9 + (2 * x2x9 - 2 * x3x8)*z8 + (2 * x1*x9 - 2 * x3x7)*z7 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z6 + (2 * x2x6 - 2 * x3x5)*z5 + (2 * x1x6 - 2 * x3x4)*z4 + (x9sqr - x8 * x8 - x7 * x7 + x6 * x6 - x5 * x5 - x4 * x4 + 3 * x3sqr + x2 * x2 + x1 * x1)*z3 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z2 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z1;
				C(2, 5) = (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z9 + (2 * y2y9 - 2 * y3y8)*z8 + (2 * y1*y9 - 2 * y3y7)*z7 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z6 + (2 * y2y6 - 2 * y3y5)*z5 + (2 * y1y6 - 2 * y3y4)*z4 + (y9sqr - y8 * y8 - y7 * y7 + y6 * y6 - y5 * y5 - y4 * y4 + 3 * y3sqr + y2 * y2 + y1 * y1)*z3 + (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z2 + (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z1;
				C(2, 6) = (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z9 + (2 * x2*y9 - 2 * x3*y8 - 2 * x8*y3 + 2 * x9*y2)*z8 + (2 * x1*y9 - 2 * x3*y7 - 2 * x7*y3 + 2 * x9*y1)*z7 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z6 + (2 * x2*y6 - 2 * x3*y5 - 2 * x5*y3 + 2 * x6*y2)*z5 + (2 * x1*y6 - 2 * x3*y4 - 2 * x4*y3 + 2 * x6*y1)*z4 + (2 * x9*y9 - 2 * x8*y8 - 2 * x7*y7 + 2 * x6*y6 - 2 * x5*y5 - 2 * x4*y4 + 6 * x3*y3 + 2 * x2*y2 + 2 * x1*y1)*z3 + (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z2 + (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z1;
				C(2, 7) = x3 * z9sqr + (2 * x2*z8 + 2 * x1*z7 + 2 * x9*z3 + 2 * x8*z2 + 2 * x7*z1)*z9 - x3 * z8sqr + (2 * x9*z2 - 2 * x8*z3)*z8 - x3 * z7sqr + (2 * x9*z1 - 2 * x7*z3)*z7 + x3 * z6sqr + (2 * x2*z5 + 2 * x1*z4 + 2 * x6*z3 + 2 * x5*z2 + 2 * x4*z1)*z6 - x3 * z5sqr + (2 * x6*z2 - 2 * x5*z3)*z5 - x3 * z4sqr + (2 * x6*z1 - 2 * x4*z3)*z4 + 3 * x3*z3sqr + (2 * x2*z2 + 2 * x1*z1)*z3 + x3 * z2sqr + x3 * z1sqr;
				C(2, 8) = y3 * z9sqr + (2 * y2*z8 + 2 * y1*z7 + 2 * y9*z3 + 2 * y8*z2 + 2 * y7*z1)*z9 - y3 * z8sqr + (2 * y9*z2 - 2 * y8*z3)*z8 - y3 * z7sqr + (2 * y9*z1 - 2 * y7*z3)*z7 + y3 * z6sqr + (2 * y2*z5 + 2 * y1*z4 + 2 * y6*z3 + 2 * y5*z2 + 2 * y4*z1)*z6 - y3 * z5sqr + (2 * y6*z2 - 2 * y5*z3)*z5 - y3 * z4sqr + (2 * y6*z1 - 2 * y4*z3)*z4 + 3 * y3*z3sqr + (2 * y2*z2 + 2 * y1*z1)*z3 + y3 * z2sqr + y3 * z1sqr;
				b(2) = -(z3*z9sqr + (2 * z2*z8 + 2 * z1*z7)*z9 - z3 * z8sqr - z3 * z7sqr + z3 * z6sqr + (2 * z2*z5 + 2 * z1*z4)*z6 - z3 * z5sqr - z3 * z4sqr + z3 * z3sqr + (z2sqr + z1 * z1)*z3);

				C(3, 0) = (-x4 * x9sqr + 2 * x6*x7x9 - x4 * x8sqr + 2 * x5*x7x8 + x4 * x7sqr + x4 * x6sqr + 2 * x1x3*x6 + x4 * x5sqr + 2 * x1x2*x5 + x4 * x4sqr + (-x3 * x3 - x2 * x2 + x1 * x1)*x4);
				C(3, 1) = (-y4 * y9sqr + 2 * y6*y7y9 - y4 * y8sqr + 2 * y5*y7y8 + y4 * y7sqr + y4 * y6sqr + 2 * y1y3*y6 + y4 * y5sqr + 2 * y1y2*y5 + y4 * y4sqr + (-y3 * y3 - y2 * y2 + y1 * y1)*y4);
				C(3, 2) = (2 * x6x7 - 2 * x4x9)*y9 + (2 * x5x7 - 2 * x4x8)*y8 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y7 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y6 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y5 + (-x9 * x9 - x8 * x8 + x7 * x7 + x6 * x6 + x5 * x5 + 3 * x4sqr - x3 * x3 - x2 * x2 + x1 * x1)*y4 + (2 * x1x6 - 2 * x3x4)*y3 + (2 * x1x5 - 2 * x2x4)*y2 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y1;
				C(3, 3) = -x4 * y9sqr + (2 * x6*y7 + 2 * x7*y6 - 2 * x9*y4)*y9 - x4 * y8sqr + (2 * x5*y7 + 2 * x7*y5 - 2 * x8*y4)*y8 + x4 * y7sqr + (2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*y7 + x4 * y6sqr + (2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*y6 + x4 * y5sqr + (2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*y5 + 3 * x4*y4sqr + (-2 * x3*y3 - 2 * x2*y2 + 2 * x1*y1)*y4 - x4 * y3sqr + 2 * x6*y1y3 - x4 * y2sqr + 2 * x5*y1y2 + x4 * y1sqr;
				C(3, 4) = (2 * x6x7 - 2 * x4x9)*z9 + (2 * x5x7 - 2 * x4x8)*z8 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z7 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z6 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z5 + (-x9 * x9 - x8 * x8 + x7 * x7 + x6 * x6 + x5 * x5 + 3 * x4sqr - x3 * x3 - x2 * x2 + x1 * x1)*z4 + (2 * x1x6 - 2 * x3x4)*z3 + (2 * x1x5 - 2 * x2x4)*z2 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z1;
				C(3, 5) = (2 * y6y7 - 2 * y4y9)*z9 + (2 * y5y7 - 2 * y4y8)*z8 + (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z7 + (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z6 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z5 + (-y9 * y9 - y8 * y8 + y7 * y7 + y6 * y6 + y5 * y5 + 3 * y4sqr - y3 * y3 - y2 * y2 + y1 * y1)*z4 + (2 * y1y6 - 2 * y3y4)*z3 + (2 * y1y5 - 2 * y2y4)*z2 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z1;
				C(3, 6) = (-2 * x4*y9 + 2 * x6*y7 + 2 * x7*y6 - 2 * x9*y4)*z9 + (-2 * x4*y8 + 2 * x5*y7 + 2 * x7*y5 - 2 * x8*y4)*z8 + (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z7 + (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z6 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z5 + (-2 * x9*y9 - 2 * x8*y8 + 2 * x7*y7 + 2 * x6*y6 + 2 * x5*y5 + 6 * x4*y4 - 2 * x3*y3 - 2 * x2*y2 + 2 * x1*y1)*z4 + (2 * x1*y6 - 2 * x3*y4 - 2 * x4*y3 + 2 * x6*y1)*z3 + (2 * x1*y5 - 2 * x2*y4 - 2 * x4*y2 + 2 * x5*y1)*z2 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z1;
				C(3, 7) = -x4 * z9sqr + (2 * x6*z7 + 2 * x7*z6 - 2 * x9*z4)*z9 - x4 * z8sqr + (2 * x5*z7 + 2 * x7*z5 - 2 * x8*z4)*z8 + x4 * z7sqr + (2 * x9*z6 + 2 * x8*z5 + 2 * x7*z4)*z7 + x4 * z6sqr + (2 * x6*z4 + 2 * x1*z3 + 2 * x3*z1)*z6 + x4 * z5sqr + (2 * x5*z4 + 2 * x1*z2 + 2 * x2*z1)*z5 + 3 * x4*z4sqr + (-2 * x3*z3 - 2 * x2*z2 + 2 * x1*z1)*z4 - x4 * z3sqr + 2 * x6*z1*z3 - x4 * z2sqr + 2 * x5*z1*z2 + x4 * z1sqr;
				C(3, 8) = -y4 * z9sqr + (2 * y6*z7 + 2 * y7*z6 - 2 * y9*z4)*z9 - y4 * z8sqr + (2 * y5*z7 + 2 * y7*z5 - 2 * y8*z4)*z8 + y4 * z7sqr + (2 * y9*z6 + 2 * y8*z5 + 2 * y7*z4)*z7 + y4 * z6sqr + (2 * y6*z4 + 2 * y1*z3 + 2 * y3*z1)*z6 + y4 * z5sqr + (2 * y5*z4 + 2 * y1*z2 + 2 * y2*z1)*z5 + 3 * y4*z4sqr + (-2 * y3*z3 - 2 * y2*z2 + 2 * y1*z1)*z4 - y4 * z3sqr + 2 * y6*z1*z3 - y4 * z2sqr + 2 * y5*z1*z2 + y4 * z1sqr;
				b(3) = -(-z4 * z9sqr + 2 * z6*z7*z9 - z4 * z8sqr + 2 * z5*z7*z8 + z4 * z7sqr + z4 * z6sqr + 2 * z1*z3*z6 + z4 * z5sqr + 2 * z1*z2*z5 + z4 * z4sqr + (-z3 * z3 - z2 * z2 + z1 * z1)*z4);

				C(4, 0) = (-x5 * x9sqr + 2 * x6x8*x9 + x5 * x8sqr + 2 * x4*x7x8 - x5 * x7sqr + x5 * x6sqr + 2 * x2x3*x6 + x5 * x5sqr + (x4sqr - x3 * x3 + x2 * x2 - x1 * x1)*x5 + 2 * x1x2*x4);
				C(4, 1) = (-y5 * y9sqr + 2 * y6y8*y9 + y5 * y8sqr + 2 * y4*y7y8 - y5 * y7sqr + y5 * y6sqr + 2 * y2y3*y6 + y5 * y5sqr + (y4sqr - y3 * y3 + y2 * y2 - y1 * y1)*y5 + 2 * y1y2*y4);
				C(4, 2) = (2 * x6x8 - 2 * x5x9)*y9 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y8 + (2 * x4x8 - 2 * x5x7)*y7 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y6 + (-x9 * x9 + x8 * x8 - x7 * x7 + x6 * x6 + 3 * x5sqr + x4 * x4 - x3 * x3 + x2 * x2 - x1 * x1)*y5 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y4 + (2 * x2x6 - 2 * x3x5)*y3 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y2 + (2 * x2x4 - 2 * x1x5)*y1;
				C(4, 3) = -x5 * y9sqr + (2 * x6*y8 + 2 * x8*y6 - 2 * x9*y5)*y9 + x5 * y8sqr + (2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*y8 - x5 * y7sqr + (2 * x8*y4 - 2 * x7*y5)*y7 + x5 * y6sqr + (2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*y6 + 3 * x5*y5sqr + (2 * x4*y4 - 2 * x3*y3 + 2 * x2*y2 - 2 * x1*y1)*y5 + x5 * y4sqr + (2 * x1*y2 + 2 * x2*y1)*y4 - x5 * y3sqr + 2 * x6*y2y3 + x5 * y2sqr + 2 * x4*y1y2 - x5 * y1sqr;
				C(4, 4) = (2 * x6x8 - 2 * x5x9)*z9 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z8 + (2 * x4x8 - 2 * x5x7)*z7 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z6 + (-x9 * x9 + x8 * x8 - x7 * x7 + x6 * x6 + 3 * x5sqr + x4 * x4 - x3 * x3 + x2 * x2 - x1 * x1)*z5 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z4 + (2 * x2x6 - 2 * x3x5)*z3 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z2 + (2 * x2x4 - 2 * x1x5)*z1;
				C(4, 5) = (2 * y6y8 - 2 * y5y9)*z9 + (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z8 + (2 * y4y8 - 2 * y5y7)*z7 + (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z6 + (-y9 * y9 + y8 * y8 - y7 * y7 + y6 * y6 + 3 * y5sqr + y4 * y4 - y3 * y3 + y2 * y2 - y1 * y1)*z5 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z4 + (2 * y2y6 - 2 * y3y5)*z3 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z2 + (2 * y2y4 - 2 * y1y5)*z1;
				C(4, 6) = (-2 * x5*y9 + 2 * x6*y8 + 2 * x8*y6 - 2 * x9*y5)*z9 + (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z8 + (2 * x4*y8 - 2 * x5*y7 - 2 * x7*y5 + 2 * x8*y4)*z7 + (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z6 + (-2 * x9*y9 + 2 * x8*y8 - 2 * x7*y7 + 2 * x6*y6 + 6 * x5*y5 + 2 * x4*y4 - 2 * x3*y3 + 2 * x2*y2 - 2 * x1*y1)*z5 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z4 + (2 * x2*y6 - 2 * x3*y5 - 2 * x5*y3 + 2 * x6*y2)*z3 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z2 + (-2 * x1*y5 + 2 * x2*y4 + 2 * x4*y2 - 2 * x5*y1)*z1;
				C(4, 7) = -x5 * z9sqr + (2 * x6*z8 + 2 * x8*z6 - 2 * x9*z5)*z9 + x5 * z8sqr + (2 * x4*z7 + 2 * x9*z6 + 2 * x8*z5 + 2 * x7*z4)*z8 - x5 * z7sqr + (2 * x8*z4 - 2 * x7*z5)*z7 + x5 * z6sqr + (2 * x6*z5 + 2 * x2*z3 + 2 * x3*z2)*z6 + 3 * x5*z5sqr + (2 * x4*z4 - 2 * x3*z3 + 2 * x2*z2 - 2 * x1*z1)*z5 + x5 * z4sqr + (2 * x1*z2 + 2 * x2*z1)*z4 - x5 * z3sqr + 2 * x6*z2*z3 + x5 * z2sqr + 2 * x4*z1*z2 - x5 * z1sqr;
				C(4, 8) = -y5 * z9sqr + (2 * y6*z8 + 2 * y8*z6 - 2 * y9*z5)*z9 + y5 * z8sqr + (2 * y4*z7 + 2 * y9*z6 + 2 * y8*z5 + 2 * y7*z4)*z8 - y5 * z7sqr + (2 * y8*z4 - 2 * y7*z5)*z7 + y5 * z6sqr + (2 * y6*z5 + 2 * y2*z3 + 2 * y3*z2)*z6 + 3 * y5*z5sqr + (2 * y4*z4 - 2 * y3*z3 + 2 * y2*z2 - 2 * y1*z1)*z5 + y5 * z4sqr + (2 * y1*z2 + 2 * y2*z1)*z4 - y5 * z3sqr + 2 * y6*z2*z3 + y5 * z2sqr + 2 * y4*z1*z2 - y5 * z1sqr;
				b(4) = -(-z5 * z9sqr + 2 * z6*z8*z9 + z5 * z8sqr + 2 * z4*z7*z8 - z5 * z7sqr + z5 * z6sqr + 2 * z2*z3*z6 + z5 * z5sqr + (z4sqr - z3 * z3 + z2 * z2 - z1 * z1)*z5 + 2 * z1*z2*z4);

				C(5, 0) = (x6*x9sqr + (2 * x5x8 + 2 * x4x7)*x9 - x6 * x8sqr - x6 * x7sqr + x6 * x6sqr + (x5sqr + x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*x6 + 2 * x2x3*x5 + 2 * x1*x3x4);
				C(5, 1) = (y6*y9sqr + (2 * y5y8 + 2 * y4y7)*y9 - y6 * y8sqr - y6 * y7sqr + y6 * y6sqr + (y5sqr + y4 * y4 + y3 * y3 - y2 * y2 - y1 * y1)*y6 + 2 * y2y3*y5 + 2 * y1y3*y4);
				C(5, 2) = (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y9 + (2 * x5x9 - 2 * x6x8)*y8 + (2 * x4x9 - 2 * x6x7)*y7 + (x9sqr - x8 * x8 - x7 * x7 + 3 * x6sqr + x5 * x5 + x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*y6 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y5 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y4 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*y3 + (2 * x3x5 - 2 * x2x6)*y2 + (2 * x3x4 - 2 * x1x6)*y1;
				C(5, 3) = x6 * y9sqr + (2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*y9 - x6 * y8sqr + (2 * x9*y5 - 2 * x8*y6)*y8 - x6 * y7sqr + (2 * x9*y4 - 2 * x7*y6)*y7 + 3 * x6*y6sqr + (2 * x5*y5 + 2 * x4*y4 + 2 * x3*y3 - 2 * x2*y2 - 2 * x1*y1)*y6 + x6 * y5sqr + (2 * x2*y3 + 2 * x3*y2)*y5 + x6 * y4sqr + (2 * x1*y3 + 2 * x3*y1)*y4 + x6 * y3sqr + (2 * x5*y2 + 2 * x4*y1)*y3 - x6 * y2sqr - x6 * y1sqr;
				C(5, 4) = (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z9 + (2 * x5x9 - 2 * x6x8)*z8 + (2 * x4x9 - 2 * x6x7)*z7 + (x9sqr - x8 * x8 - x7 * x7 + 3 * x6sqr + x5 * x5 + x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*z6 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z5 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z4 + (2 * x3x6 + 2 * x2x5 + 2 * x1x4)*z3 + (2 * x3x5 - 2 * x2x6)*z2 + (2 * x3x4 - 2 * x1x6)*z1;
				C(5, 5) = (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z9 + (2 * y5y9 - 2 * y6y8)*z8 + (2 * y4y9 - 2 * y6y7)*z7 + (y9sqr - y8 * y8 - y7 * y7 + 3 * y6sqr + y5 * y5 + y4 * y4 + y3 * y3 - y2 * y2 - y1 * y1)*z6 + (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z5 + (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z4 + (2 * y3y6 + 2 * y2y5 + 2 * y1y4)*z3 + (2 * y3y5 - 2 * y2y6)*z2 + (2 * y3y4 - 2 * y1y6)*z1;
				C(5, 6) = (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z9 + (2 * x5*y9 - 2 * x6*y8 - 2 * x8*y6 + 2 * x9*y5)*z8 + (2 * x4*y9 - 2 * x6*y7 - 2 * x7*y6 + 2 * x9*y4)*z7 + (2 * x9*y9 - 2 * x8*y8 - 2 * x7*y7 + 6 * x6*y6 + 2 * x5*y5 + 2 * x4*y4 + 2 * x3*y3 - 2 * x2*y2 - 2 * x1*y1)*z6 + (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z5 + (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z4 + (2 * x3*y6 + 2 * x2*y5 + 2 * x1*y4 + 2 * x6*y3 + 2 * x5*y2 + 2 * x4*y1)*z3 + (-2 * x2*y6 + 2 * x3*y5 + 2 * x5*y3 - 2 * x6*y2)*z2 + (-2 * x1*y6 + 2 * x3*y4 + 2 * x4*y3 - 2 * x6*y1)*z1;
				C(5, 7) = x6 * z9sqr + (2 * x5*z8 + 2 * x4*z7 + 2 * x9*z6 + 2 * x8*z5 + 2 * x7*z4)*z9 - x6 * z8sqr + (2 * x9*z5 - 2 * x8*z6)*z8 - x6 * z7sqr + (2 * x9*z4 - 2 * x7*z6)*z7 + 3 * x6*z6sqr + (2 * x5*z5 + 2 * x4*z4 + 2 * x3*z3 - 2 * x2*z2 - 2 * x1*z1)*z6 + x6 * z5sqr + (2 * x2*z3 + 2 * x3*z2)*z5 + x6 * z4sqr + (2 * x1*z3 + 2 * x3*z1)*z4 + x6 * z3sqr + (2 * x5*z2 + 2 * x4*z1)*z3 - x6 * z2sqr - x6 * z1sqr;
				C(5, 8) = y6 * z9sqr + (2 * y5*z8 + 2 * y4*z7 + 2 * y9*z6 + 2 * y8*z5 + 2 * y7*z4)*z9 - y6 * z8sqr + (2 * y9*z5 - 2 * y8*z6)*z8 - y6 * z7sqr + (2 * y9*z4 - 2 * y7*z6)*z7 + 3 * y6*z6sqr + (2 * y5*z5 + 2 * y4*z4 + 2 * y3*z3 - 2 * y2*z2 - 2 * y1*z1)*z6 + y6 * z5sqr + (2 * y2*z3 + 2 * y3*z2)*z5 + y6 * z4sqr + (2 * y1*z3 + 2 * y3*z1)*z4 + y6 * z3sqr + (2 * y5*z2 + 2 * y4*z1)*z3 - y6 * z2sqr - y6 * z1sqr;
				b(5) = -(z6*z9sqr + (2 * z5*z8 + 2 * z4*z7)*z9 - z6 * z8sqr - z6 * z7sqr + z6 * z6sqr + (z5sqr + z4 * z4 + z3 * z3 - z2 * z2 - z1 * z1)*z6 + 2 * z2*z3*z5 + 2 * z1*z3*z4);

				C(6, 0) = (x7*x9sqr + (2 * x4x6 + 2 * x1x3)*x9 + x7 * x8sqr + (2 * x4x5 + 2 * x1x2)*x8 + x7 * x7sqr + (-x6 * x6 - x5 * x5 + x4 * x4 - x3 * x3 - x2 * x2 + x1 * x1)*x7);
				C(6, 1) = (y7*y9sqr + (2 * y4y6 + 2 * y1y3)*y9 + y7 * y8sqr + (2 * y4y5 + 2 * y1y2)*y8 + y7 * y7sqr + (-y6 * y6 - y5 * y5 + y4 * y4 - y3 * y3 - y2 * y2 + y1 * y1)*y7);
				C(6, 2) = (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y9 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y8 + (x9sqr + x8 * x8 + 3 * x7sqr - x6 * x6 - x5 * x5 + x4 * x4 - x3 * x3 - x2 * x2 + x1 * x1)*y7 + (2 * x4x9 - 2 * x6x7)*y6 + (2 * x4x8 - 2 * x5x7)*y5 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y4 + (2 * x1*x9 - 2 * x3x7)*y3 + (2 * x1x8 - 2 * x2x7)*y2 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y1;
				C(6, 3) = x7 * y9sqr + (2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*y9 + x7 * y8sqr + (2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*y8 + 3 * x7*y7sqr + (-2 * x6*y6 - 2 * x5*y5 + 2 * x4*y4 - 2 * x3*y3 - 2 * x2*y2 + 2 * x1*y1)*y7 - x7 * y6sqr + 2 * x9*y4y6 - x7 * y5sqr + 2 * x8*y4y5 + x7 * y4sqr - x7 * y3sqr + 2 * x9*y1y3 - x7 * y2sqr + 2 * x8*y1y2 + x7 * y1sqr;
				C(6, 4) = (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z9 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z8 + (x9sqr + x8 * x8 + 3 * x7sqr - x6 * x6 - x5 * x5 + x4 * x4 - x3 * x3 - x2 * x2 + x1 * x1)*z7 + (2 * x4x9 - 2 * x6x7)*z6 + (2 * x4x8 - 2 * x5x7)*z5 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z4 + (2 * x1*x9 - 2 * x3x7)*z3 + (2 * x1x8 - 2 * x2x7)*z2 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z1;
				C(6, 5) = (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z9 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z8 + (y9sqr + y8 * y8 + 3 * y7sqr - y6 * y6 - y5 * y5 + y4 * y4 - y3 * y3 - y2 * y2 + y1 * y1)*z7 + (2 * y4y9 - 2 * y6y7)*z6 + (2 * y4y8 - 2 * y5y7)*z5 + (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z4 + (2 * y1*y9 - 2 * y3y7)*z3 + (2 * y1y8 - 2 * y2y7)*z2 + (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z1;
				C(6, 6) = (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z9 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z8 + (2 * x9*y9 + 2 * x8*y8 + 6 * x7*y7 - 2 * x6*y6 - 2 * x5*y5 + 2 * x4*y4 - 2 * x3*y3 - 2 * x2*y2 + 2 * x1*y1)*z7 + (2 * x4*y9 - 2 * x6*y7 - 2 * x7*y6 + 2 * x9*y4)*z6 + (2 * x4*y8 - 2 * x5*y7 - 2 * x7*y5 + 2 * x8*y4)*z5 + (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z4 + (2 * x1*y9 - 2 * x3*y7 - 2 * x7*y3 + 2 * x9*y1)*z3 + (2 * x1*y8 - 2 * x2*y7 - 2 * x7*y2 + 2 * x8*y1)*z2 + (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z1;
				C(6, 7) = x7 * z9sqr + (2 * x9*z7 + 2 * x4*z6 + 2 * x6*z4 + 2 * x1*z3 + 2 * x3*z1)*z9 + x7 * z8sqr + (2 * x8*z7 + 2 * x4*z5 + 2 * x5*z4 + 2 * x1*z2 + 2 * x2*z1)*z8 + 3 * x7*z7sqr + (-2 * x6*z6 - 2 * x5*z5 + 2 * x4*z4 - 2 * x3*z3 - 2 * x2*z2 + 2 * x1*z1)*z7 - x7 * z6sqr + 2 * x9*z4*z6 - x7 * z5sqr + 2 * x8*z4*z5 + x7 * z4sqr - x7 * z3sqr + 2 * x9*z1*z3 - x7 * z2sqr + 2 * x8*z1*z2 + x7 * z1sqr;
				C(6, 8) = y7 * z9sqr + (2 * y9*z7 + 2 * y4*z6 + 2 * y6*z4 + 2 * y1*z3 + 2 * y3*z1)*z9 + y7 * z8sqr + (2 * y8*z7 + 2 * y4*z5 + 2 * y5*z4 + 2 * y1*z2 + 2 * y2*z1)*z8 + 3 * y7*z7sqr + (-2 * y6*z6 - 2 * y5*z5 + 2 * y4*z4 - 2 * y3*z3 - 2 * y2*z2 + 2 * y1*z1)*z7 - y7 * z6sqr + 2 * y9*z4*z6 - y7 * z5sqr + 2 * y8*z4*z5 + y7 * z4sqr - y7 * z3sqr + 2 * y9*z1*z3 - y7 * z2sqr + 2 * y8*z1*z2 + y7 * z1sqr;
				b(6) = -(z7*z9sqr + (2 * z4*z6 + 2 * z1*z3)*z9 + z7 * z8sqr + (2 * z4*z5 + 2 * z1*z2)*z8 + z7 * z7sqr + (-z6 * z6 - z5 * z5 + z4 * z4 - z3 * z3 - z2 * z2 + z1 * z1)*z7);

				C(7, 0) = (x8*x9sqr + (2 * x5x6 + 2 * x2x3)*x9 + x8 * x8sqr + (x7sqr - x6 * x6 + x5 * x5 - x4 * x4 - x3 * x3 + x2 * x2 - x1 * x1)*x8 + (2 * x4x5 + 2 * x1x2)*x7);
				C(7, 1) = (y8*y9sqr + (2 * y5y6 + 2 * y2y3)*y9 + y8 * y8sqr + (y7sqr - y6 * y6 + y5 * y5 - y4 * y4 - y3 * y3 + y2 * y2 - y1 * y1)*y8 + (2 * y4y5 + 2 * y1y2)*y7);
				C(7, 2) = (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y9 + (x9sqr + 3 * x8sqr + x7 * x7 - x6 * x6 + x5 * x5 - x4 * x4 - x3 * x3 + x2 * x2 - x1 * x1)*y8 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*y7 + (2 * x5x9 - 2 * x6x8)*y6 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y5 + (2 * x5x7 - 2 * x4x8)*y4 + (2 * x2x9 - 2 * x3x8)*y3 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y2 + (2 * x2x7 - 2 * x1x8)*y1;
				C(7, 3) = x8 * y9sqr + (2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*y9 + 3 * x8*y8sqr + (2 * x7*y7 - 2 * x6*y6 + 2 * x5*y5 - 2 * x4*y4 - 2 * x3*y3 + 2 * x2*y2 - 2 * x1*y1)*y8 + x8 * y7sqr + (2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*y7 - x8 * y6sqr + 2 * x9*y5y6 + x8 * y5sqr + 2 * x7*y4y5 - x8 * y4sqr - x8 * y3sqr + 2 * x9*y2y3 + x8 * y2sqr + 2 * x7*y1y2 - x8 * y1sqr;
				C(7, 4) = (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z9 + (x9sqr + 3 * x8sqr + x7 * x7 - x6 * x6 + x5 * x5 - x4 * x4 - x3 * x3 + x2 * x2 - x1 * x1)*z8 + (2 * x7x8 + 2 * x4x5 + 2 * x1x2)*z7 + (2 * x5x9 - 2 * x6x8)*z6 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z5 + (2 * x5x7 - 2 * x4x8)*z4 + (2 * x2x9 - 2 * x3x8)*z3 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z2 + (2 * x2x7 - 2 * x1x8)*z1;
				C(7, 5) = (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z9 + (y9sqr + 3 * y8sqr + y7 * y7 - y6 * y6 + y5 * y5 - y4 * y4 - y3 * y3 + y2 * y2 - y1 * y1)*z8 + (2 * y7y8 + 2 * y4y5 + 2 * y1y2)*z7 + (2 * y5y9 - 2 * y6y8)*z6 + (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z5 + (2 * y5y7 - 2 * y4y8)*z4 + (2 * y2y9 - 2 * y3y8)*z3 + (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z2 + (2 * y2y7 - 2 * y1y8)*z1;
				C(7, 6) = (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z9 + (2 * x9*y9 + 6 * x8*y8 + 2 * x7*y7 - 2 * x6*y6 + 2 * x5*y5 - 2 * x4*y4 - 2 * x3*y3 + 2 * x2*y2 - 2 * x1*y1)*z8 + (2 * x7*y8 + 2 * x8*y7 + 2 * x4*y5 + 2 * x5*y4 + 2 * x1*y2 + 2 * x2*y1)*z7 + (2 * x5*y9 - 2 * x6*y8 - 2 * x8*y6 + 2 * x9*y5)*z6 + (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z5 + (-2 * x4*y8 + 2 * x5*y7 + 2 * x7*y5 - 2 * x8*y4)*z4 + (2 * x2*y9 - 2 * x3*y8 - 2 * x8*y3 + 2 * x9*y2)*z3 + (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z2 + (-2 * x1*y8 + 2 * x2*y7 + 2 * x7*y2 - 2 * x8*y1)*z1;
				C(7, 7) = x8 * z9sqr + (2 * x9*z8 + 2 * x5*z6 + 2 * x6*z5 + 2 * x2*z3 + 2 * x3*z2)*z9 + 3 * x8*z8sqr + (2 * x7*z7 - 2 * x6*z6 + 2 * x5*z5 - 2 * x4*z4 - 2 * x3*z3 + 2 * x2*z2 - 2 * x1*z1)*z8 + x8 * z7sqr + (2 * x4*z5 + 2 * x5*z4 + 2 * x1*z2 + 2 * x2*z1)*z7 - x8 * z6sqr + 2 * x9*z5*z6 + x8 * z5sqr + 2 * x7*z4*z5 - x8 * z4sqr - x8 * z3sqr + 2 * x9*z2*z3 + x8 * z2sqr + 2 * x7*z1*z2 - x8 * z1sqr;
				C(7, 8) = y8 * z9sqr + (2 * y9*z8 + 2 * y5*z6 + 2 * y6*z5 + 2 * y2*z3 + 2 * y3*z2)*z9 + 3 * y8*z8sqr + (2 * y7*z7 - 2 * y6*z6 + 2 * y5*z5 - 2 * y4*z4 - 2 * y3*z3 + 2 * y2*z2 - 2 * y1*z1)*z8 + y8 * z7sqr + (2 * y4*z5 + 2 * y5*z4 + 2 * y1*z2 + 2 * y2*z1)*z7 - y8 * z6sqr + 2 * y9*z5*z6 + y8 * z5sqr + 2 * y7*z4*z5 - y8 * z4sqr - y8 * z3sqr + 2 * y9*z2*z3 + y8 * z2sqr + 2 * y7*z1*z2 - y8 * z1sqr;
				b(7) = -(z8*z9sqr + (2 * z5*z6 + 2 * z2*z3)*z9 + z8 * z8sqr + (z7sqr - z6 * z6 + z5 * z5 - z4 * z4 - z3 * z3 + z2 * z2 - z1 * z1)*z8 + (2 * z4*z5 + 2 * z1*z2)*z7);

				C(8, 0) = (x9sqr*x9 + (x8sqr + x7 * x7 + x6 * x6 - x5 * x5 - x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*x9 + (2 * x5x6 + 2 * x2x3)*x8 + (2 * x4x6 + 2 * x1x3)*x7);
				C(8, 1) = (y9sqr*y9 + (y8sqr + y7 * y7 + y6 * y6 - y5 * y5 - y4 * y4 + y3 * y3 - y2 * y2 - y1 * y1)*y9 + (2 * y5y6 + 2 * y2y3)*y8 + (2 * y4y6 + 2 * y1y3)*y7);
				C(8, 2) = (3 * x9sqr + x8 * x8 + x7 * x7 + x6 * x6 - x5 * x5 - x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*y9 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*y8 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*y7 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*y6 + (2 * x6x8 - 2 * x5x9)*y5 + (2 * x6x7 - 2 * x4x9)*y4 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*y3 + (2 * x3x8 - 2 * x2x9)*y2 + (2 * x3x7 - 2 * x1*x9)*y1;
				C(8, 3) = 3 * x9*y9sqr + (2 * x8*y8 + 2 * x7*y7 + 2 * x6*y6 - 2 * x5*y5 - 2 * x4*y4 + 2 * x3*y3 - 2 * x2*y2 - 2 * x1*y1)*y9 + x9 * y8sqr + (2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*y8 + x9 * y7sqr + (2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*y7 + x9 * y6sqr + (2 * x8*y5 + 2 * x7*y4)*y6 - x9 * y5sqr - x9 * y4sqr + x9 * y3sqr + (2 * x8*y2 + 2 * x7*y1)*y3 - x9 * y2sqr - x9 * y1sqr;
				C(8, 4) = (3 * x9sqr + x8 * x8 + x7 * x7 + x6 * x6 - x5 * x5 - x4 * x4 + x3 * x3 - x2 * x2 - x1 * x1)*z9 + (2 * x8x9 + 2 * x5x6 + 2 * x2x3)*z8 + (2 * x7x9 + 2 * x4x6 + 2 * x1x3)*z7 + (2 * x6x9 + 2 * x5x8 + 2 * x4x7)*z6 + (2 * x6x8 - 2 * x5x9)*z5 + (2 * x6x7 - 2 * x4x9)*z4 + (2 * x3x9 + 2 * x2x8 + 2 * x1x7)*z3 + (2 * x3x8 - 2 * x2x9)*z2 + (2 * x3x7 - 2 * x1*x9)*z1;
				C(8, 5) = (3 * y9sqr + y8 * y8 + y7 * y7 + y6 * y6 - y5 * y5 - y4 * y4 + y3 * y3 - y2 * y2 - y1 * y1)*z9 + (2 * y8y9 + 2 * y5y6 + 2 * y2y3)*z8 + (2 * y7y9 + 2 * y4y6 + 2 * y1y3)*z7 + (2 * y6y9 + 2 * y5y8 + 2 * y4y7)*z6 + (2 * y6y8 - 2 * y5y9)*z5 + (2 * y6y7 - 2 * y4y9)*z4 + (2 * y3y9 + 2 * y2y8 + 2 * y1y7)*z3 + (2 * y3y8 - 2 * y2y9)*z2 + (2 * y3y7 - 2 * y1*y9)*z1;
				C(8, 6) = (6 * x9*y9 + 2 * x8*y8 + 2 * x7*y7 + 2 * x6*y6 - 2 * x5*y5 - 2 * x4*y4 + 2 * x3*y3 - 2 * x2*y2 - 2 * x1*y1)*z9 + (2 * x8*y9 + 2 * x9*y8 + 2 * x5*y6 + 2 * x6*y5 + 2 * x2*y3 + 2 * x3*y2)*z8 + (2 * x7*y9 + 2 * x9*y7 + 2 * x4*y6 + 2 * x6*y4 + 2 * x1*y3 + 2 * x3*y1)*z7 + (2 * x6*y9 + 2 * x5*y8 + 2 * x4*y7 + 2 * x9*y6 + 2 * x8*y5 + 2 * x7*y4)*z6 + (-2 * x5*y9 + 2 * x6*y8 + 2 * x8*y6 - 2 * x9*y5)*z5 + (-2 * x4*y9 + 2 * x6*y7 + 2 * x7*y6 - 2 * x9*y4)*z4 + (2 * x3*y9 + 2 * x2*y8 + 2 * x1*y7 + 2 * x9*y3 + 2 * x8*y2 + 2 * x7*y1)*z3 + (-2 * x2*y9 + 2 * x3*y8 + 2 * x8*y3 - 2 * x9*y2)*z2 + (-2 * x1*y9 + 2 * x3*y7 + 2 * x7*y3 - 2 * x9*y1)*z1;
				C(8, 7) = 3 * x9*z9sqr + (2 * x8*z8 + 2 * x7*z7 + 2 * x6*z6 - 2 * x5*z5 - 2 * x4*z4 + 2 * x3*z3 - 2 * x2*z2 - 2 * x1*z1)*z9 + x9 * z8sqr + (2 * x5*z6 + 2 * x6*z5 + 2 * x2*z3 + 2 * x3*z2)*z8 + x9 * z7sqr + (2 * x4*z6 + 2 * x6*z4 + 2 * x1*z3 + 2 * x3*z1)*z7 + x9 * z6sqr + (2 * x8*z5 + 2 * x7*z4)*z6 - x9 * z5sqr - x9 * z4sqr + x9 * z3sqr + (2 * x8*z2 + 2 * x7*z1)*z3 - x9 * z2sqr - x9 * z1sqr;
				C(8, 8) = 3 * y9*z9sqr + (2 * y8*z8 + 2 * y7*z7 + 2 * y6*z6 - 2 * y5*z5 - 2 * y4*z4 + 2 * y3*z3 - 2 * y2*z2 - 2 * y1*z1)*z9 + y9 * z8sqr + (2 * y5*z6 + 2 * y6*z5 + 2 * y2*z3 + 2 * y3*z2)*z8 + y9 * z7sqr + (2 * y4*z6 + 2 * y6*z4 + 2 * y1*z3 + 2 * y3*z1)*z7 + y9 * z6sqr + (2 * y8*z5 + 2 * y7*z4)*z6 - y9 * z5sqr - y9 * z4sqr + y9 * z3sqr + (2 * y8*z2 + 2 * y7*z1)*z3 - y9 * z2sqr - y9 * z1sqr;
				b(8) = -(z9sqr*z9 + (z8sqr + z7 * z7 + z6 * z6 - z5 * z5 - z4 * z4 + z3 * z3 - z2 * z2 - z1 * z1)*z9 + (2 * z5*z6 + 2 * z2*z3)*z8 + (2 * z4*z6 + 2 * z1*z3)*z7);

				C(9, 0) = (x1x5 - x2 * x4)*x9 + (x3x4 - x1 * x6)*x8 + (x2x6 - x3 * x5)*x7;
				C(9, 1) = (y1y5 - y2 * y4)*y9 + (y3y4 - y1 * y6)*y8 + (y2y6 - y3 * y5)*y7;
				C(9, 2) = (x1x5 - x2 * x4)*y9 + (x3x4 - x1 * x6)*y8 + (x2x6 - x3 * x5)*y7 + (x2x7 - x1 * x8)*y6 + (x1*x9 - x3 * x7)*y5 + (x3x8 - x2 * x9)*y4 + (x4x8 - x5 * x7)*y3 + (x6x7 - x4 * x9)*y2 + (x5x9 - x6 * x8)*y1;
				C(9, 3) = (x1*y5 - x2 * y4 - x4 * y2 + x5 * y1)*y9 + (-x1 * y6 + x3 * y4 + x4 * y3 - x6 * y1)*y8 + (x2*y6 - x3 * y5 - x5 * y3 + x6 * y2)*y7 + (x7*y2 - x8 * y1)*y6 + (x9*y1 - x7 * y3)*y5 + (x8*y3 - x9 * y2)*y4;
				C(9, 4) = (x1x5 - x2 * x4)*z9 + (x3x4 - x1 * x6)*z8 + (x2x6 - x3 * x5)*z7 + (x2x7 - x1 * x8)*z6 + (x1*x9 - x3 * x7)*z5 + (x3x8 - x2 * x9)*z4 + (x4x8 - x5 * x7)*z3 + (x6x7 - x4 * x9)*z2 + (x5x9 - x6 * x8)*z1;
				C(9, 5) = (y1y5 - y2 * y4)*z9 + (y3y4 - y1 * y6)*z8 + (y2y6 - y3 * y5)*z7 + (y2y7 - y1 * y8)*z6 + (y1*y9 - y3 * y7)*z5 + (y3y8 - y2 * y9)*z4 + (y4y8 - y5 * y7)*z3 + (y6y7 - y4 * y9)*z2 + (y5y9 - y6 * y8)*z1;
				C(9, 6) = (x1*y5 - x2 * y4 - x4 * y2 + x5 * y1)*z9 + (-x1 * y6 + x3 * y4 + x4 * y3 - x6 * y1)*z8 + (x2*y6 - x3 * y5 - x5 * y3 + x6 * y2)*z7 + (-x1 * y8 + x2 * y7 + x7 * y2 - x8 * y1)*z6 + (x1*y9 - x3 * y7 - x7 * y3 + x9 * y1)*z5 + (-x2 * y9 + x3 * y8 + x8 * y3 - x9 * y2)*z4 + (x4*y8 - x5 * y7 - x7 * y5 + x8 * y4)*z3 + (-x4 * y9 + x6 * y7 + x7 * y6 - x9 * y4)*z2 + (x5*y9 - x6 * y8 - x8 * y6 + x9 * y5)*z1;
				C(9, 7) = (x1*z5 - x2 * z4 - x4 * z2 + x5 * z1)*z9 + (-x1 * z6 + x3 * z4 + x4 * z3 - x6 * z1)*z8 + (x2*z6 - x3 * z5 - x5 * z3 + x6 * z2)*z7 + (x7*z2 - x8 * z1)*z6 + (x9*z1 - x7 * z3)*z5 + (x8*z3 - x9 * z2)*z4;
				C(9, 8) = (y1*z5 - y2 * z4 - y4 * z2 + y5 * z1)*z9 + (-y1 * z6 + y3 * z4 + y4 * z3 - y6 * z1)*z8 + (y2*z6 - y3 * z5 - y5 * z3 + y6 * z2)*z7 + (y7*z2 - y8 * z1)*z6 + (y9*z1 - y7 * z3)*z5 + (y8*z3 - y9 * z2)*z4;
				b(9) = -(z1*z5*z9 - z2 * z4*z9 - z1 * z6*z8 + z3 * z4*z8 + z2 * z6*z7 - z3 * z5*z7);

				// Solving the inhomogneous linear system.
				Eigen::Matrix<double, 9, 1> result =
					(b.transpose() * b).llt().solve(b.transpose() * C);

				//Eigen::Matrix<double, 9, 1>
				//	result = C.colPivHouseholderQr().solve(b);

				// Selecting the essential matrix which minimizes the trace constraint.
				// The system does not have multiple solutions in theory.
				// However, the elements are linearly dependent.
				// Thus, they must satisfy additional constraints.
				Eigen::Matrix3d bestEssentialMatrix;
				static const size_t indices[3] = { 7, 4, 1 };

				double minimumTraceError = std::numeric_limits<double>::max();
				double traceError;
				for (size_t i = 0; i < 3; ++i)
				{
					const double kAlpha = 
						pow(result(indices[i]), 1.0 / (i + 1));

					if (isnan(kAlpha))
						continue;

					for (int j = 0; j < 3; ++j)
					{
						const double kBeta = 
							pow(result(indices[j] + 1), 1.0 / (j + 1));

						if (isnan(kBeta))
							continue;

						Eigen::Matrix<double, 9, 1> essentialMatrixVec = 
							kAlpha * kX + kBeta * kY + kZ;

						Eigen::Matrix3d essentialMatrix;
						essentialMatrix << essentialMatrixVec(0), essentialMatrixVec(1), essentialMatrixVec(2),
							essentialMatrixVec(3), essentialMatrixVec(4), essentialMatrixVec(5),
							essentialMatrixVec(6), essentialMatrixVec(7), essentialMatrixVec(8);

						traceError = 
							(2 * essentialMatrix * essentialMatrix.transpose() * essentialMatrix - (essentialMatrix * essentialMatrix.transpose()).trace() * essentialMatrix).norm();

						if (minimumTraceError > traceError)
						{
							minimumTraceError = traceError;
							bestEssentialMatrix = essentialMatrix;
						}
					}
				}

				if (minimumTraceError < minimumTraceValue)
				{
					EssentialMatrix model;
					model.descriptor = bestEssentialMatrix;
					models_.emplace_back(model);
					return true;
				}

				return false;	
			}
		}
	}
}