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

#include <Eigen/Eigen>
#include "solver_engine.h"
#include "unsupported/Eigen/Polynomials"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating an essential matrix between two images
			// when we are given the gravity direction in the two images.
			class EssentialMatrixThreePointsGravity : public SolverEngine
			{
			public:
				EssentialMatrixThreePointsGravity() : 
					gravity_source(Eigen::Matrix3d::Identity()),
					gravity_destination(Eigen::Matrix3d::Identity())
				{
				}

				~EssentialMatrixThreePointsGravity()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				static constexpr bool needsGravity()
				{
					return true;
				}

				void setGravity(const Eigen::Matrix3d &gravity_source_,
								const Eigen::Matrix3d &gravity_destination_)
				{
					gravity_source = gravity_source_;
					gravity_destination = gravity_destination_;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return true;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat &data_,					 // The set of data points
					const size_t *sample_,					 // The sample used for the estimation
					size_t sample_number_,					 // The size of the sample
					std::vector<Model> &models_,			 // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;

				Eigen::MatrixXcd solver_3pt_caliess(const Eigen::VectorXd &data_) const;
			};

			OLGA_INLINE bool EssentialMatrixThreePointsGravity::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;

				// Building the coefficient matrices
				const double *data_ptr = reinterpret_cast<double *>(data_.data);
				const size_t cols = data_.cols;
				const size_t pointNumber = sample_number_;

				size_t r_dlt = 90,
					   t_dlt = 90;
				Eigen::MatrixXd coefficients(pointNumber, 6);
				for (size_t n = 0; n < pointNumber; ++n)
				{
					const size_t &idx = sample_[n];
					const double
						&x2o = data_.at<double>(idx, 0),
						&y2o = data_.at<double>(idx, 1),
						&x1o = data_.at<double>(idx, 2),
						&y1o = data_.at<double>(idx, 3);

					Eigen::Vector3d pt1(x2o, y2o, 1);
					Eigen::Vector3d pt2(x1o, y1o, 1);

					pt1 = gravity_source * pt1;
					pt2 = gravity_destination * pt2;

					const double
						&x2 = pt1(0),
						&y2 = pt1(1),
						&x1 = pt2(0),
						&y1 = pt2(1);

					coefficients.row(n) << x2 * x1 + 1,
						x1 * y2,
						x1 - x2,
						x2 * y1,
						y1,
						y2;

				}
				const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients.transpose() * coefficients);
				if (lu.dimensionOfKernel() != 3) 
					return false;

				const Eigen::Matrix<double, 6, 3> null_space = 
					lu.kernel();

				const Eigen::Matrix<double, 6, 1> &V4 = null_space.col(0);
				const Eigen::Matrix<double, 6, 1> &V5 = null_space.col(1);
				const Eigen::Matrix<double, 6, 1> &V6 = null_space.col(2);
				Eigen::Matrix<double, 18, 1> data2;
				data2 << V4, V5, V6;
				
				Eigen::MatrixXcd sols = solver_3pt_caliess(data2);

				for (size_t k = 0; k < 4; ++k)
				{
					if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() ||
						sols(1, k).imag() > std::numeric_limits<double>::epsilon())
						continue;

					Eigen::MatrixXd M = sols(0, k).real() * V4 + sols(1, k).real() * V5 + V6;

					Eigen::Matrix<double, 3, 3> E;
					E << M(0), M(1), M(2),
						M(3), 0, M(4),
						-M(2), M(5), M(0);

					E = gravity_destination.transpose() *
						E *
						gravity_source;

					Model model;
					model.descriptor = E;
					models_.push_back(model);
				}

				return models_.size() > 0;
			}

			Eigen::MatrixXcd EssentialMatrixThreePointsGravity::solver_3pt_caliess(const Eigen::VectorXd &data_) const
			{
				using namespace Eigen;

				// Action =  y
				// Quotient ring basis (V) = 1,x,y,y^2,
				// Available monomials (RR*V) = x*y,y^3,1,x,y,y^2,
				const double *d = data_.data();
				VectorXd coeffs(36);
				coeffs[0] = std::pow(d[1], 2) - std::pow(d[3], 2) - std::pow(d[4], 2) + std::pow(d[5], 2);
				coeffs[1] = 2 * d[1] * d[7] - 2 * d[3] * d[9] - 2 * d[4] * d[10] + 2 * d[5] * d[11];
				coeffs[2] = std::pow(d[7], 2) - std::pow(d[9], 2) - std::pow(d[10], 2) + std::pow(d[11], 2);
				coeffs[3] = 2 * d[1] * d[13] - 2 * d[3] * d[15] - 2 * d[4] * d[16] + 2 * d[5] * d[17];
				coeffs[4] = 2 * d[7] * d[13] - 2 * d[9] * d[15] - 2 * d[10] * d[16] + 2 * d[11] * d[17];
				coeffs[5] = std::pow(d[13], 2) - std::pow(d[15], 2) - std::pow(d[16], 2) + std::pow(d[17], 2);
				coeffs[6] = d[0] * d[1] * d[3] + d[1] * d[2] * d[4] - d[2] * d[3] * d[5] + d[0] * d[4] * d[5];
				coeffs[7] = d[1] * d[3] * d[6] + d[4] * d[5] * d[6] + d[0] * d[3] * d[7] + d[2] * d[4] * d[7] + d[1] * d[4] * d[8] - d[3] * d[5] * d[8] + d[0] * d[1] * d[9] - d[2] * d[5] * d[9] + d[1] * d[2] * d[10] + d[0] * d[5] * d[10] - d[2] * d[3] * d[11] + d[0] * d[4] * d[11];
				coeffs[8] = d[3] * d[6] * d[7] + d[4] * d[7] * d[8] + d[1] * d[6] * d[9] + d[0] * d[7] * d[9] - d[5] * d[8] * d[9] + d[5] * d[6] * d[10] + d[2] * d[7] * d[10] + d[1] * d[8] * d[10] + d[4] * d[6] * d[11] - d[3] * d[8] * d[11] - d[2] * d[9] * d[11] + d[0] * d[10] * d[11];
				coeffs[9] = d[6] * d[7] * d[9] + d[7] * d[8] * d[10] - d[8] * d[9] * d[11] + d[6] * d[10] * d[11];
				coeffs[10] = d[1] * d[3] * d[12] + d[4] * d[5] * d[12] + d[0] * d[3] * d[13] + d[2] * d[4] * d[13] + d[1] * d[4] * d[14] - d[3] * d[5] * d[14] + d[0] * d[1] * d[15] - d[2] * d[5] * d[15] + d[1] * d[2] * d[16] + d[0] * d[5] * d[16] - d[2] * d[3] * d[17] + d[0] * d[4] * d[17];
				coeffs[11] = d[3] * d[7] * d[12] + d[1] * d[9] * d[12] + d[5] * d[10] * d[12] + d[4] * d[11] * d[12] + d[3] * d[6] * d[13] + d[4] * d[8] * d[13] + d[0] * d[9] * d[13] + d[2] * d[10] * d[13] + d[4] * d[7] * d[14] - d[5] * d[9] * d[14] + d[1] * d[10] * d[14] - d[3] * d[11] * d[14] + d[1] * d[6] * d[15] + d[0] * d[7] * d[15] - d[5] * d[8] * d[15] - d[2] * d[11] * d[15] + d[5] * d[6] * d[16] + d[2] * d[7] * d[16] + d[1] * d[8] * d[16] + d[0] * d[11] * d[16] + d[4] * d[6] * d[17] - d[3] * d[8] * d[17] - d[2] * d[9] * d[17] + d[0] * d[10] * d[17];
				coeffs[12] = d[7] * d[9] * d[12] + d[10] * d[11] * d[12] + d[6] * d[9] * d[13] + d[8] * d[10] * d[13] + d[7] * d[10] * d[14] - d[9] * d[11] * d[14] + d[6] * d[7] * d[15] - d[8] * d[11] * d[15] + d[7] * d[8] * d[16] + d[6] * d[11] * d[16] - d[8] * d[9] * d[17] + d[6] * d[10] * d[17];
				coeffs[13] = d[3] * d[12] * d[13] + d[4] * d[13] * d[14] + d[1] * d[12] * d[15] + d[0] * d[13] * d[15] - d[5] * d[14] * d[15] + d[5] * d[12] * d[16] + d[2] * d[13] * d[16] + d[1] * d[14] * d[16] + d[4] * d[12] * d[17] - d[3] * d[14] * d[17] - d[2] * d[15] * d[17] + d[0] * d[16] * d[17];
				coeffs[14] = d[9] * d[12] * d[13] + d[10] * d[13] * d[14] + d[7] * d[12] * d[15] + d[6] * d[13] * d[15] - d[11] * d[14] * d[15] + d[11] * d[12] * d[16] + d[8] * d[13] * d[16] + d[7] * d[14] * d[16] + d[10] * d[12] * d[17] - d[9] * d[14] * d[17] - d[8] * d[15] * d[17] + d[6] * d[16] * d[17];
				coeffs[15] = d[12] * d[13] * d[15] + d[13] * d[14] * d[16] - d[14] * d[15] * d[17] + d[12] * d[16] * d[17];
				coeffs[16] = d[0] * d[3] * d[4] + d[2] * std::pow(d[4], 2) + d[0] * d[1] * d[5] - d[2] * std::pow(d[5], 2);
				coeffs[17] = d[3] * d[4] * d[6] + d[1] * d[5] * d[6] + d[0] * d[5] * d[7] + std::pow(d[4], 2) * d[8] - std::pow(d[5], 2) * d[8] + d[0] * d[4] * d[9] + d[0] * d[3] * d[10] + 2 * d[2] * d[4] * d[10] + d[0] * d[1] * d[11] - 2 * d[2] * d[5] * d[11];
				coeffs[18] = d[5] * d[6] * d[7] + d[4] * d[6] * d[9] + d[3] * d[6] * d[10] + 2 * d[4] * d[8] * d[10] + d[0] * d[9] * d[10] + d[2] * std::pow(d[10], 2) + d[1] * d[6] * d[11] + d[0] * d[7] * d[11] - 2 * d[5] * d[8] * d[11] - d[2] * std::pow(d[11], 2);
				coeffs[19] = d[6] * d[9] * d[10] + d[8] * std::pow(d[10], 2) + d[6] * d[7] * d[11] - d[8] * std::pow(d[11], 2);
				coeffs[20] = d[3] * d[4] * d[12] + d[1] * d[5] * d[12] + d[0] * d[5] * d[13] + std::pow(d[4], 2) * d[14] - std::pow(d[5], 2) * d[14] + d[0] * d[4] * d[15] + d[0] * d[3] * d[16] + 2 * d[2] * d[4] * d[16] + d[0] * d[1] * d[17] - 2 * d[2] * d[5] * d[17];
				coeffs[21] = d[5] * d[7] * d[12] + d[4] * d[9] * d[12] + d[3] * d[10] * d[12] + d[1] * d[11] * d[12] + d[5] * d[6] * d[13] + d[0] * d[11] * d[13] + 2 * d[4] * d[10] * d[14] - 2 * d[5] * d[11] * d[14] + d[4] * d[6] * d[15] + d[0] * d[10] * d[15] + d[3] * d[6] * d[16] + 2 * d[4] * d[8] * d[16] + d[0] * d[9] * d[16] + 2 * d[2] * d[10] * d[16] + d[1] * d[6] * d[17] + d[0] * d[7] * d[17] - 2 * d[5] * d[8] * d[17] - 2 * d[2] * d[11] * d[17];
				coeffs[22] = d[9] * d[10] * d[12] + d[7] * d[11] * d[12] + d[6] * d[11] * d[13] + std::pow(d[10], 2) * d[14] - std::pow(d[11], 2) * d[14] + d[6] * d[10] * d[15] + d[6] * d[9] * d[16] + 2 * d[8] * d[10] * d[16] + d[6] * d[7] * d[17] - 2 * d[8] * d[11] * d[17];
				coeffs[23] = d[5] * d[12] * d[13] + d[4] * d[12] * d[15] + d[3] * d[12] * d[16] + 2 * d[4] * d[14] * d[16] + d[0] * d[15] * d[16] + d[2] * std::pow(d[16], 2) + d[1] * d[12] * d[17] + d[0] * d[13] * d[17] - 2 * d[5] * d[14] * d[17] - d[2] * std::pow(d[17], 2);
				coeffs[24] = d[11] * d[12] * d[13] + d[10] * d[12] * d[15] + d[9] * d[12] * d[16] + 2 * d[10] * d[14] * d[16] + d[6] * d[15] * d[16] + d[8] * std::pow(d[16], 2) + d[7] * d[12] * d[17] + d[6] * d[13] * d[17] - 2 * d[11] * d[14] * d[17] - d[8] * std::pow(d[17], 2);
				coeffs[25] = d[12] * d[15] * d[16] + d[14] * std::pow(d[16], 2) + d[12] * d[13] * d[17] - d[14] * std::pow(d[17], 2);
				coeffs[26] = d[0] * std::pow(d[3], 2) + d[2] * d[3] * d[4] - d[1] * d[2] * d[5] - d[0] * std::pow(d[5], 2);
				coeffs[27] = std::pow(d[3], 2) * d[6] - std::pow(d[5], 2) * d[6] - d[2] * d[5] * d[7] + d[3] * d[4] * d[8] - d[1] * d[5] * d[8] + 2 * d[0] * d[3] * d[9] + d[2] * d[4] * d[9] + d[2] * d[3] * d[10] - d[1] * d[2] * d[11] - 2 * d[0] * d[5] * d[11];
				coeffs[28] = -d[5] * d[7] * d[8] + 2 * d[3] * d[6] * d[9] + d[4] * d[8] * d[9] + d[0] * std::pow(d[9], 2) + d[3] * d[8] * d[10] + d[2] * d[9] * d[10] - 2 * d[5] * d[6] * d[11] - d[2] * d[7] * d[11] - d[1] * d[8] * d[11] - d[0] * std::pow(d[11], 2);
				coeffs[29] = d[6] * std::pow(d[9], 2) + d[8] * d[9] * d[10] - d[7] * d[8] * d[11] - d[6] * std::pow(d[11], 2);
				coeffs[30] = std::pow(d[3], 2) * d[12] - std::pow(d[5], 2) * d[12] - d[2] * d[5] * d[13] + d[3] * d[4] * d[14] - d[1] * d[5] * d[14] + 2 * d[0] * d[3] * d[15] + d[2] * d[4] * d[15] + d[2] * d[3] * d[16] - d[1] * d[2] * d[17] - 2 * d[0] * d[5] * d[17];
				coeffs[31] = 2 * d[3] * d[9] * d[12] - 2 * d[5] * d[11] * d[12] - d[5] * d[8] * d[13] - d[2] * d[11] * d[13] - d[5] * d[7] * d[14] + d[4] * d[9] * d[14] + d[3] * d[10] * d[14] - d[1] * d[11] * d[14] + 2 * d[3] * d[6] * d[15] + d[4] * d[8] * d[15] + 2 * d[0] * d[9] * d[15] + d[2] * d[10] * d[15] + d[3] * d[8] * d[16] + d[2] * d[9] * d[16] - 2 * d[5] * d[6] * d[17] - d[2] * d[7] * d[17] - d[1] * d[8] * d[17] - 2 * d[0] * d[11] * d[17];
				coeffs[32] = std::pow(d[9], 2) * d[12] - std::pow(d[11], 2) * d[12] - d[8] * d[11] * d[13] + d[9] * d[10] * d[14] - d[7] * d[11] * d[14] + 2 * d[6] * d[9] * d[15] + d[8] * d[10] * d[15] + d[8] * d[9] * d[16] - d[7] * d[8] * d[17] - 2 * d[6] * d[11] * d[17];
				coeffs[33] = -d[5] * d[13] * d[14] + 2 * d[3] * d[12] * d[15] + d[4] * d[14] * d[15] + d[0] * std::pow(d[15], 2) + d[3] * d[14] * d[16] + d[2] * d[15] * d[16] - 2 * d[5] * d[12] * d[17] - d[2] * d[13] * d[17] - d[1] * d[14] * d[17] - d[0] * std::pow(d[17], 2);
				coeffs[34] = -d[11] * d[13] * d[14] + 2 * d[9] * d[12] * d[15] + d[10] * d[14] * d[15] + d[6] * std::pow(d[15], 2) + d[9] * d[14] * d[16] + d[8] * d[15] * d[16] - 2 * d[11] * d[12] * d[17] - d[8] * d[13] * d[17] - d[7] * d[14] * d[17] - d[6] * std::pow(d[17], 2);
				coeffs[35] = d[12] * std::pow(d[15], 2) + d[14] * d[15] * d[16] - d[13] * d[14] * d[17] - d[12] * std::pow(d[17], 2);

				static const int coeffs_ind[] = {0, 6, 16, 26, 1, 0, 7, 17, 27, 2, 1, 8, 18, 28, 3, 10, 0, 20, 30, 4, 3, 11, 1, 21, 31, 2, 9, 19, 29, 15, 5, 25, 35, 5, 13, 3, 23, 33, 5, 14, 4, 24, 34, 4, 12, 2, 22, 32};

				static const int C_ind[] = {0, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 34, 35, 38, 39, 40, 41, 42, 44, 45, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59};

				MatrixXd C = MatrixXd::Zero(6, 10);
				for (int i = 0; i < 48; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(6);
				MatrixXd C1 = C.rightCols(4);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(6, 4);
				RR << -C12.bottomRows(2), MatrixXd::Identity(4, 4);

				static const int AM_ind[] = {4, 0, 5, 1};
				MatrixXd AM(4, 4);
				for (int i = 0; i < 4; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(0).replicate(4, 1)).eval();

				MatrixXcd sols(2, 4);
				sols.row(0) = V.row(1);
				sols.row(1) = D.transpose();
				return sols;
			}
		} // namespace solver
	}	  // namespace estimator
} // namespace gcransac
