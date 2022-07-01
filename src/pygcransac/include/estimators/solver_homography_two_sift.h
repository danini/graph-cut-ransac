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
#include "estimators/homography_estimator.h"
#include "math_utils.h"

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class HomographyTwoSIFTSolver : public SolverEngine
			{
			protected:
				double *tmp_data, *RR_ptr;
				cv::Mat C, C0, C1, RR, AM;

			public:
				HomographyTwoSIFTSolver()
				{
					tmp_data = new double[48];
					C = cv::Mat(10, 6, CV_64F);
					C0 = cv::Mat(6, 6, CV_64F);
					C1 = cv::Mat(6, 4, CV_64F);
					RR = cv::Mat::zeros(6, 4, CV_64F);
					RR_ptr = reinterpret_cast<double *>(RR.data);
					RR_ptr[8] = 1;
					RR_ptr[13] = 1;
					RR_ptr[18] = 1;
					RR_ptr[23] = 1;
					
					AM = cv::Mat(4, 4, CV_64F);
				}

				~HomographyTwoSIFTSolver()
				{
					delete[] tmp_data;
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
				
				static constexpr bool needFundamentalMatrix()
				{
					return false;
				}

				void setFundamentalMatrix(Eigen::Matrix3d &kFundamentalMatrix_)
				{
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

			OLGA_INLINE bool HomographyTwoSIFTSolver::estimateModel(
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
				const size_t kRowNumber = 6; // Otherwise, use all constraints.
				// The coefficient matrix
				Eigen::MatrixXd coefficients(kRowNumber, 9);

				const size_t kColumns = data_.cols;
				const double *kDataPtr = reinterpret_cast<double *>(data_.data);
				size_t rowIdx = 0;
				double weight = 1.0;
				int data_idx = 0;

				for (size_t i = 0; i < sampleNumber_; ++i)
				{
					const size_t kIdx = 
						sample_ == nullptr ? i : sample_[i];

					const double *kPointPtr = 
						kDataPtr + kIdx * kColumns;

					const double
						&u1 = kPointPtr[0],
						&v1 = kPointPtr[1],
						&u2 = kPointPtr[2],
						&v2 = kPointPtr[3],
						&q1 = kPointPtr[4],
						&q2 = kPointPtr[5],
						&a1 = kPointPtr[6],
						&a2 = kPointPtr[7];

					if (weights_ != nullptr)
						weight = weights_[kIdx];

					const double s1 = sin(a1),
						c1 = cos(a1),
						s2 = sin(a2),
						c2 = cos(a2);

					coefficients.row(rowIdx++) << 0, 0, 0, u1, v1, 1, -u1 * v2, -v1 * v2, -v2;
					coefficients.row(rowIdx++) << u1, v1, 1, 0, 0, 0, -u1 * u2, -v1 * u2, -u2;

					tmp_data[data_idx++] = u1;
					tmp_data[data_idx++] = v1;
					tmp_data[data_idx++] = u2;
					tmp_data[data_idx++] = v2;
					tmp_data[data_idx++] = s1;
					tmp_data[data_idx++] = c1;
					tmp_data[data_idx++] = s2;
					tmp_data[data_idx++] = c2;
					tmp_data[data_idx++] = q1;
					tmp_data[data_idx++] = q2;

					const double c1s2 = c1 * s2;
					const double c1c2 = c1 * c2;
					const double s1s2 = s1 * s2;
					const double s1c2 = s1 * c2;

					coefficients.row(rowIdx++) << -c1s2, -s1s2, 0, c1c2, s1 * c2, 0, u2 * c1s2 - v2 * c1c2, u2 * s1s2 - v2 * s1c2, 0;
				}

				Eigen::JacobiSVD<Eigen::MatrixXd> svd(
					coefficients,
					Eigen::ComputeFullU | Eigen::ComputeFullV);

				Eigen::MatrixXd null_space = svd.matrixV().rightCols<3>();
				
				const double n11 = null_space(0, 0);
				const double n12 = null_space(1, 0);
				const double n13 = null_space(2, 0);
				const double n14 = null_space(3, 0);
				const double n15 = null_space(4, 0);
				const double n16 = null_space(5, 0);
				const double n17 = null_space(6, 0);
				const double n18 = null_space(7, 0);
				const double n19 = null_space(8, 0);

				const double n21 = null_space(0, 1);
				const double n22 = null_space(1, 1);
				const double n23 = null_space(2, 1);
				const double n24 = null_space(3, 1);
				const double n25 = null_space(4, 1);
				const double n26 = null_space(5, 1);
				const double n27 = null_space(6, 1);
				const double n28 = null_space(7, 1);
				const double n29 = null_space(8, 1);

				const double n31 = null_space(0, 2);
				const double n32 = null_space(1, 2);
				const double n33 = null_space(2, 2);
				const double n34 = null_space(3, 2);
				const double n35 = null_space(4, 2);
				const double n36 = null_space(5, 2);
				const double n37 = null_space(6, 2);
				const double n38 = null_space(7, 2);
				const double n39 = null_space(8, 2);

				const double d1 = tmp_data[0];
				const double d2 = tmp_data[1];
				const double d3 = tmp_data[2];
				const double d4 = tmp_data[3];
				const double d9 = tmp_data[8];
				const double d10 = tmp_data[9];
				const double d11 = tmp_data[10];
				const double d12 = tmp_data[11];
				const double d13 = tmp_data[12];
				const double d14 = tmp_data[13];
				const double d19 = tmp_data[18];
				const double d20 = tmp_data[19];

				const double d1d10 = d1 * d10;
				const double d2d10 = d2 * d10;
				const double d1d1d10 = d1 * d1d10;
				const double d1d2d10 = d1 * d2d10;
				const double d2d2d10 = d2 * d2d10;
				const double d3d9 = d3 * d9;
				const double d4d9 = d4 * d9;
				const double d13d19 = d13 * d19;
				const double d14d19 = d14 * d19;
				const double d3d9k1k1 = d3d9 * 1;
				const double d4d9k1k1 = d4d9 * 1;
				const double d19k2k2 = d19 * 1;
				const double d13d19k2k2 = d13 * d19k2k2;
				const double d14d19k2k2 = d14 * d19k2k2;
				const double d9k1k1 = d9 * 1;
				const double d11d20 = d11 * d20;
				const double d12d20 = d12 * d20;
				const double d11d11d20 = d11 * d11d20;
				const double d11d12d20 = d11 * d12d20;
				const double d12d12d20 = d12 * d12d20;
				const double n15n17 = n15 * n17;
				const double n14n18 = n14 * n18;
				const double n12n17 = n12 * n17;
				const double n11n18 = n11 * n18;
				const double n17n17 = n17 * n17;
				const double n17n18 = n17 * n18;
				const double n18n24 = n18 * n24;
				const double n17n25 = n17 * n25;
				const double n15n27 = n15 * n27;
				const double n14n28 = n14 * n28;
				const double n18n21 = n18 * n21;
				const double n17n22 = n17 * n22;
				const double n25n27 = n25 * n27;
				const double n24n28 = n24 * n28;
				const double n22n27 = n22 * n27;
				const double n21n28 = n21 * n28;
				const double n27n27 = n27 * n27;
				const double n27n28 = n27 * n28;
				const double n18n34 = n18 * n34;
				const double n17n35 = n17 * n35;
				const double n15n37 = n15 * n37;
				const double n14n38 = n14 * n38;
				const double n18n31 = n18 * n31;
				const double n28n34 = n28 * n34;
				const double n27n35 = n27 * n35;
				const double n17n32 = n17 * n32;
				const double n25n37 = n25 * n37;
				const double n24n38 = n24 * n38;
				const double n28n31 = n28 * n31;
				const double n27n32 = n27 * n32;
				const double n35n37 = n35 * n37;
				const double n34n38 = n34 * n38;
				const double n32n37 = n32 * n37;
				const double n31n38 = n31 * n38;
				const double n37n37 = n37 * n37;
				const double n37n38 = n37 * n38;
				const double n18n18 = n18 * n18;
				const double n12n14 = n12 * n14;
				const double n11n15 = n11 * n15;
				const double n17n19 = n17 * n19;
				const double n18n19 = n18 * n19;
				const double n19n19 = n19 * n19;
				const double n12n27 = n12 * n27;
				const double n11n28 = n11 * n28;
				const double n17n27 = n17 * n27;
				const double n18n27 = n18 * n27;
				const double n17n28 = n17 * n28;
				const double n18n28 = n18 * n28;
				const double n15n21 = n15 * n21;
				const double n14n22 = n14 * n22;
				const double n12n24 = n12 * n24;
				const double n11n25 = n11 * n25;
				const double n19n27 = n19 * n27;
				const double n17n29 = n17 * n29;
				const double n19n28 = n19 * n28;
				const double n18n29 = n18 * n29;
				const double n19n29 = n19 * n29;
				const double n28n28 = n28 * n28;
				const double n22n24 = n22 * n24;
				const double n21n25 = n21 * n25;
				const double n27n29 = n27 * n29;
				const double n28n29 = n28 * n29;
				const double n29n29 = n29 * n29;
				const double n12n37 = n12 * n37;
				const double n11n38 = n11 * n38;
				const double n17n37 = n17 * n37;
				const double n18n37 = n18 * n37;
				const double n17n38 = n17 * n38;
				const double n18n38 = n18 * n38;
				const double n15n31 = n15 * n31;
				const double n14n32 = n14 * n32;
				const double n12n34 = n12 * n34;
				const double n11n35 = n11 * n35;
				const double n19n37 = n19 * n37;
				const double n17n39 = n17 * n39;
				const double n19n38 = n19 * n38;
				const double n18n39 = n18 * n39;
				const double n19n39 = n19 * n39;
				const double n22n37 = n22 * n37;
				const double n21n38 = n21 * n38;
				const double n27n37 = n27 * n37;
				const double n28n37 = n28 * n37;
				const double n27n38 = n27 * n38;
				const double n28n38 = n28 * n38;
				const double n25n31 = n25 * n31;
				const double n24n32 = n24 * n32;
				const double n22n34 = n22 * n34;
				const double n21n35 = n21 * n35;
				const double n29n37 = n29 * n37;
				const double n27n39 = n27 * n39;
				const double n29n38 = n29 * n38;
				const double n28n39 = n28 * n39;
				const double n29n39 = n29 * n39;
				const double n38n38 = n38 * n38;
				const double n32n34 = n32 * n34;
				const double n31n35 = n31 * n35;
				const double n37n39 = n37 * n39;
				const double n38n39 = n38 * n39;
				const double n39n39 = n39 * n39;

				const double coeffs0 = n15n17 * d3d9k1k1 - n14n18 * d3d9k1k1 - n12n17 * d4d9k1k1 + n11n18 * d4d9k1k1 + n17n17 * d1d1d10 + 2 * n17n18 * d1d2d10 + n18n18 * d2d2d10 + n12n14 * d9k1k1 - n11n15 * d9k1k1 + 2 * n17n19 * d1d10 + 2 * n18n19 * d2d10 + n19n19 * d10;
				const double coeffs1 = -n18n24 * d3d9k1k1 + n17n25 * d3d9k1k1 + n15n27 * d3d9k1k1 - n14n28 * d3d9k1k1 + n18n21 * d4d9k1k1 - n17n22 * d4d9k1k1 - n12n27 * d4d9k1k1 + n11n28 * d4d9k1k1 + 2 * n17n27 * d1d1d10 + 2 * n18n27 * d1d2d10 + 2 * n17n28 * d1d2d10 + 2 * n18n28 * d2d2d10 - n15n21 * d9k1k1 + n14n22 * d9k1k1 + n12n24 * d9k1k1 - n11n25 * d9k1k1 + 2 * n19n27 * d1d10 + 2 * n17 * n29 * d1d10 + 2 * n19n28 * d2d10 + 2 * n18n29 * d2d10 + 2 * n19n29 * d10;
				const double coeffs2 = n25n27 * d3d9k1k1 - n24n28 * d3d9k1k1 - n22n27 * d4d9k1k1 + n21n28 * d4d9k1k1 + n27n27 * d1d1d10 + 2 * n27n28 * d1d2d10 + n28n28 * d2d2d10 + n22n24 * d9k1k1 - n21n25 * d9k1k1 + 2 * n27 * n29 * d1d10 + 2 * n28n29 * d2d10 + n29n29 * d10;
				const double coeffs3 = -n18n34 * d3d9k1k1 + n17n35 * d3d9k1k1 + n15n37 * d3d9k1k1 - n14n38 * d3d9k1k1 + n18n31 * d4d9k1k1 - n17n32 * d4d9k1k1 - n12n37 * d4d9k1k1 + n11n38 * d4d9k1k1 + 2 * n17n37 * d1d1d10 + 2 * n18n37 * d1d2d10 + 2 * n17n38 * d1d2d10 + 2 * n18n38 * d2d2d10 - n15n31 * d9k1k1 + n14n32 * d9k1k1 + n12n34 * d9k1k1 - n11n35 * d9k1k1 + 2 * n19n37 * d1d10 + 2 * n17n39 * d1d10 + 2 * n19n38 * d2d10 + 2 * n18n39 * d2d10 + 2 * n19 * n39 * d10;
				const double coeffs4 = -n28n34 * d3d9k1k1 + n27n35 * d3d9k1k1 + n25n37 * d3d9k1k1 - n24n38 * d3d9k1k1 + n28n31 * d4d9k1k1 - n27n32 * d4d9k1k1 - n22n37 * d4d9k1k1 + n21n38 * d4d9k1k1 + 2 * n27n37 * d1d1d10 + 2 * n28n37 * d1d2d10 + 2 * n27n38 * d1d2d10 + 2 * n28n38 * d2d2d10 - n25n31 * d9k1k1 + n24n32 * d9k1k1 + n22n34 * d9k1k1 - n21n35 * d9k1k1 + 2 * n29n37 * d1d10 + 2 * n27n39 * d1d10 + 2 * n29n38 * d2d10 + 2 * n28n39 * d2d10 + 2 * n29n39 * d10;
				const double coeffs5 = n35n37 * d3d9k1k1 - n34n38 * d3d9k1k1 - n32n37 * d4d9k1k1 + n31n38 * d4d9k1k1 + n37n37 * d1d1d10 + 2 * n37n38 * d1d2d10 + n38n38 * d2d2d10 + n32n34 * d9k1k1 - n31n35 * d9k1k1 + 2 * n37n39 * d1d10 + 2 * n38n39 * d2d10 + n39n39 * d10;

				const double coeffs6 = n15n17 * d13d19k2k2 - n14n18 * d13d19k2k2 - n12n17 * d14d19k2k2 + n11n18 * d14d19k2k2 + n17n17 * d11d11d20 + 2 * n17n18 * d11d12d20 + n18n18 * d12d12d20 + n12n14 * d19k2k2 - n11n15 * d19k2k2 + 2 * n17n19 * d11d20 + 2 * n18n19 * d12d20 + n19n19 * d20;
				const double coeffs7 = -n18n24 * d13d19k2k2 + n17n25 * d13d19k2k2 + n15n27 * d13d19k2k2 - n14n28 * d13d19k2k2 + n18n21 * d14d19k2k2 - n17n22 * d14d19k2k2 - n12n27 * d14d19k2k2 + n11n28 * d14d19k2k2 + 2 * n17n27 * d11d11d20 + 2 * n18n27 * d11d12d20 + 2 * n17n28 * d11d12d20 + 2 * n18n28 * d12d12d20 - n15n21 * d19k2k2 + n14n22 * d19k2k2 + n12n24 * d19k2k2 - n11n25 * d19k2k2 + 2 * n19n27 * d11d20 + 2 * n17 * n29 * d11d20 + 2 * n19n28 * d12d20 + 2 * n18n29 * d12d20 + 2 * n19n29 * d20;
				const double coeffs8 = n25n27 * d13d19k2k2 - n24n28 * d13d19k2k2 - n22n27 * d14d19k2k2 + n21n28 * d14d19k2k2 + n27n27 * d11d11d20 + 2 * n27n28 * d11d12d20 + n28n28 * d12d12d20 + n22n24 * d19k2k2 - n21n25 * d19k2k2 + 2 * n27 * n29 * d11d20 + 2 * n28n29 * d12d20 + n29n29 * d20;
				const double coeffs9 = -n18n34 * d13d19k2k2 + n17n35 * d13d19k2k2 + n15n37 * d13d19k2k2 - n14n38 * d13d19k2k2 + n18n31 * d14d19k2k2 - n17n32 * d14d19k2k2 - n12n37 * d14d19k2k2 + n11n38 * d14d19k2k2 + 2 * n17n37 * d11d11d20 + 2 * n18n37 * d11d12d20 + 2 * n17n38 * d11d12d20 + 2 * n18n38 * d12d12d20 - n15n31 * d19k2k2 + n14n32 * d19k2k2 + n12n34 * d19k2k2 - n11n35 * d19k2k2 + 2 * n19n37 * d11d20 + 2 * n17n39 * d11d20 + 2 * n19n38 * d12d20 + 2 * n18n39 * d12d20 + 2 * n19 * n39 * d20;
				const double coeffs10 = -n28n34 * d13d19k2k2 + n27n35 * d13d19k2k2 + n25n37 * d13d19k2k2 - n24n38 * d13d19k2k2 + n28n31 * d14d19k2k2 - n27n32 * d14d19k2k2 - n22n37 * d14d19k2k2 + n21n38 * d14d19k2k2 + 2 * n27n37 * d11d11d20 + 2 * n28n37 * d11d12d20 + 2 * n27n38 * d11d12d20 + 2 * n28n38 * d12d12d20 - n25n31 * d19k2k2 + n24n32 * d19k2k2 + n22n34 * d19k2k2 - n21n35 * d19k2k2 + 2 * n29n37 * d11d20 + 2 * n27n39 * d11d20 + 2 * n29n38 * d12d20 + 2 * n28n39 * d12d20 + 2 * n29n39 * d20;
				const double coeffs11 = n35n37 * d13d19k2k2 - n34n38 * d13d19k2k2 - n32n37 * d14d19k2k2 + n31n38 * d14d19k2k2 + n37n37 * d11d11d20 + 2 * n37n38 * d11d12d20 + n38n38 * d12d12d20 + n32n34 * d19k2k2 - n31n35 * d19k2k2 + 2 * n37n39 * d11d20 + 2 * n38n39 * d12d20 + n39n39 * d20;

				double * const C_ptr = reinterpret_cast<double *>(C.data);

				C_ptr[0] = coeffs0;
				C_ptr[50] = coeffs6;
				C_ptr[1] = coeffs1;
				C_ptr[11] = coeffs0;
				C_ptr[21] = coeffs6;
				C_ptr[51] = coeffs7;
				C_ptr[2] = coeffs2;
				C_ptr[12] = coeffs1;
				C_ptr[22] = coeffs7;
				C_ptr[52] = coeffs8;
				C_ptr[3] = coeffs3;
				C_ptr[33] = coeffs6;
				C_ptr[43] = coeffs0;
				C_ptr[53] = coeffs9;
				C_ptr[4] = coeffs4;
				C_ptr[14] = coeffs3;
				C_ptr[24] = coeffs9;
				C_ptr[34] = coeffs7;
				C_ptr[44] = coeffs1;
				C_ptr[54] = coeffs10;
				C_ptr[15] = coeffs2;
				C_ptr[25] = coeffs8;
				C_ptr[36] = coeffs11;
				C_ptr[46] = coeffs5;
				C_ptr[7] = coeffs5;
				C_ptr[37] = coeffs9;
				C_ptr[47] = coeffs3;
				C_ptr[57] = coeffs11;
				C_ptr[18] = coeffs5;
				C_ptr[28] = coeffs11;
				C_ptr[38] = coeffs10;
				C_ptr[48] = coeffs4;
				C_ptr[19] = coeffs4;
				C_ptr[29] = coeffs10;
				C_ptr[39] = coeffs8;
				C_ptr[49] = coeffs2;
						
				static int double_size = sizeof(double);
				static int double_size_4 = 4 * double_size;
				static int double_size_5 = 5 * double_size;
				static int double_size_6 = 6 * double_size;
				static int double_size_8 = 8 * double_size;
				static int double_size_10 = 10 * double_size;
				static int double_size_12 = 12 * double_size;
				static int double_size_16 = 16 * double_size;
				static int double_size_18 = 18 * double_size;
				static int double_size_20 = 20 * double_size;
				static int double_size_24 = 24 * double_size;
				static int double_size_26 = 26 * double_size;
				static int double_size_30 = 30 * double_size;
				static int double_size_36 = 36 * double_size;
				static int double_size_40 = 40 * double_size;
				static int double_size_46 = 46 * double_size;
				static int double_size_50 = 50 * double_size;
				static int double_size_56 = 56 * double_size;

				memcpy(C0.data, C.data, double_size_6);
				memcpy(C0.data + double_size_6, C.data + double_size_10, double_size_6);
				memcpy(C0.data + double_size_12, C.data + double_size_20, double_size_6);
				memcpy(C0.data + double_size_18, C.data + double_size_30, double_size_6);
				memcpy(C0.data + double_size_24, C.data + double_size_40, double_size_6);
				memcpy(C0.data + double_size_30, C.data + double_size_50, double_size_6);
				
				memcpy(C1.data, C.data + double_size_6, double_size_4);
				memcpy(C1.data + double_size_4, C.data + double_size_16, double_size_4);
				memcpy(C1.data + double_size_8, C.data + double_size_26, double_size_4);
				memcpy(C1.data + double_size_12, C.data + double_size_36, double_size_4);
				memcpy(C1.data + double_size_16, C.data + double_size_46, double_size_4);
				memcpy(C1.data + double_size_20, C.data + double_size_56, double_size_4);

				cv::Mat C2;
				cv::solve(C0, C1, C2);
				
				double *C2_ptr = reinterpret_cast<double *>(C2.data);
				
				memcpy(RR_ptr, C2.data, double_size_8);

				RR_ptr[0] = -C2_ptr[16]; 
				RR_ptr[1] = -C2_ptr[17]; 
				RR_ptr[2] = -C2_ptr[18]; 
				RR_ptr[3] = -C2_ptr[19]; 
				RR_ptr[4] = -C2_ptr[20]; 
				RR_ptr[5] = -C2_ptr[21]; 
				RR_ptr[6] = -C2_ptr[22]; 
				RR_ptr[7] = -C2_ptr[23]; 
				
				memcpy(AM.data, RR.data + double_size_16, double_size_4);
				memcpy(AM.data + double_size_4, RR.data, double_size_4);
				memcpy(AM.data + double_size_8, RR.data + double_size_20, double_size_4);
				memcpy(AM.data + double_size_12, RR.data + double_size_4, double_size_4);

				cv::Mat evecs, evals;
				utils::EigenvalueDecomposition eig(AM);
				evals = eig.eigenvalues();
				if (evals.rows == 0)
					return false;
				evecs = eig.eigenvectors();

				evecs.at<double>(4) /= evecs.at<double>(0);
				evecs.at<double>(5) /= evecs.at<double>(1);
				evecs.at<double>(6) /= evecs.at<double>(2);
				evecs.at<double>(7) /= evecs.at<double>(3);
								
				double *h_ptr = NULL;
				double alpha, beta;
				for (int i = 0; i < evecs.cols; ++i)
				{
					alpha = evecs.at<double>(4 + i);
					beta = evals.at<double>(i); 

					Model model;
					model.descriptor.resize(3, 3);

					model.descriptor(0, 0) = alpha * n11 + beta * n21 + n31;
					model.descriptor(0, 1) = alpha * n12 + beta * n22 + n32;
					model.descriptor(0, 2) = alpha * n13 + beta * n23 + n33;
					model.descriptor(1, 0) = alpha * n14 + beta * n24 + n34;
					model.descriptor(1, 1) = alpha * n15 + beta * n25 + n35;
					model.descriptor(1, 2) = alpha * n16 + beta * n26 + n36;
					model.descriptor(2, 0) = alpha * n17 + beta * n27 + n37;
					model.descriptor(2, 1) = alpha * n18 + beta * n28 + n38;
					model.descriptor(2, 2) = alpha * n19 + beta * n29 + n39;

					model.descriptor /= model.descriptor(2, 2);

					models_.push_back(model);
				}

				return true;
			}
		}
	}
}