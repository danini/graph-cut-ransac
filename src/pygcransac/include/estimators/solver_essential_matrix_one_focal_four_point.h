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
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EssentialOnefocal4PC : public SolverEngine
			{
			public:
				EssentialOnefocal4PC()
				{
				}

				~EssentialOnefocal4PC()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				static const char *getName()
				{
					return "E1f-4PT";
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

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				Eigen::Matrix3d gravity_source;
				Eigen::Matrix3d gravity_destination;
				Eigen::MatrixXcd solver_4pt_onefocal(const Eigen::VectorXd &data_) const;
			};

			Eigen::MatrixXcd EssentialOnefocal4PC::solver_4pt_onefocal(const Eigen::VectorXd &data) const
			{
				using namespace Eigen;
				const double *d = data.data();
				VectorXd coeffs(60);
				coeffs[0] = d[14];
				coeffs[1] = d[9];
				coeffs[2] = d[13];
				coeffs[3] = d[4];
				coeffs[4] = d[8];
				coeffs[5] = d[12];
				coeffs[6] = d[3];
				coeffs[7] = d[7];
				coeffs[8] = d[11];
				coeffs[9] = d[2];
				coeffs[10] = d[6];
				coeffs[11] = d[10];
				coeffs[12] = d[1];
				coeffs[13] = d[5];
				coeffs[14] = d[0];
				coeffs[15] = d[29];
				coeffs[16] = d[24];
				coeffs[17] = d[28];
				coeffs[18] = d[19];
				coeffs[19] = d[23];
				coeffs[20] = d[27];
				coeffs[21] = d[18];
				coeffs[22] = d[22];
				coeffs[23] = d[26];
				coeffs[24] = d[17];
				coeffs[25] = d[21];
				coeffs[26] = d[25];
				coeffs[27] = d[16];
				coeffs[28] = d[20];
				coeffs[29] = d[15];
				coeffs[30] = d[44];
				coeffs[31] = d[39];
				coeffs[32] = d[43];
				coeffs[33] = d[34];
				coeffs[34] = d[38];
				coeffs[35] = d[42];
				coeffs[36] = d[33];
				coeffs[37] = d[37];
				coeffs[38] = d[41];
				coeffs[39] = d[32];
				coeffs[40] = d[36];
				coeffs[41] = d[40];
				coeffs[42] = d[31];
				coeffs[43] = d[35];
				coeffs[44] = d[30];
				coeffs[45] = d[59];
				coeffs[46] = d[54];
				coeffs[47] = d[58];
				coeffs[48] = d[49];
				coeffs[49] = d[53];
				coeffs[50] = d[57];
				coeffs[51] = d[48];
				coeffs[52] = d[52];
				coeffs[53] = d[56];
				coeffs[54] = d[47];
				coeffs[55] = d[51];
				coeffs[56] = d[55];
				coeffs[57] = d[46];
				coeffs[58] = d[50];
				coeffs[59] = d[45];

				static const int coeffs_ind[] = {0, 15, 30, 45, 1, 16, 31, 46, 2, 17, 15, 32, 30, 0, 45, 47, 3, 18, 33, 48, 4, 19, 16, 34, 31, 1, 46, 49, 5, 20, 17, 35, 32, 2, 47, 50, 6, 21, 18, 36, 33, 3, 48, 51, 7, 22, 19, 37, 34, 4,
												 49, 52, 8, 23, 20, 38, 35, 5, 50, 53, 9, 24, 21, 39, 36, 6, 51, 54, 10, 25, 22, 40, 37, 7, 52, 55, 11, 26, 23, 41, 38, 8, 53, 56, 12, 27, 24, 42, 39, 9, 54, 57, 13, 28, 25, 43, 40, 10, 55, 58,
												 26, 41, 11, 56, 14, 29, 27, 44, 42, 12, 57, 59, 28, 43, 13, 58, 29, 44, 14, 59};

				static const int C_ind[] = {0, 1, 3, 7, 8, 9, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
											62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
											114, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 130, 132, 133, 134, 138, 140, 141, 142};

				MatrixXd C = MatrixXd::Zero(8, 18);
				for (int i = 0; i < 120; i++)
				{
					C(C_ind[i]) = coeffs(coeffs_ind[i]);
				}

				MatrixXd C0 = C.leftCols(8);
				MatrixXd C1 = C.rightCols(10);
				MatrixXd C12 = C0.fullPivLu().solve(C1);
				MatrixXd RR(13, 10);
				RR << -C12.bottomRows(3), MatrixXd::Identity(10, 10);

				static const int AM_ind[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10};
				MatrixXd AM(10, 10);
				for (int i = 0; i < 10; i++)
				{
					AM.row(i) = RR.row(AM_ind[i]);
				}

				EigenSolver<MatrixXd> es(AM);
				ArrayXcd D = es.eigenvalues();
				ArrayXXcd V = es.eigenvectors();
				V = (V / V.row(9).replicate(10, 1)).eval();

				MatrixXcd sols(2, 10);
				sols.row(0) = D.transpose();
				sols.row(1) = V.row(8);
				return sols;
			}

			OLGA_INLINE bool EssentialOnefocal4PC::estimateModel(
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

				const size_t pointNumber = sample_number_; // 4

				const double
					&u1 = data_.at<double>(sample_[0], 0),
					&v1 = data_.at<double>(sample_[0], 1),
					&u2 = data_.at<double>(sample_[0], 2),
					&v2 = data_.at<double>(sample_[0], 3),
					&u3 = data_.at<double>(sample_[1], 0),
					&v3 = data_.at<double>(sample_[1], 1),
					&u4 = data_.at<double>(sample_[1], 2),
					&v4 = data_.at<double>(sample_[1], 3),
					&u5 = data_.at<double>(sample_[2], 0),
					&v5 = data_.at<double>(sample_[2], 1),
					&u6 = data_.at<double>(sample_[2], 2),
					&v6 = data_.at<double>(sample_[2], 3),
					&u7 = data_.at<double>(sample_[3], 0),
					&v7 = data_.at<double>(sample_[3], 1),
					&u8 = data_.at<double>(sample_[3], 2),
					&v8 = data_.at<double>(sample_[3], 3);

				double b1 = gravity_destination(0, 1) / gravity_destination(0, 0);
				double b2 = gravity_destination(1, 0) / gravity_destination(0, 0);
				double b3 = gravity_destination(1, 1) / gravity_destination(0, 0);
				double b4 = gravity_destination(1, 2) / gravity_destination(0, 0);
				double b5 = gravity_destination(2, 0) / gravity_destination(0, 0);
				double b6 = gravity_destination(2, 1) / gravity_destination(0, 0);
				double b7 = gravity_destination(2, 2) / gravity_destination(0, 0);

				Eigen::Matrix<double, 3, 4> p1;
				p1 << u1, u3, u5, u7,
					v1, v3, v5, v7,
					1, 1, 1, 1;
				Eigen::Matrix<double, 3, 4> q1;
				q1 = gravity_source * p1;

				// q1 = (q1 / q1.row(2).replicate(3, 1)).eval();
				q1.row(0) = q1.row(0).array() / q1.row(2).array();
				q1.row(1) = q1.row(1).array() / q1.row(2).array();

				Eigen::Matrix<double, 2, 4> p2;
				p2 << u2, u4, u6, u8,
					v2, v4, v6, v8;

				Eigen::Matrix<double, 3, 2> align2;
				align2 << 1, b1,
					b2, b3,
					b5, b6;

				Eigen::Matrix<double, 3, 4> q2;
				q2 = align2 * p2;

				Eigen::MatrixXd M = MatrixXd::Zero(4, 18);

				for (size_t i = 0; i < 4; ++i)
				{
					M(i, 0) = q2(1, i) - q1(1, i) * q2(2, i);
					M(i, 1) = -2 * q1(0, i) * q2(1, i);
					M(i, 2) = -q2(1, i) - q1(1, i) * q2(2, i);
					M(i, 3) = b4 - b7 * q1(1, i);
					M(i, 4) = -2 * b4 * q1(0, i);
					M(i, 5) = -b4 - b7 * q1(1, i);
					M(i, 6) = q1(0, i) * q2(2, i) - q2(0, i);
					M(i, 7) = 2 * q2(2, i) + 2 * q1(0, i) * q2(0, i);
					M(i, 8) = q2(0, i) - q1(0, i) * q2(2, i);
					M(i, 9) = b7 * q1(0, i);
					M(i, 10) = 2 * b7;
					M(i, 11) = -b7 * q1(0, i);
					M(i, 12) = q1(1, i) * q2(0, i) - q1(0, i) * q2(1, i);
					M(i, 13) = -2 * q2(1, i);
					M(i, 14) = q1(0, i) * q2(1, i) + q1(1, i) * q2(0, i);
					M(i, 15) = -b4 * q1(0, i);
					M(i, 16) = -2 * b4;
					M(i, 17) = b4 * q1(0, i);
				}

				Eigen::MatrixXd N = MatrixXd::Zero(3, 18);
				N.row(0) = M.row(1);
				N.row(1) = M.row(2);
				N.row(2) = M.row(3);
				Eigen::MatrixXd D = MatrixXd::Zero(3, 9);
				D << N(0, 0) * N(2, 14) + N(0, 1) * N(2, 13) + N(0, 2) * N(2, 12) - N(0, 12) * N(2, 2) - N(0, 13) * N(2, 1) - N(0, 14) * N(2, 0) - N(0, 2) * N(2, 14) + N(0, 14) * N(2, 2), N(0, 1) * N(2, 14) + N(0, 2) * N(2, 13) - N(0, 13) * N(2, 2) - N(0, 14) * N(2, 1), N(0, 2) * N(2, 14) - N(0, 14) * N(2, 2), N(0, 0) * N(2, 17) + N(0, 1) * N(2, 16) + N(0, 2) * N(2, 15) + N(0, 3) * N(2, 14) + N(0, 4) * N(2, 13) + N(0, 5) * N(2, 12) - N(0, 12) * N(2, 5) - N(0, 13) * N(2, 4) - N(0, 14) * N(2, 3) - N(0, 15) * N(2, 2) - N(0, 16) * N(2, 1) - N(0, 17) * N(2, 0) - N(0, 2) * N(2, 17) - N(0, 5) * N(2, 14) + N(0, 14) * N(2, 5) + N(0, 17) * N(2, 2), N(0, 1) * N(2, 17) + N(0, 2) * N(2, 16) + N(0, 4) * N(2, 14) + N(0, 5) * N(2, 13) - N(0, 13) * N(2, 5) - N(0, 14) * N(2, 4) - N(0, 16) * N(2, 2) - N(0, 17) * N(2, 1), N(0, 2) * N(2, 17) + N(0, 5) * N(2, 14) - N(0, 14) * N(2, 5) - N(0, 17) * N(2, 2), N(0, 3) * N(2, 17) + N(0, 4) * N(2, 16) + N(0, 5) * N(2, 15) - N(0, 15) * N(2, 5) - N(0, 16) * N(2, 4) - N(0, 17) * N(2, 3) - N(0, 5) * N(2, 17) + N(0, 17) * N(2, 5), N(0, 4) * N(2, 17) + N(0, 5) * N(2, 16) - N(0, 16) * N(2, 5) - N(0, 17) * N(2, 4), N(0, 5) * N(2, 17) - N(0, 17) * N(2, 5),
					N(0, 12) * N(1, 2) - N(0, 1) * N(1, 13) - N(0, 2) * N(1, 12) - N(0, 0) * N(1, 14) + N(0, 13) * N(1, 1) + N(0, 14) * N(1, 0) + N(0, 2) * N(1, 14) - N(0, 14) * N(1, 2), N(0, 13) * N(1, 2) - N(0, 2) * N(1, 13) - N(0, 1) * N(1, 14) + N(0, 14) * N(1, 1), N(0, 14) * N(1, 2) - N(0, 2) * N(1, 14), N(0, 12) * N(1, 5) - N(0, 1) * N(1, 16) - N(0, 2) * N(1, 15) - N(0, 3) * N(1, 14) - N(0, 4) * N(1, 13) - N(0, 5) * N(1, 12) - N(0, 0) * N(1, 17) + N(0, 13) * N(1, 4) + N(0, 14) * N(1, 3) + N(0, 15) * N(1, 2) + N(0, 16) * N(1, 1) + N(0, 17) * N(1, 0) + N(0, 2) * N(1, 17) + N(0, 5) * N(1, 14) - N(0, 14) * N(1, 5) - N(0, 17) * N(1, 2), N(0, 13) * N(1, 5) - N(0, 2) * N(1, 16) - N(0, 4) * N(1, 14) - N(0, 5) * N(1, 13) - N(0, 1) * N(1, 17) + N(0, 14) * N(1, 4) + N(0, 16) * N(1, 2) + N(0, 17) * N(1, 1), N(0, 14) * N(1, 5) - N(0, 5) * N(1, 14) - N(0, 2) * N(1, 17) + N(0, 17) * N(1, 2), N(0, 15) * N(1, 5) - N(0, 4) * N(1, 16) - N(0, 5) * N(1, 15) - N(0, 3) * N(1, 17) + N(0, 16) * N(1, 4) + N(0, 17) * N(1, 3) + N(0, 5) * N(1, 17) - N(0, 17) * N(1, 5), N(0, 16) * N(1, 5) - N(0, 5) * N(1, 16) - N(0, 4) * N(1, 17) + N(0, 17) * N(1, 4), N(0, 17) * N(1, 5) - N(0, 5) * N(1, 17),
					N(1, 12) * N(2, 2) - N(1, 1) * N(2, 13) - N(1, 2) * N(2, 12) - N(1, 0) * N(2, 14) + N(1, 13) * N(2, 1) + N(1, 14) * N(2, 0) + N(1, 2) * N(2, 14) - N(1, 14) * N(2, 2), N(1, 13) * N(2, 2) - N(1, 2) * N(2, 13) - N(1, 1) * N(2, 14) + N(1, 14) * N(2, 1), N(1, 14) * N(2, 2) - N(1, 2) * N(2, 14), N(1, 12) * N(2, 5) - N(1, 1) * N(2, 16) - N(1, 2) * N(2, 15) - N(1, 3) * N(2, 14) - N(1, 4) * N(2, 13) - N(1, 5) * N(2, 12) - N(1, 0) * N(2, 17) + N(1, 13) * N(2, 4) + N(1, 14) * N(2, 3) + N(1, 15) * N(2, 2) + N(1, 16) * N(2, 1) + N(1, 17) * N(2, 0) + N(1, 2) * N(2, 17) + N(1, 5) * N(2, 14) - N(1, 14) * N(2, 5) - N(1, 17) * N(2, 2), N(1, 13) * N(2, 5) - N(1, 2) * N(2, 16) - N(1, 4) * N(2, 14) - N(1, 5) * N(2, 13) - N(1, 1) * N(2, 17) + N(1, 14) * N(2, 4) + N(1, 16) * N(2, 2) + N(1, 17) * N(2, 1), N(1, 14) * N(2, 5) - N(1, 5) * N(2, 14) - N(1, 2) * N(2, 17) + N(1, 17) * N(2, 2), N(1, 15) * N(2, 5) - N(1, 4) * N(2, 16) - N(1, 5) * N(2, 15) - N(1, 3) * N(2, 17) + N(1, 16) * N(2, 4) + N(1, 17) * N(2, 3) + N(1, 5) * N(2, 17) - N(1, 17) * N(2, 5), N(1, 16) * N(2, 5) - N(1, 5) * N(2, 16) - N(1, 4) * N(2, 17) + N(1, 17) * N(2, 4), N(1, 17) * N(2, 5) - N(1, 5) * N(2, 17);

				Eigen::MatrixXd D2 = MatrixXd::Zero(4, 15);
				Eigen::MatrixXd D21 = MatrixXd::Zero(1, 15);
				D21 << N(1, 6) * D(0, 0), N(1, 6) * D(0, 1) + N(1, 7) * D(0, 0), N(1, 6) * D(0, 2) + N(1, 7) * D(0, 1) + N(1, 8) * D(0, 0), N(1, 7) * D(0, 2) + N(1, 8) * D(0, 1), N(1, 8) * D(0, 2), N(1, 6) * D(0, 3) + N(1, 9) * D(0, 0), N(1, 6) * D(0, 4) + N(1, 7) * D(0, 3) + N(1, 9) * D(0, 1) + N(1, 10) * D(0, 0), N(1, 6) * D(0, 5) + N(1, 7) * D(0, 4) + N(1, 8) * D(0, 3) + N(1, 9) * D(0, 2) + N(1, 10) * D(0, 1) + N(1, 11) * D(0, 0), N(1, 7) * D(0, 5) + N(1, 8) * D(0, 4) + N(1, 10) * D(0, 2) + N(1, 11) * D(0, 1), N(1, 8) * D(0, 5) + N(1, 11) * D(0, 2), N(1, 6) * D(0, 6) + N(1, 9) * D(0, 3), N(1, 6) * D(0, 7) + N(1, 7) * D(0, 6) + N(1, 9) * D(0, 4) + N(1, 10) * D(0, 3), N(1, 6) * D(0, 8) + N(1, 7) * D(0, 7) + N(1, 8) * D(0, 6) + N(1, 9) * D(0, 5) + N(1, 10) * D(0, 4) + N(1, 11) * D(0, 3), N(1, 7) * D(0, 8) + N(1, 8) * D(0, 7) + N(1, 10) * D(0, 5) + N(1, 11) * D(0, 4), N(1, 8) * D(0, 8) + N(1, 11) * D(0, 5);

				Eigen::MatrixXd D22 = MatrixXd::Zero(1, 15);
				D22 << N(2, 6) * D(1, 0), N(2, 6) * D(1, 1) + N(2, 7) * D(1, 0), N(2, 6) * D(1, 2) + N(2, 7) * D(1, 1) + N(2, 8) * D(1, 0), N(2, 7) * D(1, 2) + N(2, 8) * D(1, 1), N(2, 8) * D(1, 2), N(2, 6) * D(1, 3) + N(2, 9) * D(1, 0), N(2, 6) * D(1, 4) + N(2, 7) * D(1, 3) + N(2, 9) * D(1, 1) + N(2, 10) * D(1, 0), N(2, 6) * D(1, 5) + N(2, 7) * D(1, 4) + N(2, 8) * D(1, 3) + N(2, 9) * D(1, 2) + N(2, 10) * D(1, 1) + N(2, 11) * D(1, 0), N(2, 7) * D(1, 5) + N(2, 8) * D(1, 4) + N(2, 10) * D(1, 2) + N(2, 11) * D(1, 1), N(2, 8) * D(1, 5) + N(2, 11) * D(1, 2), N(2, 6) * D(1, 6) + N(2, 9) * D(1, 3), N(2, 6) * D(1, 7) + N(2, 7) * D(1, 6) + N(2, 9) * D(1, 4) + N(2, 10) * D(1, 3), N(2, 6) * D(1, 8) + N(2, 7) * D(1, 7) + N(2, 8) * D(1, 6) + N(2, 9) * D(1, 5) + N(2, 10) * D(1, 4) + N(2, 11) * D(1, 3), N(2, 7) * D(1, 8) + N(2, 8) * D(1, 7) + N(2, 10) * D(1, 5) + N(2, 11) * D(1, 4), N(2, 8) * D(1, 8) + N(2, 11) * D(1, 5);

				Eigen::MatrixXd D23 = MatrixXd::Zero(1, 15);
				D23 << N(0, 6) * D(2, 0), N(0, 6) * D(2, 1) + N(0, 7) * D(2, 0), N(0, 6) * D(2, 2) + N(0, 7) * D(2, 1) + N(0, 8) * D(2, 0), N(0, 7) * D(2, 2) + N(0, 8) * D(2, 1), N(0, 8) * D(2, 2), N(0, 6) * D(2, 3) + N(0, 9) * D(2, 0), N(0, 6) * D(2, 4) + N(0, 7) * D(2, 3) + N(0, 9) * D(2, 1) + N(0, 10) * D(2, 0), N(0, 6) * D(2, 5) + N(0, 7) * D(2, 4) + N(0, 8) * D(2, 3) + N(0, 9) * D(2, 2) + N(0, 10) * D(2, 1) + N(0, 11) * D(2, 0), N(0, 7) * D(2, 5) + N(0, 8) * D(2, 4) + N(0, 10) * D(2, 2) + N(0, 11) * D(2, 1), N(0, 8) * D(2, 5) + N(0, 11) * D(2, 2), N(0, 6) * D(2, 6) + N(0, 9) * D(2, 3), N(0, 6) * D(2, 7) + N(0, 7) * D(2, 6) + N(0, 9) * D(2, 4) + N(0, 10) * D(2, 3), N(0, 6) * D(2, 8) + N(0, 7) * D(2, 7) + N(0, 8) * D(2, 6) + N(0, 9) * D(2, 5) + N(0, 10) * D(2, 4) + N(0, 11) * D(2, 3), N(0, 7) * D(2, 8) + N(0, 8) * D(2, 7) + N(0, 10) * D(2, 5) + N(0, 11) * D(2, 4), N(0, 8) * D(2, 8) + N(0, 11) * D(2, 5);

				D2.row(0) = D21 + D22 + D23;

				N.row(0) = M.row(0);
				N.row(1) = M.row(2);
				N.row(2) = M.row(3);
				D << N(0, 0) * N(2, 14) + N(0, 1) * N(2, 13) + N(0, 2) * N(2, 12) - N(0, 12) * N(2, 2) - N(0, 13) * N(2, 1) - N(0, 14) * N(2, 0) - N(0, 2) * N(2, 14) + N(0, 14) * N(2, 2), N(0, 1) * N(2, 14) + N(0, 2) * N(2, 13) - N(0, 13) * N(2, 2) - N(0, 14) * N(2, 1), N(0, 2) * N(2, 14) - N(0, 14) * N(2, 2), N(0, 0) * N(2, 17) + N(0, 1) * N(2, 16) + N(0, 2) * N(2, 15) + N(0, 3) * N(2, 14) + N(0, 4) * N(2, 13) + N(0, 5) * N(2, 12) - N(0, 12) * N(2, 5) - N(0, 13) * N(2, 4) - N(0, 14) * N(2, 3) - N(0, 15) * N(2, 2) - N(0, 16) * N(2, 1) - N(0, 17) * N(2, 0) - N(0, 2) * N(2, 17) - N(0, 5) * N(2, 14) + N(0, 14) * N(2, 5) + N(0, 17) * N(2, 2), N(0, 1) * N(2, 17) + N(0, 2) * N(2, 16) + N(0, 4) * N(2, 14) + N(0, 5) * N(2, 13) - N(0, 13) * N(2, 5) - N(0, 14) * N(2, 4) - N(0, 16) * N(2, 2) - N(0, 17) * N(2, 1), N(0, 2) * N(2, 17) + N(0, 5) * N(2, 14) - N(0, 14) * N(2, 5) - N(0, 17) * N(2, 2), N(0, 3) * N(2, 17) + N(0, 4) * N(2, 16) + N(0, 5) * N(2, 15) - N(0, 15) * N(2, 5) - N(0, 16) * N(2, 4) - N(0, 17) * N(2, 3) - N(0, 5) * N(2, 17) + N(0, 17) * N(2, 5), N(0, 4) * N(2, 17) + N(0, 5) * N(2, 16) - N(0, 16) * N(2, 5) - N(0, 17) * N(2, 4), N(0, 5) * N(2, 17) - N(0, 17) * N(2, 5),
					N(0, 12) * N(1, 2) - N(0, 1) * N(1, 13) - N(0, 2) * N(1, 12) - N(0, 0) * N(1, 14) + N(0, 13) * N(1, 1) + N(0, 14) * N(1, 0) + N(0, 2) * N(1, 14) - N(0, 14) * N(1, 2), N(0, 13) * N(1, 2) - N(0, 2) * N(1, 13) - N(0, 1) * N(1, 14) + N(0, 14) * N(1, 1), N(0, 14) * N(1, 2) - N(0, 2) * N(1, 14), N(0, 12) * N(1, 5) - N(0, 1) * N(1, 16) - N(0, 2) * N(1, 15) - N(0, 3) * N(1, 14) - N(0, 4) * N(1, 13) - N(0, 5) * N(1, 12) - N(0, 0) * N(1, 17) + N(0, 13) * N(1, 4) + N(0, 14) * N(1, 3) + N(0, 15) * N(1, 2) + N(0, 16) * N(1, 1) + N(0, 17) * N(1, 0) + N(0, 2) * N(1, 17) + N(0, 5) * N(1, 14) - N(0, 14) * N(1, 5) - N(0, 17) * N(1, 2), N(0, 13) * N(1, 5) - N(0, 2) * N(1, 16) - N(0, 4) * N(1, 14) - N(0, 5) * N(1, 13) - N(0, 1) * N(1, 17) + N(0, 14) * N(1, 4) + N(0, 16) * N(1, 2) + N(0, 17) * N(1, 1), N(0, 14) * N(1, 5) - N(0, 5) * N(1, 14) - N(0, 2) * N(1, 17) + N(0, 17) * N(1, 2), N(0, 15) * N(1, 5) - N(0, 4) * N(1, 16) - N(0, 5) * N(1, 15) - N(0, 3) * N(1, 17) + N(0, 16) * N(1, 4) + N(0, 17) * N(1, 3) + N(0, 5) * N(1, 17) - N(0, 17) * N(1, 5), N(0, 16) * N(1, 5) - N(0, 5) * N(1, 16) - N(0, 4) * N(1, 17) + N(0, 17) * N(1, 4), N(0, 17) * N(1, 5) - N(0, 5) * N(1, 17),
					N(1, 12) * N(2, 2) - N(1, 1) * N(2, 13) - N(1, 2) * N(2, 12) - N(1, 0) * N(2, 14) + N(1, 13) * N(2, 1) + N(1, 14) * N(2, 0) + N(1, 2) * N(2, 14) - N(1, 14) * N(2, 2), N(1, 13) * N(2, 2) - N(1, 2) * N(2, 13) - N(1, 1) * N(2, 14) + N(1, 14) * N(2, 1), N(1, 14) * N(2, 2) - N(1, 2) * N(2, 14), N(1, 12) * N(2, 5) - N(1, 1) * N(2, 16) - N(1, 2) * N(2, 15) - N(1, 3) * N(2, 14) - N(1, 4) * N(2, 13) - N(1, 5) * N(2, 12) - N(1, 0) * N(2, 17) + N(1, 13) * N(2, 4) + N(1, 14) * N(2, 3) + N(1, 15) * N(2, 2) + N(1, 16) * N(2, 1) + N(1, 17) * N(2, 0) + N(1, 2) * N(2, 17) + N(1, 5) * N(2, 14) - N(1, 14) * N(2, 5) - N(1, 17) * N(2, 2), N(1, 13) * N(2, 5) - N(1, 2) * N(2, 16) - N(1, 4) * N(2, 14) - N(1, 5) * N(2, 13) - N(1, 1) * N(2, 17) + N(1, 14) * N(2, 4) + N(1, 16) * N(2, 2) + N(1, 17) * N(2, 1), N(1, 14) * N(2, 5) - N(1, 5) * N(2, 14) - N(1, 2) * N(2, 17) + N(1, 17) * N(2, 2), N(1, 15) * N(2, 5) - N(1, 4) * N(2, 16) - N(1, 5) * N(2, 15) - N(1, 3) * N(2, 17) + N(1, 16) * N(2, 4) + N(1, 17) * N(2, 3) + N(1, 5) * N(2, 17) - N(1, 17) * N(2, 5), N(1, 16) * N(2, 5) - N(1, 5) * N(2, 16) - N(1, 4) * N(2, 17) + N(1, 17) * N(2, 4), N(1, 17) * N(2, 5) - N(1, 5) * N(2, 17);

				D21 << N(1, 6) * D(0, 0), N(1, 6) * D(0, 1) + N(1, 7) * D(0, 0), N(1, 6) * D(0, 2) + N(1, 7) * D(0, 1) + N(1, 8) * D(0, 0), N(1, 7) * D(0, 2) + N(1, 8) * D(0, 1), N(1, 8) * D(0, 2), N(1, 6) * D(0, 3) + N(1, 9) * D(0, 0), N(1, 6) * D(0, 4) + N(1, 7) * D(0, 3) + N(1, 9) * D(0, 1) + N(1, 10) * D(0, 0), N(1, 6) * D(0, 5) + N(1, 7) * D(0, 4) + N(1, 8) * D(0, 3) + N(1, 9) * D(0, 2) + N(1, 10) * D(0, 1) + N(1, 11) * D(0, 0), N(1, 7) * D(0, 5) + N(1, 8) * D(0, 4) + N(1, 10) * D(0, 2) + N(1, 11) * D(0, 1), N(1, 8) * D(0, 5) + N(1, 11) * D(0, 2), N(1, 6) * D(0, 6) + N(1, 9) * D(0, 3), N(1, 6) * D(0, 7) + N(1, 7) * D(0, 6) + N(1, 9) * D(0, 4) + N(1, 10) * D(0, 3), N(1, 6) * D(0, 8) + N(1, 7) * D(0, 7) + N(1, 8) * D(0, 6) + N(1, 9) * D(0, 5) + N(1, 10) * D(0, 4) + N(1, 11) * D(0, 3), N(1, 7) * D(0, 8) + N(1, 8) * D(0, 7) + N(1, 10) * D(0, 5) + N(1, 11) * D(0, 4), N(1, 8) * D(0, 8) + N(1, 11) * D(0, 5);

				D22 << N(2, 6) * D(1, 0), N(2, 6) * D(1, 1) + N(2, 7) * D(1, 0), N(2, 6) * D(1, 2) + N(2, 7) * D(1, 1) + N(2, 8) * D(1, 0), N(2, 7) * D(1, 2) + N(2, 8) * D(1, 1), N(2, 8) * D(1, 2), N(2, 6) * D(1, 3) + N(2, 9) * D(1, 0), N(2, 6) * D(1, 4) + N(2, 7) * D(1, 3) + N(2, 9) * D(1, 1) + N(2, 10) * D(1, 0), N(2, 6) * D(1, 5) + N(2, 7) * D(1, 4) + N(2, 8) * D(1, 3) + N(2, 9) * D(1, 2) + N(2, 10) * D(1, 1) + N(2, 11) * D(1, 0), N(2, 7) * D(1, 5) + N(2, 8) * D(1, 4) + N(2, 10) * D(1, 2) + N(2, 11) * D(1, 1), N(2, 8) * D(1, 5) + N(2, 11) * D(1, 2), N(2, 6) * D(1, 6) + N(2, 9) * D(1, 3), N(2, 6) * D(1, 7) + N(2, 7) * D(1, 6) + N(2, 9) * D(1, 4) + N(2, 10) * D(1, 3), N(2, 6) * D(1, 8) + N(2, 7) * D(1, 7) + N(2, 8) * D(1, 6) + N(2, 9) * D(1, 5) + N(2, 10) * D(1, 4) + N(2, 11) * D(1, 3), N(2, 7) * D(1, 8) + N(2, 8) * D(1, 7) + N(2, 10) * D(1, 5) + N(2, 11) * D(1, 4), N(2, 8) * D(1, 8) + N(2, 11) * D(1, 5);

				D23 << N(0, 6) * D(2, 0), N(0, 6) * D(2, 1) + N(0, 7) * D(2, 0), N(0, 6) * D(2, 2) + N(0, 7) * D(2, 1) + N(0, 8) * D(2, 0), N(0, 7) * D(2, 2) + N(0, 8) * D(2, 1), N(0, 8) * D(2, 2), N(0, 6) * D(2, 3) + N(0, 9) * D(2, 0), N(0, 6) * D(2, 4) + N(0, 7) * D(2, 3) + N(0, 9) * D(2, 1) + N(0, 10) * D(2, 0), N(0, 6) * D(2, 5) + N(0, 7) * D(2, 4) + N(0, 8) * D(2, 3) + N(0, 9) * D(2, 2) + N(0, 10) * D(2, 1) + N(0, 11) * D(2, 0), N(0, 7) * D(2, 5) + N(0, 8) * D(2, 4) + N(0, 10) * D(2, 2) + N(0, 11) * D(2, 1), N(0, 8) * D(2, 5) + N(0, 11) * D(2, 2), N(0, 6) * D(2, 6) + N(0, 9) * D(2, 3), N(0, 6) * D(2, 7) + N(0, 7) * D(2, 6) + N(0, 9) * D(2, 4) + N(0, 10) * D(2, 3), N(0, 6) * D(2, 8) + N(0, 7) * D(2, 7) + N(0, 8) * D(2, 6) + N(0, 9) * D(2, 5) + N(0, 10) * D(2, 4) + N(0, 11) * D(2, 3), N(0, 7) * D(2, 8) + N(0, 8) * D(2, 7) + N(0, 10) * D(2, 5) + N(0, 11) * D(2, 4), N(0, 8) * D(2, 8) + N(0, 11) * D(2, 5);

				D2.row(1) = D21 + D22 + D23;

				N.row(0) = M.row(0);
				N.row(1) = M.row(1);
				N.row(2) = M.row(3);
				D << N(0, 0) * N(2, 14) + N(0, 1) * N(2, 13) + N(0, 2) * N(2, 12) - N(0, 12) * N(2, 2) - N(0, 13) * N(2, 1) - N(0, 14) * N(2, 0) - N(0, 2) * N(2, 14) + N(0, 14) * N(2, 2), N(0, 1) * N(2, 14) + N(0, 2) * N(2, 13) - N(0, 13) * N(2, 2) - N(0, 14) * N(2, 1), N(0, 2) * N(2, 14) - N(0, 14) * N(2, 2), N(0, 0) * N(2, 17) + N(0, 1) * N(2, 16) + N(0, 2) * N(2, 15) + N(0, 3) * N(2, 14) + N(0, 4) * N(2, 13) + N(0, 5) * N(2, 12) - N(0, 12) * N(2, 5) - N(0, 13) * N(2, 4) - N(0, 14) * N(2, 3) - N(0, 15) * N(2, 2) - N(0, 16) * N(2, 1) - N(0, 17) * N(2, 0) - N(0, 2) * N(2, 17) - N(0, 5) * N(2, 14) + N(0, 14) * N(2, 5) + N(0, 17) * N(2, 2), N(0, 1) * N(2, 17) + N(0, 2) * N(2, 16) + N(0, 4) * N(2, 14) + N(0, 5) * N(2, 13) - N(0, 13) * N(2, 5) - N(0, 14) * N(2, 4) - N(0, 16) * N(2, 2) - N(0, 17) * N(2, 1), N(0, 2) * N(2, 17) + N(0, 5) * N(2, 14) - N(0, 14) * N(2, 5) - N(0, 17) * N(2, 2), N(0, 3) * N(2, 17) + N(0, 4) * N(2, 16) + N(0, 5) * N(2, 15) - N(0, 15) * N(2, 5) - N(0, 16) * N(2, 4) - N(0, 17) * N(2, 3) - N(0, 5) * N(2, 17) + N(0, 17) * N(2, 5), N(0, 4) * N(2, 17) + N(0, 5) * N(2, 16) - N(0, 16) * N(2, 5) - N(0, 17) * N(2, 4), N(0, 5) * N(2, 17) - N(0, 17) * N(2, 5),
					N(0, 12) * N(1, 2) - N(0, 1) * N(1, 13) - N(0, 2) * N(1, 12) - N(0, 0) * N(1, 14) + N(0, 13) * N(1, 1) + N(0, 14) * N(1, 0) + N(0, 2) * N(1, 14) - N(0, 14) * N(1, 2), N(0, 13) * N(1, 2) - N(0, 2) * N(1, 13) - N(0, 1) * N(1, 14) + N(0, 14) * N(1, 1), N(0, 14) * N(1, 2) - N(0, 2) * N(1, 14), N(0, 12) * N(1, 5) - N(0, 1) * N(1, 16) - N(0, 2) * N(1, 15) - N(0, 3) * N(1, 14) - N(0, 4) * N(1, 13) - N(0, 5) * N(1, 12) - N(0, 0) * N(1, 17) + N(0, 13) * N(1, 4) + N(0, 14) * N(1, 3) + N(0, 15) * N(1, 2) + N(0, 16) * N(1, 1) + N(0, 17) * N(1, 0) + N(0, 2) * N(1, 17) + N(0, 5) * N(1, 14) - N(0, 14) * N(1, 5) - N(0, 17) * N(1, 2), N(0, 13) * N(1, 5) - N(0, 2) * N(1, 16) - N(0, 4) * N(1, 14) - N(0, 5) * N(1, 13) - N(0, 1) * N(1, 17) + N(0, 14) * N(1, 4) + N(0, 16) * N(1, 2) + N(0, 17) * N(1, 1), N(0, 14) * N(1, 5) - N(0, 5) * N(1, 14) - N(0, 2) * N(1, 17) + N(0, 17) * N(1, 2), N(0, 15) * N(1, 5) - N(0, 4) * N(1, 16) - N(0, 5) * N(1, 15) - N(0, 3) * N(1, 17) + N(0, 16) * N(1, 4) + N(0, 17) * N(1, 3) + N(0, 5) * N(1, 17) - N(0, 17) * N(1, 5), N(0, 16) * N(1, 5) - N(0, 5) * N(1, 16) - N(0, 4) * N(1, 17) + N(0, 17) * N(1, 4), N(0, 17) * N(1, 5) - N(0, 5) * N(1, 17),
					N(1, 12) * N(2, 2) - N(1, 1) * N(2, 13) - N(1, 2) * N(2, 12) - N(1, 0) * N(2, 14) + N(1, 13) * N(2, 1) + N(1, 14) * N(2, 0) + N(1, 2) * N(2, 14) - N(1, 14) * N(2, 2), N(1, 13) * N(2, 2) - N(1, 2) * N(2, 13) - N(1, 1) * N(2, 14) + N(1, 14) * N(2, 1), N(1, 14) * N(2, 2) - N(1, 2) * N(2, 14), N(1, 12) * N(2, 5) - N(1, 1) * N(2, 16) - N(1, 2) * N(2, 15) - N(1, 3) * N(2, 14) - N(1, 4) * N(2, 13) - N(1, 5) * N(2, 12) - N(1, 0) * N(2, 17) + N(1, 13) * N(2, 4) + N(1, 14) * N(2, 3) + N(1, 15) * N(2, 2) + N(1, 16) * N(2, 1) + N(1, 17) * N(2, 0) + N(1, 2) * N(2, 17) + N(1, 5) * N(2, 14) - N(1, 14) * N(2, 5) - N(1, 17) * N(2, 2), N(1, 13) * N(2, 5) - N(1, 2) * N(2, 16) - N(1, 4) * N(2, 14) - N(1, 5) * N(2, 13) - N(1, 1) * N(2, 17) + N(1, 14) * N(2, 4) + N(1, 16) * N(2, 2) + N(1, 17) * N(2, 1), N(1, 14) * N(2, 5) - N(1, 5) * N(2, 14) - N(1, 2) * N(2, 17) + N(1, 17) * N(2, 2), N(1, 15) * N(2, 5) - N(1, 4) * N(2, 16) - N(1, 5) * N(2, 15) - N(1, 3) * N(2, 17) + N(1, 16) * N(2, 4) + N(1, 17) * N(2, 3) + N(1, 5) * N(2, 17) - N(1, 17) * N(2, 5), N(1, 16) * N(2, 5) - N(1, 5) * N(2, 16) - N(1, 4) * N(2, 17) + N(1, 17) * N(2, 4), N(1, 17) * N(2, 5) - N(1, 5) * N(2, 17);

				D21 << N(1, 6) * D(0, 0), N(1, 6) * D(0, 1) + N(1, 7) * D(0, 0), N(1, 6) * D(0, 2) + N(1, 7) * D(0, 1) + N(1, 8) * D(0, 0), N(1, 7) * D(0, 2) + N(1, 8) * D(0, 1), N(1, 8) * D(0, 2), N(1, 6) * D(0, 3) + N(1, 9) * D(0, 0), N(1, 6) * D(0, 4) + N(1, 7) * D(0, 3) + N(1, 9) * D(0, 1) + N(1, 10) * D(0, 0), N(1, 6) * D(0, 5) + N(1, 7) * D(0, 4) + N(1, 8) * D(0, 3) + N(1, 9) * D(0, 2) + N(1, 10) * D(0, 1) + N(1, 11) * D(0, 0), N(1, 7) * D(0, 5) + N(1, 8) * D(0, 4) + N(1, 10) * D(0, 2) + N(1, 11) * D(0, 1), N(1, 8) * D(0, 5) + N(1, 11) * D(0, 2), N(1, 6) * D(0, 6) + N(1, 9) * D(0, 3), N(1, 6) * D(0, 7) + N(1, 7) * D(0, 6) + N(1, 9) * D(0, 4) + N(1, 10) * D(0, 3), N(1, 6) * D(0, 8) + N(1, 7) * D(0, 7) + N(1, 8) * D(0, 6) + N(1, 9) * D(0, 5) + N(1, 10) * D(0, 4) + N(1, 11) * D(0, 3), N(1, 7) * D(0, 8) + N(1, 8) * D(0, 7) + N(1, 10) * D(0, 5) + N(1, 11) * D(0, 4), N(1, 8) * D(0, 8) + N(1, 11) * D(0, 5);

				D22 << N(2, 6) * D(1, 0), N(2, 6) * D(1, 1) + N(2, 7) * D(1, 0), N(2, 6) * D(1, 2) + N(2, 7) * D(1, 1) + N(2, 8) * D(1, 0), N(2, 7) * D(1, 2) + N(2, 8) * D(1, 1), N(2, 8) * D(1, 2), N(2, 6) * D(1, 3) + N(2, 9) * D(1, 0), N(2, 6) * D(1, 4) + N(2, 7) * D(1, 3) + N(2, 9) * D(1, 1) + N(2, 10) * D(1, 0), N(2, 6) * D(1, 5) + N(2, 7) * D(1, 4) + N(2, 8) * D(1, 3) + N(2, 9) * D(1, 2) + N(2, 10) * D(1, 1) + N(2, 11) * D(1, 0), N(2, 7) * D(1, 5) + N(2, 8) * D(1, 4) + N(2, 10) * D(1, 2) + N(2, 11) * D(1, 1), N(2, 8) * D(1, 5) + N(2, 11) * D(1, 2), N(2, 6) * D(1, 6) + N(2, 9) * D(1, 3), N(2, 6) * D(1, 7) + N(2, 7) * D(1, 6) + N(2, 9) * D(1, 4) + N(2, 10) * D(1, 3), N(2, 6) * D(1, 8) + N(2, 7) * D(1, 7) + N(2, 8) * D(1, 6) + N(2, 9) * D(1, 5) + N(2, 10) * D(1, 4) + N(2, 11) * D(1, 3), N(2, 7) * D(1, 8) + N(2, 8) * D(1, 7) + N(2, 10) * D(1, 5) + N(2, 11) * D(1, 4), N(2, 8) * D(1, 8) + N(2, 11) * D(1, 5);

				D23 << N(0, 6) * D(2, 0), N(0, 6) * D(2, 1) + N(0, 7) * D(2, 0), N(0, 6) * D(2, 2) + N(0, 7) * D(2, 1) + N(0, 8) * D(2, 0), N(0, 7) * D(2, 2) + N(0, 8) * D(2, 1), N(0, 8) * D(2, 2), N(0, 6) * D(2, 3) + N(0, 9) * D(2, 0), N(0, 6) * D(2, 4) + N(0, 7) * D(2, 3) + N(0, 9) * D(2, 1) + N(0, 10) * D(2, 0), N(0, 6) * D(2, 5) + N(0, 7) * D(2, 4) + N(0, 8) * D(2, 3) + N(0, 9) * D(2, 2) + N(0, 10) * D(2, 1) + N(0, 11) * D(2, 0), N(0, 7) * D(2, 5) + N(0, 8) * D(2, 4) + N(0, 10) * D(2, 2) + N(0, 11) * D(2, 1), N(0, 8) * D(2, 5) + N(0, 11) * D(2, 2), N(0, 6) * D(2, 6) + N(0, 9) * D(2, 3), N(0, 6) * D(2, 7) + N(0, 7) * D(2, 6) + N(0, 9) * D(2, 4) + N(0, 10) * D(2, 3), N(0, 6) * D(2, 8) + N(0, 7) * D(2, 7) + N(0, 8) * D(2, 6) + N(0, 9) * D(2, 5) + N(0, 10) * D(2, 4) + N(0, 11) * D(2, 3), N(0, 7) * D(2, 8) + N(0, 8) * D(2, 7) + N(0, 10) * D(2, 5) + N(0, 11) * D(2, 4), N(0, 8) * D(2, 8) + N(0, 11) * D(2, 5);

				D2.row(2) = D21 + D22 + D23;

				N.row(0) = M.row(0);
				N.row(1) = M.row(1);
				N.row(2) = M.row(2);
				D << N(0, 0) * N(2, 14) + N(0, 1) * N(2, 13) + N(0, 2) * N(2, 12) - N(0, 12) * N(2, 2) - N(0, 13) * N(2, 1) - N(0, 14) * N(2, 0) - N(0, 2) * N(2, 14) + N(0, 14) * N(2, 2), N(0, 1) * N(2, 14) + N(0, 2) * N(2, 13) - N(0, 13) * N(2, 2) - N(0, 14) * N(2, 1), N(0, 2) * N(2, 14) - N(0, 14) * N(2, 2), N(0, 0) * N(2, 17) + N(0, 1) * N(2, 16) + N(0, 2) * N(2, 15) + N(0, 3) * N(2, 14) + N(0, 4) * N(2, 13) + N(0, 5) * N(2, 12) - N(0, 12) * N(2, 5) - N(0, 13) * N(2, 4) - N(0, 14) * N(2, 3) - N(0, 15) * N(2, 2) - N(0, 16) * N(2, 1) - N(0, 17) * N(2, 0) - N(0, 2) * N(2, 17) - N(0, 5) * N(2, 14) + N(0, 14) * N(2, 5) + N(0, 17) * N(2, 2), N(0, 1) * N(2, 17) + N(0, 2) * N(2, 16) + N(0, 4) * N(2, 14) + N(0, 5) * N(2, 13) - N(0, 13) * N(2, 5) - N(0, 14) * N(2, 4) - N(0, 16) * N(2, 2) - N(0, 17) * N(2, 1), N(0, 2) * N(2, 17) + N(0, 5) * N(2, 14) - N(0, 14) * N(2, 5) - N(0, 17) * N(2, 2), N(0, 3) * N(2, 17) + N(0, 4) * N(2, 16) + N(0, 5) * N(2, 15) - N(0, 15) * N(2, 5) - N(0, 16) * N(2, 4) - N(0, 17) * N(2, 3) - N(0, 5) * N(2, 17) + N(0, 17) * N(2, 5), N(0, 4) * N(2, 17) + N(0, 5) * N(2, 16) - N(0, 16) * N(2, 5) - N(0, 17) * N(2, 4), N(0, 5) * N(2, 17) - N(0, 17) * N(2, 5),
					N(0, 12) * N(1, 2) - N(0, 1) * N(1, 13) - N(0, 2) * N(1, 12) - N(0, 0) * N(1, 14) + N(0, 13) * N(1, 1) + N(0, 14) * N(1, 0) + N(0, 2) * N(1, 14) - N(0, 14) * N(1, 2), N(0, 13) * N(1, 2) - N(0, 2) * N(1, 13) - N(0, 1) * N(1, 14) + N(0, 14) * N(1, 1), N(0, 14) * N(1, 2) - N(0, 2) * N(1, 14), N(0, 12) * N(1, 5) - N(0, 1) * N(1, 16) - N(0, 2) * N(1, 15) - N(0, 3) * N(1, 14) - N(0, 4) * N(1, 13) - N(0, 5) * N(1, 12) - N(0, 0) * N(1, 17) + N(0, 13) * N(1, 4) + N(0, 14) * N(1, 3) + N(0, 15) * N(1, 2) + N(0, 16) * N(1, 1) + N(0, 17) * N(1, 0) + N(0, 2) * N(1, 17) + N(0, 5) * N(1, 14) - N(0, 14) * N(1, 5) - N(0, 17) * N(1, 2), N(0, 13) * N(1, 5) - N(0, 2) * N(1, 16) - N(0, 4) * N(1, 14) - N(0, 5) * N(1, 13) - N(0, 1) * N(1, 17) + N(0, 14) * N(1, 4) + N(0, 16) * N(1, 2) + N(0, 17) * N(1, 1), N(0, 14) * N(1, 5) - N(0, 5) * N(1, 14) - N(0, 2) * N(1, 17) + N(0, 17) * N(1, 2), N(0, 15) * N(1, 5) - N(0, 4) * N(1, 16) - N(0, 5) * N(1, 15) - N(0, 3) * N(1, 17) + N(0, 16) * N(1, 4) + N(0, 17) * N(1, 3) + N(0, 5) * N(1, 17) - N(0, 17) * N(1, 5), N(0, 16) * N(1, 5) - N(0, 5) * N(1, 16) - N(0, 4) * N(1, 17) + N(0, 17) * N(1, 4), N(0, 17) * N(1, 5) - N(0, 5) * N(1, 17),
					N(1, 12) * N(2, 2) - N(1, 1) * N(2, 13) - N(1, 2) * N(2, 12) - N(1, 0) * N(2, 14) + N(1, 13) * N(2, 1) + N(1, 14) * N(2, 0) + N(1, 2) * N(2, 14) - N(1, 14) * N(2, 2), N(1, 13) * N(2, 2) - N(1, 2) * N(2, 13) - N(1, 1) * N(2, 14) + N(1, 14) * N(2, 1), N(1, 14) * N(2, 2) - N(1, 2) * N(2, 14), N(1, 12) * N(2, 5) - N(1, 1) * N(2, 16) - N(1, 2) * N(2, 15) - N(1, 3) * N(2, 14) - N(1, 4) * N(2, 13) - N(1, 5) * N(2, 12) - N(1, 0) * N(2, 17) + N(1, 13) * N(2, 4) + N(1, 14) * N(2, 3) + N(1, 15) * N(2, 2) + N(1, 16) * N(2, 1) + N(1, 17) * N(2, 0) + N(1, 2) * N(2, 17) + N(1, 5) * N(2, 14) - N(1, 14) * N(2, 5) - N(1, 17) * N(2, 2), N(1, 13) * N(2, 5) - N(1, 2) * N(2, 16) - N(1, 4) * N(2, 14) - N(1, 5) * N(2, 13) - N(1, 1) * N(2, 17) + N(1, 14) * N(2, 4) + N(1, 16) * N(2, 2) + N(1, 17) * N(2, 1), N(1, 14) * N(2, 5) - N(1, 5) * N(2, 14) - N(1, 2) * N(2, 17) + N(1, 17) * N(2, 2), N(1, 15) * N(2, 5) - N(1, 4) * N(2, 16) - N(1, 5) * N(2, 15) - N(1, 3) * N(2, 17) + N(1, 16) * N(2, 4) + N(1, 17) * N(2, 3) + N(1, 5) * N(2, 17) - N(1, 17) * N(2, 5), N(1, 16) * N(2, 5) - N(1, 5) * N(2, 16) - N(1, 4) * N(2, 17) + N(1, 17) * N(2, 4), N(1, 17) * N(2, 5) - N(1, 5) * N(2, 17);

				D21 << N(1, 6) * D(0, 0), N(1, 6) * D(0, 1) + N(1, 7) * D(0, 0), N(1, 6) * D(0, 2) + N(1, 7) * D(0, 1) + N(1, 8) * D(0, 0), N(1, 7) * D(0, 2) + N(1, 8) * D(0, 1), N(1, 8) * D(0, 2), N(1, 6) * D(0, 3) + N(1, 9) * D(0, 0), N(1, 6) * D(0, 4) + N(1, 7) * D(0, 3) + N(1, 9) * D(0, 1) + N(1, 10) * D(0, 0), N(1, 6) * D(0, 5) + N(1, 7) * D(0, 4) + N(1, 8) * D(0, 3) + N(1, 9) * D(0, 2) + N(1, 10) * D(0, 1) + N(1, 11) * D(0, 0), N(1, 7) * D(0, 5) + N(1, 8) * D(0, 4) + N(1, 10) * D(0, 2) + N(1, 11) * D(0, 1), N(1, 8) * D(0, 5) + N(1, 11) * D(0, 2), N(1, 6) * D(0, 6) + N(1, 9) * D(0, 3), N(1, 6) * D(0, 7) + N(1, 7) * D(0, 6) + N(1, 9) * D(0, 4) + N(1, 10) * D(0, 3), N(1, 6) * D(0, 8) + N(1, 7) * D(0, 7) + N(1, 8) * D(0, 6) + N(1, 9) * D(0, 5) + N(1, 10) * D(0, 4) + N(1, 11) * D(0, 3), N(1, 7) * D(0, 8) + N(1, 8) * D(0, 7) + N(1, 10) * D(0, 5) + N(1, 11) * D(0, 4), N(1, 8) * D(0, 8) + N(1, 11) * D(0, 5);

				D22 << N(2, 6) * D(1, 0), N(2, 6) * D(1, 1) + N(2, 7) * D(1, 0), N(2, 6) * D(1, 2) + N(2, 7) * D(1, 1) + N(2, 8) * D(1, 0), N(2, 7) * D(1, 2) + N(2, 8) * D(1, 1), N(2, 8) * D(1, 2), N(2, 6) * D(1, 3) + N(2, 9) * D(1, 0), N(2, 6) * D(1, 4) + N(2, 7) * D(1, 3) + N(2, 9) * D(1, 1) + N(2, 10) * D(1, 0), N(2, 6) * D(1, 5) + N(2, 7) * D(1, 4) + N(2, 8) * D(1, 3) + N(2, 9) * D(1, 2) + N(2, 10) * D(1, 1) + N(2, 11) * D(1, 0), N(2, 7) * D(1, 5) + N(2, 8) * D(1, 4) + N(2, 10) * D(1, 2) + N(2, 11) * D(1, 1), N(2, 8) * D(1, 5) + N(2, 11) * D(1, 2), N(2, 6) * D(1, 6) + N(2, 9) * D(1, 3), N(2, 6) * D(1, 7) + N(2, 7) * D(1, 6) + N(2, 9) * D(1, 4) + N(2, 10) * D(1, 3), N(2, 6) * D(1, 8) + N(2, 7) * D(1, 7) + N(2, 8) * D(1, 6) + N(2, 9) * D(1, 5) + N(2, 10) * D(1, 4) + N(2, 11) * D(1, 3), N(2, 7) * D(1, 8) + N(2, 8) * D(1, 7) + N(2, 10) * D(1, 5) + N(2, 11) * D(1, 4), N(2, 8) * D(1, 8) + N(2, 11) * D(1, 5);

				D23 << N(0, 6) * D(2, 0), N(0, 6) * D(2, 1) + N(0, 7) * D(2, 0), N(0, 6) * D(2, 2) + N(0, 7) * D(2, 1) + N(0, 8) * D(2, 0), N(0, 7) * D(2, 2) + N(0, 8) * D(2, 1), N(0, 8) * D(2, 2), N(0, 6) * D(2, 3) + N(0, 9) * D(2, 0), N(0, 6) * D(2, 4) + N(0, 7) * D(2, 3) + N(0, 9) * D(2, 1) + N(0, 10) * D(2, 0), N(0, 6) * D(2, 5) + N(0, 7) * D(2, 4) + N(0, 8) * D(2, 3) + N(0, 9) * D(2, 2) + N(0, 10) * D(2, 1) + N(0, 11) * D(2, 0), N(0, 7) * D(2, 5) + N(0, 8) * D(2, 4) + N(0, 10) * D(2, 2) + N(0, 11) * D(2, 1), N(0, 8) * D(2, 5) + N(0, 11) * D(2, 2), N(0, 6) * D(2, 6) + N(0, 9) * D(2, 3), N(0, 6) * D(2, 7) + N(0, 7) * D(2, 6) + N(0, 9) * D(2, 4) + N(0, 10) * D(2, 3), N(0, 6) * D(2, 8) + N(0, 7) * D(2, 7) + N(0, 8) * D(2, 6) + N(0, 9) * D(2, 5) + N(0, 10) * D(2, 4) + N(0, 11) * D(2, 3), N(0, 7) * D(2, 8) + N(0, 8) * D(2, 7) + N(0, 10) * D(2, 5) + N(0, 11) * D(2, 4), N(0, 8) * D(2, 8) + N(0, 11) * D(2, 5);

				D2.row(3) = D21 + D22 + D23;

				Eigen::Matrix<double, 1, 60> data2;
				data2 << D2.row(0), D2.row(1), D2.row(2), D2.row(3);

				Eigen::MatrixXcd sols = solver_4pt_onefocal(data2);

				for (size_t k = 0; k < 10; ++k)
				{
					if (sols(0, k).imag() > std::numeric_limits<double>::epsilon() ||
						sols(1, k).imag() > std::numeric_limits<double>::epsilon())
						continue;

					double r = sols(0, k).real();
					double rr = r * r;

					Eigen::Matrix3d Ry;
					Ry << 1.0 - rr, 0.0, 2.0 * r,
						0.0, 1.0 + rr, 0.0,
						-2.0 * r, 0.0, 1.0 - rr;
					Ry = Ry / (1.0 + rr);

					double focal = sols(1, k).real();
					Eigen::Matrix3d Rot = gravity_destination.transpose() * Ry * gravity_source;
					Eigen::Matrix<double, 3, 1> p22;
					p22 << u2,
						v2,
						focal;

					Eigen::Matrix<double, 3, 1> p44;
					p44 << u4,
						v4,
						focal;

					Eigen::Matrix<double, 3, 1> p66;
					p66 << u6,
						v6,
						focal;

					Eigen::Matrix<double, 3, 1> p11;
					p11 << u1,
						v1,
						1;

					Eigen::Matrix<double, 3, 1> p33;
					p33 << u3,
						v3,
						1;

					Eigen::Matrix<double, 3, 1> p55;
					p55 << u5,
						v5,
						1;

					Eigen::Matrix<double, 3, 3> A;
					Eigen::Matrix<double, 3, 1> A11;
					A11 = Rot * p11;
					p22.cross(A11);
					A.col(0) = p22.cross(A11);
					A11 = Rot * p33;
					A.col(1) = p44.cross(A11);
					A11 = Rot * p55;
					A.col(2) = p66.cross(A11);

					A = A.transpose().eval();

					Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
					const Eigen::VectorXd &ker = svd.matrixV().rightCols(1);

					Eigen::Matrix3d tx;
					tx << 0, -ker(2, 0), ker(1, 0),
						ker(2, 0), 0, -ker(0, 0),
						-ker(1, 0), ker(0, 0), 0;

					Eigen::Matrix<double, 3, 3> E;
					E = tx * Rot;

					Model model;
					model.descriptor = Eigen::MatrixXd(3, 4);
					model.descriptor.block<3, 3>(0, 0) = E; // full essential matrix
					model.descriptor(2, 3) = focal;			// focal length

					models_.push_back(model);
				}
				return models_.size() > 0;
			}
		}
	}
}