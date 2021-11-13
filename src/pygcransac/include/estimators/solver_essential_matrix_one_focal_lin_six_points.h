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
			class EssentialOnefocalLIN6PC : public SolverEngine
			{
			public:
				EssentialOnefocalLIN6PC()
				{
				}

				~EssentialOnefocalLIN6PC()
				{
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 6;
				}

				static const char *getName()
				{
					return "E1f-LIN6PT";
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
					return 1;
				}

				static constexpr bool needsGravity()
				{
					return true;
				}

				static constexpr bool acceptsPriorModel() 
				{
					return false;
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

			OLGA_INLINE bool EssentialOnefocalLIN6PC::estimateModel(
				const cv::Mat &data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				using namespace Eigen;
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
					&v8 = data_.at<double>(sample_[3], 3),
					&u9 = data_.at<double>(sample_[4], 0),
					&v9 = data_.at<double>(sample_[4], 1),
					&u10 = data_.at<double>(sample_[4], 2),
					&v10 = data_.at<double>(sample_[4], 3),
					&u11 = data_.at<double>(sample_[5], 0),
					&v11 = data_.at<double>(sample_[5], 1),
					&u12 = data_.at<double>(sample_[5], 2),
					&v12 = data_.at<double>(sample_[5], 3);

				double b1 = gravity_destination(0, 1) / gravity_destination(0, 0);
				double b2 = gravity_destination(1, 0) / gravity_destination(0, 0);
				double b3 = gravity_destination(1, 1) / gravity_destination(0, 0);
				double b4 = gravity_destination(1, 2) / gravity_destination(0, 0);
				double b5 = gravity_destination(2, 0) / gravity_destination(0, 0);
				double b6 = gravity_destination(2, 1) / gravity_destination(0, 0);
				double b7 = gravity_destination(2, 2) / gravity_destination(0, 0);

				Eigen::Matrix<double, 3, 6> p1;
				p1 << u1, u3, u5, u7, u9, u11,
					v1, v3, v5, v7, v9, v11,
					1, 1, 1, 1, 1, 1;
				Eigen::Matrix<double, 3, 6> q1;
				q1 = gravity_source * p1;

				q1.row(0) = q1.row(0).array() / q1.row(2).array();
				q1.row(1) = q1.row(1).array() / q1.row(2).array();

				Eigen::Matrix<double, 2, 6> p2;
				p2 << u2, u4, u6, u8, u10, u12,
					v2, v4, v6, v8, v10, v12;

				Eigen::Matrix<double, 3, 2> align2;
				align2 << 1, b1,
					b2, b3,
					b5, b6;

				Eigen::Matrix<double, 3, 6> q2;
				q2 = align2 * p2;

				Eigen::MatrixXd M = MatrixXd::Zero(6, 18);

				for (size_t i = 0; i < 6; ++i)
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
				};

				Eigen::MatrixXd D2 = MatrixXd::Zero(20, 15);
				int mm = 0;

				for (size_t k = 0; k < 4; ++k)
				{
					for (size_t i = k + 1; i < 5; ++i)
					{
						for (size_t j = i + 1; j < 6; ++j)
						{

							Eigen::MatrixXd N = MatrixXd::Zero(3, 18);
							N.row(0) = M.row(k);
							N.row(1) = M.row(i);
							N.row(2) = M.row(j);
							Eigen::MatrixXd D = MatrixXd::Zero(3, 9);
							D << N(0, 0) * N(2, 14) + N(0, 1) * N(2, 13) + N(0, 2) * N(2, 12) - N(0, 12) * N(2, 2) - N(0, 13) * N(2, 1) - N(0, 14) * N(2, 0) - N(0, 2) * N(2, 14) + N(0, 14) * N(2, 2), N(0, 1) * N(2, 14) + N(0, 2) * N(2, 13) - N(0, 13) * N(2, 2) - N(0, 14) * N(2, 1), N(0, 2) * N(2, 14) - N(0, 14) * N(2, 2), N(0, 0) * N(2, 17) + N(0, 1) * N(2, 16) + N(0, 2) * N(2, 15) + N(0, 3) * N(2, 14) + N(0, 4) * N(2, 13) + N(0, 5) * N(2, 12) - N(0, 12) * N(2, 5) - N(0, 13) * N(2, 4) - N(0, 14) * N(2, 3) - N(0, 15) * N(2, 2) - N(0, 16) * N(2, 1) - N(0, 17) * N(2, 0) - N(0, 2) * N(2, 17) - N(0, 5) * N(2, 14) + N(0, 14) * N(2, 5) + N(0, 17) * N(2, 2), N(0, 1) * N(2, 17) + N(0, 2) * N(2, 16) + N(0, 4) * N(2, 14) + N(0, 5) * N(2, 13) - N(0, 13) * N(2, 5) - N(0, 14) * N(2, 4) - N(0, 16) * N(2, 2) - N(0, 17) * N(2, 1), N(0, 2) * N(2, 17) + N(0, 5) * N(2, 14) - N(0, 14) * N(2, 5) - N(0, 17) * N(2, 2), N(0, 3) * N(2, 17) + N(0, 4) * N(2, 16) + N(0, 5) * N(2, 15) - N(0, 15) * N(2, 5) - N(0, 16) * N(2, 4) - N(0, 17) * N(2, 3) - N(0, 5) * N(2, 17) + N(0, 17) * N(2, 5), N(0, 4) * N(2, 17) + N(0, 5) * N(2, 16) - N(0, 16) * N(2, 5) - N(0, 17) * N(2, 4), N(0, 5) * N(2, 17) - N(0, 17) * N(2, 5),
								N(0, 12) * N(1, 2) - N(0, 1) * N(1, 13) - N(0, 2) * N(1, 12) - N(0, 0) * N(1, 14) + N(0, 13) * N(1, 1) + N(0, 14) * N(1, 0) + N(0, 2) * N(1, 14) - N(0, 14) * N(1, 2), N(0, 13) * N(1, 2) - N(0, 2) * N(1, 13) - N(0, 1) * N(1, 14) + N(0, 14) * N(1, 1), N(0, 14) * N(1, 2) - N(0, 2) * N(1, 14), N(0, 12) * N(1, 5) - N(0, 1) * N(1, 16) - N(0, 2) * N(1, 15) - N(0, 3) * N(1, 14) - N(0, 4) * N(1, 13) - N(0, 5) * N(1, 12) - N(0, 0) * N(1, 17) + N(0, 13) * N(1, 4) + N(0, 14) * N(1, 3) + N(0, 15) * N(1, 2) + N(0, 16) * N(1, 1) + N(0, 17) * N(1, 0) + N(0, 2) * N(1, 17) + N(0, 5) * N(1, 14) - N(0, 14) * N(1, 5) - N(0, 17) * N(1, 2), N(0, 13) * N(1, 5) - N(0, 2) * N(1, 16) - N(0, 4) * N(1, 14) - N(0, 5) * N(1, 13) - N(0, 1) * N(1, 17) + N(0, 14) * N(1, 4) + N(0, 16) * N(1, 2) + N(0, 17) * N(1, 1), N(0, 14) * N(1, 5) - N(0, 5) * N(1, 14) - N(0, 2) * N(1, 17) + N(0, 17) * N(1, 2), N(0, 15) * N(1, 5) - N(0, 4) * N(1, 16) - N(0, 5) * N(1, 15) - N(0, 3) * N(1, 17) + N(0, 16) * N(1, 4) + N(0, 17) * N(1, 3) + N(0, 5) * N(1, 17) - N(0, 17) * N(1, 5), N(0, 16) * N(1, 5) - N(0, 5) * N(1, 16) - N(0, 4) * N(1, 17) + N(0, 17) * N(1, 4), N(0, 17) * N(1, 5) - N(0, 5) * N(1, 17),
								N(1, 12) * N(2, 2) - N(1, 1) * N(2, 13) - N(1, 2) * N(2, 12) - N(1, 0) * N(2, 14) + N(1, 13) * N(2, 1) + N(1, 14) * N(2, 0) + N(1, 2) * N(2, 14) - N(1, 14) * N(2, 2), N(1, 13) * N(2, 2) - N(1, 2) * N(2, 13) - N(1, 1) * N(2, 14) + N(1, 14) * N(2, 1), N(1, 14) * N(2, 2) - N(1, 2) * N(2, 14), N(1, 12) * N(2, 5) - N(1, 1) * N(2, 16) - N(1, 2) * N(2, 15) - N(1, 3) * N(2, 14) - N(1, 4) * N(2, 13) - N(1, 5) * N(2, 12) - N(1, 0) * N(2, 17) + N(1, 13) * N(2, 4) + N(1, 14) * N(2, 3) + N(1, 15) * N(2, 2) + N(1, 16) * N(2, 1) + N(1, 17) * N(2, 0) + N(1, 2) * N(2, 17) + N(1, 5) * N(2, 14) - N(1, 14) * N(2, 5) - N(1, 17) * N(2, 2), N(1, 13) * N(2, 5) - N(1, 2) * N(2, 16) - N(1, 4) * N(2, 14) - N(1, 5) * N(2, 13) - N(1, 1) * N(2, 17) + N(1, 14) * N(2, 4) + N(1, 16) * N(2, 2) + N(1, 17) * N(2, 1), N(1, 14) * N(2, 5) - N(1, 5) * N(2, 14) - N(1, 2) * N(2, 17) + N(1, 17) * N(2, 2), N(1, 15) * N(2, 5) - N(1, 4) * N(2, 16) - N(1, 5) * N(2, 15) - N(1, 3) * N(2, 17) + N(1, 16) * N(2, 4) + N(1, 17) * N(2, 3) + N(1, 5) * N(2, 17) - N(1, 17) * N(2, 5), N(1, 16) * N(2, 5) - N(1, 5) * N(2, 16) - N(1, 4) * N(2, 17) + N(1, 17) * N(2, 4), N(1, 17) * N(2, 5) - N(1, 5) * N(2, 17);
							Eigen::MatrixXd D21 = MatrixXd::Zero(1, 15);
							D21 << N(1, 6) * D(0, 0), N(1, 6) * D(0, 1) + N(1, 7) * D(0, 0), N(1, 6) * D(0, 2) + N(1, 7) * D(0, 1) + N(1, 8) * D(0, 0), N(1, 7) * D(0, 2) + N(1, 8) * D(0, 1), N(1, 8) * D(0, 2), N(1, 6) * D(0, 3) + N(1, 9) * D(0, 0), N(1, 6) * D(0, 4) + N(1, 7) * D(0, 3) + N(1, 9) * D(0, 1) + N(1, 10) * D(0, 0), N(1, 6) * D(0, 5) + N(1, 7) * D(0, 4) + N(1, 8) * D(0, 3) + N(1, 9) * D(0, 2) + N(1, 10) * D(0, 1) + N(1, 11) * D(0, 0), N(1, 7) * D(0, 5) + N(1, 8) * D(0, 4) + N(1, 10) * D(0, 2) + N(1, 11) * D(0, 1), N(1, 8) * D(0, 5) + N(1, 11) * D(0, 2), N(1, 6) * D(0, 6) + N(1, 9) * D(0, 3), N(1, 6) * D(0, 7) + N(1, 7) * D(0, 6) + N(1, 9) * D(0, 4) + N(1, 10) * D(0, 3), N(1, 6) * D(0, 8) + N(1, 7) * D(0, 7) + N(1, 8) * D(0, 6) + N(1, 9) * D(0, 5) + N(1, 10) * D(0, 4) + N(1, 11) * D(0, 3), N(1, 7) * D(0, 8) + N(1, 8) * D(0, 7) + N(1, 10) * D(0, 5) + N(1, 11) * D(0, 4), N(1, 8) * D(0, 8) + N(1, 11) * D(0, 5);

							Eigen::MatrixXd D22 = MatrixXd::Zero(1, 15);
							D22 << N(2, 6) * D(1, 0), N(2, 6) * D(1, 1) + N(2, 7) * D(1, 0), N(2, 6) * D(1, 2) + N(2, 7) * D(1, 1) + N(2, 8) * D(1, 0), N(2, 7) * D(1, 2) + N(2, 8) * D(1, 1), N(2, 8) * D(1, 2), N(2, 6) * D(1, 3) + N(2, 9) * D(1, 0), N(2, 6) * D(1, 4) + N(2, 7) * D(1, 3) + N(2, 9) * D(1, 1) + N(2, 10) * D(1, 0), N(2, 6) * D(1, 5) + N(2, 7) * D(1, 4) + N(2, 8) * D(1, 3) + N(2, 9) * D(1, 2) + N(2, 10) * D(1, 1) + N(2, 11) * D(1, 0), N(2, 7) * D(1, 5) + N(2, 8) * D(1, 4) + N(2, 10) * D(1, 2) + N(2, 11) * D(1, 1), N(2, 8) * D(1, 5) + N(2, 11) * D(1, 2), N(2, 6) * D(1, 6) + N(2, 9) * D(1, 3), N(2, 6) * D(1, 7) + N(2, 7) * D(1, 6) + N(2, 9) * D(1, 4) + N(2, 10) * D(1, 3), N(2, 6) * D(1, 8) + N(2, 7) * D(1, 7) + N(2, 8) * D(1, 6) + N(2, 9) * D(1, 5) + N(2, 10) * D(1, 4) + N(2, 11) * D(1, 3), N(2, 7) * D(1, 8) + N(2, 8) * D(1, 7) + N(2, 10) * D(1, 5) + N(2, 11) * D(1, 4), N(2, 8) * D(1, 8) + N(2, 11) * D(1, 5);

							Eigen::MatrixXd D23 = MatrixXd::Zero(1, 15);
							D23 << N(0, 6) * D(2, 0), N(0, 6) * D(2, 1) + N(0, 7) * D(2, 0), N(0, 6) * D(2, 2) + N(0, 7) * D(2, 1) + N(0, 8) * D(2, 0), N(0, 7) * D(2, 2) + N(0, 8) * D(2, 1), N(0, 8) * D(2, 2), N(0, 6) * D(2, 3) + N(0, 9) * D(2, 0), N(0, 6) * D(2, 4) + N(0, 7) * D(2, 3) + N(0, 9) * D(2, 1) + N(0, 10) * D(2, 0), N(0, 6) * D(2, 5) + N(0, 7) * D(2, 4) + N(0, 8) * D(2, 3) + N(0, 9) * D(2, 2) + N(0, 10) * D(2, 1) + N(0, 11) * D(2, 0), N(0, 7) * D(2, 5) + N(0, 8) * D(2, 4) + N(0, 10) * D(2, 2) + N(0, 11) * D(2, 1), N(0, 8) * D(2, 5) + N(0, 11) * D(2, 2), N(0, 6) * D(2, 6) + N(0, 9) * D(2, 3), N(0, 6) * D(2, 7) + N(0, 7) * D(2, 6) + N(0, 9) * D(2, 4) + N(0, 10) * D(2, 3), N(0, 6) * D(2, 8) + N(0, 7) * D(2, 7) + N(0, 8) * D(2, 6) + N(0, 9) * D(2, 5) + N(0, 10) * D(2, 4) + N(0, 11) * D(2, 3), N(0, 7) * D(2, 8) + N(0, 8) * D(2, 7) + N(0, 10) * D(2, 5) + N(0, 11) * D(2, 4), N(0, 8) * D(2, 8) + N(0, 11) * D(2, 5);

							D2.row(mm) = D21 + D22 + D23;
							mm = mm + 1;
						}
					}
				}

				Eigen::JacobiSVD<Eigen::MatrixXd> svd2(D2, Eigen::ComputeFullV);

				const Eigen::Matrix<double, 15, 1> &V15 = svd2.matrixV().col(14);

				double r = V15(1, 0) / V15(0, 0);
				double focal = V15(5, 0) / V15(0, 0);
				double rr = r * r;

				Eigen::Matrix3d Ry;
				Ry << 1.0 - rr, 0.0, 2.0 * r,
					0.0, 1.0 + rr, 0.0,
					-2.0 * r, 0.0, 1.0 - rr;
				Ry = Ry / (1.0 + rr);

				Eigen::MatrixXd Rot = gravity_destination.transpose() * Ry * gravity_source;
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
				return true;
			}
		}
	}
}