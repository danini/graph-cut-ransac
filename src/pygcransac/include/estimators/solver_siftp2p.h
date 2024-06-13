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
#include "sturm8.h"
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class SIFTP2PQuerySolver : public SolverEngine
			{
			public:
				SIFTP2PQuerySolver()
				{
				}

				~SIFTP2PQuerySolver()
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
					return 8;
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
				void solver(
					const double s1[], // sin of angle in first view
					const double c1[], // cos of angle in first view
					const double s2[], // sin of angle in second view
					const double c2[], // cos of angle in second view
					const Eigen::Vector2d x[],
					const double d[],
					const Eigen::Vector3d n[],
					const Eigen::Vector2d y[],
					const Eigen::Matrix3d R_ref[],
					const Eigen::Vector3d t_ref[],
					std::vector<Eigen::Matrix3d> &Rsolns, 
					std::vector<Eigen::Vector3d> &tsolns
					/*// -- first correspondence --
					const double s11, // sin of angle in first view
					const double c11, // cos of angle in first view
					const double s21, // sin of angle in second view
					const double c21, // cos of angle in second view
					const Eigen::Vector2d &u1,
					const double d1,
					const Eigen::Vector3d &n1,
					const Eigen::Vector2d &v1,
					// -- second correspondence --
					const double s12, // sin of angle in first view
					const double c12, // cos of angle in first view
					const double s22, // sin of angle in second view
					const double c22, // cos of angle in second view
					const Eigen::Vector2d &u2,
					const double d2,
					const Eigen::Vector3d &n2,
					const Eigen::Vector2d &v2,
					std::vector<Eigen::Matrix3d> &Rsolns, std::vector<Eigen::Vector3d> &tsolns*/
				) const;

				int re3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions, bool try_random_var_change = true) const;
				void refine_3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions, int n_sols) const ;

				void cayley(const Eigen::Vector3d &r, Eigen::Matrix3d &R, double &s ) const;
				Eigen::MatrixXd build_constraints(
					const Eigen::Matrix3d &R_ref,
					const Eigen::Vector3d &t_ref,
					const double s1, //sin of angle in first view
					const double c1, //cos of angle in first view
					const double s2, //sin of angle in second view
					const double c2, //cos of angle in second view
					const Eigen::Vector2d &u,
					const double d,
					const Eigen::Vector3d &n,
					const Eigen::Vector2d &v ) const;
			};
			
			OLGA_INLINE bool SIFTP2PQuerySolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				//const double * data_ptr_1 = reinterpret_cast<double *>(data_.row(sample_[0]).data);
				//const double * data_ptr_2 = reinterpret_cast<double *>(data_.row(sample_[1]).data);
				const size_t columns = data_.cols;
							
				double s1[2];
				double c1[2];
				double s2[2];
				double c2[2];
				Eigen::Vector2d x[2];
				double d[2];
				Eigen::Vector3d n[2];
				Eigen::Vector2d y[2];
  			    Eigen::Matrix3d R_ref[2];
       		 	Eigen::Vector3d t_ref[2];
				
				
				for ( int k = 0; k < 2; k++ )
				{
					const double * data_ptr = reinterpret_cast<double *>(data_.row(sample_[k]).data);

					s1[k] = sin(data_ptr[8]);
					c1[k] = cos(data_ptr[8]);
					x[k] = Eigen::Vector2d(data_ptr[0], data_ptr[1]);
					d[k] = data_ptr[4];
					n[k] = Eigen::Vector3d(data_ptr[5], data_ptr[6], data_ptr[7]);

					s2[k] = sin(data_ptr[10]);
					c2[k] = cos(data_ptr[10]);
					y[k] = Eigen::Vector2d(data_ptr[2] / d[k], data_ptr[3] / d[k]);

					// The next 3*3 elements are the rotation matrix of the reference frame in a row-major order
					R_ref[k] << data_ptr[12], data_ptr[13], data_ptr[14],
						data_ptr[15], data_ptr[16], data_ptr[17],
						data_ptr[18], data_ptr[19], data_ptr[20];

					// The next 3 elements are the translation vector of the reference frame
					t_ref[k] << data_ptr[21], data_ptr[22], data_ptr[23];
				}

				std::vector<Eigen::Matrix3d> Rsolns; 
				std::vector<Eigen::Vector3d> tsolns;
				
				solver(
					s1, // sin of angle in first view
					c1, // cos of angle in first view
					s2, // sin of angle in second view
					c2, // cos of angle in second view
					x,
					d,
					n,
					y,
					R_ref,
					t_ref,
					Rsolns, 
					tsolns);
				
				for ( int i = 0; i < Rsolns.size(); i++ )
				{
					Model model;
					model.descriptor.resize(3, 4);

					model.descriptor << Rsolns[i], tsolns[i];
					models_.emplace_back(model);
				}
				return models_.size();
			}
		
			Eigen::MatrixXd SIFTP2PQuerySolver::build_constraints(
				const Eigen::Matrix3d &R_ref,
				const Eigen::Vector3d &t_ref,
				const double s1, //sin of angle in first view
				const double c1, //cos of angle in first view
				const double s2, //sin of angle in second view
				const double c2, //cos of angle in second view
				const Eigen::Vector2d &x,
				const double d,
				const Eigen::Vector3d &n_ref,
				const Eigen::Vector2d &y ) const
			{
				Eigen::MatrixXd M(3,13);
				M << -1, 0, y(0), R_ref(0,0)*t_ref(0) - R_ref(2,0)*d + R_ref(1,0)*t_ref(1) + R_ref(2,0)*t_ref(2) - R_ref(0,0)*d*x(0) - R_ref(1,0)*d*x(1) - R_ref(2,2)*d*y(0) + R_ref(0,2)*t_ref(0)*y(0) + R_ref(1,2)*t_ref(1)*y(0) + R_ref(2,2)*t_ref(2)*y(0) - R_ref(0,2)*d*x(0)*y(0) - R_ref(1,2)*d*x(1)*y(0), 2*R_ref(0,1)*t_ref(0) - 2*R_ref(2,1)*d + 2*R_ref(1,1)*t_ref(1) + 2*R_ref(2,1)*t_ref(2) - 2*R_ref(0,1)*d*x(0) - 2*R_ref(1,1)*d*x(1), 2*R_ref(0,2)*t_ref(0) - 2*R_ref(2,2)*d + 2*R_ref(1,2)*t_ref(1) + 2*R_ref(2,2)*t_ref(2) - 2*R_ref(0,2)*d*x(0) - 2*R_ref(1,2)*d*x(1) + 2*R_ref(2,0)*d*y(0) - 2*R_ref(0,0)*t_ref(0)*y(0) - 2*R_ref(1,0)*t_ref(1)*y(0) - 2*R_ref(2,0)*t_ref(2)*y(0) + 2*R_ref(0,0)*d*x(0)*y(0) + 2*R_ref(1,0)*d*x(1)*y(0), R_ref(2,0)*d - R_ref(0,0)*t_ref(0) - R_ref(1,0)*t_ref(1) - R_ref(2,0)*t_ref(2) + R_ref(0,0)*d*x(0) + R_ref(1,0)*d*x(1) - R_ref(2,2)*d*y(0) + R_ref(0,2)*t_ref(0)*y(0) + R_ref(1,2)*t_ref(1)*y(0) + R_ref(2,2)*t_ref(2)*y(0) - R_ref(0,2)*d*x(0)*y(0) - R_ref(1,2)*d*x(1)*y(0), 2*R_ref(2,1)*d*y(0) - 2*R_ref(0,1)*t_ref(0)*y(0) - 2*R_ref(1,1)*t_ref(1)*y(0) - 2*R_ref(2,1)*t_ref(2)*y(0) + 2*R_ref(0,1)*d*x(0)*y(0) + 2*R_ref(1,1)*d*x(1)*y(0), R_ref(2,0)*d - R_ref(0,0)*t_ref(0) - R_ref(1,0)*t_ref(1) - R_ref(2,0)*t_ref(2) + R_ref(0,0)*d*x(0) + R_ref(1,0)*d*x(1) + R_ref(2,2)*d*y(0) - R_ref(0,2)*t_ref(0)*y(0) - R_ref(1,2)*t_ref(1)*y(0) - R_ref(2,2)*t_ref(2)*y(0) + R_ref(0,2)*d*x(0)*y(0) + R_ref(1,2)*d*x(1)*y(0), 2*R_ref(2,1)*d*y(0) - 2*R_ref(0,1)*t_ref(0)*y(0) - 2*R_ref(1,1)*t_ref(1)*y(0) - 2*R_ref(2,1)*t_ref(2)*y(0) + 2*R_ref(0,1)*d*x(0)*y(0) + 2*R_ref(1,1)*d*x(1)*y(0), 2*R_ref(0,2)*t_ref(0) - 2*R_ref(2,2)*d + 2*R_ref(1,2)*t_ref(1) + 2*R_ref(2,2)*t_ref(2) - 2*R_ref(0,2)*d*x(0) - 2*R_ref(1,2)*d*x(1) - 2*R_ref(2,0)*d*y(0) + 2*R_ref(0,0)*t_ref(0)*y(0) + 2*R_ref(1,0)*t_ref(1)*y(0) + 2*R_ref(2,0)*t_ref(2)*y(0) - 2*R_ref(0,0)*d*x(0)*y(0) - 2*R_ref(1,0)*d*x(1)*y(0), 2*R_ref(2,1)*d - 2*R_ref(0,1)*t_ref(0) - 2*R_ref(1,1)*t_ref(1) - 2*R_ref(2,1)*t_ref(2) + 2*R_ref(0,1)*d*x(0) + 2*R_ref(1,1)*d*x(1), R_ref(0,0)*t_ref(0) - R_ref(2,0)*d + R_ref(1,0)*t_ref(1) + R_ref(2,0)*t_ref(2) - R_ref(0,0)*d*x(0) - R_ref(1,0)*d*x(1) + R_ref(2,2)*d*y(0) - R_ref(0,2)*t_ref(0)*y(0) - R_ref(1,2)*t_ref(1)*y(0) - R_ref(2,2)*t_ref(2)*y(0) + R_ref(0,2)*d*x(0)*y(0) + R_ref(1,2)*d*x(1)*y(0),
				0, -1, y(1), R_ref(2,1)*d - R_ref(0,1)*t_ref(0) - R_ref(1,1)*t_ref(1) - R_ref(2,1)*t_ref(2) + R_ref(0,1)*d*x(0) + R_ref(1,1)*d*x(1) - R_ref(2,2)*d*y(1) + R_ref(0,2)*t_ref(0)*y(1) + R_ref(1,2)*t_ref(1)*y(1) + R_ref(2,2)*t_ref(2)*y(1) - R_ref(0,2)*d*x(0)*y(1) - R_ref(1,2)*d*x(1)*y(1), 2*R_ref(0,0)*t_ref(0) - 2*R_ref(2,0)*d + 2*R_ref(1,0)*t_ref(1) + 2*R_ref(2,0)*t_ref(2) - 2*R_ref(0,0)*d*x(0) - 2*R_ref(1,0)*d*x(1), 2*R_ref(2,0)*d*y(1) - 2*R_ref(0,0)*t_ref(0)*y(1) - 2*R_ref(1,0)*t_ref(1)*y(1) - 2*R_ref(2,0)*t_ref(2)*y(1) + 2*R_ref(0,0)*d*x(0)*y(1) + 2*R_ref(1,0)*d*x(1)*y(1), R_ref(0,1)*t_ref(0) - R_ref(2,1)*d + R_ref(1,1)*t_ref(1) + R_ref(2,1)*t_ref(2) - R_ref(0,1)*d*x(0) - R_ref(1,1)*d*x(1) - R_ref(2,2)*d*y(1) + R_ref(0,2)*t_ref(0)*y(1) + R_ref(1,2)*t_ref(1)*y(1) + R_ref(2,2)*t_ref(2)*y(1) - R_ref(0,2)*d*x(0)*y(1) - R_ref(1,2)*d*x(1)*y(1), 2*R_ref(0,2)*t_ref(0) - 2*R_ref(2,2)*d + 2*R_ref(1,2)*t_ref(1) + 2*R_ref(2,2)*t_ref(2) - 2*R_ref(0,2)*d*x(0) - 2*R_ref(1,2)*d*x(1) + 2*R_ref(2,1)*d*y(1) - 2*R_ref(0,1)*t_ref(0)*y(1) - 2*R_ref(1,1)*t_ref(1)*y(1) - 2*R_ref(2,1)*t_ref(2)*y(1) + 2*R_ref(0,1)*d*x(0)*y(1) + 2*R_ref(1,1)*d*x(1)*y(1), R_ref(2,1)*d - R_ref(0,1)*t_ref(0) - R_ref(1,1)*t_ref(1) - R_ref(2,1)*t_ref(2) + R_ref(0,1)*d*x(0) + R_ref(1,1)*d*x(1) + R_ref(2,2)*d*y(1) - R_ref(0,2)*t_ref(0)*y(1) - R_ref(1,2)*t_ref(1)*y(1) - R_ref(2,2)*t_ref(2)*y(1) + R_ref(0,2)*d*x(0)*y(1) + R_ref(1,2)*d*x(1)*y(1), 2*R_ref(2,2)*d - 2*R_ref(0,2)*t_ref(0) - 2*R_ref(1,2)*t_ref(1) - 2*R_ref(2,2)*t_ref(2) + 2*R_ref(0,2)*d*x(0) + 2*R_ref(1,2)*d*x(1) + 2*R_ref(2,1)*d*y(1) - 2*R_ref(0,1)*t_ref(0)*y(1) - 2*R_ref(1,1)*t_ref(1)*y(1) - 2*R_ref(2,1)*t_ref(2)*y(1) + 2*R_ref(0,1)*d*x(0)*y(1) + 2*R_ref(1,1)*d*x(1)*y(1), 2*R_ref(0,0)*t_ref(0)*y(1) - 2*R_ref(2,0)*d*y(1) + 2*R_ref(1,0)*t_ref(1)*y(1) + 2*R_ref(2,0)*t_ref(2)*y(1) - 2*R_ref(0,0)*d*x(0)*y(1) - 2*R_ref(1,0)*d*x(1)*y(1), 2*R_ref(0,0)*t_ref(0) - 2*R_ref(2,0)*d + 2*R_ref(1,0)*t_ref(1) + 2*R_ref(2,0)*t_ref(2) - 2*R_ref(0,0)*d*x(0) - 2*R_ref(1,0)*d*x(1), R_ref(0,1)*t_ref(0) - R_ref(2,1)*d + R_ref(1,1)*t_ref(1) + R_ref(2,1)*t_ref(2) - R_ref(0,1)*d*x(0) - R_ref(1,1)*d*x(1) + R_ref(2,2)*d*y(1) - R_ref(0,2)*t_ref(0)*y(1) - R_ref(1,2)*t_ref(1)*y(1) - R_ref(2,2)*t_ref(2)*y(1) + R_ref(0,2)*d*x(0)*y(1) + R_ref(1,2)*d*x(1)*y(1),
				0, 0, 0, R_ref(0,1)*c1*c2*d*n_ref(2) - R_ref(2,1)*c1*c2*d*n_ref(0) + R_ref(0,0)*c1*d*n_ref(2)*s2 + R_ref(1,1)*c2*d*n_ref(2)*s1 - R_ref(2,0)*c1*d*n_ref(0)*s2 - R_ref(2,1)*c2*d*n_ref(1)*s1 + R_ref(1,0)*d*n_ref(2)*s1*s2 - R_ref(2,0)*d*n_ref(1)*s1*s2 + R_ref(0,1)*c1*c2*d*n_ref(1)*x(1) - R_ref(1,1)*c1*c2*d*n_ref(0)*x(1) - R_ref(0,2)*c1*c2*d*n_ref(2)*y(1) + R_ref(2,2)*c1*c2*d*n_ref(0)*y(1) + R_ref(0,0)*c1*d*n_ref(1)*s2*x(1) - R_ref(0,1)*c2*d*n_ref(1)*s1*x(0) - R_ref(1,0)*c1*d*n_ref(0)*s2*x(1) + R_ref(1,1)*c2*d*n_ref(0)*s1*x(0) + R_ref(0,2)*c1*d*n_ref(2)*s2*y(0) - R_ref(1,2)*c2*d*n_ref(2)*s1*y(1) - R_ref(2,2)*c1*d*n_ref(0)*s2*y(0) + R_ref(2,2)*c2*d*n_ref(1)*s1*y(1) - R_ref(0,0)*d*n_ref(1)*s1*s2*x(0) + R_ref(1,0)*d*n_ref(0)*s1*s2*x(0) + R_ref(1,2)*d*n_ref(2)*s1*s2*y(0) - R_ref(2,2)*d*n_ref(1)*s1*s2*y(0) - R_ref(0,2)*c1*c2*d*n_ref(1)*x(1)*y(1) + R_ref(1,2)*c1*c2*d*n_ref(0)*x(1)*y(1) + R_ref(0,2)*c1*d*n_ref(1)*s2*x(1)*y(0) + R_ref(0,2)*c2*d*n_ref(1)*s1*x(0)*y(1) - R_ref(1,2)*c1*d*n_ref(0)*s2*x(1)*y(0) - R_ref(1,2)*c2*d*n_ref(0)*s1*x(0)*y(1) - R_ref(0,2)*d*n_ref(1)*s1*s2*x(0)*y(0) + R_ref(1,2)*d*n_ref(0)*s1*s2*x(0)*y(0), 2*R_ref(2,0)*c1*c2*d*n_ref(0) - 2*R_ref(0,0)*c1*c2*d*n_ref(2) + 2*R_ref(0,1)*c1*d*n_ref(2)*s2 - 2*R_ref(1,0)*c2*d*n_ref(2)*s1 + 2*R_ref(2,0)*c2*d*n_ref(1)*s1 - 2*R_ref(2,1)*c1*d*n_ref(0)*s2 + 2*R_ref(1,1)*d*n_ref(2)*s1*s2 - 2*R_ref(2,1)*d*n_ref(1)*s1*s2 - 2*R_ref(0,0)*c1*c2*d*n_ref(1)*x(1) + 2*R_ref(1,0)*c1*c2*d*n_ref(0)*x(1) + 2*R_ref(0,0)*c2*d*n_ref(1)*s1*x(0) + 2*R_ref(0,1)*c1*d*n_ref(1)*s2*x(1) - 2*R_ref(1,0)*c2*d*n_ref(0)*s1*x(0) - 2*R_ref(1,1)*c1*d*n_ref(0)*s2*x(1) - 2*R_ref(0,1)*d*n_ref(1)*s1*s2*x(0) + 2*R_ref(1,1)*d*n_ref(0)*s1*s2*x(0), 2*R_ref(0,2)*c1*d*n_ref(2)*s2 - 2*R_ref(2,2)*c1*d*n_ref(0)*s2 + 2*R_ref(1,2)*d*n_ref(2)*s1*s2 - 2*R_ref(2,2)*d*n_ref(1)*s1*s2 + 2*R_ref(0,0)*c1*c2*d*n_ref(2)*y(1) - 2*R_ref(2,0)*c1*c2*d*n_ref(0)*y(1) + 2*R_ref(0,2)*c1*d*n_ref(1)*s2*x(1) - 2*R_ref(1,2)*c1*d*n_ref(0)*s2*x(1) - 2*R_ref(0,0)*c1*d*n_ref(2)*s2*y(0) + 2*R_ref(1,0)*c2*d*n_ref(2)*s1*y(1) + 2*R_ref(2,0)*c1*d*n_ref(0)*s2*y(0) - 2*R_ref(2,0)*c2*d*n_ref(1)*s1*y(1) - 2*R_ref(0,2)*d*n_ref(1)*s1*s2*x(0) + 2*R_ref(1,2)*d*n_ref(0)*s1*s2*x(0) - 2*R_ref(1,0)*d*n_ref(2)*s1*s2*y(0) + 2*R_ref(2,0)*d*n_ref(1)*s1*s2*y(0) + 2*R_ref(0,0)*c1*c2*d*n_ref(1)*x(1)*y(1) - 2*R_ref(1,0)*c1*c2*d*n_ref(0)*x(1)*y(1) - 2*R_ref(0,0)*c1*d*n_ref(1)*s2*x(1)*y(0) - 2*R_ref(0,0)*c2*d*n_ref(1)*s1*x(0)*y(1) + 2*R_ref(1,0)*c1*d*n_ref(0)*s2*x(1)*y(0) + 2*R_ref(1,0)*c2*d*n_ref(0)*s1*x(0)*y(1) + 2*R_ref(0,0)*d*n_ref(1)*s1*s2*x(0)*y(0) - 2*R_ref(1,0)*d*n_ref(0)*s1*s2*x(0)*y(0), R_ref(2,1)*c1*c2*d*n_ref(0) - R_ref(0,1)*c1*c2*d*n_ref(2) - R_ref(0,0)*c1*d*n_ref(2)*s2 - R_ref(1,1)*c2*d*n_ref(2)*s1 + R_ref(2,0)*c1*d*n_ref(0)*s2 + R_ref(2,1)*c2*d*n_ref(1)*s1 - R_ref(1,0)*d*n_ref(2)*s1*s2 + R_ref(2,0)*d*n_ref(1)*s1*s2 - R_ref(0,1)*c1*c2*d*n_ref(1)*x(1) + R_ref(1,1)*c1*c2*d*n_ref(0)*x(1) - R_ref(0,2)*c1*c2*d*n_ref(2)*y(1) + R_ref(2,2)*c1*c2*d*n_ref(0)*y(1) - R_ref(0,0)*c1*d*n_ref(1)*s2*x(1) + R_ref(0,1)*c2*d*n_ref(1)*s1*x(0) + R_ref(1,0)*c1*d*n_ref(0)*s2*x(1) - R_ref(1,1)*c2*d*n_ref(0)*s1*x(0) + R_ref(0,2)*c1*d*n_ref(2)*s2*y(0) - R_ref(1,2)*c2*d*n_ref(2)*s1*y(1) - R_ref(2,2)*c1*d*n_ref(0)*s2*y(0) + R_ref(2,2)*c2*d*n_ref(1)*s1*y(1) + R_ref(0,0)*d*n_ref(1)*s1*s2*x(0) - R_ref(1,0)*d*n_ref(0)*s1*s2*x(0) + R_ref(1,2)*d*n_ref(2)*s1*s2*y(0) - R_ref(2,2)*d*n_ref(1)*s1*s2*y(0) - R_ref(0,2)*c1*c2*d*n_ref(1)*x(1)*y(1) + R_ref(1,2)*c1*c2*d*n_ref(0)*x(1)*y(1) + R_ref(0,2)*c1*d*n_ref(1)*s2*x(1)*y(0) + R_ref(0,2)*c2*d*n_ref(1)*s1*x(0)*y(1) - R_ref(1,2)*c1*d*n_ref(0)*s2*x(1)*y(0) - R_ref(1,2)*c2*d*n_ref(0)*s1*x(0)*y(1) - R_ref(0,2)*d*n_ref(1)*s1*s2*x(0)*y(0) + R_ref(1,2)*d*n_ref(0)*s1*s2*x(0)*y(0), 2*R_ref(2,2)*c1*c2*d*n_ref(0) - 2*R_ref(0,2)*c1*c2*d*n_ref(2) - 2*R_ref(1,2)*c2*d*n_ref(2)*s1 + 2*R_ref(2,2)*c2*d*n_ref(1)*s1 - 2*R_ref(0,2)*c1*c2*d*n_ref(1)*x(1) + 2*R_ref(1,2)*c1*c2*d*n_ref(0)*x(1) + 2*R_ref(0,1)*c1*c2*d*n_ref(2)*y(1) - 2*R_ref(2,1)*c1*c2*d*n_ref(0)*y(1) + 2*R_ref(0,2)*c2*d*n_ref(1)*s1*x(0) - 2*R_ref(1,2)*c2*d*n_ref(0)*s1*x(0) - 2*R_ref(0,1)*c1*d*n_ref(2)*s2*y(0) + 2*R_ref(1,1)*c2*d*n_ref(2)*s1*y(1) + 2*R_ref(2,1)*c1*d*n_ref(0)*s2*y(0) - 2*R_ref(2,1)*c2*d*n_ref(1)*s1*y(1) - 2*R_ref(1,1)*d*n_ref(2)*s1*s2*y(0) + 2*R_ref(2,1)*d*n_ref(1)*s1*s2*y(0) + 2*R_ref(0,1)*c1*c2*d*n_ref(1)*x(1)*y(1) - 2*R_ref(1,1)*c1*c2*d*n_ref(0)*x(1)*y(1) - 2*R_ref(0,1)*c1*d*n_ref(1)*s2*x(1)*y(0) - 2*R_ref(0,1)*c2*d*n_ref(1)*s1*x(0)*y(1) + 2*R_ref(1,1)*c1*d*n_ref(0)*s2*x(1)*y(0) + 2*R_ref(1,1)*c2*d*n_ref(0)*s1*x(0)*y(1) + 2*R_ref(0,1)*d*n_ref(1)*s1*s2*x(0)*y(0) - 2*R_ref(1,1)*d*n_ref(0)*s1*s2*x(0)*y(0), R_ref(0,1)*c1*c2*d*n_ref(2) - R_ref(2,1)*c1*c2*d*n_ref(0) - R_ref(0,0)*c1*d*n_ref(2)*s2 + R_ref(1,1)*c2*d*n_ref(2)*s1 + R_ref(2,0)*c1*d*n_ref(0)*s2 - R_ref(2,1)*c2*d*n_ref(1)*s1 - R_ref(1,0)*d*n_ref(2)*s1*s2 + R_ref(2,0)*d*n_ref(1)*s1*s2 + R_ref(0,1)*c1*c2*d*n_ref(1)*x(1) - R_ref(1,1)*c1*c2*d*n_ref(0)*x(1) + R_ref(0,2)*c1*c2*d*n_ref(2)*y(1) - R_ref(2,2)*c1*c2*d*n_ref(0)*y(1) - R_ref(0,0)*c1*d*n_ref(1)*s2*x(1) - R_ref(0,1)*c2*d*n_ref(1)*s1*x(0) + R_ref(1,0)*c1*d*n_ref(0)*s2*x(1) + R_ref(1,1)*c2*d*n_ref(0)*s1*x(0) - R_ref(0,2)*c1*d*n_ref(2)*s2*y(0) + R_ref(1,2)*c2*d*n_ref(2)*s1*y(1) + R_ref(2,2)*c1*d*n_ref(0)*s2*y(0) - R_ref(2,2)*c2*d*n_ref(1)*s1*y(1) + R_ref(0,0)*d*n_ref(1)*s1*s2*x(0) - R_ref(1,0)*d*n_ref(0)*s1*s2*x(0) - R_ref(1,2)*d*n_ref(2)*s1*s2*y(0) + R_ref(2,2)*d*n_ref(1)*s1*s2*y(0) + R_ref(0,2)*c1*c2*d*n_ref(1)*x(1)*y(1) - R_ref(1,2)*c1*c2*d*n_ref(0)*x(1)*y(1) - R_ref(0,2)*c1*d*n_ref(1)*s2*x(1)*y(0) - R_ref(0,2)*c2*d*n_ref(1)*s1*x(0)*y(1) + R_ref(1,2)*c1*d*n_ref(0)*s2*x(1)*y(0) + R_ref(1,2)*c2*d*n_ref(0)*s1*x(0)*y(1) + R_ref(0,2)*d*n_ref(1)*s1*s2*x(0)*y(0) - R_ref(1,2)*d*n_ref(0)*s1*s2*x(0)*y(0), 2*R_ref(0,2)*c1*c2*d*n_ref(2) - 2*R_ref(2,2)*c1*c2*d*n_ref(0) + 2*R_ref(1,2)*c2*d*n_ref(2)*s1 - 2*R_ref(2,2)*c2*d*n_ref(1)*s1 + 2*R_ref(0,2)*c1*c2*d*n_ref(1)*x(1) - 2*R_ref(1,2)*c1*c2*d*n_ref(0)*x(1) + 2*R_ref(0,1)*c1*c2*d*n_ref(2)*y(1) - 2*R_ref(2,1)*c1*c2*d*n_ref(0)*y(1) - 2*R_ref(0,2)*c2*d*n_ref(1)*s1*x(0) + 2*R_ref(1,2)*c2*d*n_ref(0)*s1*x(0) - 2*R_ref(0,1)*c1*d*n_ref(2)*s2*y(0) + 2*R_ref(1,1)*c2*d*n_ref(2)*s1*y(1) + 2*R_ref(2,1)*c1*d*n_ref(0)*s2*y(0) - 2*R_ref(2,1)*c2*d*n_ref(1)*s1*y(1) - 2*R_ref(1,1)*d*n_ref(2)*s1*s2*y(0) + 2*R_ref(2,1)*d*n_ref(1)*s1*s2*y(0) + 2*R_ref(0,1)*c1*c2*d*n_ref(1)*x(1)*y(1) - 2*R_ref(1,1)*c1*c2*d*n_ref(0)*x(1)*y(1) - 2*R_ref(0,1)*c1*d*n_ref(1)*s2*x(1)*y(0) - 2*R_ref(0,1)*c2*d*n_ref(1)*s1*x(0)*y(1) + 2*R_ref(1,1)*c1*d*n_ref(0)*s2*x(1)*y(0) + 2*R_ref(1,1)*c2*d*n_ref(0)*s1*x(0)*y(1) + 2*R_ref(0,1)*d*n_ref(1)*s1*s2*x(0)*y(0) - 2*R_ref(1,1)*d*n_ref(0)*s1*s2*x(0)*y(0), 2*R_ref(0,2)*c1*d*n_ref(2)*s2 - 2*R_ref(2,2)*c1*d*n_ref(0)*s2 + 2*R_ref(1,2)*d*n_ref(2)*s1*s2 - 2*R_ref(2,2)*d*n_ref(1)*s1*s2 - 2*R_ref(0,0)*c1*c2*d*n_ref(2)*y(1) + 2*R_ref(2,0)*c1*c2*d*n_ref(0)*y(1) + 2*R_ref(0,2)*c1*d*n_ref(1)*s2*x(1) - 2*R_ref(1,2)*c1*d*n_ref(0)*s2*x(1) + 2*R_ref(0,0)*c1*d*n_ref(2)*s2*y(0) - 2*R_ref(1,0)*c2*d*n_ref(2)*s1*y(1) - 2*R_ref(2,0)*c1*d*n_ref(0)*s2*y(0) + 2*R_ref(2,0)*c2*d*n_ref(1)*s1*y(1) - 2*R_ref(0,2)*d*n_ref(1)*s1*s2*x(0) + 2*R_ref(1,2)*d*n_ref(0)*s1*s2*x(0) + 2*R_ref(1,0)*d*n_ref(2)*s1*s2*y(0) - 2*R_ref(2,0)*d*n_ref(1)*s1*s2*y(0) - 2*R_ref(0,0)*c1*c2*d*n_ref(1)*x(1)*y(1) + 2*R_ref(1,0)*c1*c2*d*n_ref(0)*x(1)*y(1) + 2*R_ref(0,0)*c1*d*n_ref(1)*s2*x(1)*y(0) + 2*R_ref(0,0)*c2*d*n_ref(1)*s1*x(0)*y(1) - 2*R_ref(1,0)*c1*d*n_ref(0)*s2*x(1)*y(0) - 2*R_ref(1,0)*c2*d*n_ref(0)*s1*x(0)*y(1) - 2*R_ref(0,0)*d*n_ref(1)*s1*s2*x(0)*y(0) + 2*R_ref(1,0)*d*n_ref(0)*s1*s2*x(0)*y(0), 2*R_ref(2,0)*c1*c2*d*n_ref(0) - 2*R_ref(0,0)*c1*c2*d*n_ref(2) - 2*R_ref(0,1)*c1*d*n_ref(2)*s2 - 2*R_ref(1,0)*c2*d*n_ref(2)*s1 + 2*R_ref(2,0)*c2*d*n_ref(1)*s1 + 2*R_ref(2,1)*c1*d*n_ref(0)*s2 - 2*R_ref(1,1)*d*n_ref(2)*s1*s2 + 2*R_ref(2,1)*d*n_ref(1)*s1*s2 - 2*R_ref(0,0)*c1*c2*d*n_ref(1)*x(1) + 2*R_ref(1,0)*c1*c2*d*n_ref(0)*x(1) + 2*R_ref(0,0)*c2*d*n_ref(1)*s1*x(0) - 2*R_ref(0,1)*c1*d*n_ref(1)*s2*x(1) - 2*R_ref(1,0)*c2*d*n_ref(0)*s1*x(0) + 2*R_ref(1,1)*c1*d*n_ref(0)*s2*x(1) + 2*R_ref(0,1)*d*n_ref(1)*s1*s2*x(0) - 2*R_ref(1,1)*d*n_ref(0)*s1*s2*x(0), R_ref(2,1)*c1*c2*d*n_ref(0) - R_ref(0,1)*c1*c2*d*n_ref(2) + R_ref(0,0)*c1*d*n_ref(2)*s2 - R_ref(1,1)*c2*d*n_ref(2)*s1 - R_ref(2,0)*c1*d*n_ref(0)*s2 + R_ref(2,1)*c2*d*n_ref(1)*s1 + R_ref(1,0)*d*n_ref(2)*s1*s2 - R_ref(2,0)*d*n_ref(1)*s1*s2 - R_ref(0,1)*c1*c2*d*n_ref(1)*x(1) + R_ref(1,1)*c1*c2*d*n_ref(0)*x(1) + R_ref(0,2)*c1*c2*d*n_ref(2)*y(1) - R_ref(2,2)*c1*c2*d*n_ref(0)*y(1) + R_ref(0,0)*c1*d*n_ref(1)*s2*x(1) + R_ref(0,1)*c2*d*n_ref(1)*s1*x(0) - R_ref(1,0)*c1*d*n_ref(0)*s2*x(1) - R_ref(1,1)*c2*d*n_ref(0)*s1*x(0) - R_ref(0,2)*c1*d*n_ref(2)*s2*y(0) + R_ref(1,2)*c2*d*n_ref(2)*s1*y(1) + R_ref(2,2)*c1*d*n_ref(0)*s2*y(0) - R_ref(2,2)*c2*d*n_ref(1)*s1*y(1) - R_ref(0,0)*d*n_ref(1)*s1*s2*x(0) + R_ref(1,0)*d*n_ref(0)*s1*s2*x(0) - R_ref(1,2)*d*n_ref(2)*s1*s2*y(0) + R_ref(2,2)*d*n_ref(1)*s1*s2*y(0) + R_ref(0,2)*c1*c2*d*n_ref(1)*x(1)*y(1) - R_ref(1,2)*c1*c2*d*n_ref(0)*x(1)*y(1) - R_ref(0,2)*c1*d*n_ref(1)*s2*x(1)*y(0) - R_ref(0,2)*c2*d*n_ref(1)*s1*x(0)*y(1) + R_ref(1,2)*c1*d*n_ref(0)*s2*x(1)*y(0) + R_ref(1,2)*c2*d*n_ref(0)*s1*x(0)*y(1) + R_ref(0,2)*d*n_ref(1)*s1*s2*x(0)*y(0) - R_ref(1,2)*d*n_ref(0)*s1*s2*x(0)*y(0);
				return M;
			}
			 
			void SIFTP2PQuerySolver::cayley(const Eigen::Vector3d &r, Eigen::Matrix3d &R, double &s ) const
			{
				const double x = r(0);
				const double y = r(1);
				const double z = r(2);
				
				const double w = 1;
				const double xx = x*x;
				const double xy = x*y;
				const double xz = x*z;
				const double xw = x*w;
				const double yy = y*y;
				const double yz = y*z;
				const double yw = y*w;
				const double zz = z*z;
				const double zw = z*w;
				const double ww = w*w;

				s = w*w+x*x+y*y+z*z;
				R << ww+xx-yy-zz, 2*(xy-zw), 2*(yw+xz),
					2*(xy+zw), ww-xx+yy-zz, 2*(yz-xw),
					2*(xz-yw), 2*(xw+yz), ww-xx-yy+zz;
			}

			void SIFTP2PQuerySolver::solver(
				const double s1[], // sin of angle in first view
				const double c1[], // cos of angle in first view
				const double s2[], // sin of angle in second view
				const double c2[], // cos of angle in second view
				const Eigen::Vector2d x[],
				const double d[],
				const Eigen::Vector3d n[],
				const Eigen::Vector2d y[],
				const Eigen::Matrix3d R_ref[],
				const Eigen::Vector3d t_ref[],
				std::vector<Eigen::Matrix3d> &Rsolns, 
				std::vector<Eigen::Vector3d> &tsolns) const
			{
				Eigen::MatrixXd M1 = build_constraints(R_ref[0],t_ref[0],s1[0],c1[0],s2[0],c2[0],x[0],d[0],n[0],y[0]);
				Eigen::MatrixXd M2 = build_constraints(R_ref[1],t_ref[1],s1[1],c1[1],s2[1],c2[1],x[1],d[1],n[1],y[1]);
				Eigen::Matrix<double,6,13> M;
				M << M1, M2;

				// G-J elimination
				Eigen::Matrix<double,6,10> G = M.block<6,6>(0,0).partialPivLu().solve(M.block<6,10>(0,3));
				Eigen::Matrix<double,3,10> T = G.block<3,10>(0,0);
				Eigen::Matrix<double,3,10> C = G.block<3,10>(3,0);
				
				Eigen::Matrix<double,3,8> cayley_solutions;
				int nsolns = re3q3(C, &cayley_solutions);
				
				for ( int i = 0; i < nsolns; i++ )
				{
					const Eigen::Vector3d r = cayley_solutions.col(i);
					Eigen::Matrix3d R;
					double s;
					cayley(r,R,s);
					
					Eigen::Matrix<double,10,1> X;
					X << r(0)*r(0), r(0)*r(1), r(0)*r(2), r(1)*r(1), r(1)*r(2), r(2)*r(2), r(0), r(1), r(2), 1;
					Eigen::Vector3d t = -T*X;
					
					Rsolns.push_back(R/s);
					tsolns.push_back(t/s);
				}
			}

			void SIFTP2PQuerySolver::refine_3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions, int n_sols) const 
			{
				Eigen::Matrix3d J;
				Eigen::Vector3d r;
				Eigen::Vector3d dx;
				double x, y, z;

				for (int i = 0; i < n_sols; ++i) {
					x = (*solutions)(0, i);
					y = (*solutions)(1, i);
					z = (*solutions)(2, i);

					// [x^2, x*y, x*z, y^2, y*z, z^2, x, y, z, 1.0]
					for (int iter = 0; iter < 5; ++iter) {
						r = coeffs.col(0) * x * x + coeffs.col(1) * x * y + coeffs.col(2) * x * z + coeffs.col(3) * y * y +
							coeffs.col(4) * y * z + coeffs.col(5) * z * z + coeffs.col(6) * x + coeffs.col(7) * y +
							coeffs.col(8) * z + coeffs.col(9);

						if (r.cwiseAbs().maxCoeff() < 1e-8)
							break;

						J.col(0) = 2.0 * coeffs.col(0) * x + coeffs.col(1) * y + coeffs.col(2) * z + coeffs.col(6);
						J.col(1) = coeffs.col(1) * x + 2.0 * coeffs.col(3) * y + coeffs.col(4) * z + coeffs.col(7);
						J.col(2) = coeffs.col(2) * x + coeffs.col(4) * y + 2.0 * coeffs.col(5) * z + coeffs.col(8);

						dx = J.inverse() * r;

						x -= dx(0);
						y -= dx(1);
						z -= dx(2);
					}

					(*solutions)(0, i) = x;
					(*solutions)(1, i) = y;
					(*solutions)(2, i) = z;
				}
			}

			/*
			* Order of coefficients is:  x^2, xy, xz, y^2, yz, z^2, x, y, z, 1.0;
			*
			*/
			int SIFTP2PQuerySolver::re3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions, bool try_random_var_change) const 
			{

				Eigen::Matrix<double, 3, 3> Ax, Ay, Az;
				Ax << coeffs.col(3), coeffs.col(5), coeffs.col(4); // y^2, z^2, yz
				Ay << coeffs.col(0), coeffs.col(5), coeffs.col(2); // x^2, z^2, xz
				Az << coeffs.col(3), coeffs.col(0), coeffs.col(1); // y^2, x^2, yx

				// We check det(A) as a cheaper proxy for condition number
				int elim_var = 0;
				double detx = std::abs(Ax.determinant());
				double dety = std::abs(Ay.determinant());
				double detz = std::abs(Az.determinant());
				double det = detx;
				if (det < dety) {
					det = dety;
					elim_var = 1;
				}
				if (det < detz) {
					det = detz;
					elim_var = 2;
				}

				if (try_random_var_change && det < 1e-10) {
					Eigen::Matrix<double, 3, 4> A;
					A.block<3, 3>(0, 0) = Eigen::Quaternion<double>::UnitRandom().toRotationMatrix();
					A.block<3, 1>(0, 3).setRandom().normalize();

					Eigen::Matrix<double, 10, 10> B;
					B << A(0, 0) * A(0, 0), 2 * A(0, 0) * A(0, 1), 2 * A(0, 0) * A(0, 2), A(0, 1) * A(0, 1), 2 * A(0, 1) * A(0, 2), A(0, 2) * A(0, 2), 2 * A(0, 0) * A(0, 3), 2 * A(0, 1) * A(0, 3), 2 * A(0, 2) * A(0, 3), A(0, 3) * A(0, 3),
						A(0, 0) * A(1, 0), A(0, 0) * A(1, 1) + A(0, 1) * A(1, 0), A(0, 0) * A(1, 2) + A(0, 2) * A(1, 0), A(0, 1) * A(1, 1), A(0, 1) * A(1, 2) + A(0, 2) * A(1, 1), A(0, 2) * A(1, 2), A(0, 0) * A(1, 3) + A(0, 3) * A(1, 0), A(0, 1) * A(1, 3) + A(0, 3) * A(1, 1), A(0, 2) * A(1, 3) + A(0, 3) * A(1, 2), A(0, 3) * A(1, 3),
						A(0, 0) * A(2, 0), A(0, 0) * A(2, 1) + A(0, 1) * A(2, 0), A(0, 0) * A(2, 2) + A(0, 2) * A(2, 0), A(0, 1) * A(2, 1), A(0, 1) * A(2, 2) + A(0, 2) * A(2, 1), A(0, 2) * A(2, 2), A(0, 0) * A(2, 3) + A(0, 3) * A(2, 0), A(0, 1) * A(2, 3) + A(0, 3) * A(2, 1), A(0, 2) * A(2, 3) + A(0, 3) * A(2, 2), A(0, 3) * A(2, 3),
						A(1, 0) * A(1, 0), 2 * A(1, 0) * A(1, 1), 2 * A(1, 0) * A(1, 2), A(1, 1) * A(1, 1), 2 * A(1, 1) * A(1, 2), A(1, 2) * A(1, 2), 2 * A(1, 0) * A(1, 3), 2 * A(1, 1) * A(1, 3), 2 * A(1, 2) * A(1, 3), A(1, 3) * A(1, 3),
						A(1, 0) * A(2, 0), A(1, 0) * A(2, 1) + A(1, 1) * A(2, 0), A(1, 0) * A(2, 2) + A(1, 2) * A(2, 0), A(1, 1) * A(2, 1), A(1, 1) * A(2, 2) + A(1, 2) * A(2, 1), A(1, 2) * A(2, 2), A(1, 0) * A(2, 3) + A(1, 3) * A(2, 0), A(1, 1) * A(2, 3) + A(1, 3) * A(2, 1), A(1, 2) * A(2, 3) + A(1, 3) * A(2, 2), A(1, 3) * A(2, 3),
						A(2, 0) * A(2, 0), 2 * A(2, 0) * A(2, 1), 2 * A(2, 0) * A(2, 2), A(2, 1) * A(2, 1), 2 * A(2, 1) * A(2, 2), A(2, 2) * A(2, 2), 2 * A(2, 0) * A(2, 3), 2 * A(2, 1) * A(2, 3), 2 * A(2, 2) * A(2, 3), A(2, 3) * A(2, 3),
						0, 0, 0, 0, 0, 0, A(0, 0), A(0, 1), A(0, 2), A(0, 3),
						0, 0, 0, 0, 0, 0, A(1, 0), A(1, 1), A(1, 2), A(1, 3),
						0, 0, 0, 0, 0, 0, A(2, 0), A(2, 1), A(2, 2), A(2, 3),
						0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
					Eigen::Matrix<double, 3, 10> coeffsB = coeffs * B;

					int n_sols = re3q3(coeffsB, solutions, false);

					// Revert change of variables
					for (int k = 0; k < n_sols; k++) {
						solutions->col(k) = A.block<3, 3>(0, 0) * solutions->col(k) + A.col(3);
					}

					// In some cases the numerics are quite poor after the change of variables, so we do some newton steps with the original coefficients.
					//refine_3q3(coeffs, solutions, n_sols);

					return n_sols;
				}

				Eigen::Matrix<double, 3, 7> P;

				if (elim_var == 0) {
					// re-order columns to eliminate x (target: y^2 z^2 yz x^2 xy xz x y z 1)
					P << coeffs.col(0), coeffs.col(1), coeffs.col(2), coeffs.col(6), coeffs.col(7), coeffs.col(8), coeffs.col(9);
					P = -Ax.inverse() * P;
				} else if (elim_var == 1) {
					// re-order columns to eliminate y (target: x^2 z^2 xz y^2 xy yz y x z 1)
					P << coeffs.col(3), coeffs.col(1), coeffs.col(4), coeffs.col(7), coeffs.col(6), coeffs.col(8), coeffs.col(9);
					P = -Ay.inverse() * P;
				} else if (elim_var == 2) {
					// re-order columns to eliminate z (target: y^2 x^2 yx z^2 zy z y x 1)
					P << coeffs.col(5), coeffs.col(4), coeffs.col(2), coeffs.col(8), coeffs.col(7), coeffs.col(6), coeffs.col(9);
					P = -Az.inverse() * P;
				}

				double a11 = P(0, 1) * P(2, 1) + P(0, 2) * P(1, 1) - P(2, 1) * P(0, 1) - P(2, 2) * P(2, 1) - P(2, 0);
				double a12 = P(0, 1) * P(2, 4) + P(0, 4) * P(2, 1) + P(0, 2) * P(1, 4) + P(0, 5) * P(1, 1) - P(2, 1) * P(0, 4) - P(2, 4) * P(0, 1) - P(2, 2) * P(2, 4) - P(2, 5) * P(2, 1) - P(2, 3);
				double a13 = P(0, 4) * P(2, 4) + P(0, 5) * P(1, 4) - P(2, 4) * P(0, 4) - P(2, 5) * P(2, 4) - P(2, 6);
				double a14 = P(0, 1) * P(2, 2) + P(0, 2) * P(1, 2) - P(2, 1) * P(0, 2) - P(2, 2) * P(2, 2) + P(0, 0);
				double a15 = P(0, 1) * P(2, 5) + P(0, 4) * P(2, 2) + P(0, 2) * P(1, 5) + P(0, 5) * P(1, 2) - P(2, 1) * P(0, 5) - P(2, 4) * P(0, 2) - P(2, 2) * P(2, 5) - P(2, 5) * P(2, 2) + P(0, 3);
				double a16 = P(0, 4) * P(2, 5) + P(0, 5) * P(1, 5) - P(2, 4) * P(0, 5) - P(2, 5) * P(2, 5) + P(0, 6);
				double a17 = P(0, 1) * P(2, 0) + P(0, 2) * P(1, 0) - P(2, 1) * P(0, 0) - P(2, 2) * P(2, 0);
				double a18 = P(0, 1) * P(2, 3) + P(0, 4) * P(2, 0) + P(0, 2) * P(1, 3) + P(0, 5) * P(1, 0) - P(2, 1) * P(0, 3) - P(2, 4) * P(0, 0) - P(2, 2) * P(2, 3) - P(2, 5) * P(2, 0);
				double a19 = P(0, 1) * P(2, 6) + P(0, 4) * P(2, 3) + P(0, 2) * P(1, 6) + P(0, 5) * P(1, 3) - P(2, 1) * P(0, 6) - P(2, 4) * P(0, 3) - P(2, 2) * P(2, 6) - P(2, 5) * P(2, 3);
				double a110 = P(0, 4) * P(2, 6) + P(0, 5) * P(1, 6) - P(2, 4) * P(0, 6) - P(2, 5) * P(2, 6);

				double a21 = P(2, 1) * P(2, 1) + P(2, 2) * P(1, 1) - P(1, 1) * P(0, 1) - P(1, 2) * P(2, 1) - P(1, 0);
				double a22 = P(2, 1) * P(2, 4) + P(2, 4) * P(2, 1) + P(2, 2) * P(1, 4) + P(2, 5) * P(1, 1) - P(1, 1) * P(0, 4) - P(1, 4) * P(0, 1) - P(1, 2) * P(2, 4) - P(1, 5) * P(2, 1) - P(1, 3);
				double a23 = P(2, 4) * P(2, 4) + P(2, 5) * P(1, 4) - P(1, 4) * P(0, 4) - P(1, 5) * P(2, 4) - P(1, 6);
				double a24 = P(2, 1) * P(2, 2) + P(2, 2) * P(1, 2) - P(1, 1) * P(0, 2) - P(1, 2) * P(2, 2) + P(2, 0);
				double a25 = P(2, 1) * P(2, 5) + P(2, 4) * P(2, 2) + P(2, 2) * P(1, 5) + P(2, 5) * P(1, 2) - P(1, 1) * P(0, 5) - P(1, 4) * P(0, 2) - P(1, 2) * P(2, 5) - P(1, 5) * P(2, 2) + P(2, 3);
				double a26 = P(2, 4) * P(2, 5) + P(2, 5) * P(1, 5) - P(1, 4) * P(0, 5) - P(1, 5) * P(2, 5) + P(2, 6);
				double a27 = P(2, 1) * P(2, 0) + P(2, 2) * P(1, 0) - P(1, 1) * P(0, 0) - P(1, 2) * P(2, 0);
				double a28 = P(2, 1) * P(2, 3) + P(2, 4) * P(2, 0) + P(2, 2) * P(1, 3) + P(2, 5) * P(1, 0) - P(1, 1) * P(0, 3) - P(1, 4) * P(0, 0) - P(1, 2) * P(2, 3) - P(1, 5) * P(2, 0);
				double a29 = P(2, 1) * P(2, 6) + P(2, 4) * P(2, 3) + P(2, 2) * P(1, 6) + P(2, 5) * P(1, 3) - P(1, 1) * P(0, 6) - P(1, 4) * P(0, 3) - P(1, 2) * P(2, 6) - P(1, 5) * P(2, 3);
				double a210 = P(2, 4) * P(2, 6) + P(2, 5) * P(1, 6) - P(1, 4) * P(0, 6) - P(1, 5) * P(2, 6);

				double t2 = P(2, 1) * P(2, 1);
				double t3 = P(2, 2) * P(2, 2);
				double t4 = P(0, 1) * P(1, 4);
				double t5 = P(0, 4) * P(1, 1);
				double t6 = t4 + t5;
				double t7 = P(0, 2) * P(1, 5);
				double t8 = P(0, 5) * P(1, 2);
				double t9 = t7 + t8;
				double t10 = P(0, 1) * P(1, 5);
				double t11 = P(0, 4) * P(1, 2);
				double t12 = t10 + t11;
				double t13 = P(0, 2) * P(1, 4);
				double t14 = P(0, 5) * P(1, 1);
				double t15 = t13 + t14;
				double t16 = P(2, 1) * P(2, 5);
				double t17 = P(2, 2) * P(2, 4);
				double t18 = t16 + t17;
				double t19 = P(2, 4) * P(2, 4);
				double t20 = P(2, 5) * P(2, 5);
				double a31 = P(0, 0) * P(1, 1) + P(0, 1) * P(1, 0) - P(2, 0) * P(2, 1) * 2.0 - P(0, 1) * t2 - P(1, 1) * t3 - P(2, 2) * t2 * 2.0 + (P(0, 1) * P(0, 1)) * P(1, 1) + P(0, 2) * P(1, 1) * P(1, 2) + P(0, 1) * P(1, 2) * P(2, 1) + P(0, 2) * P(1, 1) * P(2, 1);
				double a32 = P(0, 0) * P(1, 4) + P(0, 1) * P(1, 3) + P(0, 3) * P(1, 1) + P(0, 4) * P(1, 0) - P(2, 0) * P(2, 4) * 2.0 - P(2, 1) * P(2, 3) * 2.0 - P(0, 4) * t2 + P(0, 1) * t6 - P(1, 4) * t3 + P(1, 1) * t9 + P(2, 1) * t12 + P(2, 1) * t15 - P(2, 1) * t18 * 2.0 + P(0, 1) * P(0, 4) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 4) + P(0, 1) * P(1, 2) * P(2, 4) + P(0, 2) * P(1, 1) * P(2, 4) - P(0, 1) * P(2, 1) * P(2, 4) * 2.0 - P(1, 1) * P(2, 2) * P(2, 5) * 2.0 - P(2, 1) * P(2, 2) * P(2, 4) * 2.0;
				double a33 = P(0, 1) * P(1, 6) + P(0, 3) * P(1, 4) + P(0, 4) * P(1, 3) + P(0, 6) * P(1, 1) - P(2, 1) * P(2, 6) * 2.0 - P(2, 3) * P(2, 4) * 2.0 + P(0, 4) * t6 - P(0, 1) * t19 + P(1, 4) * t9 - P(1, 1) * t20 + P(2, 4) * t12 + P(2, 4) * t15 - P(2, 4) * t18 * 2.0 + P(0, 1) * P(0, 4) * P(1, 4) + P(0, 5) * P(1, 1) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 1) + P(0, 5) * P(1, 4) * P(2, 1) - P(0, 4) * P(2, 1) * P(2, 4) * 2.0 - P(1, 4) * P(2, 2) * P(2, 5) * 2.0 - P(2, 1) * P(2, 4) * P(2, 5) * 2.0;
				double a34 = P(0, 4) * P(1, 6) + P(0, 6) * P(1, 4) - P(2, 4) * P(2, 6) * 2.0 - P(0, 4) * t19 - P(1, 4) * t20 - P(2, 5) * t19 * 2.0 + (P(0, 4) * P(0, 4)) * P(1, 4) + P(0, 5) * P(1, 4) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 4) + P(0, 5) * P(1, 4) * P(2, 4);
				double a35 = P(0, 0) * P(1, 2) + P(0, 2) * P(1, 0) - P(2, 0) * P(2, 2) * 2.0 - P(0, 2) * t2 - P(1, 2) * t3 - P(2, 1) * t3 * 2.0 + P(0, 2) * (P(1, 2) * P(1, 2)) + P(0, 1) * P(0, 2) * P(1, 1) + P(0, 1) * P(1, 2) * P(2, 2) + P(0, 2) * P(1, 1) * P(2, 2);
				double a36 = P(0, 0) * P(1, 5) + P(0, 2) * P(1, 3) + P(0, 3) * P(1, 2) + P(0, 5) * P(1, 0) - P(2, 0) * P(2, 5) * 2.0 - P(2, 2) * P(2, 3) * 2.0 - P(0, 5) * t2 + P(0, 2) * t6 - P(1, 5) * t3 + P(1, 2) * t9 + P(2, 2) * t12 + P(2, 2) * t15 - P(2, 2) * t18 * 2.0 + P(0, 1) * P(0, 5) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 5) + P(0, 1) * P(1, 2) * P(2, 5) + P(0, 2) * P(1, 1) * P(2, 5) - P(0, 2) * P(2, 1) * P(2, 4) * 2.0 - P(1, 2) * P(2, 2) * P(2, 5) * 2.0 - P(2, 1) * P(2, 2) * P(2, 5) * 2.0;
				double a37 = P(0, 2) * P(1, 6) + P(0, 3) * P(1, 5) + P(0, 5) * P(1, 3) + P(0, 6) * P(1, 2) - P(2, 2) * P(2, 6) * 2.0 - P(2, 3) * P(2, 5) * 2.0 + P(0, 5) * t6 - P(0, 2) * t19 + P(1, 5) * t9 - P(1, 2) * t20 + P(2, 5) * t12 + P(2, 5) * t15 - P(2, 5) * t18 * 2.0 + P(0, 2) * P(0, 4) * P(1, 4) + P(0, 5) * P(1, 2) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 2) + P(0, 5) * P(1, 4) * P(2, 2) - P(0, 5) * P(2, 1) * P(2, 4) * 2.0 - P(1, 5) * P(2, 2) * P(2, 5) * 2.0 - P(2, 2) * P(2, 4) * P(2, 5) * 2.0;
				double a38 = P(0, 5) * P(1, 6) + P(0, 6) * P(1, 5) - P(2, 5) * P(2, 6) * 2.0 - P(0, 5) * t19 - P(1, 5) * t20 - P(2, 4) * t20 * 2.0 + P(0, 5) * (P(1, 5) * P(1, 5)) + P(0, 4) * P(0, 5) * P(1, 4) + P(0, 4) * P(1, 5) * P(2, 5) + P(0, 5) * P(1, 4) * P(2, 5);
				double a39 = P(0, 0) * P(1, 0) - P(0, 0) * t2 - P(1, 0) * t3 - P(2, 0) * P(2, 0) + P(0, 0) * P(0, 1) * P(1, 1) + P(0, 2) * P(1, 0) * P(1, 2) + P(0, 1) * P(1, 2) * P(2, 0) + P(0, 2) * P(1, 1) * P(2, 0) - P(2, 0) * P(2, 1) * P(2, 2) * 2.0;
				double a310 = P(0, 0) * P(1, 3) + P(0, 3) * P(1, 0) - P(2, 0) * P(2, 3) * 2.0 - P(0, 3) * t2 + P(0, 0) * t6 - P(1, 3) * t3 + P(1, 0) * t9 + P(2, 0) * t12 + P(2, 0) * t15 - P(2, 0) * t18 * 2.0 + P(0, 1) * P(0, 3) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 3) + P(0, 1) * P(1, 2) * P(2, 3) + P(0, 2) * P(1, 1) * P(2, 3) - P(0, 0) * P(2, 1) * P(2, 4) * 2.0 - P(1, 0) * P(2, 2) * P(2, 5) * 2.0 - P(2, 1) * P(2, 2) * P(2, 3) * 2.0;
				double a311 = P(0, 0) * P(1, 6) + P(0, 3) * P(1, 3) + P(0, 6) * P(1, 0) - P(2, 0) * P(2, 6) * 2.0 - P(0, 6) * t2 + P(0, 3) * t6 - P(0, 0) * t19 - P(1, 6) * t3 + P(1, 3) * t9 - P(1, 0) * t20 + P(2, 3) * t12 + P(2, 3) * t15 - P(2, 3) * t18 * 2.0 - P(2, 3) * P(2, 3) + P(0, 0) * P(0, 4) * P(1, 4) + P(0, 1) * P(0, 6) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 6) + P(0, 5) * P(1, 0) * P(1, 5) + P(0, 1) * P(1, 2) * P(2, 6) + P(0, 2) * P(1, 1) * P(2, 6) + P(0, 4) * P(1, 5) * P(2, 0) + P(0, 5) * P(1, 4) * P(2, 0) - P(0, 3) * P(2, 1) * P(2, 4) * 2.0 - P(1, 3) * P(2, 2) * P(2, 5) * 2.0 - P(2, 0) * P(2, 4) * P(2, 5) * 2.0 - P(2, 1) * P(2, 2) * P(2, 6) * 2.0;
				double a312 = P(0, 3) * P(1, 6) + P(0, 6) * P(1, 3) - P(2, 3) * P(2, 6) * 2.0 + P(0, 6) * t6 - P(0, 3) * t19 + P(1, 6) * t9 - P(1, 3) * t20 + P(2, 6) * t12 + P(2, 6) * t15 - P(2, 6) * t18 * 2.0 + P(0, 3) * P(0, 4) * P(1, 4) + P(0, 5) * P(1, 3) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 3) + P(0, 5) * P(1, 4) * P(2, 3) - P(0, 6) * P(2, 1) * P(2, 4) * 2.0 - P(1, 6) * P(2, 2) * P(2, 5) * 2.0 - P(2, 3) * P(2, 4) * P(2, 5) * 2.0;
				double a313 = P(0, 6) * P(1, 6) - P(0, 6) * t19 - P(1, 6) * t20 - P(2, 6) * P(2, 6) + P(0, 4) * P(0, 6) * P(1, 4) + P(0, 5) * P(1, 5) * P(1, 6) + P(0, 4) * P(1, 5) * P(2, 6) + P(0, 5) * P(1, 4) * P(2, 6) - P(2, 4) * P(2, 5) * P(2, 6) * 2.0;

				// det(M(x))
				double c8 = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39 - a14 * a21 * a39;
				double c7 = a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31 - a17 * a24 * a32 - a17 * a25 * a31 - a18 * a24 * a31 - a11 * a27 * a36 - a11 * a28 * a35 - a12 * a27 * a35 + a17 * a21 * a36 + a17 * a22 * a35 + a18 * a21 * a35 + a11 * a25 * a39 + a12 * a24 * a39 - a14 * a22 * a39 - a15 * a21 * a39 + a11 * a24 * a310 - a14 * a21 * a310;
				double c6 = a14 * a27 * a33 + a14 * a28 * a32 + a14 * a29 * a31 + a15 * a27 * a32 + a15 * a28 * a31 + a16 * a27 * a31 - a17 * a24 * a33 - a17 * a25 * a32 - a17 * a26 * a31 - a18 * a24 * a32 - a18 * a25 * a31 - a19 * a24 * a31 - a11 * a27 * a37 - a11 * a28 * a36 - a11 * a29 * a35 - a12 * a27 * a36 - a12 * a28 * a35 - a13 * a27 * a35 + a17 * a21 * a37 + a17 * a22 * a36 + a17 * a23 * a35 + a18 * a21 * a36 + a18 * a22 * a35 + a19 * a21 * a35 + a11 * a26 * a39 + a12 * a25 * a39 + a13 * a24 * a39 - a14 * a23 * a39 - a15 * a22 * a39 - a16 * a21 * a39 + a11 * a24 * a311 + a11 * a25 * a310 + a12 * a24 * a310 - a14 * a21 * a311 - a14 * a22 * a310 - a15 * a21 * a310;
				double c5 = a14 * a27 * a34 + a14 * a28 * a33 + a14 * a29 * a32 + a15 * a27 * a33 + a15 * a28 * a32 + a15 * a29 * a31 + a16 * a27 * a32 + a16 * a28 * a31 - a17 * a24 * a34 - a17 * a25 * a33 - a17 * a26 * a32 - a18 * a24 * a33 - a18 * a25 * a32 - a18 * a26 * a31 - a19 * a24 * a32 - a19 * a25 * a31 - a11 * a27 * a38 - a11 * a28 * a37 - a11 * a29 * a36 - a12 * a27 * a37 - a12 * a28 * a36 - a12 * a29 * a35 - a13 * a27 * a36 - a13 * a28 * a35 + a17 * a21 * a38 + a17 * a22 * a37 + a17 * a23 * a36 + a18 * a21 * a37 + a18 * a22 * a36 + a18 * a23 * a35 + a19 * a21 * a36 + a19 * a22 * a35 + a12 * a26 * a39 + a13 * a25 * a39 - a15 * a23 * a39 - a16 * a22 * a39 - a24 * a31 * a110 + a21 * a35 * a110 + a14 * a31 * a210 - a11 * a35 * a210 + a11 * a24 * a312 + a11 * a25 * a311 + a11 * a26 * a310 + a12 * a24 * a311 + a12 * a25 * a310 + a13 * a24 * a310 - a14 * a21 * a312 - a14 * a22 * a311 - a14 * a23 * a310 - a15 * a21 * a311 - a15 * a22 * a310 - a16 * a21 * a310;
				double c4 = a14 * a28 * a34 + a14 * a29 * a33 + a15 * a27 * a34 + a15 * a28 * a33 + a15 * a29 * a32 + a16 * a27 * a33 + a16 * a28 * a32 + a16 * a29 * a31 - a17 * a25 * a34 - a17 * a26 * a33 - a18 * a24 * a34 - a18 * a25 * a33 - a18 * a26 * a32 - a19 * a24 * a33 - a19 * a25 * a32 - a19 * a26 * a31 - a11 * a28 * a38 - a11 * a29 * a37 - a12 * a27 * a38 - a12 * a28 * a37 - a12 * a29 * a36 - a13 * a27 * a37 - a13 * a28 * a36 - a13 * a29 * a35 + a17 * a22 * a38 + a17 * a23 * a37 + a18 * a21 * a38 + a18 * a22 * a37 + a18 * a23 * a36 + a19 * a21 * a37 + a19 * a22 * a36 + a19 * a23 * a35 + a13 * a26 * a39 - a16 * a23 * a39 - a24 * a32 * a110 - a25 * a31 * a110 + a21 * a36 * a110 + a22 * a35 * a110 + a14 * a32 * a210 + a15 * a31 * a210 - a11 * a36 * a210 - a12 * a35 * a210 + a11 * a24 * a313 + a11 * a25 * a312 + a11 * a26 * a311 + a12 * a24 * a312 + a12 * a25 * a311 + a12 * a26 * a310 + a13 * a24 * a311 + a13 * a25 * a310 - a14 * a21 * a313 - a14 * a22 * a312 - a14 * a23 * a311 - a15 * a21 * a312 - a15 * a22 * a311 - a15 * a23 * a310 - a16 * a21 * a311 - a16 * a22 * a310;
				double c3 = a14 * a29 * a34 + a15 * a28 * a34 + a15 * a29 * a33 + a16 * a27 * a34 + a16 * a28 * a33 + a16 * a29 * a32 - a17 * a26 * a34 - a18 * a25 * a34 - a18 * a26 * a33 - a19 * a24 * a34 - a19 * a25 * a33 - a19 * a26 * a32 - a11 * a29 * a38 - a12 * a28 * a38 - a12 * a29 * a37 - a13 * a27 * a38 - a13 * a28 * a37 - a13 * a29 * a36 + a17 * a23 * a38 + a18 * a22 * a38 + a18 * a23 * a37 + a19 * a21 * a38 + a19 * a22 * a37 + a19 * a23 * a36 - a24 * a33 * a110 - a25 * a32 * a110 - a26 * a31 * a110 + a21 * a37 * a110 + a22 * a36 * a110 + a23 * a35 * a110 + a14 * a33 * a210 + a15 * a32 * a210 + a16 * a31 * a210 - a11 * a37 * a210 - a12 * a36 * a210 - a13 * a35 * a210 + a11 * a25 * a313 + a11 * a26 * a312 + a12 * a24 * a313 + a12 * a25 * a312 + a12 * a26 * a311 + a13 * a24 * a312 + a13 * a25 * a311 + a13 * a26 * a310 - a14 * a22 * a313 - a14 * a23 * a312 - a15 * a21 * a313 - a15 * a22 * a312 - a15 * a23 * a311 - a16 * a21 * a312 - a16 * a22 * a311 - a16 * a23 * a310;
				double c2 = a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33 - a18 * a26 * a34 - a19 * a25 * a34 - a19 * a26 * a33 - a12 * a29 * a38 - a13 * a28 * a38 - a13 * a29 * a37 + a18 * a23 * a38 + a19 * a22 * a38 + a19 * a23 * a37 - a24 * a34 * a110 - a25 * a33 * a110 - a26 * a32 * a110 + a21 * a38 * a110 + a22 * a37 * a110 + a23 * a36 * a110 + a14 * a34 * a210 + a15 * a33 * a210 + a16 * a32 * a210 - a11 * a38 * a210 - a12 * a37 * a210 - a13 * a36 * a210 + a11 * a26 * a313 + a12 * a25 * a313 + a12 * a26 * a312 + a13 * a24 * a313 + a13 * a25 * a312 + a13 * a26 * a311 - a14 * a23 * a313 - a15 * a22 * a313 - a15 * a23 * a312 - a16 * a21 * a313 - a16 * a22 * a312 - a16 * a23 * a311;
				double c1 = a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38 - a25 * a34 * a110 - a26 * a33 * a110 + a22 * a38 * a110 + a23 * a37 * a110 + a15 * a34 * a210 + a16 * a33 * a210 - a12 * a38 * a210 - a13 * a37 * a210 + a12 * a26 * a313 + a13 * a25 * a313 + a13 * a26 * a312 - a15 * a23 * a313 - a16 * a22 * a313 - a16 * a23 * a312;
				double c0 = -a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210 + a13 * a26 * a313 - a16 * a23 * a313;

				double roots[8];

				int n_roots = re3q3::bisect_sturm(c0, c1, c2, c3, c4, c5, c6, c7, c8, roots, 1e-10);

				Eigen::Matrix<double, 3, 3> A;
				for (int i = 0; i < n_roots; ++i) {
					double xs1 = roots[i];
					double xs2 = xs1 * xs1;
					double xs3 = xs1 * xs2;
					double xs4 = xs1 * xs3;

					A << a11 * xs2 + a12 * xs1 + a13, a14 * xs2 + a15 * xs1 + a16, a17 * xs3 + a18 * xs2 + a19 * xs1 + a110,
						a21 * xs2 + a22 * xs1 + a23, a24 * xs2 + a25 * xs1 + a26, a27 * xs3 + a28 * xs2 + a29 * xs1 + a210,
						a31 * xs3 + a32 * xs2 + a33 * xs1 + a34, a35 * xs3 + a36 * xs2 + a37 * xs1 + a38, a39 * xs4 + a310 * xs3 + a311 * xs2 + a312 * xs1 + a313;

					(*solutions)(0, i) = xs1;
					(*solutions)(1, i) = (A(1, 2) * A(0, 1) - A(0, 2) * A(1, 1)) / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
					(*solutions)(2, i) = (A(1, 2) * A(0, 0) - A(0, 2) * A(1, 0)) / (A(0, 1) * A(1, 0) - A(1, 1) * A(0, 0));
				}
				if (elim_var == 1) {
					solutions->row(0).swap(solutions->row(1));
				} else if (elim_var == 2) {
					solutions->row(0).swap(solutions->row(2));
				}

				refine_3q3(coeffs, solutions, n_roots);

				return n_roots;
			}
		}
	}
}