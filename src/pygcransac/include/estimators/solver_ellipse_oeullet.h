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

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EllipseOuelletSolver : public SolverEngine
			{
			public:
				EllipseOuelletSolver()
				{
				}

				~EllipseOuelletSolver()
				{
				}

				// Determines if there is a chance of returning multiple models
				// when function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 1;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 3;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point

			protected:
				OLGA_INLINE bool fitEllipseFromNPoints(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_) const;

				OLGA_INLINE bool normalizePoints(
					const cv::Mat& data_, // The set of data points
					cv::Mat& normalized_data_, // The normalized data points
					double &x_mean_, // The mean of the x coordinates
					double &y_mean_, // The mean of the y coordinates
					double &x_stddev_, // The standard deviation of the x coordinates
					double &y_stddev_, // The standard deviation of the y coordinates
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_) const; // The size of the sample
					
				OLGA_INLINE bool getConicCenter(
					double A, double B, double C, double D, double E,
					double &xc, double &yc) const;

				OLGA_INLINE bool getConicAxes(
					double A, double B, double C, double D, double E, double F,
					double x0, double y0,
					double &a, double &b,
					double &orientation) const;
			};

			OLGA_INLINE bool EllipseOuelletSolver::getConicAxes(
				double A, double B, double C, double D, double E, double F,
				double x0, double y0,
				double &a, double &b,
				double &orientation) const
			{
				// 1) Translate x -> X+x0, y->Y+y0 => get new eq in X,Y:
				//    A'X^2 + B'X Y + C'Y^2 + F' = 0, no linear terms if (x0,y0) is correct center.
				//
				// Let's do that explicitly by substituting x=X+x0, y=Y+y0 in the original eq
				// For a short snippet, we can do it manually or systematically:
				//   Expand => gather X^2, X Y, Y^2, constant => A',B',C',F'.
				// For brevity, let's do a direct approach.

				// SHIFT:
				// x = X + x0, y = Y + y0
				// x^2 => X^2 + 2 x0 X + x0^2, etc.

				// We'll define short helpers: 
				auto sq = [](double v){return v*v;};

				// Coeff. of X^2: A'
				double A_prime = A; 
				// Coeff. of X Y: B_prime
				double B_prime = B; 
				// Coeff. of Y^2: C_prime
				double C_prime = C;

				// Now there's a cross-term from X^2 if we expand D*x = D*(X+x0), etc.
				// But we expect the linear terms vanish if (x0,y0) is truly the center.
				// The only leftover constant term is:
				//
				// F' = A*x0^2 + B*x0*y0 + C*y0^2 + D*x0 + E*y0 + F
				// because the boundary eq at (X,Y) => 0 => must match the old eq => 0 at (x,y).
				double F_prime = A*sq(x0) + B*x0*y0 + C*sq(y0)
							+ D*x0 + E*y0 + F; // no negative sign yet

				// If the ellipse is correctly centered, there should be no linear X or Y left.
				// We skip verifying that here. If not zero, x0,y0 wasn't correct.

				// 2) We want a factor alpha so alpha*(A'X^2 + B'X Y + C'Y^2) = 1 on the boundary.
				// The boundary in the new coords satisfies A'X^2 + B'X Y + C'Y^2 + F' = 0 => 
				// => A'X^2 + B'X Y + C'Y^2 = -F'.
				// So define alpha = 1 / (-F')  (assuming F'<0 for typical ellipse).
				// That yields alpha*(A'X^2 + B'X Y + C'Y^2) = 1.
				if (std::fabs(F_prime) < 1e-14) {
					std::cerr << "Degenerate or invalid center => cannot compute axes.\n";
					return false;
				}

				double signF = (F_prime < 0.0) ? -1.0 : +1.0; // typically F'<0 => signF = -1
				double alpha = 1.0 / (-F_prime); // if F' < 0, alpha>0 => valid

				// 3) Build the 2x2 matrix Q = alpha * [[A', B'/2],[B'/2, C']]
				//    Then we find the eigenvalues => 1/a^2, 1/b^2.
				Eigen::Matrix2d Q;
				Q << alpha*A_prime, alpha*(B_prime*0.5),
					alpha*(B_prime*0.5), alpha*C_prime;

				// 4) Eigen decomposition
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Q);
				if (es.info() != Eigen::Success) {
					std::cerr << "Eigen decomposition failed.\n";
					return false;
				}

				// The eigenvalues => lam1, lam2
				double lam1 = es.eigenvalues()(0);
				double lam2 = es.eigenvalues()(1);

				// If either is <= 0 => not an ellipse
				if (lam1 <= 1e-14 || lam2 <= 1e-14) {
					std::cerr << "Not an ellipse or degenerate.\n";
					return false;
				}

				double axis1 = std::sqrt(1.0 / lam1);
				double axis2 = std::sqrt(1.0 / lam2);

				// The corresponding eigenvectors
				Eigen::Vector2d evec1 = es.eigenvectors().col(0); // normalized
				Eigen::Vector2d evec2 = es.eigenvectors().col(1);

				// 5) Identify which eigenvalue => major axis vs. minor axis
				//    lam1 < lam2 => axis1 > axis2 => axis1 is the major axis
				//    orientation = angle of evec1 w.r.t. global X-axis if axis1 is major.
				if (axis1 >= axis2)
				{
					a = axis1;  // major
					b = axis2;  // minor

					// Orientation = angle of evec1 (the larger axis) in the XY-plane
					double ang = std::atan2(evec1(1), evec1(0));
					// keep angle in [0, 2π) if desired
					if (ang < 0) ang += 2.0 * M_PI;
					orientation = ang;
				}
				else
				{
					a = axis2;  // major
					b = axis1;  // minor

					// orientation = angle of evec2
					double ang = std::atan2(evec2(1), evec2(0));
					if (ang < 0) ang += 2.0 * M_PI;
					orientation = ang;
				}

				return true;
			}

			OLGA_INLINE bool EllipseOuelletSolver::getConicCenter(double A, double B, double C, double D, double E,
                           double &xc, double &yc) const
			{
				// Solve:
				//  [2A   B ] [xc] = [-D]
				//  [ B  2C ] [yc]   [-E]
				double det = 4.0 * A * C - B * B;
				if (std::fabs(det) < 1e-14)
					return false; // Degenerate or no unique center

				xc = ( -D*2.0*C - (-E)*B ) / (2.0 * C * 2.0 * A - B * B);
				yc = ( (-E)*2.0*A - (-D)*B ) / (2.0 * A * 2.0 * C - B * B);

				return true;
			}

			OLGA_INLINE bool EllipseOuelletSolver::normalizePoints(
				const cv::Mat& data_, // The set of data points
				cv::Mat& normalized_data_,
				double &x_mean_,
				double &y_mean_,
				double &x_stddev_,
				double &y_stddev_,
				const size_t *sample_, // The sample used for the estimation
				size_t sample_number_) const // The size of the sample
			{
				// The number of columns in the data matrix
				const size_t columns = data_.cols;
				// The pointer to the data matrix
				const double* data_ptr = reinterpret_cast<const double*>(data_.data);

				// Calculate the mean x and y
				x_mean_ = 0.0;
				y_mean_ = 0.0;
				for (size_t i = 0; i < sample_number_; ++i) {
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double* point_ptr = data_ptr + idx * columns;
					x_mean_ += point_ptr[0];
					y_mean_ += point_ptr[1];
				}
				x_mean_ /= sample_number_;
				y_mean_ /= sample_number_;

				// Calculate the standard deviation of x and y
				x_stddev_ = 0.0;
				y_stddev_ = 0.0;
				for (size_t i = 0; i < sample_number_; ++i) {
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double* point_ptr = data_ptr + idx * columns;
					double x = point_ptr[0] - x_mean_;  // Center first
					double y = point_ptr[1] - y_mean_;  // Center first
					x_stddev_ += x * x;
					y_stddev_ += y * y;
				}
				x_stddev_ = std::sqrt(x_stddev_ / sample_number_);
				y_stddev_ = std::sqrt(y_stddev_ / sample_number_);

				if (x_stddev_ < 1e-8 || y_stddev_ < 1e-8) { // Prevent division by zero
					return false;
				}

				normalized_data_.create(sample_number_, columns, CV_64F);
				double* normalized_data_ptr = reinterpret_cast<double*>(normalized_data_.data);

				// Normalize the points
				for (size_t i = 0; i < sample_number_; ++i) {
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double* point_ptr = data_ptr + idx * columns;
					normalized_data_ptr[i * columns] = (point_ptr[0] - x_mean_) / x_stddev_ ;
					normalized_data_ptr[i * columns + 1] = (point_ptr[1] - y_mean_) / y_stddev_;

					for (size_t j = 2; j < columns; ++j)
						normalized_data_ptr[i * columns + j] = point_ptr[j];
				}

				return true;
			}
					
			OLGA_INLINE bool EllipseOuelletSolver::fitEllipseFromNPoints(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Number of points
				const int N = static_cast<int>(sample_number_);
			
				// Initialize design matrix A (N x 6)
				Eigen::MatrixXd A(N, 6);
				A.setZero();
			
				const size_t columns = data_.cols;
				const double* data_ptr = reinterpret_cast<const double*>(data_.data);
			
				for (int i = 0; i < N; ++i)
				{
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double* point_ptr = data_ptr + idx * columns;
			
					double x = point_ptr[0];
					double y = point_ptr[1];
					double nx = point_ptr[2];
					double ny = point_ptr[3];
			
					// Construct the Ouellet constraint equation row
					double l1 = nx;
					double l2 = ny;
					double l3 = -nx * x - ny * y;
			
					A(i, 0) = l1 * l1;           // A
					A(i, 1) = 2 * l1 * l2;       // B
					A(i, 2) = 2 * l1 * l3;       // C
					A(i, 3) = l2 * l2;           // D
					A(i, 4) = 2 * l2 * l3;       // E
					A(i, 5) = l3 * l3;           // F
				}
			
				// Compute SVD of A
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
				Eigen::VectorXd params = svd.matrixV().col(5); // Last column of V (smallest singular value)
			
				// Construct the 3x3 conic matrix from parameters
				Eigen::Matrix3d Cs;
				Cs << params(0), params(1), params(2),
					  params(1), params(3), params(4),
					  params(2), params(4), params(5);
			
				// Invert to obtain the estimated conic parameters
				Eigen::Matrix3d C = Cs.inverse();
			
				// Extract the conic parameters
				double A_ = C(0, 0);
				double B_ = 2 * C(0, 1);
				double C_ = C(1, 1);
				double D_ = 2 * C(0, 2);
				double E_ = 2 * C(1, 2);
				double F_ = C(2, 2);
			
				// Ensure it defines an ellipse (B^2 - 4AC < 0)
				if (B_ * B_ - 4.0 * A_ * C_ >= 0.0)
					return false; // Not an ellipse
			
				// Compute ellipse center
				double xc, yc;
				if (!getConicCenter(A_, B_, C_, D_, E_, xc, yc))
					return false;
			
				// Compute ellipse axes and orientation
				double a_major, a_minor, phi;
				if (!getConicAxes(A_, B_, C_, D_, E_, F_, xc, yc, a_major, a_minor, phi))
					return false;
			
				// Store the model
				Model model;
				model.descriptor.resize(11, 1);
				model.descriptor << A_, B_, C_, D_, E_, F_, xc, yc, a_major, a_minor, phi;
				models_.emplace_back(model);
			
				return true;
			}

			OLGA_INLINE bool EllipseOuelletSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{			
				if (sample_number_ < sampleSize())
					return false;
				return fitEllipseFromNPoints(data_, sample_, sample_number_, models_, weights_);
				
			}
		}
	}
}