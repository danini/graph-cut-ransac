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
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class EllipsePointBasedSolver : public SolverEngine
			{
			public:
				EllipsePointBasedSolver()
				{
				}

				~EllipsePointBasedSolver()
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
					return 5;
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
				OLGA_INLINE bool fitEllipseFrom5Points(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_) const;
					
				OLGA_INLINE bool fitEllipseFromNPoints(
					const cv::Mat& data_,
					const size_t *sample_,
					size_t sample_number_,
					std::vector<Model> &models_,
					const double *weights_) const;
					
				OLGA_INLINE bool getConicCenter(
					double A, double B, double C, double D, double E,
					double &xc, double &yc) const;

				OLGA_INLINE bool getConicAxes(
					double A, double B, double C, double D, double E, double F,
					double x0, double y0,
					double &a, double &b,
					double &orientation) const;
			};

			OLGA_INLINE bool EllipsePointBasedSolver::getConicAxes(
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

			OLGA_INLINE bool EllipsePointBasedSolver::getConicCenter(double A, double B, double C, double D, double E,
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
					
			OLGA_INLINE bool EllipsePointBasedSolver::fitEllipseFromNPoints(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{					
				// 2) Build the design matrix D (N x 6), row i = [x_i^2, x_i y_i, y_i^2, x_i, y_i, 1].
				const int N = static_cast<int>(sample_number_);
				Eigen::MatrixXd D(N, 6);
				D.setZero();

				const size_t columns = data_.cols;
				const double* data_ptr = reinterpret_cast<const double*>(data_.data);

				for (int i = 0; i < N; ++i)
				{
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double* point_ptr = data_ptr + idx * columns;

					double x = point_ptr[0];
					double y = point_ptr[1];

					D(i,0) = x*x;  // A
					D(i,1) = x*y;  // B
					D(i,2) = y*y;  // C
					D(i,3) = x;    // D
					D(i,4) = y;    // E
					D(i,5) = 1.0;  // F
				}

				// 3) Scatter matrix S = D^T * D (6x6)
				Eigen::MatrixXd S = D.transpose() * D;

				// Partition S into blocks:
				//   Sqq = S(0..2,0..2), S(0..2,3..5),
				//   S(3..5,0..2), S(3..5,3..5).
				// where the first 3 columns = {A,B,C}, last 3 = {D,E,F}.
				Eigen::Matrix3d Sqq = S.block<3,3>(0,0);
				Eigen::Matrix3d Sql = S.block<3,3>(0,3);
				Eigen::Matrix3d Slq = S.block<3,3>(3,0);
				Eigen::Matrix3d Sll = S.block<3,3>(3,3);

				// 4) Solve for the "quadratic part" a_q = (A,B,C) by:
				//      (Sqq - Sql * Sll^-1 * Slq)* a_q = lambda*C * a_q
				// where C = [ [0,  0,  2],
				//             [0, -1,  0],
				//             [2,  0,  0] ]
				//
				// Let M = Sqq - Sql * (Sll^-1) * Slq.
				Eigen::Matrix3d Sll_inv = Sll.inverse();
				Eigen::Matrix3d M = Sqq - Sql * Sll_inv * Slq;

				// Fitzgibbon's constraint matrix
				Eigen::Matrix3d Cc; // "C" in the literature
				Cc << 0,  0,  2,
					0, -1,  0,
					2,  0,  0;

				// We want to solve the generalized eigenproblem:
				//    M a_q = lambda * Cc a_q
				// => M^-1 Cc a_q = lambda a_q
				// We'll do an ordinary Eigen decomposition on T = M^-1 * Cc
				Eigen::Matrix3d Minv = M.inverse();
				Eigen::Matrix3d T = Minv * Cc;

				Eigen::EigenSolver<Eigen::Matrix3d> es(T);
				if (es.info() != Eigen::Success)
					return false;

				// We'll look through the three eigenvalues and pick the real, positive solution
				// that yields an ellipse (4AC - B^2 > 0).
				Eigen::Vector3cd evals = es.eigenvalues();
				Eigen::Matrix3cd evecs = es.eigenvectors();

				int bestIndex = -1;
				double bestVal = 1e20;
				for (int i = 0; i < 3; ++i)
				{
					// We only consider real eigenvalues
					if (std::abs(evals[i].imag()) > 1e-12) 
						continue;
					double lambda = evals[i].real();
					if (lambda > 0.0 && lambda < bestVal)
					{
						bestVal = lambda;
						bestIndex = i;
					}
				}
				if (bestIndex < 0)
					return false; // no valid ellipse eigenvalue found

				Eigen::Vector3d a_q = evecs.col(bestIndex).real();

				// 5) Solve for the linear part a_l = (D,E,F) = -Sll^-1 * Slq * a_q
				Eigen::Vector3d a_l = - Sll_inv * Slq * a_q;

				// Our raw solution is a = (A,B,C,D,E,F)^T = (a_q, a_l).
				Eigen::VectorXd a_conic(6);
				a_conic << a_q(0), a_q(1), a_q(2), a_l(0), a_l(1), a_l(2);

				// 6) Scale so that 4AC - B^2 = 1
				double A = a_conic(0);
				double B = a_conic(1);
				double C = a_conic(2);
				double val = 4.0 * A * C - B*B;
				if (std::fabs(val) < 1e-14)
					return false; // degenerate
				double scale = 1.0 / std::sqrt(std::fabs(val));
				// If val < 0 => not an ellipse => you can skip or forcibly flip. We'll skip:
				if (val < 0.0)
					return false; // indicates hyperbola or degeneracy

				a_conic *= scale;  // now 4AC - B^2 = 1

				// 7) Extract final (A,B,C,D,E,F)
				A = a_conic(0);
				B = a_conic(1);
				C = a_conic(2);
				double D_ = a_conic(3);
				double E_ = a_conic(4);
				double F_ = a_conic(5);

				// 8) Quick ellipse check => B^2 - 4AC < 0
				if (B*B - 4.0*A*C >= 0.0)
					return false;

				// 9) (Optional) compute center, axes, orientation for convenience
				double xc, yc;
				if (!getConicCenter(A, B, C, D_, E_, xc, yc))
					return false;

				double a_major, a_minor, phi;
				if (!getConicAxes(A, B, C, D_, E_, F_, xc, yc, a_major, a_minor, phi))
					return false;

				// 10) Store model
				Model model;
				// We'll store (A,B,C,D,E,F,xc,yc,a_major,a_minor,phi) in descriptor
				model.descriptor.resize(11,1);
				model.descriptor << A, B, C, D_, E_, F_, 
									xc, yc, 
									a_major, a_minor, 
									phi;
				models_.emplace_back(model);

				return true;
			}

			// Fit an *exact* ellipse through 5 points by solving the linear system
			// D * a = 0,  then enforcing  4AC - B^2 = 1  for the final scale.
			OLGA_INLINE bool EllipsePointBasedSolver::fitEllipseFrom5Points(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// --- 1. Build the design matrix D (5 x 6) ---
				// Each row i = [ x_i^2,  x_i y_i,  y_i^2,  x_i,  y_i,  1 ]
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> D(sample_number_, 6);
				D.setZero();

				const size_t columns = data_.cols;
				const double *data_ptr = reinterpret_cast<const double*>(data_.data);
				for (size_t i = 0; i < sample_number_; ++i)
				{
					const size_t idx = (sample_ == nullptr) ? i : sample_[i];
					const double *point_ptr = data_ptr + idx * columns;

					double x = point_ptr[0];
					double y = point_ptr[1];

					// Fill row i of D
					D(i, 0) = x*x;   // A
					D(i, 1) = x*y;   // B
					D(i, 2) = y*y;   // C
					D(i, 3) = x;     // D
					D(i, 4) = y;     // E
					D(i, 5) = 1.0;   // F
				}

				// --- 2. Solve D * a = 0 via SVD to find the null space vector 'a' ---
				// We'll get a 6-vector (A,B,C,D,E,F) up to a scalar factor.
				// If the rank of D is exactly 5, the null space is 1-dimensional => last column of V.
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullV);
				// The SVD is: D = U * S * V^T
				// The null space vector is the column of V corresponding to the smallest singular value.
				// For a 5x6 matrix of rank 5, that should be V.col(5).
				Eigen::VectorXd a = svd.matrixV().col(5);

				// Optional: If sample_number_ > 5 (overdetermined),
				// you might pick the column for the smallest singular value. 
				// This code remains the same in JacobiSVD as long as 'ComputeFullV' is used.

				// a = [ A, B, C, D, E, F ]
				double A = a(0);
				double B = a(1);
				double C = a(2);

				// --- 3. Fix scale so that 4AC - B^2 = 1. ---
				// The raw 'a' is determined only up to a nonzero scale factor.
				const double val = 4.0 * A * C - B * B;  // = (4AC - B^2)
				if (std::fabs(val) < 1e-14) // Degenerate conic => no valid ellipse
					return false;

				// We want 4AC - B^2 = +1 => scale the vector
				double scale = 1.0 / std::sqrt(std::fabs(val));
				// If val < 0 => it would be a hyperbola. If you *know* you have an ellipse,
				// but the sign is negative, you can do scale = -scale. 
				// Typically, if val < 0 => no ellipse. We'll just check after scaling:
				if (val < 0) scale = -scale;

				a *= scale;

				// Now A, B, C, D, E, F are scaled so that 4AC - B^2 = 1
				A = a(0); // updated
				B = a(1);
				C = a(2);
				double D_ = a(3);
				double E_ = a(4);
				double F_ = a(5);

				/*std::cout << D << std::endl;
				std::cout << "A: " << A << " B: " << B << " C: " << C << " D: " << D_ << " E: " << E_ << " F: " << F_ << std::endl;
				std::cout << 4.0 * A * C - B * B << std::endl;
				std::cout << B * B - 4.0 * A * C << std::endl;*/

				// --- 4. Check ellipse condition => B^2 - 4AC < 0. ---
				// Because we enforced 4AC - B^2 = 1 => B^2 - 4AC = -1 < 0 => ellipse
				// Just do a safety check:
				if (B * B - 4.0 * A * C >= 0.0)
					return false; // Not an ellipse (or degenerate)

				// 9) Compute center, axes, orientation for convenience
				double xc, yc;
				if (!getConicCenter(A, B, C, D_, E_, xc, yc))
					return false;

				double a_major, a_minor, phi;
				if (!getConicAxes(A, B, C, D_, E_, F_, xc, yc, a_major, a_minor, phi))
					return false;

				//std::cout << "Center: " << xc << " " << yc << std::endl;
				//std::cout << "Major: " << a_major << " Minor: " << a_minor << std::endl;

				// --- 5. Store the final conic in the model ---
				Model model;
				model.descriptor.resize(11, 1);
				model.descriptor << A, B, C, D_, E_, F_, xc, yc, a_major, a_minor, phi;
				models_.emplace_back(model);

				return true;
			}


			OLGA_INLINE bool EllipsePointBasedSolver::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{			
				if (sample_number_ < sampleSize())
					return false;
				if (sample_number_ == sampleSize())
					return fitEllipseFrom5Points(data_, sample_, sample_number_, models_, weights_);
				else
					return fitEllipseFromNPoints(data_, sample_, sample_number_, models_, weights_);
				return false;
				
			}
		}
	}
}