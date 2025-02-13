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

#define _USE_MATH_DEFINES

#include <math.h>
#include <cmath>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "solver_linear_model.h"
#include "model.h"
#include "../neighborhood/grid_neighborhood_graph.h"
#include "../samplers/uniform_sampler.h"

#include "GCRANSAC.h"


namespace gcransac
{
	namespace estimator
	{
		// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
		template<class _MinimalSolverEngine,  // The solver used for estimating the model from a minimal sample
			class _NonMinimalSolverEngine> // The solver used for estimating the model from a non-minimal sample
			class EllipseEstimator : public Estimator < cv::Mat, Model >
		{
		protected:
			// Minimal solver engine used for estimating a model from a minimal sample
			const std::shared_ptr<_MinimalSolverEngine> minimal_solver;

			// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
			const std::shared_ptr<_NonMinimalSolverEngine> non_minimal_solver;

		public:
			EllipseEstimator() :
				// Minimal solver engine used for estimating a model from a minimal sample
				minimal_solver(std::make_shared<_MinimalSolverEngine>()),
				// Non-minimal solver engine used for estimating a model from a bigger than minimal sample
				non_minimal_solver(std::make_shared<_NonMinimalSolverEngine>())
			{}

			~EllipseEstimator() {}

			// Return the minimal solver
			const _MinimalSolverEngine *getMinimalSolver() const
			{
				return minimal_solver.get();
			}

			// Return a mutable minimal solver
			_MinimalSolverEngine *getMutableMinimalSolver()
			{
				return minimal_solver.get();
			}

			// Return the minimal solver
			const _NonMinimalSolverEngine *getNonMinimalSolver() const
			{
				return non_minimal_solver.get();
			}

			// Return a mutable minimal solver
			_NonMinimalSolverEngine *getMutableNonMinimalSolver()
			{
				return non_minimal_solver.get();
			}

			// The size of a minimal sample_ required for the estimation
			static constexpr size_t maximumMinimalSolutions() {
				return _MinimalSolverEngine::maximumSolutions();
			}

			// The size of a non-minimal sample required for the estimation
			static constexpr size_t nonMinimalSampleSize() {
				return _NonMinimalSolverEngine::sampleSize();
			}

			// The size of a minimal sample required for the estimation
			static constexpr size_t sampleSize() {
				return _MinimalSolverEngine::sampleSize();
			}

			// A flag deciding if the points can be weighted when the non-minimal fitting is applied 
			static constexpr bool isWeightingApplicable() {
				return true;
			}

			// The size of a sample when doing inner RANSAC on a non-minimal sample
			OLGA_INLINE size_t inlierLimit() const {
				return 7 * sampleSize();
			}

			OLGA_INLINE bool estimateModel(const cv::Mat& data,
				const size_t *sample,
				std::vector<Model>* models) const
			{
				// Model calculation by the seven point algorithm
				constexpr size_t sample_size = sampleSize();

				// Estimate the model parameters by the minimal solver
				minimal_solver->estimateModel(data,
					sample,
					sample_size,
					*models);

				// The estimation was successfull if at least one model is kept
				return models->size() > 0;
			}

			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Model& model_) const
			{
				return squaredResidual(point_, model_.descriptor);
			}

			// The squared residual function used for deciding which points are inliers
			OLGA_INLINE double squaredResidual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				// descriptor_ = [A, B, C, D, E, F] or your stored (x_center, y_center, a, b, phi) 
				// For bounding-circle, you ideally want (x_center, y_center) and a = major axis.
				double px = point_.at<double>(0);
				double py = point_.at<double>(1);

				// 0) parse ellipse center and major axis from 'descriptor_'
				double x0 = descriptor_(6); // center x coordinate
				double y0 = descriptor_(7); // center y coordinate
				double a = descriptor_(8); // major axis
				double b = descriptor_(9); // minor axis

				const double threshold = 1.0; // or something based on your inlier tolerance

				// 1) Quick bounding circle reject
				if (quickRejectByBoundingCircle(px, py, x0, y0, a, threshold))
					return 1e6; // Return a "huge" distance => definitely outlier

				// 2) If we pass the filter, do the more expensive exact Euclidean distance
				double ed = euclideanDistanceToEllipse(point_, descriptor_);
				return ed * ed;
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Model& model_) const
			{
				return residual(point_, model_.descriptor);
			}

			OLGA_INLINE double algebraicDistance(
				const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				// descriptor_ = [A, B, C, D, E, F] in a 6x1 vector (or similar)
				double A = descriptor_(0);
				double B = descriptor_(1);
				double C = descriptor_(2);
				double D = descriptor_(3);
				double E = descriptor_(4);
				double F = descriptor_(5);

				// Extract the point coordinates from point_ (assuming CV_64F, single row)
				double x = point_.at<double>(0);
				double y = point_.at<double>(1);

				double val = A * x * x
						+ B * x * y
						+ C * y * y
						+ D * x
						+ E * y
						+ F;
				return std::fabs(val);
			}

			OLGA_INLINE double euclideanDistanceToEllipse(
				const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				// descriptor_ has 8 values: [A, B, C, D, E, F, xc, yc]
				double A = descriptor_(0);
				double B = descriptor_(1);
				double C = descriptor_(2);
				double D = descriptor_(3); // not used here, except for completeness
				double E = descriptor_(4); // not used here
				double F = descriptor_(5); // not used here
				double xc = descriptor_(6);
				double yc = descriptor_(7);
				double a = descriptor_(8);
				double b = descriptor_(9);
				double theta = descriptor_(10);

				// 2) Transform the point to the ellipse local frame
				double px = point_.at<double>(0);
				double py = point_.at<double>(1);

				// Translate by -(xc, yc)
				double X = px - xc;
				double Y = py - yc;

				// Rotate by -theta
				double cosT = std::cos(theta);
				double sinT = std::sin(theta);

				double xprime =  X*cosT +  Y*sinT;
				double yprime = -X*sinT +  Y*cosT;

				// 5) Use numeric solve (Newton) to find the boundary point (a cos t, b sin t)
				//    that minimizes the squared distance to (xprime, yprime).

				// A small helper function:
				auto distanceToEllipseLocal = [&](double Xp, double Yp, double aa, double bb)
				{
					// Minimize f(t) = (aa cos t - Xp)^2 + (bb sin t - Yp)^2.
					// We'll do ~5 Newton iterations starting from t0 = atan2(bb*Yp, aa*Xp).
					// If (Xp,Yp)=(0,0), just pick t0=0.

					double xguess = aa != 0.0 ? Xp/aa : 0.0;
					double yguess = bb != 0.0 ? Yp/bb : 0.0;
					double t = 0.0;
					if (std::fabs(xguess) + std::fabs(yguess) > 1e-12)
						t = std::atan2(yguess, xguess);

					const int MAX_ITERS = 5;
					for (int i = 0; i < MAX_ITERS; ++i)
					{
						double ct = std::cos(t), st = std::sin(t);
						double fx = aa*ct - Xp; 
						double fy = bb*st - Yp;
						// f(t) = fx^2 + fy^2
						// f'(t) = 2 [fx * d(fx)/dt + fy * d(fy)/dt]
						// d(fx)/dt = -aa sin t
						// d(fy)/dt =  bb cos t
						double f_prime = 2.0*( fx*(-aa*st) + fy*( bb*ct ) );

						// f''(t) = 2 [ ( d(fx)/dt )^2 + fx * d^2(fx)/dt^2 + ( d(fy)/dt )^2 + fy * d^2(fy)/dt^2 ]
						// But we'll do a simpler approximate Newton: just denominator ~ 2 [(-aa st)^2 + (bb ct)^2].
						double dfx_dt = -aa*st;
						double dfy_dt =  bb*ct;
						double f_denom_approx = 2.0*( dfx_dt*dfx_dt + dfy_dt*dfy_dt );

						if (std::fabs(f_denom_approx) < 1e-14)
							break;

						double delta = f_prime / f_denom_approx;
						t -= delta;

						// keep t in [-pi, pi, or any real], it doesn't matter for cos/sin
					}

					// After iteration, compute final distance
					double ct = std::cos(t), st = std::sin(t);
					double dx = aa*ct - Xp;
					double dy = bb*st - Yp;
					return std::sqrt(dx*dx + dy*dy);
				};

				// 6) Return the distance in the local frame
				double dist = distanceToEllipseLocal(xprime, yprime, a, b);
				return dist;

				// 3) Sample many points on ellipse boundary and track min. distance to (xprime, yprime)
				/*const int M = 180; // e.g. sample every 4 degrees => 90 steps over [0..2π)
				double minDist = 1e15;

				// We'll sample t from 0..2π in M steps
				for (int i = 0; i < M; ++i) 
				{
					double t = (2.0 * M_PI * i) / M;
					double ex = a * std::cos(t);  // ellipse boundary X'
					double ey = b * std::sin(t);  // ellipse boundary Y'
					double dx = (ex - xprime);
					double dy = (ey - yprime);
					double dist = dx*dx + dy*dy;  // squared distance
					if (dist < minDist)
						minDist = dist;
				}*/

				// Return the sqrt of the minimal squared distance
				//return std::sqrt(minDist);
			}

			OLGA_INLINE bool quickRejectByBoundingCircle(
				double px, double py,   // point coords
				double x_center, 
				double y_center,
				double a_majorAxis,     // the bigger of (a,b)
				double threshold) const
			{
				// 1) Compute squared distance from point to ellipse center
				double dx = px - x_center;
				double dy = py - y_center;
				double distSq = dx*dx + dy*dy;

				// 2) Compare with (a + threshold)^2
				//    or maybe (a + threshold*2) if you want more conservative margin
				double bound = a_majorAxis + threshold;
				double boundSq = bound * bound;

				// If the point is outside that bounding circle by enough margin,
				// it cannot be on or near the ellipse.
				return distSq > boundSq;
			}

			// The residual function used for deciding which points are inliers
			OLGA_INLINE double residual(const cv::Mat& point_,
				const Eigen::MatrixXd& descriptor_) const
			{
				// descriptor_ = [A, B, C, D, E, F] or your stored (x_center, y_center, a, b, phi) 
				// For bounding-circle, you ideally want (x_center, y_center) and a = major axis.
				double px = point_.at<double>(0);
				double py = point_.at<double>(1);

				// 0) parse ellipse center and major axis from 'descriptor_'
				double x0 = descriptor_(6); // center x coordinate
				double y0 = descriptor_(7); // center y coordinate
				double a = descriptor_(8); // major axis
				double b = descriptor_(9); // minor axis

				const double threshold = 1.0; // or something based on your inlier tolerance

				// 1) Quick bounding circle reject
				if (quickRejectByBoundingCircle(px, py, x0, y0, a, threshold))
					return 1e6; // Return a "huge" distance => definitely outlier

				// 2) If we pass the filter, do the more expensive exact Euclidean distance
				double ed = euclideanDistanceToEllipse(point_, descriptor_);
				return ed;
			}

			// Validate the model by checking the number of inlier with symmetric epipolar distance
			// instead of Sampson distance. In general, Sampson distance is more accurate but less
			// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
			// every so-far-the-best model is checked if it has enough inlier with symmetric
			// epipolar distance as well. 
			bool isValidModel(Model& model_,
				const cv::Mat& data_,
				const std::vector<size_t> &inliers_,
				const size_t *minimal_sample_,
				const double threshold_,
				bool &model_updated_) const
			{
				return true;
			}

			inline bool estimateModelNonminimal(
				const cv::Mat& data_,
				const size_t *sample_,
				const size_t &sample_number_,
				std::vector<Model>* models_,
				const double *weights_ = nullptr) const
			{
				// Estimate the model parameters by the minimal solver
				non_minimal_solver->estimateModel(data_,
					sample_,
					sample_number_,
					*models_);

				// The estimation was successfull if at least one model is kept
				return models_->size() > 0;
			}

		};
	}
}