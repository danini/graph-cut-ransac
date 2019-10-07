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

#include <vector>
#include <numeric>

#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"
#include "perspective_n_point_estimator.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_p3p.h"
#include "solver_dls_pnp.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"

namespace gcransac
{
	namespace utils
	{
		// The default estimator for fundamental matrix fitting
		typedef estimator::FundamentalMatrixEstimator<estimator::solver::FundamentalMatrixSevenPointSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultFundamentalMatrixEstimator;

		// The default estimator for homography fitting
		typedef estimator::RobustHomographyEstimator<estimator::solver::HomographyFourPointSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::HomographyFourPointSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultHomographyEstimator;

		// The default estimator for essential matrix fitting
		typedef estimator::EssentialMatrixEstimator<estimator::solver::EssentialMatrixFivePointSteweniusSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::EssentialMatrixFivePointSteweniusSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultEssentialMatrixEstimator;

		// The default estimator for PnP fitting
		typedef estimator::PerspectiveNPointEstimator<estimator::solver::P3PSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::DLSPnP> // The solver used for fitting a model to a non-minimal sample
			DefaultPnPEstimator;

		struct Settings {
			bool do_final_iterated_least_squares, // Flag to decide a final iterated least-squares fitting is needed to polish the output model parameters.
				do_local_optimization, // Flag to decide if local optimization is needed
				do_graph_cut, // Flag to decide of graph-cut is used in the local optimization
				use_inlier_limit; // Flag to decide if an inlier limit is used in the local optimization to speed up the procedure

			size_t desired_fps; // The desired FPS

			size_t max_local_optimization_number, // Maximum number of local optimizations
				min_iteration_number_before_lo, // Minimum number of RANSAC iterations before applying local optimization
				min_iteration_number, // Minimum number of RANSAC iterations
				max_iteration_number, // Maximum number of RANSAC iterations
				max_unsuccessful_model_generations, // Maximum number of unsuccessful model generations
				max_least_squares_iterations, // Maximum number of iterated least-squares iterations
				max_graph_cut_number, // Maximum number of graph-cuts applied in each iteration
				core_number; // Number of parallel threads

			double confidence, // Required confidence in the result
				neighborhood_sphere_radius, // The radius of the ball used for creating the neighborhood graph
				threshold, // The inlier-outlier threshold
				spatial_coherence_weight; // The weight of the spatial coherence term

			Settings() :
				do_final_iterated_least_squares(true),
				do_local_optimization(true),
				do_graph_cut(true),
				use_inlier_limit(false),
				desired_fps(-1),
				max_local_optimization_number(10),
				max_graph_cut_number(10),
				max_least_squares_iterations(10),
				min_iteration_number_before_lo(20),
				min_iteration_number(20),
				neighborhood_sphere_radius(20),
				max_iteration_number(std::numeric_limits<size_t>::max()),
				max_unsuccessful_model_generations(100),
				core_number(1),
				spatial_coherence_weight(0.14),
				threshold(2.0),
				confidence(0.95)
			{

			}
		};

		struct RANSACStatistics
		{
			size_t graph_cut_number,
				local_optimization_number,
				iteration_number,
				neighbor_number;

			std::string main_sampler_name,
				local_optimizer_sampler_name;

			double processing_time;

			std::vector<size_t> inliers;

			RANSACStatistics() :
				graph_cut_number(0),
				local_optimization_number(0),
				iteration_number(0),
				neighbor_number(0),
				processing_time(0.0)
			{

			}
		};
	}
}