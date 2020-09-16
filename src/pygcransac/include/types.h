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
#include "rigid_transformation_estimator.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_p3p.h"
#include "solver_epnp_lm.h"
#include "solver_dls_pnp.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"
#include "solver_rigid_transformation_svd.h"

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
			estimator::solver::EPnPLM> // The solver used for fitting a model to a non-minimal sample
			DefaultPnPEstimator;

		// The default estimator for PnP fitting
		typedef estimator::RigidTransformationEstimator<estimator::solver::RigidTransformationSVDBasedSolver, // The solver used for fitting a model to a minimal sample
			estimator::solver::RigidTransformationSVDBasedSolver> // The solver used for fitting a model to a non-minimal sample
			DefaultRigidTransformationEstimator;
	}
}