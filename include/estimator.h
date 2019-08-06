// Copyright (C) 2013 The Regents of the University of California (Regents).
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
//     * Neither the name of The Regents or University of California nor the
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
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#ifndef THEIA_SOLVERS_ESTIMATOR_H_
#define THEIA_SOLVERS_ESTIMATOR_H_

#include <vector>

namespace theia
{
	// Templated class for estimating a model for RANSAC. This class is purely a
	// virtual class and should be implemented for the specific task that RANSAC is
	// being used for. Two methods must be implemented: estimateModel and residual. All
	// other methods are optional, but will likely enhance the quality of the RANSAC
	// output.
	//
	// NOTE: RANSAC, ARRSAC, and other solvers work best if Datum and Model are
	// lightweight classes or structs.
	template <typename DatumType, typename ModelType> class Estimator
	{
	public:
		typedef DatumType Datum;
		typedef ModelType Model;

		Estimator() {}
		virtual ~Estimator() {}

		// Get the minimum number of samples needed to generate a model.
		virtual size_t sampleSize() const = 0;
		virtual size_t inlierLimit() const = 0;

		// Given a set of data points, estimate the model. Users should implement this
		// function appropriately for the task being solved. Returns true for
		// successful model estimation (and outputs model), false for failed
		// estimation. Typically, this is a minimal set, but it is not required to be.
		virtual bool estimateModel(const Datum& data,
			const int *sample, 
			std::vector<Model>* model) const = 0;

		// Estimate a model from a non-minimal sampling of the data. E.g. for a line,
		// use SVD on a set of points instead of constructing a line from two points.
		// By default, this simply implements the minimal case.
		virtual bool estimateModelNonminimal(const Datum& data,
			const int *sample,
			size_t sample_number,
			std::vector<Model>* model) const = 0;

		// Refine the model based on an updated subset of data, and a pre-computed
		// model. Can be optionally implemented.
		virtual bool refineModel(const std::vector<Datum>& data, Model* model) const 
		{
			return true;
		}

		// Given a model and a data point, calculate the error. Users should implement
		// this function appropriately for the task being solved.
		virtual double residual(const Datum& data, const Model& model) const = 0;
		virtual double squaredResidual(const Datum& data, const Model& model) const = 0;

		// Compute the residuals of many data points. By default this is just a loop
		// that calls residual() on each data point, but this function can be useful if
		// the errors of multiple points may be estimated simultanesously (e.g.,
		// matrix multiplication to compute the reprojection error of many points at
		// once).
		virtual std::vector<double> residuals(const std::vector<Datum>& data,
			const Model& model) const 
		{
			std::vector<double> residuals(data.size());
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
			for (int i = 0; i < data.size(); i++) 
			{
				residuals[i] = residual(data[i], model);
			}
			return residuals;
		}

		// Returns the set inliers of the data set based on the error threshold
		// provided.
		std::vector<int> getInliers(const std::vector<Datum>& data,
			const Model& model,
			double error_threshold) const 
		{
			std::vector<int> inliers;
			inliers.reserve(data.size());

			vector<bool> isInlier(data.size(), false); 
#ifdef _WIN32
			concurrency::parallel_for(0, (int)data.size(), [&](int i)
#else
			for (int i = 0; i < data.size(); i++)
#endif
			{
				isInlier[i] = residual(data[i], model) < error_threshold;
			}
#ifdef _WIN32
			);
#endif

			for (int i = 0; i < data.size(); ++i)
				if (isInlier[i])
					inliers.emplace_back(i);

			return inliers;
		}

		// Enable a quick check to see if the model is valid. This can be a geometric
		// check or some other verification of the model structure.
		virtual bool isValidModel(const Model& model) const { return true; }

		// Enable a quick check to see if the model is valid. This can be a geometric
		// check or some other verification of the model structure.
		virtual bool isValidModel(const Model& model,
			const Datum& data,
			const std::vector<int> &inliers,
			const double threshold) const
		{ return true; }
	};

}  // namespace theia

#endif  // THEIA_SOLVERS_ESTIMATOR_H_
