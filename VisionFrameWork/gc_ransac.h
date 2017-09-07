#ifndef THEIA_SOLVERS_GC_RANSAC_H_
#define THEIA_SOLVERS_GC_RANSAC_H_

#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "estimator.h"
#include "inline_support.h"
#include "prosac_sampler.h"
#include "sample_consensus_estimator.h"

namespace theia
{
	// Estimate a model using PROSAC. The Estimate method is inherited, but for
	// PROSAC requires the data to be in sorted order by quality (with highest
	// quality at index 0).
	template <class ModelEstimator>
	class GCRansac : public SampleConsensusEstimator<ModelEstimator>
	{
	public:
		typedef typename ModelEstimator::Datum Datum;
		typedef typename ModelEstimator::Model Model;

		GCRansac(const RansacParameters& ransac_params, const ModelEstimator& estimator)
			: SampleConsensusEstimator<ModelEstimator>(ransac_params, estimator) {}
		~GCRansac() {}

		bool Initialize()
		{
			Sampler<Datum>* prosac_sampler =
				new ProsacSampler<Datum>(this->estimator_.SampleSize());
			QualityMeasurement* inlier_support =
				new InlierSupport(this->ransac_params_.error_thresh);
			return SampleConsensusEstimator<ModelEstimator>::Initialize(
				prosac_sampler, inlier_support);
		}
	};
}  // namespace theia

#endif  // THEIA_SOLVERS_PROSAC_H_
