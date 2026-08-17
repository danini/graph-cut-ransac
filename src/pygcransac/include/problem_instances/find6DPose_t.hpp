#pragma once
#include "gcransac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"
#include "samplers/importance_sampler.h"
#include "samplers/adaptive_reordering_sampler.h"
#include "samplers/single_point_sampler.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;


// A method for estimating a the absolute pose given 2D-3D correspondences
template<class Pose6DEstimator>
int find6DPose_(
	// The 2D-3D correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &pose, 
	// Number of elements per point in the data matrix
    int element_number,
	// The spatial coherence weight used in the local optimization
	double spatial_coherence_weight, 
	// The inlier-outlier threshold
	double threshold, 
	// The RANSAC confidence. Typical values are 0.95, 0.99.
	double conf, 
	// Maximum iteration number. I do not suggest setting it to lower than 1000.
	int max_iters, 
	// Minimum iteration number.
	int min_iters, 
	// A flag to decide if SPRT should be used to speed up the model verification. 
	// It is not suggested if the inlier ratio is expected to be very low - it will fail in that case.
	// Otherwise, it leads to a significant speed-up. 
	bool use_sprt, 
	// Expected inlier ratio for SPRT. Default: 0.1
	double min_inlier_ratio_for_sprt,
	// The identifier of the used sampler. 
	// Options: 
	//	(0) Uniform sampler 
	// 	(1) PROSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function. 
	int sampler_id,
	// The identifier of the used neighborhood structure. 
	// 	(0) FLANN-based neighborhood. 
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	double neighborhood_size,
	// The variance parameter of the AR-Sampler. It is only used if that particular sampler is selected.
	double sampler_variance,
	// The number of RANSAC iterations done in the local optimization
	int lo_number, 
	// Refine the estimation with bundle adjustment
	bool do_bundle_adj)
{
	size_t num_tents = correspondences.size() / element_number;
	cv::Mat points(num_tents, element_number, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// Initializing the neighborhood graph if needed
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else
	{
		cv::Mat pointsForNeighborhood;
		// Copy the first 5 columns of the points to the pointsForNeighborhood
		points.colRange(0, 4).copyTo(pointsForNeighborhood);

		if (neighborhood_id == 0) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&pointsForNeighborhood, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	}

	// Apply Graph-cut RANSAC
	Pose6DEstimator estimator;
	Pose6D model;

	// Initialize the samplers	
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Progressive NAPSAC is not usable for the absolute pose problem.\n",
			sampler_id);
		return 0;
	}
	else if (sampler_id == 3)
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4)
    {
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            sampler_variance));
	}
	else if (sampler_id == 5)
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::SinglePointSampler(&points, 1));
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<Pose6DEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<Pose6DEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<Pose6DEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<Pose6DEstimator>,
			preemption::SPRTPreemptiveVerfication<Pose6DEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{
		GCRANSAC<Pose6DEstimator,
			AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}
	
	if (do_bundle_adj && statistics.inliers.size() > 3)
	{
		std::vector<Model> models = { model };
		estimator::solver::PnPBundleAdjustment bundle_adjuster;
		bundle_adjuster.estimateModel(
			points,
			&statistics.inliers[0],
			statistics.inliers.size(),
			models,
			nullptr);
		model.descriptor = models[0].descriptor;
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	pose.resize(12);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 4; j++) {
			pose[i * 4 + j] = (double)model.descriptor(i, j);
		}
	}

	// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
	// Therefore, the derived class's objects are not deleted automatically. 
	// This causes a memory leaking. I hate C++.
	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;

	AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
	delete neighborhood_graph_ptr;

	// The number of inliers found
	return num_inliers;
}

