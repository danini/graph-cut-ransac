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


// A method for estimating an essential matrix given 2D-2D correspondences
template<class EssentialMatrixEstimator>
int findEssentialMatrix_(
	// The 2D-2D point correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &essential_matrix, 
	// The intrinsic camera matrix of the source image
	std::vector<double> &source_intrinsics, 
	// The intrinsic camera matrix of the destination image
	std::vector<double> &destination_intrinsics, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	// Minimum iteration number. I do not suggest setting it to lower than 50.
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
	//	(2) Progressive NAPSAC sampler. The correspondences should be ordered by quality (e.g., SNN ratio) prior to calling this function. 
	//	(3) Importance sampler from NG-RANSAC. The point probabilities should be provided.
	//	(4) Adaptive re-ordering sampler from Deep MAGSAC++. The point probabilities should be provided. 
	int sampler_id,
	// The identifier of the used neighborhood structure. 
	// 	(0) FLANN-based neighborhood. 
	// 	(1) Grid-based neighborhood.
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	// If (1) Grid is used, the size is the division number, e.g., 2 if we want to divide the image to 2 in along each axes (2*2 = 4 cells in total)
	double neighborhood_size,
	// The variance parameter of the AR-Sampler. It is only used if that particular sampler is selected.
	double sampler_variance,
	// The number of RANSAC iterations done in the local optimization
	int lo_number,
    // When usint SPRT
    bool do_final_iterated_least_squares)
{
	int num_tents = correspondences.size() / element_number;
	cv::Mat points(num_tents, element_number, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Using only the point coordinates and not the affine elements when constructing the neighborhood.
		cv::Mat point_coordinates = points(cv::Rect(0, 0, 4, points.rows));

		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_coordinates,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&point_coordinates, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
		
		// Checking if the neighborhood graph is initialized successfully.
		if (!neighborhood_graph->isInitialized())
		{
			AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
			delete neighborhood_graph_ptr;

			fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
			return 0;
		}
	}

	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_src(&source_intrinsics[0]);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> intrinsics_dst(&destination_intrinsics[0]);

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double threshold_normalizer =
		0.25 * (intrinsics_src(0,0) + intrinsics_src(1,1) + intrinsics_dst(0,0) + intrinsics_dst(1,1));

	cv::Mat normalized_points(points.size(), CV_64F);
	utils::normalizeCorrespondences(points,
		intrinsics_src,
		intrinsics_dst,
		normalized_points);

	// Apply Graph-cut RANSAC
	EssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	EssentialMatrix model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&points,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w1), // The width of the source image
				static_cast<double>(h1), // The height of the source image
				static_cast<double>(w2), // The width of the destination image
				static_cast<double>(h2) },  // The height of the destination image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
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
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling), 3 (NG-RANSAC sampler), 4 (AR-Sampler)\n",
			sampler_id);
		return 0;
	}
	
	// The local optimization sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points);

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}
	
	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<EssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<EssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<EssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<EssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<EssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
		if (!do_final_iterated_least_squares) {
			gcransac.settings.do_final_iterated_least_squares = false;
			gcransac.settings.max_graph_cut_number = 100;
        }

		// Start GC-RANSAC
		gcransac.run(normalized_points,
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
		GCRANSAC<EssentialMatrixEstimator, 
			AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(normalized_points,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	// Running bundle adjustment minimizing the pose error on the found inliers
	const size_t &inlier_number = statistics.inliers.size();
	if (statistics.inliers.size() > 5)
	{
		// The truncated least-squares threshold
		const double truncated_threshold = 3.0 / 2.0 * threshold / threshold_normalizer; 
		// The squared least-squares threshold
		const double squared_truncated_threshold = truncated_threshold * truncated_threshold; 

		// Initializing all weights to be zero
		std::vector<double> weights(inlier_number, 0.0);

		// Setting a weight for all inliers
		for (size_t inlier_idx = 0; inlier_idx < inlier_number; ++inlier_idx)
		{
			const size_t point_idx = statistics.inliers[inlier_idx];
			weights[inlier_idx] = 
				MAX(0, 1.0 - estimator.squaredResidual(normalized_points.row(point_idx), model) / squared_truncated_threshold);
		}

		std::vector<gcransac::Model> models;
		estimator::solver::EssentialMatrixBundleAdjustmentSolver bundleOptimizer;
		bundleOptimizer.estimateModel(
			normalized_points,
			&statistics.inliers[0],
			statistics.inliers.size(),
			models,
			&weights[0]);
			
		// Scoring function
		MSACScoringFunction<EssentialMatrixEstimator> scoring;
		scoring.initialize(squared_truncated_threshold, normalized_points.rows);
		// Score of the new model
		Score score;
		// Inliers of the new model
		std::vector<size_t> current_inliers;

		// Select the best model and update the inliers
		for (auto &tmp_model : models)
		{			
			current_inliers.clear();
			score = scoring.getScore(normalized_points, // All points
				tmp_model, // The current model parameters
				estimator, // The estimator 
				squared_truncated_threshold, // The current threshold
				current_inliers); // The current inlier set

			// Check if the updated model is better than the best so far
			if (statistics.score < score.value)
			{
				model.descriptor = tmp_model.descriptor;
				statistics.score = score.value;
				statistics.inliers.swap(current_inliers);
			}
		}
	}
	else
		model.descriptor = Eigen::Matrix3d::Identity();

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	essential_matrix.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			essential_matrix[i * 3 + j] = (double)model.descriptor(i, j);
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
