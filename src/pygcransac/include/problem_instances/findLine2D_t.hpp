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


// A method for estimating a 2D line from a set of 2D points
template<class Line2dEstimator>
int findLine2D_(
 	// The 2D points in the image
	std::vector<double>& points,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 2d line
	std::vector<double> &line, 
	// The image size
	int w, int h,
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
	int sampler_id,
	// The identifier of the used neighborhood structure. 
	// 	(0) FLANN-based neighborhood. 
	// 	(1) Grid-based neighborhood.
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	// If (1) Grid is used, the size is the division number, e.g., 2 if we want to divide the image to 2 in along each axes (2*2 = 4 cells in total)
	double neighborhood_size,
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	// The number of points provided
	const size_t &num_points = points.size() / element_number;
	// The matrix containing the points that will be passed to GC-RANSAC
	cv::Mat point_matrix(num_points, element_number, CV_64F, &points[0]);

	// Initializing the neighborhood structure based on the provided paramereters
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// The cell size or radius-search radius of the neighborhood graph
	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat empty_point_matrix(0, 2, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<2>(&empty_point_matrix, // The input points
			{ 0, // The cell size along axis X
				0 }, // The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_matrix, // The input points
				{ w / static_cast<double>(cell_number_in_neighborhood_graph_), // The cell size along axis X
					h / static_cast<double>(cell_number_in_neighborhood_graph_) }, // The cell size along axis Y
				cell_number_in_neighborhood_graph_)); // The cell number along every axis
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&point_matrix, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (Grid-based), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	}

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood_graph->isInitialized())
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Initializing the line estimator
	Line2dEstimator estimator;

	// Initializing the model object
	Line2D model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&point_matrix));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&point_matrix, estimator.sampleSize()));
	else if (sampler_id == 2) // Initializing a Progressive NAPSAC sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProgressiveNapsacSampler<4>(&point_matrix,
			{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
								// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
			estimator.sampleSize(), // The size of a minimal sample
			{ static_cast<double>(w), // The width of the image
				static_cast<double>(h) },  // The height of the image
			0.5)); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 
	else
	{
		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);

		// It is ugly: the unique_ptr does not check for virtual descructors in the base class.
		// Therefore, the derived class's objects are not deleted automatically. 
		// This causes a memory leaking. I hate C++.
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

 	// The local optimization sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&point_matrix);

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<Line2dEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<Line2dEstimator> preemptive_verification(
			point_matrix,
			estimator);

		GCRANSAC<Line2dEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<Line2dEstimator>,
			preemption::SPRTPreemptiveVerfication<Line2dEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(point_matrix,
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
		GCRANSAC<Line2dEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

		// Start GC-RANSAC
		gcransac.run(point_matrix,
			estimator,
			main_sampler.get(),
			&local_optimization_sampler,
			neighborhood_graph.get(),
			model);

		statistics = gcransac.getRansacStatistics();
	}

	line.resize(3);

	for (int i = 0; i < 3; i++) {
		line[i] = model.descriptor(i);
	}

	inliers.resize(num_points);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_points; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
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
