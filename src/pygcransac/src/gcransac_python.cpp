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

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/homography_affine_correspondence_estimator.h"
#include "estimators/essential_estimator.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

// A method for estimating a 2D line from a set of 2D points
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
	const size_t &num_points = points.size() / 2;

	// The matrix containing the points that will be passed to GC-RANSAC
	cv::Mat point_matrix(num_points, 2, CV_64F, &points[0]);

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
	utils::Default2DLineEstimator estimator;

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
	inlier_selector::EmptyInlierSelector<utils::Default2DLineEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator> preemptive_verification(
			point_matrix,
			estimator);

		GCRANSAC<utils::Default2DLineEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::Default2DLineEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::Default2DLineEstimator>> gcransac;
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
		GCRANSAC<utils::Default2DLineEstimator, AbstractNeighborhood> gcransac;
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

// A method for estimating a the absolute pose given 2D-3D correspondences
int find6DPose_(
	// The 2D-3D correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &pose, 
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
	int lo_number)
{
	size_t num_tents = correspondences.size() / 5;
	cv::Mat points(num_tents, 5, CV_64F, &correspondences[0]);

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
		if (neighborhood_id == 0) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are 0 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	}

	// Apply Graph-cut RANSAC
	utils::DefaultPnPEstimator estimator;
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
	inlier_selector::EmptyInlierSelector<utils::DefaultPnPEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultPnPEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultPnPEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultPnPEstimator>> gcransac;
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
		GCRANSAC<utils::DefaultPnPEstimator,
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

// A method for estimating a rigid translation between two point clouds
int findRigidTransform_(
	// The first point cloud
	std::vector<double>& correspondences, 
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &pose, 
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
	int sampler_id,
	// The identifier of the used neighborhood structure. 
	// 	(0) FLANN-based neighborhood. 
	int neighborhood_id,
	// The size of the neighborhood.
	// If (0) FLANN is used, the size if the Euclidean distance in the correspondence space
	double neighborhood_size,
	// The variance parameter of the AR-Sampler. It is only used if that particular sampler is selected.
	double sampler_variance,
	// A flag determining if space partitioning from 
	// Barath, Daniel, and Gabor Valasek. "Space-Partitioning RANSAC." arXiv preprint arXiv:2111.12385 (2021).
	// should be used to speed up the model verification.
	bool use_space_partitioning,
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	size_t num_tents = correspondences.size() / 6;
	cv::Mat points(num_tents, 6, CV_64F, &correspondences[0]);
	size_t iterations = 0;
	double box_size[6];

	if (neighborhood_id == 1 &&
		use_space_partitioning)
	{
		fprintf(stderr, "When space partitioning is turned on, only neighborhood_id == 0 is accepted.\n");
		return 0;		
	}

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);
	
	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon() &&
		sampler_id != 2 &&
		!use_space_partitioning)
	{
		cv::Mat emptyPoints(0, 6, CV_64F);

		if (neighborhood_id == 0) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<6>(&points,
				{ 	0.0 / static_cast<double>(cell_number_in_neighborhood_graph_),
					0.0 / static_cast<double>(cell_number_in_neighborhood_graph_),
					0.0 / static_cast<double>(cell_number_in_neighborhood_graph_),
					0.0 / static_cast<double>(cell_number_in_neighborhood_graph_),
					0.0 / static_cast<double>(cell_number_in_neighborhood_graph_),
					0.0 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&emptyPoints, neighborhood_size));
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are: 0 (Grid-based neighborhood), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	} else // Initializing a grid-based neighborhood graph
	{
		if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
		else if (neighborhood_id == 0)
		{
			// Calculating the box dimensions.
			size_t coord_idx = 0;
			for (size_t dimension = 0; dimension < 6; ++dimension)
				box_size[dimension] = points.at<double>(0, dimension); //correspondences[coord_idx++];
			for (size_t point_idx = 1; point_idx < num_tents; ++point_idx)
				for (size_t dimension = 0; dimension < 6; ++dimension)
					box_size[dimension] = MAX(points.at<double>(point_idx, dimension), box_size[dimension]);

			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<6>(&points,
				{ 	box_size[0] / static_cast<double>(cell_number_in_neighborhood_graph_),
					box_size[1] / static_cast<double>(cell_number_in_neighborhood_graph_),
					box_size[2] / static_cast<double>(cell_number_in_neighborhood_graph_),
					box_size[3] / static_cast<double>(cell_number_in_neighborhood_graph_),
					box_size[4] / static_cast<double>(cell_number_in_neighborhood_graph_),
					box_size[5] / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		}
		else
		{
			fprintf(stderr, "Unknown neighborhood-graph identifier: %d. The accepted values are: 0 (Grid-based neighborhood), 1 (FLANN-based neighborhood)\n",
				neighborhood_id);
			return 0;
		}
	}

	// Apply Graph-cut RANSAC
	utils::DefaultRigidTransformationEstimator estimator;
	RigidTransformation model;

	// Initialize the samplers	
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
	else if (sampler_id == 2)  // Initializing a NAPSAC sampler. This requires the points to be ordered according to the quality.
	{
		if (neighborhood_id != 1 ||
			neighborhood_size <= std::numeric_limits<double>::epsilon())
		{
			fprintf(stderr, "For NAPSAC sampler, the neighborhood type must be 1 and its size >0.\n");
			return 0;
		}

		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::NapsacSampler<neighborhood::FlannNeighborhoodGraph>(
			&points, dynamic_cast<neighborhood::FlannNeighborhoodGraph *>(neighborhood_graph.get())));
	} else if (sampler_id == 3) // Initializing the importance sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::ImportanceSampler(&points, 
            point_probabilities,
            estimator.sampleSize()));
	else if (sampler_id == 4) // Initializing the adaptive re-ordering sampler
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (NAPSAC sampling), 3 (Importance sampler), 4 (Adaptive re-ordering sampler)\n",
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
	inlier_selector::EmptyInlierSelector<utils::DefaultRigidTransformationEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultRigidTransformationEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultRigidTransformationEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultRigidTransformationEstimator>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations

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
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<utils::DefaultRigidTransformationEstimator> preemptive_verification;

		if (use_space_partitioning)
		{			
			// The space partitioning algorithm to accelerate inlier selection
			inlier_selector::SpacePartitioningRANSAC<utils::DefaultRigidTransformationEstimator, AbstractNeighborhood> inlier_selector(
				neighborhood_graph.get());

			GCRANSAC<utils::DefaultRigidTransformationEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultRigidTransformationEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultRigidTransformationEstimator>,
				inlier_selector::SpacePartitioningRANSAC<utils::DefaultRigidTransformationEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations

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
		} else
		{
			// Initializing the fast inlier selector object
			inlier_selector::EmptyInlierSelector<utils::DefaultRigidTransformationEstimator, 
				AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultRigidTransformationEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultRigidTransformationEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultRigidTransformationEstimator>,
				inlier_selector::EmptyInlierSelector<utils::DefaultRigidTransformationEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
				
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
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	pose.resize(16);

	for (int i = 0; i < 4; i++) {
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

// A method for estimating a fundamental matrix given 2D-2D correspondences
int findFundamentalMatrix_(
	// The 2D-2D point correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &fundamental_matrix, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	int lo_number)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

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
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
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

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	// Apply Graph-cut RANSAC
	utils::DefaultFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

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
		// This causes a memory leaking.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::DefaultFundamentalMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultFundamentalMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultFundamentalMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultFundamentalMatrixEstimator>> gcransac;
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
		GCRANSAC<utils::DefaultFundamentalMatrixEstimator, AbstractNeighborhood> gcransac;
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

	// Running bundle adjustment minimizing the pose error on the found inliers
	const size_t &inlier_number = statistics.inliers.size();
	if (statistics.inliers.size() > 7)
	{
		// The truncated least-squares threshold
		const double truncated_threshold = 3.0 / 2.0 * threshold; 
		// The squared least-squares threshold
		const double squared_truncated_threshold = truncated_threshold * truncated_threshold; 

		// Initializing all weights to be zero
		std::vector<double> weights(inlier_number, 0.0);

		// Setting a weight for all inliers
		for (size_t inlier_idx = 0; inlier_idx < inlier_number; ++inlier_idx)
		{
			const size_t point_idx = statistics.inliers[inlier_idx];
			weights[inlier_idx] = 
				MAX(0, 1.0 - estimator.squaredResidual(points.row(point_idx), model) / squared_truncated_threshold);
		}

		std::vector<gcransac::Model> models;
		estimator::solver::FundamentalMatrixBundleAdjustmentSolver bundleOptimizer;
		bundleOptimizer.estimateModel(
			points,
			&statistics.inliers[0],
			statistics.inliers.size(),
			models,
			&weights[0]);
			
		// Scoring function
		MSACScoringFunction<utils::DefaultFundamentalMatrixEstimator> scoring;
		scoring.initialize(squared_truncated_threshold, points.rows);
		// Score of the new model
		Score score;
		// Inliers of the new model
		std::vector<size_t> current_inliers;

		// Select the best model and update the inliers
		for (auto &tmp_model : models)
		{			
			current_inliers.clear();
			score = scoring.getScore(points, // All points
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

	fundamental_matrix.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamental_matrix[i * 3 + j] = (double)model.descriptor(i, j);
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

// A method for estimating a fundamental matrix given 2D-2D correspondences
int findFundamentalMatrixAC_(
	// The 2D-2D point correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &fundamental_matrix, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	int lo_number)
{
	// Each row contains 8 elements as [x1, y1, x2, y2, a11, a12, a21, a22]
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);

	// The neighborhood structure used for the graph-cut-based local optimization
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

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
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	// Apply Graph-cut RANSAC
	utils::ACBasedFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampler), 4 (AR-Sampler)\n",
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
		// This causes a memory leaking.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::ACBasedFundamentalMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::ACBasedFundamentalMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::ACBasedFundamentalMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::ACBasedFundamentalMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::ACBasedFundamentalMatrixEstimator>> gcransac;
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
		GCRANSAC<utils::ACBasedFundamentalMatrixEstimator, AbstractNeighborhood> gcransac;
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

	// Running bundle adjustment minimizing the pose error on the found inliers
	if (statistics.inliers.size() > 7)
	{
		std::vector<gcransac::Model> models;
		estimator::solver::FundamentalMatrixBundleAdjustmentSolver bundleOptimizer;
		bundleOptimizer.estimateModel(
			points,
			&statistics.inliers[0],
			statistics.inliers.size(),
			models);

		// The truncated least-squares threshold
		const double truncated_threshold = 3.0 / 2.0 * threshold; 
		// The squared least-squares threshold
		const double squared_truncated_threshold = truncated_threshold * truncated_threshold; 
		// Scoring function
		MSACScoringFunction<utils::ACBasedFundamentalMatrixEstimator> scoring;
		scoring.initialize(squared_truncated_threshold, points.rows);
		// Score of the new model
		Score score;
		// Inliers of the new model
		std::vector<size_t> inliers;

		// Select the best model and update the inliers
		for (auto &tmp_model : models)
		{			
			inliers.clear();
			score = scoring.getScore(points, // All points
				tmp_model, // The current model parameters
				estimator, // The estimator 
				squared_truncated_threshold, // The current threshold
				inliers); // The current inlier set

			// Check if the updated model is better than the best so far
			if (statistics.score < score.value)
			{
				model.descriptor = tmp_model.descriptor;
				statistics.score = score.value;
				statistics.inliers.swap(inliers);
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

	fundamental_matrix.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamental_matrix[i * 3 + j] = (double)model.descriptor(i, j);
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


// A method for estimating a fundamental matrix given 2D-2D correspondences
int findFundamentalMatrixSIFT_(
	// The 2D-2D point correspondences
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &fundamental_matrix, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	int lo_number)
{
	// Each row contains 8 elements as [x1, y1, x2, y2, size1, size2, ori1, ori2]
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);

	// The neighborhood structure used for the graph-cut-based local optimization
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

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
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	// Apply Graph-cut RANSAC
	utils::SIFTBasedFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampler), 4 (AR-Sampler)\n",
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
		// This causes a memory leaking.
		AbstractSampler *sampler_ptr = main_sampler.release();
		delete sampler_ptr;

		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		return 0;
	}

	utils::RANSACStatistics statistics;
	
	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<utils::SIFTBasedFundamentalMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::SIFTBasedFundamentalMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::SIFTBasedFundamentalMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::SIFTBasedFundamentalMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::SIFTBasedFundamentalMatrixEstimator>> gcransac;
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
		GCRANSAC<utils::SIFTBasedFundamentalMatrixEstimator, AbstractNeighborhood> gcransac;
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

	// Running bundle adjustment minimizing the pose error on the found inliers
	if (statistics.inliers.size() > 7)
	{
		std::vector<gcransac::Model> models = { model };
		estimator::solver::FundamentalMatrixBundleAdjustmentSolver bundleOptimizer;
		bundleOptimizer.estimateModel(
			points,
			&statistics.inliers[0],
			statistics.inliers.size(),
			models);

		// The truncated least-squares threshold
		const double truncated_threshold = 3.0 / 2.0 * threshold; 
		// The squared least-squares threshold
		const double squared_truncated_threshold = truncated_threshold * truncated_threshold; 
		// Scoring function
		MSACScoringFunction<utils::SIFTBasedFundamentalMatrixEstimator> scoring;
		scoring.initialize(squared_truncated_threshold, points.rows);
		// Score of the new model
		Score score;
		// Inliers of the new model
		std::vector<size_t> tmp_inliers;

		// Select the best model and update the inliers
		for (auto &tmp_model : models)
		{			
			tmp_inliers.clear();
			score = scoring.getScore(points, // All points
				tmp_model, // The current model parameters
				estimator, // The estimator 
				squared_truncated_threshold, // The current threshold
				tmp_inliers); // The current inlier set

			// Check if the updated model is better than the best so far
			if (statistics.score < score.value)
			{
				model.descriptor = tmp_model.descriptor;
				statistics.score = score.value;
				statistics.inliers.swap(tmp_inliers);
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

	fundamental_matrix.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			fundamental_matrix[i * 3 + j] = (double)model.descriptor(i, j);
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

// A method for estimating an essential matrix given 2D-2D correspondences
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
	int lo_number)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

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
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
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
	utils::DefaultEssentialMatrixEstimator estimator(intrinsics_src,
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
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
	inlier_selector::EmptyInlierSelector<utils::DefaultEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultEssentialMatrixEstimator>> gcransac;
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
			model,
			preemptive_verification,
			inlier_selector);

		statistics = gcransac.getRansacStatistics();
	}
	else
	{		
		GCRANSAC<utils::DefaultEssentialMatrixEstimator, 
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
		MSACScoringFunction<utils::DefaultEssentialMatrixEstimator> scoring;
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

// A method for estimating an essential matrix given 2D-2D correspondences
int findEssentialMatrixAC_(
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
	int lo_number)
{
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

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
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> 
		intrinsics_src(&source_intrinsics[0]);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> 
		intrinsics_dst(&destination_intrinsics[0]);

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
	utils::ACBasedEssentialMatrixEstimator estimator(intrinsics_src,
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
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
	inlier_selector::EmptyInlierSelector<utils::ACBasedEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::ACBasedEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::ACBasedEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::ACBasedEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::ACBasedEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		GCRANSAC<utils::ACBasedEssentialMatrixEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		MSACScoringFunction<utils::ACBasedEssentialMatrixEstimator> scoring;
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


// A method for estimating an essential matrix given 2D-2D correspondences
int findEssentialMatrixSIFT_(
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
	int lo_number)
{
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);

	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat emptyPoints(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&emptyPoints, // The input points
			{ 0, 		// The cell size along axis X
				0,  	// The cell size along axis Y
				0, 		// The cell size along axis X
				0 }, 	// The cell size along axis Y
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Using only the point coordinates and not the affine elements when constructing the neighborhood.
		cv::Mat point_coordinates = points(cv::Rect(0, 0, 4, points.rows));

		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_coordinates,
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> 
		intrinsics_src(&source_intrinsics[0]);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> 
		intrinsics_dst(&destination_intrinsics[0]);

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
	utils::SIFTBasedEssentialMatrixEstimator estimator(intrinsics_src,
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampling), 4 (AR-Sampler)\n",
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
	inlier_selector::EmptyInlierSelector<utils::SIFTBasedEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::SIFTBasedEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::SIFTBasedEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::SIFTBasedEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::SIFTBasedEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		GCRANSAC<utils::SIFTBasedEssentialMatrixEstimator, AbstractNeighborhood> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		MSACScoringFunction<utils::SIFTBasedEssentialMatrixEstimator> scoring;
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

// A method for estimating an essential matrix given 2D-2D correspondences
int findPlanarEssentialMatrix_(
	// The 2D-2D point correspondences.
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
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

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
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
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
	utils::DefaultPlanarEssentialMatrixEstimator estimator(intrinsics_src,
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
		double variance = 0.1;
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
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
	inlier_selector::EmptyInlierSelector<utils::DefaultPlanarEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultPlanarEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultPlanarEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultPlanarEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultPlanarEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
		gcransac.settings.do_final_iterated_least_squares = false;
		gcransac.settings.max_graph_cut_number = 100;

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
		
		GCRANSAC<utils::DefaultPlanarEssentialMatrixEstimator, AbstractNeighborhood> gcransac;
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
		MSACScoringFunction<utils::DefaultPlanarEssentialMatrixEstimator> scoring;
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

// A method for estimating an essential matrix given 2D-2D correspondences
int findGravityEssentialMatrix_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The 2D-2D point correspondences.
	std::vector<double>& gravity_source,
	// The 2D-2D point correspondences.
	std::vector<double>& gravity_destination,
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
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);

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
		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&points,
				{ w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
					w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
					h2 / static_cast<double>(cell_number_in_neighborhood_graph_) },
				cell_number_in_neighborhood_graph_));
		else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
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

	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> gravity_src(&gravity_source[0]);
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> gravity_dst(&gravity_destination[0]);

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
	utils::DefaultGravityEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	EssentialMatrix model;

	// Setting the gravity directions to the solver
	estimator.getMutableMinimalSolver()->setGravity(gravity_src, gravity_dst);

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
		double variance = 0.1;
        double max_prob = 0;
        for (const auto &prob : point_probabilities)
            max_prob = MAX(max_prob, prob);
        for (auto &prob : point_probabilities)
            prob /= max_prob;
		main_sampler = std::unique_ptr<AbstractSampler>(new gcransac::sampler::AdaptiveReorderingSampler(&points, 
            point_probabilities,
            estimator.sampleSize(),
            variance));
	}
	else
	{
		AbstractNeighborhood *neighborhood_graph_ptr = neighborhood_graph.release();
		delete neighborhood_graph_ptr;

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
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
	inlier_selector::EmptyInlierSelector<utils::DefaultGravityEssentialMatrixEstimator, 
		AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultGravityEssentialMatrixEstimator> preemptive_verification(
			points,
			estimator,
			min_inlier_ratio_for_sprt);

		GCRANSAC<utils::DefaultGravityEssentialMatrixEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::DefaultGravityEssentialMatrixEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::DefaultGravityEssentialMatrixEstimator>> gcransac;
		gcransac.settings.threshold = threshold / threshold_normalizer; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
		gcransac.settings.do_final_iterated_least_squares = false;
		gcransac.settings.max_graph_cut_number = 100;

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
		
		GCRANSAC<utils::DefaultGravityEssentialMatrixEstimator, AbstractNeighborhood> gcransac;
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
		MSACScoringFunction<utils::DefaultGravityEssentialMatrixEstimator> scoring;
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

// A method for estimating a homography matrix given 2D-2D correspondences
int findHomography_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &homography, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	// A flag determining if space partitioning from 
	// Barath, Daniel, and Gabor Valasek. "Space-Partitioning RANSAC." arXiv preprint arXiv:2111.12385 (2021).
	// should be used to speed up the model verification.
	bool use_space_partitioning,
	// The variance parameter of the AR-Sampler. It is only used if that particular sampler is selected.
	double sampler_variance,
	// The number of RANSAC iterations done in the local optimization
	int lo_number)
{
	int num_tents = correspondences.size() / 4;
	cv::Mat points(num_tents, 4, CV_64F, &correspondences[0]);
	
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	if (use_space_partitioning && neighborhood_id != 0)
	{
		fprintf(stderr, "Space Partitioning only works with Grid neighorbood yet. Thus, setting neighborhood_id = 0.\n");
		neighborhood_id = 0;
	}

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	// Initializing a grid-based neighborhood graph
	if (neighborhood_id == 0)
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&points,
			{ (w1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(h1 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(w2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_),
				(h2 + std::numeric_limits<double>::epsilon()) / static_cast<double>(cell_number_in_neighborhood_graph_) },
			cell_number_in_neighborhood_graph_));
	else if (neighborhood_id == 1) // Initializing the neighbhood graph by FLANN
		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::FlannNeighborhoodGraph(&points, neighborhood_size));
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

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	utils::DefaultHomographyEstimator estimator;
	Homography model;

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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 2 (P-NAPSAC sampling)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

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

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification(
			points,
			estimator);

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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
		} else
		{
			inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::SPRTPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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
	}
	else
	{
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator> preemptive_verification;

		if (use_space_partitioning)
		{			
			inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::SpacePartitioningRANSAC<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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
		} else
		{
			inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

			GCRANSAC<utils::DefaultHomographyEstimator,
				AbstractNeighborhood,
				MSACScoringFunction<utils::DefaultHomographyEstimator>,
				preemption::EmptyPreemptiveVerfication<utils::DefaultHomographyEstimator>,
				inlier_selector::EmptyInlierSelector<utils::DefaultHomographyEstimator, AbstractNeighborhood>> gcransac;
			gcransac.settings.threshold = threshold; // The inlier-outlier threshold
			gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
			gcransac.settings.confidence = conf; // The required confidence in the results
			gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
			gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
			gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
			gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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
	}

	homography.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			homography[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
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

	// Return the number of inliers found
	return num_inliers;
}

// A method for estimating a homography matrix given 2D-2D correspondences
int findHomographyAC_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &homography, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	int lo_number)
{
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);
	
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat empty_point_matrix(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&empty_point_matrix, // The input points
			{ 	0, // The cell size along axis X in the source image
				0, // The cell size along axis Y in the source image
				0, // The cell size along axis X in the destination image
				0 }, // The cell size along axis Y in the destination image
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Using only the point coordinates and not the affine elements when constructing the neighborhood.
		cv::Mat point_coordinates = points(cv::Rect(0, 0, 4, points.rows));

		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_coordinates,
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	utils::ACBasedHomographyEstimator estimator;
	Homography model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampling), 4 (AR-Sampler)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

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
	inlier_selector::EmptyInlierSelector<utils::ACBasedHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::ACBasedHomographyEstimator> preemptive_verification(
			points,
			estimator);

		GCRANSAC<utils::ACBasedHomographyEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::ACBasedHomographyEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::ACBasedHomographyEstimator>,
			inlier_selector::EmptyInlierSelector<utils::ACBasedHomographyEstimator, AbstractNeighborhood>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<utils::ACBasedHomographyEstimator> preemptive_verification;

		GCRANSAC<utils::ACBasedHomographyEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::ACBasedHomographyEstimator>,
			preemption::EmptyPreemptiveVerfication<utils::ACBasedHomographyEstimator>,
			inlier_selector::EmptyInlierSelector<utils::ACBasedHomographyEstimator, AbstractNeighborhood>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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

	homography.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			homography[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
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

	// Return the number of inliers found
	return num_inliers;
}


// A method for estimating a homography matrix given 2D-2D correspondences
int findHomographySIFT_(
	// The 2D-2D point correspondences.
	std::vector<double>& correspondences,
	// The probabilities for each 3D-3D point correspondence if available
	std::vector<double> &point_probabilities,
	// Output: the found inliers 
	std::vector<bool>& inliers, 
	// Output: the found 6D pose
	std::vector<double> &homography, 
	// The images' sizes
	int h1, int w1, int h2, int w2,
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
	int lo_number)
{
	int num_tents = correspondences.size() / 8;
	cv::Mat points(num_tents, 8, CV_64F, &correspondences[0]);
	
	typedef neighborhood::NeighborhoodGraph<cv::Mat> AbstractNeighborhood;
	std::unique_ptr<AbstractNeighborhood> neighborhood_graph;

	const size_t cell_number_in_neighborhood_graph_ = 
		static_cast<size_t>(neighborhood_size);

	// If the spatial weight is 0.0, the neighborhood graph should not be created 
	if (spatial_coherence_weight <= std::numeric_limits<double>::epsilon())
	{
		cv::Mat empty_point_matrix(0, 4, CV_64F);

		neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
			new neighborhood::GridNeighborhoodGraph<4>(&empty_point_matrix, // The input points
			{ 	0, // The cell size along axis X in the source image
				0, // The cell size along axis Y in the source image
				0, // The cell size along axis X in the destination image
				0 }, // The cell size along axis Y in the destination image
			1)); // The cell number along every axis
	} else // Initializing a grid-based neighborhood graph
	{
		// Using only the point coordinates and not the affine elements when constructing the neighborhood.
		cv::Mat point_coordinates = points(cv::Rect(0, 0, 4, points.rows));

		// Initializing a grid-based neighborhood graph
		if (neighborhood_id == 0)
			neighborhood_graph = std::unique_ptr<AbstractNeighborhood>(
				new neighborhood::GridNeighborhoodGraph<4>(&point_coordinates,
				{ w1 / neighborhood_size,
					h1 / neighborhood_size,
					w2 / neighborhood_size,
					h2 / neighborhood_size },
				static_cast<size_t>(neighborhood_size)));
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

	utils::SIFTBasedHomographyEstimator estimator;
	Homography model;

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
	if (sampler_id == 0) // Initializing a RANSAC-like uniformly random sampler
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&points));
	else if (sampler_id == 1)  // Initializing a PROSAC sampler. This requires the points to be ordered according to the quality.
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::ProsacSampler(&points, estimator.sampleSize()));
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

		fprintf(stderr, "Unknown sampler identifier: %d. The accepted samplers are 0 (uniform sampling), 1 (PROSAC sampling), 3 (NG-RANSAC sampling), 4 (AR-Sampler)\n",
			sampler_id);
		return 0;
	}

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

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
	inlier_selector::EmptyInlierSelector<utils::SIFTBasedHomographyEstimator, AbstractNeighborhood> inlier_selector(neighborhood_graph.get());

	if (use_sprt)
	{
		// Initializing SPRT test
		preemption::SPRTPreemptiveVerfication<utils::SIFTBasedHomographyEstimator> preemptive_verification(
			points,
			estimator);

		GCRANSAC<utils::SIFTBasedHomographyEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::SIFTBasedHomographyEstimator>,
			preemption::SPRTPreemptiveVerfication<utils::SIFTBasedHomographyEstimator>,
			inlier_selector::EmptyInlierSelector<utils::SIFTBasedHomographyEstimator, AbstractNeighborhood>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = neighborhood_size; // The radius of the neighborhood ball

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
		// Initializing an empty preemption
		preemption::EmptyPreemptiveVerfication<utils::SIFTBasedHomographyEstimator> preemptive_verification;

		GCRANSAC<utils::SIFTBasedHomographyEstimator,
			AbstractNeighborhood,
			MSACScoringFunction<utils::SIFTBasedHomographyEstimator>,
			preemption::EmptyPreemptiveVerfication<utils::SIFTBasedHomographyEstimator>,
			inlier_selector::EmptyInlierSelector<utils::SIFTBasedHomographyEstimator, AbstractNeighborhood>> gcransac;
		gcransac.settings.threshold = threshold; // The inlier-outlier threshold
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = conf; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = lo_number; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball

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

	homography.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			homography[i * 3 + j] = model.descriptor(i, j);
		}
	}

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
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

	// Return the number of inliers found
	return num_inliers;
}
