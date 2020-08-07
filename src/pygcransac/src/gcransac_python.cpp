#include "gcransac_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"
#include "flann_neighborhood_graph.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"
#include "solver_epnp_lm.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

using namespace gcransac;

int find6DPose_(std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters)
{
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 7, CV_64F);
	size_t iterations = 0;

	for (size_t i = 0; i < num_tents; ++i) {
		points.at<double>(i, 5) = imagePoints[2 * i];
		points.at<double>(i, 6) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}

	const size_t cell_number_in_neighborhood_graph_ = 8;
	neighborhood::FlannNeighborhoodGraph neighborhood1(&points, 20);

	typedef estimator::PerspectiveNPointEstimator<estimator::solver::P3PSolver, // The solver used for fitting a model to a minimal sample
		estimator::solver::EPnPLM> // The solver used for fitting a model to a non-minimal sample
		PnPEstimator;

	// Apply Graph-cut RANSAC
	PnPEstimator estimator;
	Pose6D model;

	// Initialize the samplers	
	sampler::UniformSampler main_sampler(&points);  // The main sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	GCRANSAC<PnPEstimator, neighborhood::FlannNeighborhoodGraph> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = 1; // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();

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
	return num_inliers;
}

int find6DPoseEPOS_(
	std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<double>& cameraParams,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double &score,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int min_iters,
	int max_iters,
	double sphere_radius,
	double scaling_from_millimeters,
	double min_coverage,
	double min_triangle_area,
	bool apply_numerical_optimization,
	int num_models)
{
	// Calculate the inverse of the intrinsic camera parameters
	Eigen::Matrix3d K;
	K << cameraParams[0], cameraParams[1], cameraParams[2],
		cameraParams[3], cameraParams[4], cameraParams[5],
		cameraParams[6], cameraParams[7], cameraParams[8];
	const Eigen::Matrix3d Kinv =
		K.inverse();

	size_t num_tents = imagePoints.size() / 2; // Number of points
	cv::Mat points(num_tents, 7, CV_64F); // The data matrix. Each row is of format "nu nv x y z u v", where (u,v) is the image point and (nu, nv) is the normalized image point
	Eigen::Vector3d vec; // Temporary variable for normalizing the points
	vec(2) = 1.0;
	
	cv::Mat points_for_neighborhood(points.rows, 5, CV_64F); // The matrix containing the data from which the neighborhoods are calculated
	std::map<std::pair<int, int>, int> pixels; // A helper variable to count how many pixels are occupied

	// Normalizing the data and counting the occupied pixels
	for (size_t i = 0; i < num_tents; ++i) {
		points.at<double>(i, 5) = imagePoints[2 * i];
		points.at<double>(i, 6) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];

		vec(0) = points.at<double>(i, 5);
		vec(1) = points.at<double>(i, 6);

		points.at<double>(i, 0) = Kinv.row(0) * vec;
		points.at<double>(i, 1) = Kinv.row(1) * vec;

		points_for_neighborhood.at<double>(i, 0) = points.at<double>(i, 5);
		points_for_neighborhood.at<double>(i, 1) = points.at<double>(i, 6);
		points_for_neighborhood.at<double>(i, 2) = points.at<double>(i, 2) * scaling_from_millimeters;
		points_for_neighborhood.at<double>(i, 3) = points.at<double>(i, 3) * scaling_from_millimeters;
		points_for_neighborhood.at<double>(i, 4) = points.at<double>(i, 4) * scaling_from_millimeters;

		const int x = points.at<double>(i, 5),
			y = points.at<double>(i, 6);
		pixels[std::make_pair(x, y)] = 1;
	}

	// Normalize the threshold
	const double f = 0.5 * (K(0, 0) + K(1, 1));
	const double normalized_threshold = threshold / f;

	// Estimate the neighborhood structure
	neighborhood::FlannNeighborhoodGraph neighborhood1(&points_for_neighborhood, sphere_radius);

	typedef estimator::PerspectiveNPointEstimator<estimator::solver::P3PSolver, // The solver used for fitting a model to a minimal sample
		estimator::solver::EPnPLM> // The solver used for fitting a model to a non-minimal sample
		PnPEstimator;

	// Apply Graph-cut RANSAC
	PnPEstimator estimator(min_triangle_area);
	Pose6D model;

	// Initialize the samplers	
	sampler::UniformSampler main_sampler(&points);  // The main sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	GCRANSAC<PnPEstimator, neighborhood::FlannNeighborhoodGraph, EPOSScoringFunction<PnPEstimator>> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = normalized_threshold; // The inlier-outlier threshold
	gcransac.settings.minimum_pixel_coverage = min_coverage;
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = min_iters; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = 1; // The number of parallel processes
	gcransac.settings.used_pixels = pixels.size();

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	const size_t inlier_number = statistics.inliers.size();

	// If numerical optimization is needed, apply the Levenberg-Marquardt 
	// implementation of OpenCV.
	if (apply_numerical_optimization && inlier_number >= 6)
	{
		// The estimated rotation matrix
		Eigen::Matrix3d rotation =
			model.descriptor.leftCols<3>();
		// The estimated translation
		Eigen::Vector3d translation =
			model.descriptor.rightCols<1>();

		// Copy the data into two matrices containing the image and object points. 
		// This would not be necessary, but selecting the submatrices by cv::Rect
		// leads to an error in cv::solvePnP().
		cv::Mat inlier_image_points(inlier_number, 2, CV_64F),
			inlier_object_points(inlier_number, 3, CV_64F);

		for (size_t i = 0; i < inlier_number; ++i)
		{
			const size_t &idx = statistics.inliers[i];
			inlier_image_points.at<double>(i, 0) = points.at<double>(idx, 0);
			inlier_image_points.at<double>(i, 1) = points.at<double>(idx, 1);
			inlier_object_points.at<double>(i, 0) = points.at<double>(idx, 2);
			inlier_object_points.at<double>(i, 1) = points.at<double>(idx, 3);
			inlier_object_points.at<double>(i, 2) = points.at<double>(idx, 4);
		}

		// Converting the estimated pose parameters OpenCV format
		cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
			cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format
		
		// Convert the rotation matrix by the rodrigues formula
		cv::Mat cv_rodrigues(3, 1, CV_64F);
		cv::Rodrigues(cv_rotation.t(), cv_rodrigues);

		// Applying numerical optimization to the estimated pose parameters
		cv::solvePnP(inlier_object_points, // The object points
			inlier_image_points, // The image points
			cv::Mat::eye(3, 3, CV_64F), // The camera's intrinsic parameters 
			cv::Mat(), // An empty vector since the radial distortion is not known
			cv_rodrigues, // The initial rotation
			cv_translation, // The initial translation
			true, // Use the initial values
			cv::SOLVEPNP_ITERATIVE); // Apply numerical refinement
		
		// Convert the rotation vector back to a rotation matrix
		cv::Rodrigues(cv_rodrigues, cv_rotation);

		// Transpose the rotation matrix back
		//cv_rotation = cv_rotation.t();

		// Calculate the error of the refined pose
		model.descriptor <<
			cv_rotation.at<double>(0, 0), cv_rotation.at<double>(0, 1), cv_rotation.at<double>(0, 2), cv_translation.at<double>(0),
			cv_rotation.at<double>(1, 0), cv_rotation.at<double>(1, 1), cv_rotation.at<double>(1, 2), cv_translation.at<double>(1),
			cv_rotation.at<double>(2, 0), cv_rotation.at<double>(2, 1), cv_rotation.at<double>(2, 2), cv_translation.at<double>(2);
	}
	
	score = statistics.score;
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
	return num_inliers;
}

int findFundamentalMatrix_(std::vector<double>& srcPts,
	std::vector<double>& dstPts,
	std::vector<bool>& inliers,
	std::vector<double>&F,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters)
{
	int num_tents = srcPts.size() / 2;
	cv::Mat points(num_tents, 4, CV_64F);
	int iterations = 0;
	for (int i = 0; i < num_tents; ++i) {
		points.at<double>(i, 0) = srcPts[2 * i];
		points.at<double>(i, 1) = srcPts[2 * i + 1];
		points.at<double>(i, 2) = dstPts[2 * i];
		points.at<double>(i, 3) = dstPts[2 * i + 1];
	}
	const size_t cell_number_in_neighborhood_graph_ = 8;
	neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	// Apply Graph-cut RANSAC
	utils::DefaultFundamentalMatrixEstimator estimator;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	GCRANSAC<utils::DefaultFundamentalMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = 0.0005 * threshold * max_image_diagonal; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));

	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;
	}

	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}

	F.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			F[i * 3 + j] = (double)model.descriptor(i, j);
		}
	}
	return num_inliers;
}

int findEssentialMatrix_(std::vector<double>& srcPts,
	std::vector<double>& dstPts,
	std::vector<bool>& inliers,
	std::vector<double>&E,
	std::vector<double>& src_K,
	std::vector<double>& dst_K,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters)
{
	int num_tents = srcPts.size() / 2;
	cv::Mat points(num_tents, 4, CV_64F);
	int iterations = 0;
	for (int i = 0; i < num_tents; ++i) {
		points.at<double>(i, 0) = srcPts[2 * i];
		points.at<double>(i, 1) = srcPts[2 * i + 1];
		points.at<double>(i, 2) = dstPts[2 * i];
		points.at<double>(i, 3) = dstPts[2 * i + 1];
	}
	const size_t cell_number_in_neighborhood_graph_ = 8;
	neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	Eigen::Matrix3d intrinsics_src,
		intrinsics_dst;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			intrinsics_src(i, j) = src_K[i * 3 + j];
		}
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			intrinsics_dst(i, j) = dst_K[i * 3 + j];
		}
	}

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
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	GCRANSAC<utils::DefaultEssentialMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(normalized_points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	// Get the statistics of the results
	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();


	inliers.resize(num_tents);

	const int num_inliers = statistics.inliers.size();
	for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
		inliers[pt_idx] = 0;

	}
	for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
		inliers[statistics.inliers[pt_idx]] = 1;
	}


	E.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			E[i * 3 + j] = (double)model.descriptor(i, j);
		}
	}
	return num_inliers;
}


int findHomography_(std::vector<double>& srcPts,
	std::vector<double>& dstPts,
	std::vector<bool>& inliers,
	std::vector<double>& H,
	int h1, int w1, int h2, int w2,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters)

{
	const size_t cell_number_in_neighborhood_graph_ = 8;

	int num_tents = srcPts.size() / 2;
	cv::Mat points(num_tents, 4, CV_64F);
	int iterations = 0;
	for (int i = 0; i < num_tents; ++i) {
		points.at<double>(i, 0) = srcPts[2 * i];
		points.at<double>(i, 1) = srcPts[2 * i + 1];
		points.at<double>(i, 2) = dstPts[2 * i];
		points.at<double>(i, 3) = dstPts[2 * i + 1];
	}

	neighborhood::GridNeighborhoodGraph neighborhood1(&points,
		w1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h1 / static_cast<double>(cell_number_in_neighborhood_graph_),
		w2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		h2 / static_cast<double>(cell_number_in_neighborhood_graph_),
		//		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		//		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		//		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		//		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood1.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return 0;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(w1, w2), 2) + pow(MAX(h1, h2), 2));

	utils::DefaultHomographyEstimator estimator;
	Homography model;

	GCRANSAC<utils::DefaultHomographyEstimator, neighborhood::GridNeighborhoodGraph> gcransac;

	gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = threshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
	gcransac.settings.confidence = conf; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(w1), // The width of the source image
		static_cast<double>(h1), // The height of the source image
		static_cast<double>(w2), // The width of the destination image
		static_cast<double>(h2),  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood1,
		model);

	const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));

	H.resize(9);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			H[i * 3 + j] = model.descriptor(i, j);
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

	return num_inliers;
}
