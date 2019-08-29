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

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
	#include <direct.h>
#endif 

struct stat info;

enum Problem {
	FundamentalMatrixFitting,
	EssentialMatrixFitting,
	HomographyFitting
};

void testEssentialMatrixFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &source_intrinsics_path_,
	const std::string &destination_intrinsics_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_);

void testFundamentalMatrixFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_);

void testHomographyFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_);

std::vector<std::string> getAvailableTestScenes(Problem problem_);

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_);

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &src_intrinsics_path_,
	std::string &dst_intrinsics_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_);

using namespace gcransac;

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	const double confidence = 0.99; // The RANSAC confidence value
	const int fps = -1; // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
	const double inlier_outlier_threshold_essential_matrix = 0.0003; // The used inlier-outlier threshold in GC-RANSAC for essential matrix estimation.
	const double inlier_outlier_threshold_fundamental_matrix = 0.0005; // The used adaptive inlier-outlier threshold in GC-RANSAC for fundamental matrix estimation.
	const double inlier_outlier_threshold_homography = 2.00; // The used inlier-outlier threshold in GC-RANSAC for homography estimation.
	const double spatial_coherence_weight = 0.14; // The weight of the spatial coherence term in the graph-cut energy minimization.
	const size_t cell_number_in_neighborhood_graph = 8; // The number of cells along each axis in the neighborhood graph.

	printf("------------------------------------------------------------\nFundamental matrix fitting\n------------------------------------------------------------\n");
	for (const std::string &scene : getAvailableTestScenes(Problem::FundamentalMatrixFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path; // Path where the matched image is saved

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testFundamentalMatrixFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_fundamental_matrix, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	printf("------------------------------------------------------------\nHomography fitting\n------------------------------------------------------------\n");
	for (const std::string &scene : getAvailableTestScenes(Problem::HomographyFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path; // Path where the matched image is saved

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testHomographyFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_homography, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	printf("------------------------------------------------------------\nEssential matrix fitting\n------------------------------------------------------------\n");
	for (const std::string &scene : getAvailableTestScenes(Problem::EssentialMatrixFitting))
	{
		printf("Processed scene = '%s'\n", scene.c_str());
		std::string src_image_path, // Path of the source image
			dst_image_path, // Path of the destination image
			input_correspondence_path, // Path where the detected correspondences are saved
			output_correspondence_path, // Path where the inlier correspondences are saved
			output_matched_image_path, // Path where the matched image is saved
			src_intrinsics_path, // Path where the intrinsics camera matrix of the source image is
			dst_intrinsics_path; // Path where the intrinsics camera matrix of the destination image is

		// Initializing the paths
		initializeScene(scene,
			src_image_path,
			dst_image_path,
			src_intrinsics_path,
			dst_intrinsics_path,
			input_correspondence_path,
			output_correspondence_path,
			output_matched_image_path);

		// Estimating the fundamental matrix by the Graph-Cut RANSAC algorithm
		testEssentialMatrixFitting(
			src_image_path, // The source image's path
			dst_image_path, // The destination image's path
			src_intrinsics_path, // Path where the intrinsics camera matrix of the source image is
			dst_intrinsics_path, // Path where the intrinsics camera matrix of the destination image is
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			output_matched_image_path, // The path where the matched image pair will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold_essential_matrix, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			cell_number_in_neighborhood_graph, // The radius of the neighborhood ball for determining the neighborhoods.
			fps); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.
		printf("\n------------------------------------------------------------\n");
	}

	return 0;
}

std::vector<std::string> getAvailableTestScenes(Problem problem_)
{
	switch (problem_)
	{
	case Problem::FundamentalMatrixFitting:
		return { "head", "johnssona", "Kyoto" };
	case Problem::HomographyFitting:
		return { "graf", "Eiffel", "adam" };
	default:
		return { "fountain" };
	}
}

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_)
{
	// The root directory where the "results" and "data" folder are
	const std::string root_dir = "";

	// The directory to which the results will be saved
	std::string dir = root_dir + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
	{
#ifdef _WIN32 // Create a directory on Windows
<<<<<<< HEAD
		if (_mkdir(dir.c_str()) != 0) // Create it, if	 not
=======
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
>>>>>>> 2e3d812602e7140a01c5d281f95ffe27a052de4c
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#else // Create a directory on Linux
		if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}
#endif
	}

	// The source image's path
	src_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", src_image_path_.c_str());
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", dst_image_path_.c_str());
		return false;
	}

	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	input_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		root_dir + "results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
}

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &src_intrinsics_path_,
	std::string &dst_intrinsics_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_)
{
	// The root directory where the "results" and "data" folder are
	const std::string root_dir = "";

	// The directory to which the results will be saved
	std::string dir = root_dir + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
#ifdef _WIN32 // Create a directory on Windows
	if (_mkdir(dir.c_str()) != 0) // Create it, if	 not
	{
		fprintf(stderr, "Error while creating a new folder in 'results'\n");
		return false;
	}
#else // Create a directory on Linux
	if (mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
	{
		fprintf(stderr, "Error while creating a new folder in 'results'\n");
		return false;
	}
#endif

	// The source image's path
	src_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", src_image_path_.c_str());
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", dst_image_path_.c_str());
		return false;
	}

	// The path where the intrinsics camera matrix of the source camera can be found
	src_intrinsics_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.K";
	// The path where the intrinsics camera matrix of the destination camera can be found
	dst_intrinsics_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.K";
	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	input_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		root_dir + "results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
}

void testHomographyFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_)
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_); // The source image
	cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph neighborhood(&points,
		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	DefaultHomographyEstimator estimator;
	std::vector<int> inliers;
	Homography model;

	GCRANSAC<DefaultHomographyEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = inlier_outlier_threshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximm number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(source_image.cols), // The width of the source image
		static_cast<double>(source_image.rows), // The height of the source image
		static_cast<double>(destination_image.cols), // The width of the destination image
		static_cast<double>(destination_image.rows));  // The height of the destination image

	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model);

	// Get the statistics of the results
	const RANSACStatistics &statistics = gcransac.getRansacStatistics();

	// Write statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));

	// Draw the inlier matches to the images
	cv::Mat match_image;
	drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n", output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n", out_correspondence_path_.c_str());
	savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}

void testFundamentalMatrixFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_)
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_);
	cv::Mat destination_image = cv::imread(destination_path_);

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph neighborhood(&points,
		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Calculating the maximum image diagonal to be used for setting the threshold
	// adaptively for each image pair. 
	const double max_image_diagonal =
		sqrt(pow(MAX(source_image.cols, destination_image.cols), 2) + pow(MAX(source_image.rows, destination_image.rows), 2));
	
	// Apply Graph-cut RANSAC
	DefaultFundamentalMatrixEstimator estimator;
	std::vector<int> inliers;
	FundamentalMatrix model;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(source_image.cols), // The width of the source image
		static_cast<double>(source_image.rows), // The height of the source image
		static_cast<double>(destination_image.cols), // The width of the destination image
		static_cast<double>(destination_image.rows));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}
	
	GCRANSAC<DefaultFundamentalMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = inlier_outlier_threshold_ * max_image_diagonal; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model);

	// Get the statistics of the results
	const RANSACStatistics &statistics = gcransac.getRansacStatistics();

	// Write statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));
	
	// Draw the inlier matches to the images
	cv::Mat match_image;
	drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n", 
		output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n",
		out_correspondence_path_.c_str());
	savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}

void testEssentialMatrixFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &source_intrinsics_path_,
	const std::string &destination_intrinsics_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_)
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_);
	cv::Mat destination_image = cv::imread(destination_path_);

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"

	// Load the intrinsic camera matrices
	Eigen::Matrix3d intrinsics_src,
		intrinsics_dst;

	if (!loadMatrix<double, 3, 3>(source_intrinsics_path_,
		intrinsics_src))
	{
		printf("An error occured when loading the intrinsics camera matrix from '%s'\n",
			source_intrinsics_path_.c_str());
		return;
	}

	if (!loadMatrix<double, 3, 3>(destination_intrinsics_path_,
		intrinsics_dst))
	{
		printf("An error occured when loading the intrinsics camera matrix from '%s'\n",
			destination_intrinsics_path_.c_str());
		return;
	}

	// Normalize the point coordinate by the intrinsic matrices
	cv::Mat normalized_points(points.size(), CV_64F);
	normalizeCorrespondences(points, 
		intrinsics_src,
		intrinsics_dst,
		normalized_points);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph neighborhood(&points,
		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	DefaultEssentialMatrixEstimator estimator(intrinsics_src,
		intrinsics_dst);
	std::vector<int> inliers;
	EssentialMatrix model;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::ProgressiveNapsacSampler main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		static_cast<double>(source_image.cols), // The width of the source image
		static_cast<double>(source_image.rows), // The height of the source image
		static_cast<double>(destination_image.cols), // The width of the destination image
		static_cast<double>(destination_image.rows));  // The height of the destination image
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}
	
	GCRANSAC<DefaultEssentialMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
	gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
	gcransac.settings.threshold = inlier_outlier_threshold_; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = confidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
	gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

	// Start GC-RANSAC
	gcransac.run(normalized_points,
		estimator,
		&main_sampler,
		&local_optimization_sampler,
		&neighborhood,
		model);

	// Get the statistics of the results
	const RANSACStatistics &statistics = gcransac.getRansacStatistics();

	// Print the statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", statistics.local_optimization_number);
	printf("Applied number of graph-cuts = %d\n", statistics.graph_cut_number);
	printf("Number of iterations = %d\n\n", statistics.iteration_number);

	// Draw the inlier matches to the images
	cv::Mat match_image;
	drawMatches(points,
		statistics.inliers,
		source_image,
		destination_image,
		match_image);

	printf("Saving the matched images to file '%s'.\n", output_match_image_path_.c_str());
	imwrite(output_match_image_path_, match_image); // Save the matched image to file
	printf("Saving the inlier correspondences to file '%s'.\n", out_correspondence_path_.c_str());
	savePointsToFile(points, out_correspondence_path_.c_str(), &statistics.inliers); // Save the inliers to file

	printf("Press a button to continue...\n");

	// Showing the image
	showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}
