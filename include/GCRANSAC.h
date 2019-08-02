#pragma once

#include <opencv2\highgui\highgui.hpp>
#include "GCoptimization.h"
#include "prosac_sampler.h"
#include "uniform_random_generator.h"

/* RANSAC Scoring */
struct Score {
	/* number of inliers_, rectangular gain function */
	unsigned I;
	/* MSAC scoring, truncated quadratic gain function */
	double J;

	Score() :
		I(0),
		J(0.0)
	{

	}
};

struct Settings {
	bool do_final_iterated_least_squares, // Flag to decide a final iterated least-squares fitting is needed to polish the output model parameters.
		do_local_optimization, // Flag to decide if local optimization is needed
		do_graph_cut, // Flag to decide of graph-cut is used in the local optimization
		use_inlier_limit; // Flag to decide if an inlier limit is used in the local optimization to speed up the procedure

	int desired_fps; // The desired FPS

	size_t max_local_optimization_number, // Maximum number of local optimizations
		min_iteration_number_before_lo, // Minimum number of RANSAC iterations before applying local optimization
		min_iteration_number, // Minimum number of RANSAC iterations
		max_iteration_number, // Maximum number of RANSAC iterations
		max_unsuccessful_model_generations, // Maximum number of unsuccessful model generations
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
		max_local_optimization_number(std::numeric_limits<size_t>::max()),
		max_graph_cut_number(std::numeric_limits<size_t>::max()),
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

	double processing_time;

	std::vector<int> inliers;

	RANSACStatistics() :
		graph_cut_number(0),
		local_optimization_number(0),
		iteration_number(0),
		neighbor_number(0),
		processing_time(0.0)
	{

	}
};

template <class ModelEstimator, class Model>
class GCRANSAC
{
public:
	Settings settings;

	GCRANSAC() :
		time_limit(std::numeric_limits<double>::max())
	{
		if (random_generator == nullptr)
			random_generator = std::make_unique<UniformRandomGenerator>();
	}
	~GCRANSAC() { }

	// The main method applying Graph-Cut RANSAC to the input data points
	void run(const cv::Mat &points_,  // Data points
		ModelEstimator estimator_, // The model estimator
		Model &obtained_model_);  // The output model

	bool sample(const cv::Mat &points_, 
		const std::vector<int> &pool_,
		const std::vector<std::vector<cv::DMatch>> &neighbors_, 
		int sample_number_, 
		int *sample_);

	void setFPS(int fps_) { settings.desired_fps = fps_; time_limit = 1.0 / fps_; } // Set a desired FPS value

	const RANSACStatistics &getRansacStatistics() { return statistics; }

	// Return the score of a model_ w.r.t. the data points_ and the threshold_
	Score getScore(const cv::Mat &points_, // The data points_
		const Model &model_, // The current model_
		const ModelEstimator &estimator_, // The model_ estimator_
		const double threshold_, // The threshold_ for model_ estimation
		std::vector<int> &inliers_, // The inlier set
		bool store_inliers_ = true); // Store the inliers_ or not

	// Decides whether score s2 is higher than s1
	int isScoreLess(const Score &s1_, // Input score
		const Score &s2_, // Input score
		int type_ = 1) // Decision type_: 1 -- RANSAC-like, 2 -- MSAC-like 
	{ 
		return type_ == 1 ? (s1_.I < s2_.I || (s1_.I == s2_.I && s1_.J < s2_.J)) : s1_.J < s2_.J; 
	}

protected:
	std::unique_ptr<theia::ProsacSampler<cv::Mat>> prosac_sampler; // The PROSAC sampler
	std::unique_ptr<UniformRandomGenerator> random_generator; // The random generator for the sampling
	double time_limit; // The desired time limit
	std::vector<std::vector<cv::DMatch>> neighbours; // The neighborhood structure
	RANSACStatistics statistics; // RANSAC statistics
	int point_number; // The point number
	double sqr_threshold_2; // 2 * threshold_^2
	double truncated_threshold; // 3 / 2 * threshold_
	double truncated_threshold_2; // 9 / 4 * threshold_^2
	int step_size; // Step size per processes
	double log_probability; // The logarithm of 1 - confidence

	Graph<double, double, double> *graph; // The graph for graph-cut

	// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
	int getIterationNumber(int inlier_number_, // The inlier number
		int point_number_, // The point number
		int sample_size_, // The current_sample size
		double log_probability_); // The logarithm of the desired probability

	// Returns a labeling w.r.t. the current model and point set
	void labeling(const cv::Mat &points_, // The input data points
		int neighbor_number_, // The neighbor number in the graph
		const std::vector<std::vector<cv::DMatch>> &neighbors_, // The neighborhood
		Model &model_, // The current model_
		ModelEstimator estimator_, // The model estimator
		double lambda_, // The weight for the spatial coherence term
		double threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<int> &inliers_, // The resulting inlier set
		double &energy_); // The resulting energy

	// Apply the graph-cut optimization for GC-RANSAC
	bool graphCutLocalOptimization(const cv::Mat &points_, // The input data points
		std::vector<int> &so_far_the_best_inliers_, // The input, than the resulting inlier set
		Model &so_far_the_best_model_, // The current model
		Score &so_far_the_best_score_, // The current score
		const ModelEstimator &estimator_, // The model estimator
		const int trial_number_); // The max trial number
	
	// Model fitting by the iterated least squares method
	bool iteratedLeastSquaresFitting(
		const cv::Mat &points_, // The input data points
		const ModelEstimator &estimator_, // The model estimator
		const double threshold_, // The inlier-outlier threshold
		std::vector<int> &inliers_, // The resulting inlier set
		Model &model_); // The estimated model
};

// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
template <class ModelEstimator, class Model>
int GCRANSAC<ModelEstimator, Model>::getIterationNumber(int inlier_number_, 
	int point_number_, 
	int sample_size_,  
	double log_probability_)
{
	const double q = pow(static_cast<double>(inlier_number_) / point_number_, sample_size_);
	const double log2 = log(1 - q);

	if (abs(log2) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<int>::max();

	const double iter = log_probability_ / log2;
	return static_cast<int>(iter) + 1;
}

// The main method applying Graph-Cut RANSAC to the input data points_
template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::run(
	const cv::Mat &points_,  // Data points
	ModelEstimator estimator_, // The model estimator
	Model &obtained_model_)  // The output model 
{
	/*
		Initialization
	*/
	// Variables for measuring the processing time
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	statistics.iteration_number = 0;
	statistics.graph_cut_number = 0;
	statistics.local_optimization_number = 0;
	statistics.neighbor_number = 0;
	statistics.processing_time = 0.0;

	step_size = points_.rows / settings.core_number;

	// The size of a minimal sample used for the estimation
	const size_t sample_number = estimator_.sampleSize();
	   	
	// log(1 - confidence) used for determining the required number of iterations
	log_probability = log(1.0 - settings.confidence);
	// Maximum number of iterations
	auto max_iteration = 
		getIterationNumber(1, points_.rows, sample_number, log_probability);

	std::unique_ptr<int[]> current_sample(new int[sample_number]); // Minimal sample for model fitting
	bool do_local_optimization = false; // Flag to show if local optimization should be applied

	size_t inl_offset = 0; // Index to show which inlier vector is currently in use
	Model so_far_the_best_model; // The current so-far-the-best model parameters
	Score so_far_the_best_score; // The score of the current so-far-the-best model
	std::vector<std::vector<int>> temp_inner_inliers(2); // The inliers of the current and previous best models 

	// The PROSAC sampler
	prosac_sampler = std::make_unique<theia::ProsacSampler<cv::Mat>>(estimator_.sampleSize());
	prosac_sampler->initialize();

	point_number = points_.rows; // Number of points in the dataset
	truncated_threshold = 3.0 / 2.0 * settings.threshold; // The truncated least-squares threshold
	truncated_threshold_2 = truncated_threshold * truncated_threshold; // The squared least-squares threshold
	sqr_threshold_2 = 2.0 * settings.threshold * settings.threshold; // Two times squared least-squares threshold
	random_generator->resetGenerator(0, point_number); // Initializing the used random generator

	// Initialize the pool for sampling
	std::vector<int> pool(point_number);
	for (auto i = 0; i < point_number; ++i)
		pool[i] = i;
	
	// Compute the neighborhood graph
	cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));
	neighbours.resize(0);
	if (settings.spatial_coherence_weight > 0.0) // Compute the neighborhood if the weight is not zero
	{
		// TODO: replace by nanoflann
		cv::Mat tmp_points;
		points_.convertTo(tmp_points, CV_32F); // OpenCV's FLANN dies if the points are doubles
		flann.radiusMatch(tmp_points, // The point set converted to floats
			tmp_points, // The point set converted to floats
			neighbours, // The estimated neighborhood graph
			settings.neighborhood_sphere_radius); // The radius of the neighborhood ball
		 
		// Count the edges in the neighborhood graph
		for (auto i = 0; i < neighbours.size(); ++i)
			statistics.neighbor_number += static_cast<int>(neighbours[i].size()) - 1;
	}
	
	// Initialize the starting time if there is a desired FPS set
	if (settings.desired_fps > -1)
		start = std::chrono::system_clock::now();

	/*
		The main RANSAC iteration
	*/
	while (settings.min_iteration_number > statistics.iteration_number ||
		statistics.iteration_number < MIN(max_iteration, settings.max_iteration_number))
	{
		// Do not apply local optimization if not needed
		do_local_optimization = false;

		// Increase the iteration number counter
		++statistics.iteration_number;

		// current_sample a minimal subset
		std::vector<Model> models;

		size_t unsuccessful_model_generations = 0;
		// Select a minimal sample and estimate the implied model parameters if possible.
		// If, after a certain number of sample selections, there is no success, terminate.
		while (++unsuccessful_model_generations < settings.max_unsuccessful_model_generations)
		{
			// If the sampling is not successful, try again.
			if (!sample(points_, // All points
				pool, // The current pool from which the points are chosen
				neighbours, // The neighborhood structure
				sample_number, // Number of points to select
				current_sample.get())) // The current sample
				continue;
			 
			// Estimate the model parameters using the current sample
 			if (estimator_.estimateModel(points_,  // All points
				current_sample.get(), // The current sample
				&models)) // The estimated model parameters
				break; 
		}

		// Select the so-far-the-best from the estimated models
		for (const auto &model : models)
		{
			// Get the inliers and the score of the non-optimized model
			Score score = getScore(points_, // All points
				model, // The current model parameters
				estimator_, // The estimator 
				settings.threshold, // The current threshold
				temp_inner_inliers[inl_offset], // The current inlier set
				true); // Flag to decide if the inliers are needed
			
			// Store the model of its score is higher than that of the previous best
			if (isScoreLess(so_far_the_best_score, // The so-far-the-best model's score
				score) && // The current model's score
				estimator_.isValidModel(model, // The current model parameters
					points_, // All input points
					temp_inner_inliers[inl_offset], // The inliers of the current model
					truncated_threshold)) // The truncated inlier-outlier threshold
			{				
				inl_offset = (inl_offset + 1) % 2;
				so_far_the_best_model = model; // The new so-far-the-best model
				so_far_the_best_score = score; // The new so-far-the-best model's score
				// Decide if local optimization is needed. The current criterion requires a minimum number of iterations
				// and number of inliers before applying GC.
				do_local_optimization = statistics.iteration_number > settings.min_iteration_number_before_lo &&
					so_far_the_best_score.I > sample_number;

				// Update the number of maximum iterations
				max_iteration = getIterationNumber(so_far_the_best_score.I, // The inlier number of the current best model
					point_number, // The number of points
					sample_number, // The sample size
					log_probability); // The logarithm of 1 - confidence
			}
		}
		
		// Apply local optimziation
		if (settings.do_local_optimization && // Input flag to decide if local optimization is needed
			do_local_optimization) // A flag to decide if all the criteria meet to apply local optimization
		{
			// Increase the number of local optimizations applied
			++statistics.local_optimization_number;

			// Graph-cut-based local optimization 
			graphCutLocalOptimization(points_, // All points
				temp_inner_inliers[inl_offset], // Inlier set of the current so-far-the-best model
				so_far_the_best_model, // Best model parameters
				so_far_the_best_score, // Best model score
				estimator_, // Estimator
				settings.max_local_optimization_number); // Maximum local optimization steps

			// Update the maximum number of iterations variable
			max_iteration = 
				getIterationNumber(so_far_the_best_score.I, // The current inlier number
					point_number, // The number of points
					sample_number, // The sample size
					log_probability); // log(1 - confidence)
		}

		// Apply time limit if there is a required FPS set
		if (settings.desired_fps > -1)
		{
			end = std::chrono::system_clock::now(); // The current time
			elapsed_seconds = end - start; // Time elapsed since the algorithm started

			// Interrupt the algorithm if the time limit is exceeded
			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}

	// Apply a final local optimization if it hasn't been applied yet
	if (settings.do_local_optimization && 
		statistics.local_optimization_number == 0)
	{
		// Increase the number of local optimizations applied
		++statistics.local_optimization_number;

		// Graph-cut-based local optimization 
		graphCutLocalOptimization(points_, // All points
			temp_inner_inliers[inl_offset], // Inlier set of the current so-far-the-best model
			so_far_the_best_model, // Best model parameters
			so_far_the_best_score, // Best model score
			estimator_, // Estimator
			settings.max_local_optimization_number); // Maximum local optimization steps
	}

	// Recalculate the score if needed (i.e. there is some inconstistency in
	// in the number of inliers stored and calculated).
	if (temp_inner_inliers[inl_offset].size() != so_far_the_best_score.I)
		Score score = getScore(points_, // All points
			so_far_the_best_model, // Best model parameters
			estimator_, // The estimator
			settings.threshold, // The inloer-outlier threshold
			temp_inner_inliers[inl_offset]); // The current inliers
	 
	// Apply iteration least-squares fitting to get the final model parameters if needed
	if (settings.do_final_iterated_least_squares)
	{
		Model model;
		bool success = iteratedLeastSquaresFitting(
			points_, // The input data points
			estimator_, // The model estimator
			settings.threshold, // The inlier-outlier threshold
			temp_inner_inliers[inl_offset], // The resulting inlier set
			model); // The estimated model

		if (success)
			so_far_the_best_model.descriptor = model.descriptor;
	} 
	else // Otherwise, do only one least-squares fitting on all of the inliers
	{
		// Estimate the final model using the full inlier set
		std::vector<Model> models;
		estimator_.estimateModelNonminimal(points_,
			&(temp_inner_inliers[inl_offset])[0],
			so_far_the_best_score.I,
			&models);

		if (models.size() > 0)
			so_far_the_best_model.descriptor = models[0].descriptor;
	}

	// Return the inlier set and the estimated model parameters
	statistics.inliers.swap(temp_inner_inliers[inl_offset]);
	obtained_model_ = so_far_the_best_model;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::iteratedLeastSquaresFitting(
	const cv::Mat &points_,
	const ModelEstimator &estimator_,
	const double threshold_,
	std::vector<int> &inliers_,
	Model &model_)
{
	const size_t sample_size = estimator_.sampleSize(); // The minimal sample size
	if (inliers_.size() <= sample_size) // Return if there are not enough points
		return false;

	constexpr size_t max_iterations = std::numeric_limits<size_t>::max(); // Maximum number of least-squares iterations
	size_t iterations = 0; // Number of least-squares iterations
	std::vector<int> tmp_inliers; // Inliers of the current model

	// Iterated least-squares model fitting
	while (++iterations < max_iterations)
	{
		std::vector<Model> models; // Estimated models

		// Estimate the model from the current inlier set
		estimator_.estimateModelNonminimal(points_,
			&(inliers_)[0], // The current inliers
			inliers_.size(), // The number of inliers
			&models); // The estimated model parameters

		if (models.size() == 0) // If there is no model estimated, interrupt the procedure
			break;
		if (models.size() == 1) // If a single model is estimated we do not have to care about selecting the best
		{
			// Calculate the score of the current model
			tmp_inliers.resize(0);
			Score score = getScore(points_, // All points
				models[0], // The current model parameters
				estimator_, // The estimator 
				threshold_, // The current threshold
				tmp_inliers); // The current inlier set

			// Break if the are not enough inliers
			if (tmp_inliers.size() < sample_size)
				break;
			
			// Interrupt the procedure if the inlier number has not changed.
			// Therefore, the previous and current model parameters are likely the same.
			if (score.I == inliers_.size())
				break;

			// Update the output model
			model_ = models[0];
			// Store the inliers of the new model
			inliers_.swap(tmp_inliers);
		}
		else // If multiple models are estimated select the best (i.e. the one having the highest score) one
		{
			Score best_score; // The score of the best estimated model
			bool updated = false; // A flag determining if the model is updated

			// Evaluate all the estimated models to find the best
			for (const auto &model : models)
			{
				// Calculate the score of the current model
				tmp_inliers.resize(0);
				Score score = getScore(points_, // The input data points
					model, // The model parameters
					estimator_, // The estimator
					threshold_, // The inlier-outlier threshold
					tmp_inliers); // The inliers of the current model

				// Continue if the are not enough inliers
				if (tmp_inliers.size() < sample_size)
					continue;

				// Do not test the model if the inlier number has not changed.
				// Therefore, the previous and current model parameters are likely the same.
				if (score.I == inliers_.size())
					continue;

				// Update the model if its score is higher than that of the current best
				if (score.J > best_score.J)
				{
					updated = true; // Set a flag saying that the model is updated, so the process should continue
					best_score = score; // Store the new score
					model_ = model; // Store the new model
					inliers_.swap(tmp_inliers); // Store the inliers of the new model
				}
			}

			// If the model has not been updated, interrupt the procedure
			if (!updated)
				break;
		}
	}

	// If there were more than one iterations, the procedure is considered successfull
	return iterations > 1;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::sample(
	const cv::Mat &points_, // The input data points
	const std::vector<int> &pool_, // The pool if indices determining which point can be selected
	const std::vector<std::vector<cv::DMatch>> &neighbors_,
	int sample_number_,
	int *sample_)
{
	if (pool_.size() == points_.rows) // Apply PROSAC sampler when sampling from the whole point set
	{
		prosac_sampler->sample(
			points_, // All data points
			sample_); // The currently selected sample
	}
	else // Apply uniform random sampling otherwise
	{
		random_generator->generateUniqueRandomSet(sample_,
			sample_number_,
			pool_.size() - 1);
		for (auto sample_idx = 0; sample_idx < sample_number_; ++sample_idx)
			sample_[sample_idx] = pool_[sample_[sample_idx]];
	}
	return true;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::graphCutLocalOptimization(const cv::Mat &points_, 
	std::vector<int> &so_far_the_best_inliers_,
	Model &so_far_the_best_model_,
	Score &so_far_the_best_score_,
	const ModelEstimator &estimator_, 
	const int trial_number_)
{
	const auto inlier_limit = estimator_.inlierLimit(); // Number of points used in the inner RANSAC
	Score max_score = so_far_the_best_score_; // The current best score
	Model best_model = so_far_the_best_model_; // The current best model
	std::vector<Model> models; // The estimated models' parameters
	std::vector<int> best_inliers, // Inliers of the best model
		inliers; // The inliers of the current model
	bool updated; // A flag to see if the model is updated
	double energy; // The energy after applying the graph-cut algorithm

	// Increase the number of the local optimizations applied
	++statistics.local_optimization_number; 

	// Apply the graph-cut-based local optimization
	while (++statistics.graph_cut_number < settings.max_graph_cut_number)
	{
		// In the beginning, the best model is not updated
		updated = false;

		// Clear the inliers
		inliers.resize(0);

		// Apply the graph-cut-based inlier/outlier labeling.
		// The inlier set will contain the points closer than the threshold and
		// their neighbors depending on the weight of the spatial coherence term.
		labeling(
			points_, // The input points
			statistics.neighbor_number, // The number of neighbors, i.e. the edge number of the graph 
			neighbours, // The neighborhood graph
			best_model, // The best model parameters
			estimator_, // The model estimator
			settings.spatial_coherence_weight, // The weight of the spatial coherence term
			settings.threshold, // The inlier-outlier threshold
			inliers, // The selected inliers
			energy); // The energy after the procedure

		// Number of points (i.e. the sample size) used in the inner RANSAC
		size_t sample_size = static_cast<int>(MIN(inlier_limit, inliers.size()));

		// The current sample used in the inner RANSAC
		std::unique_ptr<int[]> current_sample(new int[sample_size]);

		// Run an inner RANSAC on the inliers coming from the graph-cut algorithm
		for (auto trial = 0; trial < trial_number_; ++trial)
		{
			// Reset the model vector
			models.resize(0);
			if (sample_size < inliers.size()) // If there are more inliers available than the minimum number, sample randomly.
			{
				sample(points_, // The input data points
					inliers, // The inliers used for the selection
					neighbours, // The neighborhood structure
					sample_size, // The size of the minimal sample
					current_sample.get()); // The selected sample

				// Apply least-squares model fitting to the selected points.
				// If it fails, continue the for cycle and, thus, the sampling.
				if (!estimator_.estimateModelNonminimal(points_,  // The input data points
					current_sample.get(),  // The selected sample
					sample_size, // The size of the sample
					&models)) // The estimated model parameter
					continue;
			}
			else if (estimator_.sampleSize() < inliers.size()) // If there are enough inliers to estimate the model, use all of them
			{
				// Clear the current sample and use all of the inliers
				current_sample.release();
				current_sample = std::unique_ptr<int[]>(&inliers[0]);

				// Apply least-squares model fitting to the selected points.
				// If it fails, break the for cycle since we have used all inliers for this step.
				if (!estimator_.estimateModelNonminimal(points_, // The input data points
					current_sample.get(),  // The selected sample
					inliers.size(), // The size of the sample
					&models)) // The estimated model parameter
					break;
			}
			else // Otherwise, break the for cycle.
				break;

			// Select the best model from the estimated set
			for (const auto &model : models)
			{
				std::vector<int> tmp_inliers;

				// Calculate the score of the current model
				Score score = getScore(points_, // The input data points
					model, // The estimated model parameters
					estimator_, // The model estimator
					settings.threshold, // The inlier-outlier threshold
					tmp_inliers, // The inliers of the estimated model
					true); // Flag saying that we do not need the inlier set

				// If this model is better than the previous best, update.
				if (isScoreLess(max_score, // The best score
					score)) // The current score
				{
					updated = true; // Flag saying that we have updated the model parameters
					max_score = score; // Store the new best score
					best_model = model; // Store the new best model parameters
					best_inliers.swap(tmp_inliers);
					tmp_inliers.clear();
				}
			}
		}

		// If the model is not updated, interrupt the procedure
		if (!updated)
			break;
	}

	// If the new best score is better than the original one, update the model parameters.
	if (isScoreLess(so_far_the_best_score_, // The original best score
		max_score)) // The best score of the local optimization
	{
		so_far_the_best_score_ = max_score; // Store the new best score
		so_far_the_best_model_ = best_model;
		so_far_the_best_inliers_.swap(best_inliers);
		best_inliers.clear();
		return true;
	}
	return false;
}


template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::getScore(const cv::Mat &points_, // The input data points
	const Model &model_, // The current model parameters
	const ModelEstimator &estimator_, // The model estimator
	const double threshold_, // The inlier-outlier threshold
	std::vector<int> &inliers_, // The selected inliers
	bool store_inliers_) // A flag determining if the inliers should be stored
{
	Score score; // The current score
	if (store_inliers_) // If the inlier should be stored, clear the variables
		inliers_.clear();
	
	// Containers for the parallel threads
	std::vector<std::vector<int>> process_inliers;
	if (store_inliers_) // If the inlier should be stored, set the container sizes
		process_inliers.resize(settings.core_number);

	// Scores for the parallel threads
	std::vector<Score> process_scores(settings.core_number);

#ifdef USE_OPENMP
#pragma omp for
#endif
  for (auto process = 0; process < settings.core_number; process++) {
		if (store_inliers_) // If the inlier should be stored, occupy the memory for the inliers
			process_inliers[process].reserve(step_size);
		const int start_idx = process * step_size; // The starting point's index
		const int end_idx = MIN(points_.rows - 1, (process + 1) * step_size); // The last point's index
		
		double residual; // The point-to-model residual
		// Iterate through all points, calculate the residuals and store the points as inliers if needed.
		for (auto point_idx = start_idx; point_idx < end_idx; ++point_idx)
		{
			// Calculate the point-to-model residual
			residual = estimator_.residual(points_.row(point_idx), model_);
			
			// If the residual is smaller than the threshold, store it as an inlier and
			// increase the score.
			if (residual < truncated_threshold)
			{
				if (store_inliers_) // Store the point as an inlier if needed.
					process_inliers[process].emplace_back(point_idx);

				// Increase the inlier number
				++(process_scores[process].I);
				// Increase the score. The original truncated quadratic loss is as follows: 
				// 1 - residual^2 / threshold^2. For RANSAC, -residual^2 is enough.
				process_scores[process].J += -residual * residual; // Truncated quadratic cost
			}
		}
	}

	// Merge the results of the parallel threads
	for (auto i = 0; i < settings.core_number; ++i)
	{
		score.I += process_scores[i].I;
		score.J += process_scores[i].J;

		if (store_inliers_)
			copy(process_inliers[i].begin(), process_inliers[i].end(), back_inserter(inliers_));
	}

	// Return the final score
	return score;
}


template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::labeling(const cv::Mat &points_, 
	int neighbor_number_, 
	const std::vector<std::vector<cv::DMatch>> &neighbors_,
	Model &model_,
	ModelEstimator estimator_,
	double lambda_,
	double threshold_,
	std::vector<int> &inliers_,
	double &energy_)
{
	// Initializing the problem graph for the graph-cut algorithm.
	Energy<double, double, double> *problem_graph = 
		new Energy<double, double, double>(points_.rows, // The number of vertices
			neighbor_number_, // The number of edges
			NULL);

	// Add a vertex for each point
	for (auto i = 0; i < points_.rows; ++i)
		problem_graph->add_node();

	// The distance and energy for each point
	double tmp_distance, 
		tmp_energy;

	// Estimate the vertex capacities
	for (auto i = 0; i < points_.rows; ++i)
	{
		tmp_distance = static_cast<double>(estimator_.residual(points_.row(i), model_));
		tmp_energy = 1.0 - tmp_distance * tmp_distance / sqr_threshold_2;
		problem_graph->add_term1(i, tmp_energy, 0);
	}

	if (lambda_ > 0)
	{
		double distance1, distance2;
		double energy1, energy2;
		double e00, e01, e10, e11;
		int actual_neighbor_idx;

		// Iterate through all points and set their edges
		for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
		{
			distance1 = static_cast<double>(estimator_.residual(points_.row(point_idx), model_));
			energy1 = MAX(0, 1 - distance1 * distance1 / truncated_threshold_2); // Truncated quadratic cost

			// Iterate through  all neighbors
			for (auto neighbor_idx = 0; neighbor_idx < neighbors_[point_idx].size(); ++neighbor_idx)
			{
				actual_neighbor_idx = neighbors_[point_idx][neighbor_idx].trainIdx;
				
				if (actual_neighbor_idx == point_idx)
					continue;

				distance2 = static_cast<double>(estimator_.residual(points_.row(actual_neighbor_idx), model_));
				energy2 = MAX(0, 1 - distance2 * distance2 / truncated_threshold_2); // Truncated quadratic cost
								
				e00 = 0.5 * (energy1 + energy2);
				e01 = 1.0;
				e10 = 1.0;
				e11 = 1.0 - 0.5 * (energy1 + energy2);

				if (e00 + e11 > e01 + e10)
					printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

				problem_graph->add_term2(point_idx, // The current point's index
					actual_neighbor_idx, // The current neighbor's index
					e00 * lambda_, 
					e01 * lambda_, 
					e10 * lambda_, 
					e11 * lambda_);
			}
		}
	}

	// Run the standard st-graph-cut algorithm
	problem_graph->minimize();

	// Select the inliers, i.e. the points labeled as SINK.
	inliers_.reserve(points_.rows);
	for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
		if (problem_graph->what_segment(point_idx) == Graph<double, double, double>::SINK)
			inliers_.emplace_back(point_idx);
	 
	// Clean the memory
	delete problem_graph;
}