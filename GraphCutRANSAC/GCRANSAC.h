#pragma once

#include <opencv2\highgui\highgui.hpp>
#include "GCoptimization.h"
#include "prosac_sampler.h"
#include "uniform_random_generator.h"

#define PRINT_TIMES 0
#define USE_CONCURRENCY 1

/* RANSAC Scoring */
typedef struct {
	/* number of inliers_, rectangular gain function */
	unsigned I;
	/* MSAC scoring, truncated quadratic gain function */
	float J;
} Score;

template <class ModelEstimator, class Model>
class GCRANSAC
{
public:

	GCRANSAC() : time_limit(FLT_MAX), 
		desired_fps(-1)
	{
		if (random_generator == nullptr)
			random_generator = std::make_unique<UniformRandomGenerator>();
	}
	~GCRANSAC() { }

	// The main method applying Graph-Cut RANSAC to the input data points_
	void run(const cv::Mat &points_,  // Data points_
		ModelEstimator estimator_, // The model_ estimator
		Model &obtained_model_,  // The output model
		std::vector<int> &obtained_inliers_, // The output inlier set
		int &iteration_number_, // The number of iterations required
		float threshold_,  // The threshold for mode_ estimation
		float spatial_coherence_weight_, // The weight of the spatial coherence term
		float sphere_size_, // The sphere'score radius for neighborhood computation 
		float probability_ = 0.05f, // The probability of finding an all inlier current_sample, default 
		bool use_graph_cut_ = true,  // True - GC-RANSAC; false - LO-RANSAC 
		bool use_inlier_limit_ = true,  // Use a subset of the inliers_ for the local optimization step or not
		int local_optimization_limit_ = 20,  // The iteration limit for the local optimization step
		bool apply_local_optimization_ = true);  // Apply local optimization or not (basically a simple RANSAC)

	bool sample(const cv::Mat &points_, 
		std::vector<int> &pool_, 
		const std::vector<std::vector<cv::DMatch>> &neighbors_, 
		int sample_number_, 
		int *sample_);

	float get_energy() { return obtained_energy; } // Return the energy_ of the obtained solution
	void set_fps(int fps_) { desired_fps = fps_; time_limit = 1.0f / fps_; } // Set a desired FPS value

	int get_lo_number() { return number_of_local_optimizations; } // Return the number of the applied local optimization steps
	int get_gc_number() { return gc_number; } // Return the number of the applied graph-cuts

	// Return the score of a model_ w.r.t. the data points_ and the threshold_
	Score get_score(const cv::Mat &points_, // The data points_
		const Model &model_, // The current model_
		const ModelEstimator &estimator_, // The model_ estimator_
		const float threshold_, // The threshold_ for model_ estimation
		std::vector<int> &inliers_, // The inlier set
		bool store_inliers_ = true); // Store the inliers_ or not

	// Decides whether score s2_ is higher than s1_
	int is_score_less(const Score &s1_, // Input score
		const Score &s2_, // Input score
		int type_ = 1) // Decision type_: 1 -- RANSAC-like, 2 -- MSAC-like 
	{ 
		return type_ == 1 ? (s1_.I < s2_.I || (s1_.I == s2_.I && s1_.J < s2_.J)) : s1_.J < s2_.J; 
	}

protected:
	std::unique_ptr<theia::ProsacSampler<cv::Mat>> prosac_sampler;
	std::unique_ptr<UniformRandomGenerator> random_generator;
	int										gc_number, number_of_local_optimizations;		// The applied LO and GC steps
	float									time_limit;					// The desired time limit
	int										desired_fps;				// The desired fps
	bool									apply_local_optimization;	// Apply LO or not
	int										local_optimization_limit;	// The limit for the local optimization step
	bool									use_inlier_limit;			// Use inlier limit for LO or not
	float									lambda;						// The weight for the spatial coherence term
	float									threshold;					// The threshold_ for the inlier decision
	float									obtained_energy;			// The energy_ of the result
	std::vector<std::vector<cv::DMatch>>	neighbours;					// The neighborhood structure
	int										neighbor_number;			// The number of neighbors_
	int										iteration_limit;			// The iteration limit
	int										point_number;				// The point number

	float									sqr_threshold_2;			// 2 * threshold_^2
	float									truncated_threshold;		// 3 / 2 * threshold_
	float									truncated_threshold_2;		// 9 / 4 * threshold_^2
	int										step_size;					// Step size per processes
	int										process_number;				// Number of parallel processes

	Graph<float, float, float>				*graph;						// The graph for graph-cut

	// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
	int get_iteration_number(int inlier_number_, // The inlier number
		int point_number_, // The point number
		int sample_size_, // The current_sample size
		float probability_); // The desired probability_

	// Returns a labeling w.r.t. the current model and point set
	void labeling(const cv::Mat &points_, // The input data points
		int neighbor_number_, // The neighbor number in the graph
		const std::vector<std::vector<cv::DMatch>> &neighbors_, // The neighborhood
		Model &model_, // The current model_
		ModelEstimator estimator_, // The model estimator
		float lambda_, // The weight for the spatial coherence term
		float threshold_, // The threshold_ for the inlier-outlier decision
		std::vector<int> &inliers_, // The resulting inlier set
		float &energy_); // The resulting energy

	// Apply the local optimization step of LO-RANSAC
	bool full_local_optimization(const cv::Mat &points_, // The input data points_
		std::vector<int> &so_far_the_best_inliers_, // The input, than the resulting inlier set
		Model &so_far_the_best_model_, // The current model
		Score &so_far_the_best_score_, // The current score
		int &max_iteration_, // The iteration limit
		const ModelEstimator &estimator_, // The model_ estimator
		const int trial_number_, // The max trial number
		const float probability_); // The desired probability

	// Apply one local optimization step of LO-RANSAC
	Score one_step_local_optimization(const cv::Mat &points_, // The input data points_
		const float threshold_, // The threshold_ for the inlier-outlier decision
		float inner_threshold_, // The inner threshold_ for LO
		const int inlier_limit_, // The inlier limit
		Model &model_, // The resulting model_
		const ModelEstimator &estimator_, // The model_ estimator_
		std::vector<int> &inliers_, // The resulting inlier set
		int lsq_number_ = 4); // The number of least-squares fitting

	// Apply the graph-cut optimization for GC-RANSAC
	bool graph_cut_local_optimization(const cv::Mat &points_, // The input data points_
		std::vector<int> &so_far_the_best_inliers_, // The input, than the resulting inlier set
		Model &so_far_the_best_model_, // The current model_
		Score &so_far_the_best_score_, // The current score
		int &max_iteration_, // The iteration limit
		const ModelEstimator &estimator_, // The model_ estimator_
		const int trial_number_, // The max trial number
		const float probability_); // The desired probability_
};

// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
template <class ModelEstimator, class Model>
int GCRANSAC<ModelEstimator, Model>::get_iteration_number(int inlier_number_, 
	int point_number_, 
	int sample_size_,  
	float probability_)
{
	float q = pow(static_cast<float>(inlier_number_) / point_number_, sample_size_);

	float iter = log(probability_) / log(1 - q);
	if (iter < 0)
		return INT_MAX;
	return static_cast<int>(iter) + 1;
}

// The main method applying Graph-Cut RANSAC to the input data points_
template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::run(const cv::Mat &points_, 
	ModelEstimator estimator_,
	Model &obtained_model_,
	std::vector<int> &obtained_inliers_,
	int &iteration_number_,
	float threshold_, 
	float spatial_coherence_weight_, 
	float sphere_size_,
	float probability_, 
	bool use_graph_cut_, 
	bool use_inlier_limit_,
	int local_optimization_limit_,
	bool apply_local_optimization_)
{
	// Initialization
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	const int min_iteration = 0;
	const int knn = 6;
	const int iter_before_lo = 50;
	const int sample_number = estimator_.SampleSize();
	const float distance = sphere_size_;

	this->apply_local_optimization = apply_local_optimization_;
	this->local_optimization_limit = local_optimization_limit_;
	this->use_inlier_limit = use_inlier_limit_;
	this->threshold = threshold_;
	this->lambda = spatial_coherence_weight_;

	float final_energy = 0;
	auto iteration = 0;
	auto max_iteration = get_iteration_number(1, points_.rows, sample_number, probability_);
	auto lo_count = 0;
	int *current_sample = new int[sample_number];
	bool do_local_optimization = false;
	int inl_offset = 0;
	int counter_for_inlier_vecs[] = { 0,0 };
	Model so_far_the_best_model;
	Score so_far_the_best_score = { 0,0 };
	std::vector<std::vector<int>> temp_inner_inliers(2);
	std::vector<int> *so_far_the_best_inlier_indices = NULL;

	prosac_sampler = std::make_unique<theia::ProsacSampler<cv::Mat>>(estimator_.SampleSize());
	prosac_sampler->Initialize();

	iteration_limit = 5000;
	gc_number = 0;
	number_of_local_optimizations = 0;
	point_number = points_.rows;
	truncated_threshold = 3.0f / 2.0f * threshold_;
	truncated_threshold_2 = truncated_threshold * truncated_threshold;
	sqr_threshold_2 = 2 * threshold_ * threshold_;
	process_number = 8;
	step_size = points_.rows / process_number;
	random_generator->reset_generator(0, point_number);

	// Initialize the pool_ for sampling (TODO: change sampler)
	std::vector<int> pool(point_number);
	for (auto i = 0; i < point_number; ++i)
		pool[i] = i;
	
	// Compute the neighborhood graph
	cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));
	neighbours.resize(0);
	if (spatial_coherence_weight_ > 0.0) // Compute the neighborhood if the weight is not zero
	{
		flann.radiusMatch(points_, points_, neighbours, distance);

		neighbor_number = 0;
		for (auto i = 0; i < neighbours.size(); ++i)
			neighbor_number += static_cast<int>(neighbours[i].size()) - 1;
	}

	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	// The main RANSAC iteration
	while (min_iteration > iteration || iteration < MIN(max_iteration, iteration_limit))
	{
		do_local_optimization = false;
		++iteration;

		// current_sample a minimal subset
		std::vector<Model> models;

		while (1) // current_sample while needed
		{
			if (!sample(points_, pool, neighbours, sample_number, current_sample)) // Sampling
				continue;
			 
 			if (estimator_.EstimateModel(points_, current_sample, &models)) // Estimate (and validate) models using the current current_sample
				break; 
		}      

		// Select the so-far-the-best from the estimated models
		for (auto model_idx = 0; model_idx < models.size(); ++model_idx)
		{
			// Get the inliers_ of the non-optimized model_
			Score score = get_score(points_, 
				models[model_idx], 
				estimator_, 
				threshold_, 
				temp_inner_inliers[inl_offset], 
				false);
			
			// Store the model_ of its score is higher than that of the previous best
			if (is_score_less(so_far_the_best_score, score))
			{
				inl_offset = (inl_offset + 1) % 2;
				
				so_far_the_best_model = models[model_idx];
				so_far_the_best_score = score;
				do_local_optimization = iteration > iter_before_lo;
				max_iteration = get_iteration_number(so_far_the_best_score.I, points_.rows, sample_number, probability_);
			}
		}

		// Decide whether a local optimization is needed or not
		if (iteration > iter_before_lo && lo_count == 0 && so_far_the_best_score.I > 7)
			do_local_optimization = true;

		// Apply local optimziation
		if (apply_local_optimization_ && do_local_optimization)
		{
			++lo_count;

			/* Graph-Cut-based Local Optimization */
			if (use_graph_cut_)
				graph_cut_local_optimization(points_,
					temp_inner_inliers[inl_offset],
					so_far_the_best_model,
					so_far_the_best_score,
					max_iteration, 
					estimator_,
					local_optimization_limit_,
					probability_);
			else
				full_local_optimization(points_,
					temp_inner_inliers[inl_offset],
					so_far_the_best_model,
					so_far_the_best_score,
					max_iteration,
					estimator_,
					local_optimization_limit_,
					probability_);

			max_iteration = get_iteration_number(so_far_the_best_score.I, point_number, sample_number, probability_);
		}

		// Apply time limit
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			if (elapsed_seconds.count() > time_limit)
				break;
		}
	}
	delete current_sample;

	// Apply a final local optimization if it hasn't been applied yet
	if (apply_local_optimization_ && lo_count == 0)
	{
		++lo_count;

		/* Graph-Cut-based Local Optimization */
		if (use_graph_cut_)
			graph_cut_local_optimization(points_,
				temp_inner_inliers[inl_offset],
				so_far_the_best_model,
				so_far_the_best_score,
				max_iteration,
				estimator_,
				local_optimization_limit_,
				probability_);
		else
			full_local_optimization(points_,
				temp_inner_inliers[inl_offset],
				so_far_the_best_model,
				so_far_the_best_score,
				max_iteration,
				estimator_,
				local_optimization_limit_,
				probability_);
	}

	// Recalculate the score if needed
	if (temp_inner_inliers[inl_offset].size() != so_far_the_best_score.I)
		Score score = get_score(points_, so_far_the_best_model, estimator_, threshold_, temp_inner_inliers[inl_offset]);
	 
	// Estimate the final model_ using the full inlier set
	std::vector<Model> models;
	estimator_.EstimateModelNonminimal(points_, &(temp_inner_inliers[inl_offset])[0], so_far_the_best_score.I, &models);

	if (models.size() > 0)
		so_far_the_best_model.descriptor = models[0].descriptor;

	obtained_inliers_ = temp_inner_inliers[inl_offset];
	obtained_model_ = so_far_the_best_model;
	iteration_number_ = iteration;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::sample(const cv::Mat &points_,
	std::vector<int> &pool_,
	const std::vector<std::vector<cv::DMatch>> &neighbors_,
	int sample_number_,
	int *sample_)
{
	if (pool_.size() == points_.rows) // Apply PROSAC sampler when sampling from the whole point set
	{
		prosac_sampler->Sample(points_,
			sample_);
	}
	else // Apply uniform random sampling otherwise
	{
		random_generator->generate_unique_random_set(sample_,
			sample_number_,
			pool_.size() - 1);
		for (auto sample_idx = 0; sample_idx < sample_number_; ++sample_idx)
			sample_[sample_idx] = pool_[sample_[sample_idx]];
	}
	return true;
}

template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::one_step_local_optimization(const cv::Mat &points_,
	const float threshold_,
	float inner_threshold_,
	const int inlier_limit_,
	Model &model_,
	const ModelEstimator &estimator_,
	std::vector<int> &inliers_,
	int lsq_number_)
{
	Score score = {0,0}, tmp_score, max_score;

	max_score = get_score(points_, 
		model_, 
		estimator_, 
		threshold_, 
		inliers_);

	if (max_score.I < 8)
		return score;

	float dth = (inner_threshold_ - threshold_) / 4.0f;

	std::vector<Model> models;
	if (max_score.I < inlier_limit_)
	{
		int *sample = &inliers_[0];
		if (!estimator_.EstimateModelNonminimal(points_, 
			sample, 
			static_cast<int>(inliers_.size()), 
			&models))
			return max_score;
	} else
	{
		int *sample = new int[inlier_limit_];
		std::vector<int> pool = inliers_;
		for (auto i = 0; i < inlier_limit_; ++i)
		{
			int idx = static_cast<int>(round((pool.size() - 1) * static_cast<float>(rand()) / RAND_MAX));
			sample[i] = pool[idx];
			pool.erase(pool.begin() + idx);
		}

		if (!estimator_.EstimateModelNonminimal(points_, sample, inlier_limit_, &models))
		{
			delete sample;
			return max_score;
		}
	}

	/* iterate */
	for (auto it = 0; it < lsq_number_; ++it)
	{
		score = get_score(points_, models[0], estimator_, threshold_, inliers_);
		tmp_score = get_score(points_, models[0], estimator_, inner_threshold_, inliers_);

		if (is_score_less(max_score, score)) 
		{
			max_score = score;
			model_ = models[0];
		}

		if (tmp_score.I < 8) {
			return max_score;
		}

		if (tmp_score.I <= inlier_limit_) { /* if we are under the limit, just use what we have without shuffling */

			models.resize(0);
			int *sample = &inliers_[0];
			if (!estimator_.EstimateModelNonminimal(points_, sample, inliers_.size(), &models))
				return max_score;
			
		}
		else {
			int *sample = new int[inlier_limit_];
			std::vector<int> pool = inliers_;
			for (auto i = 0; i < inlier_limit_; ++i)
			{
				int idx = static_cast<int>(round((pool.size() - 1) * static_cast<float>(rand()) / RAND_MAX));
				sample[i] = pool[idx];
				pool.erase(pool.begin() + idx);
			}

			if (!estimator_.EstimateModelNonminimal(points_, sample, inlier_limit_, &models))
			{
				delete sample;
				return max_score;
			}
		}

		inner_threshold_ -= dth;
	}
	return max_score;
}


template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::full_local_optimization(const cv::Mat &points_,
	std::vector<int> &so_far_the_best_inliers_,
	Model &so_far_the_best_model_,
	Score &so_far_the_best_score_,
	int &max_iteration_,
	const ModelEstimator &estimator_,
	const int trial_number_,
	const float probability_)
{
	Score score, max_score = { 0,0 };
	Model best_model;
	std::vector<int> best_inliers;

	++number_of_local_optimizations;

	if (so_far_the_best_score_.I < 16) 
		return false;
	
	int sample_size = static_cast<int>(MIN(14, so_far_the_best_inliers_.size() / 2));

	get_score(points_, 
		so_far_the_best_model_, 
		estimator_, 
		threshold, 
		so_far_the_best_inliers_, 
		true);
	std::vector<int> pool = so_far_the_best_inliers_;
	
	int *current_sample = new int[sample_size];
	std::vector<int> inliers;
	std::vector<Model> models;
	for (auto i = 0; i < trial_number_; ++i)
	{
		inliers.resize(0);
		models.resize(0);
		sample(points_, 
			pool, 
			neighbours, 
			sample_size, 
			current_sample);

		if (!estimator_.EstimateModelNonminimal(points_, 
			current_sample, 
			sample_size, 
			&models))
			continue;

		if (models.size() == 0 || models[0].descriptor.rows != 3)
			continue;

		score = one_step_local_optimization(points_, 
			threshold, 
			4.0f * threshold, 
			use_inlier_limit ? estimator_.InlierLimit() : INT_MAX, 
			models[0], 
			estimator_, 
			inliers, 
			local_optimization_limit ? 1 : 4);

		if (is_score_less(max_score, score))
		{
			max_score = score;
			best_model = models[0];
			best_inliers = inliers;
		}
	}

	inliers.resize(0);
	models.resize(0);

	if (is_score_less(so_far_the_best_score_, 
		max_score))
	{
		so_far_the_best_model_.descriptor = best_model.descriptor;
		so_far_the_best_inliers_ = best_inliers;
		so_far_the_best_score_ = max_score;
		so_far_the_best_score_.I = static_cast<unsigned int>(so_far_the_best_inliers_.size());
	}
}


template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::graph_cut_local_optimization(const cv::Mat &points_, 
	std::vector<int> &so_far_the_best_inliers_,
	Model &so_far_the_best_model_,
	Score &so_far_the_best_score_,
	int &max_number_of_iterations_, 
	const ModelEstimator &estimator_, 
	const int trial_number_,
	const float probability_)
{
	const auto inlier_limit = estimator_.InlierLimit();
	Score max_score = so_far_the_best_score_;
	Model best_model = so_far_the_best_model_;
	std::vector<int> best_inliers;

	std::vector<int> inliers;
	bool has_changed;
	std::vector<Model> models;
	float energy;

	++number_of_local_optimizations;

	while (1)
	{
		has_changed = false;
		inliers.resize(0);
		labeling(points_, neighbor_number, neighbours, best_model, estimator_, lambda, threshold, inliers, energy);

		++gc_number;

		int used_points = static_cast<int>(MIN(inlier_limit, inliers.size()));
		std::unique_ptr<int[]> current_sample = std::make_unique<int[]>(used_points);

		for (auto trial = 0; trial < trial_number_; ++trial)
		{
			models.resize(0);
			if (used_points < inliers.size())
			{
				sample(points_,
					inliers,
					neighbours,
					used_points,
					current_sample.get());
			}
			else
			{
				current_sample.release();
				current_sample = std::unique_ptr<int[]>(&inliers[0]);
			}

			if (!estimator_.EstimateModelNonminimal(points_, 
				current_sample.get(), 
				used_points, 
				&models))
				break;

			for (auto i = 0; i < models.size(); ++i)
			{
				Score score = get_score(points_, 
					models.at(i), 
					estimator_, 
					threshold, 
					inliers, 
					false);

				if (is_score_less(max_score, score))
				{
					has_changed = true;

					max_score = score;
					max_score.I = static_cast<int>(score.I);
					best_model = models[i];
				}
			}
		}

		if (!has_changed)
			break;
	}

	if (is_score_less(so_far_the_best_score_, 
		max_score))
	{
		so_far_the_best_score_ = max_score;
		so_far_the_best_model_.descriptor = best_model.descriptor;

		so_far_the_best_inliers_.resize(best_inliers.size());
		for (auto point_idx = 0; point_idx < best_inliers.size(); ++point_idx)
			so_far_the_best_inliers_[point_idx] = best_inliers[point_idx];
		return true;
	}
	return false;
}


template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::get_score(const cv::Mat &points_, 
	const Model &model_, 
	const ModelEstimator &estimator_, 
	const float threshold_, 
	std::vector<int> &inliers_, 
	bool store_inliers_)
{
	Score score = { 0,0 };
	if (store_inliers_)
		inliers_.resize(0);
	
	std::vector<std::vector<int>> process_inliers;
	if (store_inliers_)
		process_inliers.resize(process_number);

	std::vector<Score> process_scores(process_number, { 0,0 });

	concurrency::parallel_for(0, process_number, [&](int process)
	{
		if (store_inliers_)
			process_inliers[process].reserve(step_size);
		const int start_idx = process * step_size;
		const int end_idx = MIN(points_.rows - 1, (process + 1) * step_size);
		float distance;

		for (auto point_idx = start_idx; point_idx < end_idx; ++point_idx)
		{
			distance = static_cast<float>(estimator_.Error(points_.row(point_idx), model_));
			
			if (distance < truncated_threshold)
			{
				if (store_inliers_)
					process_inliers[process].push_back(point_idx);

				++(process_scores[process].I);
				process_scores[process].J += 1.0f - distance * distance / truncated_threshold_2; // Truncated quadratic cost
			}
		}
	});

	for (auto i = 0; i < process_number; ++i)
	{
		score.I += process_scores[i].I;
		score.J += process_scores[i].J;

		if (store_inliers_)
			copy(process_inliers[i].begin(), process_inliers[i].end(), back_inserter(inliers_));
	}

	return score;
}


template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::labeling(const cv::Mat &points_, 
	int neighbor_number_, 
	const std::vector<std::vector<cv::DMatch>> &neighbors_,
	Model &model_,
	ModelEstimator estimator_,
	float lambda_,
	float threshold_,
	std::vector<int> &inliers_,
	float &energy_)
{
	Energy<float, float, float> *problem_graph = new Energy<float, float, float>(points_.rows, // poor guess at number of pairwise terms needed :(
		neighbor_number_,
		NULL);

	for (auto i = 0; i < points_.rows; ++i)
		problem_graph->add_node();

	float tmp_distance, tmp_energy;
	for (auto i = 0; i < points_.rows; ++i)
	{
		tmp_distance = static_cast<float>(estimator_.Error(points_.row(i), model_));
		tmp_energy = exp(-(tmp_distance*tmp_distance) / sqr_threshold_2);

		problem_graph->add_term1(i, tmp_energy, 0);
	}

	if (lambda_ > 0)
	{
		float distance1, distance2;
		float energy1, energy2;
		float e00, e01, e10, e11;
		int actual_neighbor_idx;

		for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
		{
			distance1 = static_cast<float>(estimator_.Error(points_.row(point_idx), model_));
			energy1 = MAX(0, 1 - distance1 * distance1 / truncated_threshold_2); // Truncated quadratic cost

			for (auto neighbor_idx = 0; neighbor_idx < neighbors_[point_idx].size(); ++neighbor_idx)
			{
				actual_neighbor_idx = neighbors_[point_idx][neighbor_idx].trainIdx;
				
				if (actual_neighbor_idx == point_idx)
					continue;

				distance2 = static_cast<float>(estimator_.Error(points_.row(actual_neighbor_idx), model_));
				energy2 = MAX(0, 1 - distance2 * distance2 / truncated_threshold_2); // Truncated quadratic cost
								
				e00 = 0.5f * (energy1 + energy2);
				e01 = 1;
				e10 = 1;
				e11 = 1 - 0.5f * (energy1 + energy2);

				if (e00 + e11 > e01 + e10)
					printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

				problem_graph->add_term2(point_idx, 
					actual_neighbor_idx, 
					e00 * lambda_, 
					e01 * lambda_, 
					e10 * lambda_, 
					e11 * lambda_);
			}
		}
	}

	problem_graph->minimize();
	inliers_.reserve(points_.rows);
	for (auto point_idx = 0; point_idx < points_.rows; ++point_idx)
		if (problem_graph->what_segment(point_idx) == Graph<float, float, float>::SINK)
			inliers_.push_back(point_idx);
	 
	delete problem_graph;
}