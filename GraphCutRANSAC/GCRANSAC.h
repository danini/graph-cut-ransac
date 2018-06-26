#pragma once

#include <opencv2\highgui\highgui.hpp>
#include "GCoptimization.h"
#include "prosac_sampler.h"

#define PRINT_TIMES 0
#define USE_CONCURRENCY 1

/* RANSAC Scoring */
typedef struct {
	/* number of inliers, rectangular gain function */
	unsigned I;
	/* MSAC scoring, truncated quadratic gain function */
	float J;
} Score;

template <class ModelEstimator, class Model>
class GCRANSAC
{
public:

	GCRANSAC() { time_limit = FLT_MAX; desired_fps = -1; }
	~GCRANSAC() { }

	// The main method applying Graph-Cut RANSAC to the input data points
	void Run(const cv::Mat &points, // Data points
		ModelEstimator estimator, // The model estimator
		Model &obtained_model, // The output model
		std::vector<int> &obtained_inliers, // The output inlier set
		int &iteration_number,
		float threshold, // The threshold for model estimation
		float lambda, // The weight of the spatial coherence term
		float sphere_size, // The sphere's radius for neighborhood computation 
		float probability = 0.05, // The probability of finding an all inlier sample, default 
		bool user_graph_cut = true, // True - GC-RANSAC; false - LO-RANSAC 
		bool use_inlier_limit = true, // Use a subset of the inliers for the local optimization step or not
		int local_optimization_limit = 20, // The iteration limit for the local optimization step
		bool apply_local_optimization = true); // Apply local optimization or not (basically a simple RANSAC)

	bool Sample(const cv::Mat &points, 
		std::vector<int> &pool, 
		const std::vector<std::vector<cv::DMatch>> &neighbors, 
		int sample_number, 
		int *sample);

	float GetEnergy() { return obtained_energy; } // Return the energy of the obtained solution
	void SetFPS(int fps) { desired_fps = fps; time_limit = 1.0f / fps; } // Set a desired FPS value

	int GetLONumber() { return lo_number; } // Return the number of the applied local optimization steps
	int GetGCNumber() { return gc_number; } // Return the number of the applied graph-cuts

	// Return the score of a model w.r.t. the data points and the threshold
	Score GetScore(const cv::Mat &points, // The data points
		const Model &model, // The current model
		const ModelEstimator &estimator, // The model estimator
		const float threshold, // The threshold for model estimation
		std::vector<int> &inliers, // The inlier set
		bool store_inliers = true); // Store the inliers or not

	// Decides whether score s2 is higher than s1
	int ScoreLess(const Score &s1, // Input score
		const Score &s2, // Input score
		int type = 1) // Decision type: 1 -- RANSAC-like, 2 -- MSAC-like 
	{ 
		return type == 1 ? (s1.I < s2.I || (s1.I == s2.I && s1.J < s2.J)) : s1.J < s2.J; 
	}

protected:
	int										gc_number, lo_number;		// The applied LO and GC steps
	float									time_limit;					// The desired time limit
	int										desired_fps;				// The desired fps
	bool									apply_local_optimization;	// Apply LO or not
	int										local_optimization_limit;	// The limit for the local optimization step
	bool									use_inlier_limit;			// Use inlier limit for LO or not
	float									lambda;						// The weight for the spatial coherence term
	float									threshold;					// The threshold for the inlier decision
	float									obtained_energy;			// The energy of the result
	std::vector<std::vector<cv::DMatch>>	neighbours;					// The neighborhood structure
	int										neighbor_number;			// The number of neighbors
	int										iteration_limit;			// The iteration limit
	int										point_number;				// The point number

	Graph<float, float, float>				*graph;						// The graph for graph-cut

	// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
	int DesiredIterationNumber(int inlier_number, // The inlier number
		int point_number, // The point number
		int sample_size, // The sample size
		float probability); // The desired probability

	// Returns a labeling w.r.t. the current model and point set
	void Labeling(const cv::Mat &points, // The input data points
		int neighbor_number, // The neighbor number in the graph
		const std::vector<std::vector<cv::DMatch>> &neighbors, // The neighborhood
		Model &model, // The current model
		ModelEstimator estimator, // The model estimator
		float lambda, // The weight for the spatial coherence term
		float threshold, // The threshold for the inlier-outlier decision
		std::vector<int> &inliers, // The resulting inlier set
		float &energy); // The resulting energy

	// Apply the local optimization step of LO-RANSAC
	bool FullLocalOptimization(const cv::Mat &points, // The input data points
		std::vector<int> &so_far_the_best_inliers, // The input, than the resulting inlier set
		Model &so_far_the_best_model, // The current model
		Score &so_far_the_best_score, // The current score
		int &max_iteration, // The iteration limit
		const ModelEstimator &estimator, // The model estimator
		const int trial_number, // The max trial number
		const float probability); // The desired probability

	// Apply one local optimization step of LO-RANSAC
	Score OneStepLocalOptimization(const cv::Mat &points, // The input data points
		const float threshold, // The threshold for the inlier-outlier decision
		float inner_threshold, // The inner threshold for LO
		const int inlier_limit, // The inlier limit
		Model &model, // The resulting model
		const ModelEstimator &estimator, // The model estimator
		std::vector<int> &inliers, // The resulting inlier set
		int lsq_number = 4); // The number of least-squares fitting

	// Apply the graph-cut optimization for GC-RANSAC
	bool LocalOptimizationWithGraphCut(const cv::Mat &points, // The input data points
		std::vector<int> &so_far_the_best_inliers, // The input, than the resulting inlier set
		Model &so_far_the_best_model, // The current model
		Score &so_far_the_best_score, // The current score
		int &max_iteration, // The iteration limit
		const ModelEstimator &estimator, // The model estimator
		const int trial_number, // The max trial number
		const float probability); // The desired probability

	// Calculate the weight for the spatial coherence term w.r.t. to the current inlier number
	float CalculateLambda(int inlier_number);
};

// Computes the desired iteration number for RANSAC w.r.t. to the current inlier number
template <class ModelEstimator, class Model>
int GCRANSAC<ModelEstimator, Model>::DesiredIterationNumber(int inlier_number, int point_number, int sample_size,  float probability)
{
	float q = pow(static_cast<float>(inlier_number) / point_number, sample_size);

	float iter = log(probability) / log(1 - q);
	if (iter < 0)
		return INT_MAX;
	return static_cast<int>(iter) + 1;
}

// The main method applying Graph-Cut RANSAC to the input data points
template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::Run(const cv::Mat &points, 
	ModelEstimator estimator,
	Model &obtained_model,
	std::vector<int> &obtained_inliers,
	int &iteration_number,
	float threshold, 
	float lambda, 
	float sphere_size,
	float probability, 
	bool use_graph_cut, 
	bool use_inlier_limit,
	int local_optimization_limit,
	bool apply_local_optimization)
{
	// Initialization
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	const int min_iteration = 0;
	const int knn = 6;
	const int iter_before_lo = 50;
	const int sample_number = estimator.SampleSize();
	const float distance = sphere_size;

	this->apply_local_optimization = apply_local_optimization;
	this->local_optimization_limit = local_optimization_limit;
	this->use_inlier_limit = use_inlier_limit;
	this->threshold = threshold;
	this->lambda = lambda;

	float final_energy = 0;
	int iteration = 0;
	int max_iteration = DesiredIterationNumber(1, points.rows, sample_number, probability);
	int lo_count = 0;
	int *sample = new int[sample_number];
	bool do_local_optimization = false;
	int inl_offset = 0;
	int counter_for_inlier_vecs[] = { 0,0 };
	Model so_far_the_best_model;
	Score so_far_the_best_score = { 0,0 };
	std::vector<std::vector<int>> temp_inner_inliers(2);
	std::vector<int> *so_far_the_best_inlier_indices = NULL;

	iteration_limit = 5000;
	gc_number = 0;
	lo_number = 0;
	point_number = points.rows;

	// Initialize the pool for sampling (TODO: change sampler)
	std::vector<int> pool(point_number);
	for (int i = 0; i < point_number; ++i)
		pool[i] = i;
	
	// Compute the neighborhood graph
	cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));
	neighbours.resize(0);
	if (lambda > 0.0) // Compute the neighborhood if the weight is not zero
	{
		flann.radiusMatch(points, points, neighbours, distance);

		neighbor_number = 0;
		for (int i = 0; i < neighbours.size(); ++i)
			neighbor_number += static_cast<int>(neighbours[i].size()) - 1;
	}

	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	// The main RANSAC iteration
	while (min_iteration > iteration || iteration < MIN(max_iteration, iteration_limit))
	{
		do_local_optimization = false;
		++iteration;

		// Sample a minimal subset
		std::vector<Model> models;

		while (1) // Sample while needed
		{
			if (!Sample(points, pool, neighbours, sample_number, sample)) // Sampling
				continue;
			 
 			if (estimator.EstimateModel(points, sample, &models)) // Estimate (and validate) models using the current sample
				break; 
		}            

		// Select the so-far-the-best from the estimated models
		for (int model_idx = 0; model_idx < models.size(); ++model_idx)
		{
			// Get the inliers of the non-optimized model
			Score score = GetScore(points, models[model_idx], estimator, threshold, temp_inner_inliers[inl_offset], false);
			
			// Store the model of its score is higher than that of the previous best
			if (ScoreLess(so_far_the_best_score, score))
			{
				inl_offset = (inl_offset + 1) % 2;
				
				so_far_the_best_model = models[model_idx];
				so_far_the_best_score = score;
				do_local_optimization = iteration > iter_before_lo;
				max_iteration = DesiredIterationNumber(so_far_the_best_score.I, points.rows, sample_number, probability);
			}
		}

		// Decide whether a local optimization is needed or not
		if (iteration > iter_before_lo && lo_count == 0 && so_far_the_best_score.I > 7)
			do_local_optimization = true;

		// Apply local optimziation
		if (apply_local_optimization && do_local_optimization)
		{
			++lo_count;

			/* Graph-Cut-based Local Optimization */
			if (use_graph_cut)
				LocalOptimizationWithGraphCut(points,
					temp_inner_inliers[inl_offset],
					so_far_the_best_model,
					so_far_the_best_score,
					max_iteration, 
					estimator,
					local_optimization_limit,
					probability);
			else
				FullLocalOptimization(points,
					temp_inner_inliers[inl_offset],
					so_far_the_best_model,
					so_far_the_best_score,
					max_iteration,
					estimator,
					local_optimization_limit,
					probability);

			max_iteration = DesiredIterationNumber(so_far_the_best_score.I, point_number, sample_number, probability);
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
	delete sample;

	// Apply a final local optimization if it hasn't been applied yet
	if (apply_local_optimization && lo_count == 0)
	{
		++lo_count;

		/* Graph-Cut-based Local Optimization */
		if (use_graph_cut)
			LocalOptimizationWithGraphCut(points,
				temp_inner_inliers[inl_offset],
				so_far_the_best_model,
				so_far_the_best_score,
				max_iteration,
				estimator,
				local_optimization_limit,
				probability);
		else
			FullLocalOptimization(points,
				temp_inner_inliers[inl_offset],
				so_far_the_best_model,
				so_far_the_best_score,
				max_iteration,
				estimator,
				local_optimization_limit,
				probability);
	}

	// Recalculate the score if needed
	if (temp_inner_inliers[inl_offset].size() != so_far_the_best_score.I)
		Score score = GetScore(points, so_far_the_best_model, estimator, threshold, temp_inner_inliers[inl_offset]);
	 
	// Estimate the final model using the full inlier set
	std::vector<Model> models;
	estimator.EstimateModelNonminimal(points, &(temp_inner_inliers[inl_offset])[0], so_far_the_best_score.I, &models);

	if (models.size() > 0)
		so_far_the_best_model.descriptor = models[0].descriptor;

	obtained_inliers = temp_inner_inliers[inl_offset];
	obtained_model = so_far_the_best_model;
	iteration_number = iteration;
}

template <class ModelEstimator, class Model>
float GCRANSAC<ModelEstimator, Model>::CalculateLambda(int inlier_number)
{
	float lambda = (point_number * (static_cast<float>(inlier_number) / point_number)) / static_cast<float>(neighbor_number);
	return lambda;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::Sample(const cv::Mat &points, std::vector<int> &pool, const std::vector<std::vector<cv::DMatch>> &neighbors, int sample_number, int *sample)
{
	// TODO: replacable sampler
	for (int i = 0; i < sample_number; ++i)
	{
		int idx = static_cast<int>((pool.size() - 1) * static_cast<float>(rand()) / RAND_MAX);
		sample[i] = pool[idx];
		pool.erase(pool.begin() + idx);
	}

	pool.reserve(pool.size() + sample_number);
	for (int i = 0; i < sample_number; ++i)
		pool.push_back(sample[i]);
	return true;
}

template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::OneStepLocalOptimization(const cv::Mat &points,
	const float threshold,
	float inner_threshold,
	const int inlier_limit,
	Model &model,
	const ModelEstimator &estimator,
	std::vector<int> &inliers,
	int lsq_number)
{
	Score S = {0,0}, Ss, maxS;

	maxS = GetScore(points, model, estimator, threshold, inliers);

	if (maxS.I < 8)
		return S;

	float dth = (inner_threshold - threshold) / 4;

	std::vector<Model> models;
	if (maxS.I < inlier_limit)
	{
		int *sample = &inliers[0];
		if (!estimator.EstimateModelNonminimal(points, sample, static_cast<int>(inliers.size()), &models))
			return maxS;
	} else
	{
		int *sample = new int[inlier_limit];
		std::vector<int> pool = inliers;
		for (int i = 0; i < inlier_limit; ++i)
		{
			int idx = static_cast<int>(round((pool.size() - 1) * static_cast<float>(rand()) / RAND_MAX));
			sample[i] = pool[idx];
			pool.erase(pool.begin() + idx);
		}

		if (!estimator.EstimateModelNonminimal(points, sample, inlier_limit, &models))
		{
			delete sample;
			return maxS;
		}
	}

	/* iterate */
	for (int it = 0; it < lsq_number; ++it)
	{
		S = GetScore(points, models[0], estimator, threshold, inliers);
		Ss = GetScore(points, models[0], estimator, inner_threshold, inliers);

		if (ScoreLess(maxS, S)) 
		{
			maxS = S;
			model = models[0];
		}

		if (Ss.I < 8) {
			return maxS;
		}

		if (Ss.I <= inlier_limit) { /* if we are under the limit, just use what we have without shuffling */

			models.resize(0);
			int *sample = &inliers[0];
			if (!estimator.EstimateModelNonminimal(points, sample, inliers.size(), &models))
				return maxS;
			
		}
		else {
			int *sample = new int[inlier_limit];
			std::vector<int> pool = inliers;
			for (int i = 0; i < inlier_limit; ++i)
			{
				int idx = static_cast<int>(round((pool.size() - 1) * static_cast<float>(rand()) / RAND_MAX));
				sample[i] = pool[idx];
				pool.erase(pool.begin() + idx);
			}

			if (!estimator.EstimateModelNonminimal(points, sample, inlier_limit, &models))
			{
				delete sample;
				return maxS;
			}
		}

		inner_threshold -= dth;
	}
	return maxS;
}


template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::FullLocalOptimization(const cv::Mat &points,
	std::vector<int> &so_far_the_best_inliers,
	Model &so_far_the_best_model,
	Score &so_far_the_best_score,
	int &max_iteration,
	const ModelEstimator &estimator,
	const int trial_number,
	const float probability)
{
	Score S, maxS = { 0,0 };
	Model best_model;
	std::vector<int> best_inliers;

	++lo_number;

	if (so_far_the_best_score.I < 16) {
		return false;
	}

	int sample_number = static_cast<int>(MIN(14, so_far_the_best_inliers.size() / 2));

	GetScore(points, so_far_the_best_model, estimator, threshold, so_far_the_best_inliers, true);
	std::vector<int> pool = so_far_the_best_inliers;
	
	int *sample = new int[sample_number];
	std::vector<int> inliers;
	std::vector<Model> models;
	for (int i = 0; i < trial_number; ++i)
	{
		inliers.resize(0);
		models.resize(0);
		Sample(points, pool, neighbours, sample_number, sample);

		if (!estimator.EstimateModelNonminimal(points, sample, sample_number, &models))
			continue;

		if (models.size() == 0 || models[0].descriptor.rows != 3)
			continue;

		S = OneStepLocalOptimization(points, threshold, 4 * threshold, use_inlier_limit ? estimator.InlierLimit() : INT_MAX, models[0], estimator, inliers, local_optimization_limit ? 1 : 4);

		if (ScoreLess(maxS, S))
		{
			maxS = S;
			best_model = models[0];
			best_inliers = inliers;
		}
	}

	inliers.resize(0);
	models.resize(0);

	if (ScoreLess(so_far_the_best_score, maxS))
	{
		so_far_the_best_model.descriptor = best_model.descriptor;
		so_far_the_best_inliers = best_inliers;
		so_far_the_best_score = maxS;
		so_far_the_best_score.I = static_cast<unsigned int>(so_far_the_best_inliers.size());
	}
}


template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::LocalOptimizationWithGraphCut(const cv::Mat &points, 
	std::vector<int> &so_far_the_best_inliers,
	Model &so_far_the_best_model,
	Score &so_far_the_best_score,
	int &max_iteration, 
	const ModelEstimator &estimator, 
	const int trial_number,
	const float probability)
{
	Score maxS = so_far_the_best_score;
	Model best_model = so_far_the_best_model;
	std::vector<int> best_inliers;

	std::vector<int> inliers;
	bool there_is_change;
	std::vector<Model> models;
	float energy;
	int inlier_limit = estimator.InlierLimit();

	++lo_number;

	while (1)
	{
		models.resize(0);
		there_is_change = false;
		inliers.resize(0);
		Labeling(points, neighbor_number, neighbours, best_model, estimator, lambda, threshold, inliers, energy);

		++gc_number;

		int used_points = static_cast<int>(MIN(inlier_limit, inliers.size()));
		int *sample = new int[used_points];

		for (int trial = 0; trial < trial_number; ++trial)
		{
			if (used_points < inliers.size())
			{
				for (int i = 0; i < used_points; ++i)
				{
					int idx = static_cast<int>(round((inliers.size() - 1) * static_cast<float>(rand()) / RAND_MAX));
					sample[i] = inliers[idx];
					inliers.erase(inliers.begin() + idx);
				}

				inliers.reserve(inliers.size() + used_points);
				for (int i = 0; i < used_points; ++i)
					inliers.push_back(sample[i]);
			}
			else
				sample = &inliers[0];

			if (!estimator.EstimateModelNonminimal(points, sample, used_points, &models))
				break;

			for (int i = 0; i < models.size(); ++i)
			{
				Score s = GetScore(points, models.at(i), estimator, threshold, inliers, false);

				if (ScoreLess(maxS, s))
				{
					there_is_change = true;

					maxS = s;
					maxS.I = static_cast<int>(inliers.size());
					best_inliers = inliers;
					best_model = models[i];
				}
			}
		}

		if (!there_is_change)
			break;
	}

	if (ScoreLess(so_far_the_best_score, maxS))
	{
		so_far_the_best_score = maxS;
		so_far_the_best_model.descriptor = best_model.descriptor;

		so_far_the_best_inliers.resize(best_inliers.size());
		for (int i = 0; i < best_inliers.size(); ++i)
			so_far_the_best_inliers[i] = best_inliers[i];
		return true;
	}
	return false;
}

template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::GetScore(const cv::Mat &points, const Model &model, const ModelEstimator &estimator, const float threshold, std::vector<int> &inliers, bool store_inliers)
{	
	Score s = { 0,0 };
	float sqrThr = 2 * threshold * threshold;
	if (store_inliers)
		inliers.resize(0);
	
#if USE_CONCURRENCY // TODO: Implement the non-concurrent case
	int process_number = 8; 
	int step_size = points.rows / process_number;

	std::vector<std::vector<int>> process_inliers;
	if (store_inliers)
		process_inliers.resize(process_number);

	std::vector<Score> process_scores(process_number, {0,0});

	concurrency::parallel_for(0, process_number, [&](int process)
	{
		if (store_inliers)
			process_inliers[process].reserve(step_size);
		const int start_idx = process * step_size;
		const int end_idx = MIN(points.rows - 1, (process + 1) * step_size);
		float dist;

		for (int i = start_idx; i < end_idx; ++i)
		{
			dist = static_cast<float>(estimator.Error(points.row(i), model));
			dist = exp(-dist*dist / sqrThr);

			if (dist > 1e-3) {
				if (store_inliers)
					process_inliers[process].push_back(i);

				++(process_scores[process].I);
				process_scores[process].J += dist;
			}
		}
	});

	for (int i = 0; i < process_number; ++i)
	{
		s.I += process_scores[i].I;
		s.J += process_scores[i].J;

		if (store_inliers)
			copy(process_inliers[i].begin(), process_inliers[i].end(), back_inserter(inliers));
	}
#endif
	return s;
}


template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::Labeling(const cv::Mat &points, 
	int neighbor_number, 
	const std::vector<std::vector<cv::DMatch>> &neighbors,
	Model &model,
	ModelEstimator estimator,
	float lambda,
	float threshold,
	std::vector<int> &inliers,
	float &energy)
{
	Energy<float, float, float> *e = new Energy<float, float, float>(points.rows, // poor guess at number of pairwise terms needed :(
		neighbor_number,
		NULL);

	for (int i = 0; i < points.rows; ++i)
		e->add_node();

	const float sqr_thr = 2 * threshold * threshold;
	for (int i = 0; i < points.rows; ++i)
	{
		float distance = static_cast<float>(estimator.Error(points.row(i), model));
		float energy = exp(-(distance*distance) / sqr_thr);

		e->add_term1(i, energy, 0);
	}

	if (lambda > 0)
	{
		for (int i = 0; i < points.rows; ++i)
		{
			float distance1 = static_cast<float>(estimator.Error(points.row(i), model));
			float energy1 = exp(-(distance1*distance1) / sqr_thr);

			for (int j = 0; j < neighbors[i].size(); ++j)
			{
				int n_idx = neighbors[i][j].trainIdx;
				
				if (n_idx == i)
					continue;

				float distance2 = static_cast<float>(estimator.Error(points.row(n_idx), model));
				float energy2 = exp(-(distance2*distance2) / sqr_thr);
				
				const float e00 = 0.5f * (energy1 + energy2);
				const float e01 = 1;
				const float e10 = 1;
				const float e11 = 1 - 0.5f * (energy1 + energy2);

				if (e00 + e11 > e01 + e10)
					printf("Non-submodular expansion term detected; smooth costs must be a metric for expansion\n");

				e->add_term2(i, n_idx, e00*lambda, e01*lambda, e10*lambda, e11*lambda);
			}
		}
	}

	e->minimize();
	for (int i = 0; i < points.rows; ++i)
		if (e->what_segment(i) == Graph<float, float, float>::SINK)
			inliers.push_back(i);
	 
	delete e;
}