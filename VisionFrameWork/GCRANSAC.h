#pragma once

#include <cv.h>
#include <opencv2\highgui\highgui.hpp>
#include "GCoptimization.h"
#include "Regression.h"
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
	enum ModelType { Line2d, Homography, FundamentalMatrix };

	GCRANSAC(ModelType _type) { type = _type; time_limit = FLT_MAX; desired_fps = -1; }
	~GCRANSAC() { }

	void Run(const Mat &points, ModelEstimator estimator, Model &obtained_model, vector<int> &obtained_inliers, float threshold, float lambda, float sphere_size, float probability, int &iteration_number, bool user_graph_cut, bool use_inlier_limit = true, bool only_one_lo = false, bool apply_local_optimization = true);
	bool Sample(const Mat &points, vector<int> &pool, const vector<vector<DMatch>> &neighbors, int sample_number, int *sample);
	float GetEnergy() { return obtained_energy; }
	void SetFPS(int fps) { desired_fps = fps; time_limit = 1.0f / fps; }

	int GetLONumber() { return lo_number; }
	int GetGCNumber() { return gc_number; }

	Score GetScore(const Mat &points, const Model &model, const ModelEstimator &estimator, const float threshold, vector<int> &inliers, bool store_inliers = true);
	int ScoreLess(const Score &s1, const Score &s2, int type = 1) { return type == 1 ? (s1.I < s2.I || (s1.I == s2.I && s1.J < s2.J)) : s1.J < s2.J; }

protected:
	int gc_number, lo_number;
	float time_limit;
	int desired_fps;
	bool apply_local_optimization;
	bool only_one_lo;
	bool use_inlier_limit;
	float lambda;
	float threshold;
	int knn;
	ModelType type;
	float obtained_energy;
	vector<vector<DMatch>> neighbours;
	int neighbor_number;
	int iteration_limit;
	int point_number;

	Graph<float, float, float> *graph;

	int DesiredIterationNumber(int inlier_number, int point_number, int sample_size, float probability);
	void Labeling(const Mat &points,
		int neighbor_number,
		const vector<vector<DMatch>> &neighbors,
		Model &model,
		ModelEstimator estimator,
		float lambda,
		float threshold,
		vector<int> &inliers,
		float &energy);

	bool FullLocalOptimization(const Mat &points,
		vector<int> &so_far_the_best_inliers,
		Model &so_far_the_best_model,
		Score &so_far_the_best_score,
		int &max_iteration,
		const ModelEstimator &estimator,
		const int trial_number,
		const float probability);

	Score OneStepLocalOptimization(const Mat &points,
		const float threshold,
		float inner_threshold,
		const int inlier_limit,
		Model &model,
		const ModelEstimator &estimator,
		vector<int> &inliers,
		int lsq_number = 4);

	bool LocalOptimizationWithGraphCut(const Mat &points,
		vector<int> &so_far_the_best_inliers,
		Model &so_far_the_best_model,
		Score &so_far_the_best_score,
		int &max_iteration,
		const ModelEstimator &estimator,
		const int trial_number,
		const float probability);

	float CalculateLambda(int inlier_number);
};

template <class ModelEstimator, class Model>
int GCRANSAC<ModelEstimator, Model>::DesiredIterationNumber(int inlier_number, int point_number, int sample_size,  float probability)
{
	float q = pow((float)inlier_number / point_number, sample_size);

	float iter = log(probability) / log(1 - q);
	if (iter < 0)
		return INT_MAX;
	return (int)iter + 1;
}

template <class ModelEstimator, class Model>
void GCRANSAC<ModelEstimator, Model>::Run(const Mat &points, 
	ModelEstimator estimator,
	Model &obtained_model,
	vector<int> &obtained_inliers,
	float threshold, 
	float lambda, 
	float sphere_size,
	float probability, 
	int &iteration_number,
	bool use_graph_cut, 
	bool use_inlier_limit,
	bool only_one_lo,
	bool apply_local_optimization)
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	std::chrono::duration<double> elapsed_seconds;

	this->apply_local_optimization = apply_local_optimization;
	this->only_one_lo = only_one_lo;
	this->use_inlier_limit = use_inlier_limit;
	this->threshold = threshold;
	this->lambda = lambda;
	gc_number = 0;
	lo_number = 0;
	point_number = points.rows;

	const int min_iteration = 0;
	const int knn = 6;
	const int iter_before_lo = 50;
	const int sample_number = estimator.SampleSize();
	iteration_limit = 5e3;
	const float distance = sphere_size;
	
	// Determine neighborhood
#if PRINT_TIMES
	start = std::chrono::system_clock::now();
#endif
	FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));
	neighbours.resize(0);
	if (lambda > 0.0)
		flann.radiusMatch(points, points, neighbours, distance);

	// Initialize graph-cut
	neighbor_number = 0;
	if (lambda > 0.0)
	{
		for (int i = 0; i < neighbours.size(); ++i)
			neighbor_number += neighbours[i].size() - 1;
		lambda = CalculateLambda(0.50 * point_number);
	}

#if PRINT_TIMES
	end = std::chrono::system_clock::now();
	elapsed_seconds = end - start;
	
	ofstream file("time_distribution.csv", ios::app);
	file << elapsed_seconds.count() << ";";
	//printf("[TIME] FLANN = %f secs\n", elapsed_seconds);

	float time_sampling = 0;
	float time_graph_cut = 0;
	float time_lo = 0;
#endif

	// Main iteration
	float final_energy = 0;
	int iteration = 0;
	int max_iteration = DesiredIterationNumber(1, points.rows, sample_number, probability);
	vector<int> *so_far_the_best_inlier_indices = NULL;
	Model so_far_the_best_model;
	Score so_far_the_best_score = {0,0};
	int lo_count = 0;
	int *sample = new int[sample_number];
	bool do_local_optimization = false;
	int inl_offset = 0;
	int counter_for_inlier_vecs[] = { 0,0 };
	vector<vector<int>> temp_inner_inliers(2);

	vector<int> pool(points.rows);
	for (int i = 0; i < points.rows; ++i)
		pool[i] = i; 

	if (desired_fps > -1)
		start = std::chrono::system_clock::now();

	while (min_iteration > iteration || iteration < MIN(max_iteration, iteration_limit))
	{
		do_local_optimization = false;
		++iteration;

		// Sample a minimal subset
		vector<Model> models;
#if PRINT_TIMES
		start = std::chrono::system_clock::now();
#endif 
		while (1)
		{
			if (!Sample(points, pool, neighbours, sample_number, sample))
				continue;
			 
 			if (estimator.EstimateModel(points, sample, &models))
				break; 
		}                  

#if PRINT_TIMES
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		time_sampling += (float)elapsed_seconds.count();
#endif

		// Select the so-far-the-best from the estimated models
#if PRINT_TIMES 
		start = std::chrono::system_clock::now();
#endif
		for (int model_idx = 0; model_idx < models.size(); ++model_idx)
		{
			// Get the inliers of the non-optimized model
			Score score = GetScore(points, models[model_idx], estimator, threshold, temp_inner_inliers[inl_offset], false);
			
			if (ScoreLess(so_far_the_best_score, score))
			{
				inl_offset = (inl_offset + 1) % 2;
				
				so_far_the_best_model = models[model_idx];
				so_far_the_best_score = score;
				do_local_optimization = iteration > iter_before_lo;
				max_iteration = DesiredIterationNumber(so_far_the_best_score.I, points.rows, sample_number, probability);
				if (lambda > 0.0)
					lambda = CalculateLambda(so_far_the_best_score.I);
			}
		}

#if PRINT_TIMES
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		time_graph_cut += (float)elapsed_seconds.count();

		start = std::chrono::system_clock::now(); 
#endif

		if (iteration > iter_before_lo && lo_count == 0 && so_far_the_best_score.I > 7)
			do_local_optimization = true;

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
					only_one_lo ? 1 : 20,
					probability);
			else
				FullLocalOptimization(points,
					temp_inner_inliers[inl_offset],
					so_far_the_best_model,
					so_far_the_best_score,
					max_iteration,
					estimator,
					only_one_lo ? 1 : 20,
					probability);

			max_iteration = DesiredIterationNumber(so_far_the_best_score.I, points.rows, sample_number, probability);
			if (lambda > 0.0)
				lambda = CalculateLambda(so_far_the_best_score.I);
		}

		// Apply time limit
		if (desired_fps > -1)
		{
			end = std::chrono::system_clock::now();
			elapsed_seconds = end - start;

			if (elapsed_seconds.count() > time_limit)
				break;
		}

#if PRINT_TIMES
		end = std::chrono::system_clock::now();
		elapsed_seconds = end - start;
		time_lo += (float)elapsed_seconds.count();
#endif
	}
	delete sample;

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
				only_one_lo ? 1 : 20,
				probability);
		else
			FullLocalOptimization(points,
				temp_inner_inliers[inl_offset],
				so_far_the_best_model,
				so_far_the_best_score,
				max_iteration,
				estimator,
				only_one_lo ? 1 : 20,
				probability);
	}

	if (temp_inner_inliers[inl_offset].size() != so_far_the_best_score.I)
		Score score = GetScore(points, so_far_the_best_model, estimator, threshold, temp_inner_inliers[inl_offset]);
	 
	vector<Model> models;
	estimator.EstimateModelNonminimal(points, &(temp_inner_inliers[inl_offset])[0], so_far_the_best_score.I, &models);

	if (models.size() > 0)
		so_far_the_best_model.descriptor = models[0].descriptor;

#if PRINT_TIMES
	//printf("[TIME] Sampling = %f secs\n", time_sampling);
	//printf("[TIME] Graph Cut = %f secs\n", time_graph_cut);
	//printf("[TIME] Local Optimization = %f secs\n", time_lo);
	
	file << time_sampling << ";" << time_graph_cut << "; " << time_lo << "\n";
	file.close();
#endif

	obtained_inliers = temp_inner_inliers[inl_offset];
	obtained_model = so_far_the_best_model;
	iteration_number = iteration;
}

template <class ModelEstimator, class Model>
float GCRANSAC<ModelEstimator, Model>::CalculateLambda(int inlier_number)
{
	float lambda = (point_number * ((float)inlier_number / point_number)) / (float)neighbor_number;
	return lambda;
}

template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::Sample(const Mat &points, vector<int> &pool, const vector<vector<DMatch>> &neighbors, int sample_number, int *sample)
{
	// TODO: replacable sampler
	for (int i = 0; i < sample_number; ++i)
	{
		int idx = (pool.size() - 1) * (float)rand() / RAND_MAX;
		sample[i] = pool[idx];
		pool.erase(pool.begin() + idx);
	}

	pool.reserve(pool.size() + sample_number);
	for (int i = 0; i < sample_number; ++i)
		pool.push_back(sample[i]);
}

template <class ModelEstimator, class Model>
Score GCRANSAC<ModelEstimator, Model>::OneStepLocalOptimization(const Mat &points,
	const float threshold,
	float inner_threshold,
	const int inlier_limit,
	Model &model,
	const ModelEstimator &estimator,
	vector<int> &inliers,
	int lsq_number)
{
	Score S = {0,0}, Ss, maxS;

	maxS = GetScore(points, model, estimator, threshold, inliers);

	if (maxS.I < 8)
		return S;

	float dth = (inner_threshold - threshold) / 4;

	vector<Model> models;
	if (maxS.I < inlier_limit)
	{
		int *sample = &inliers[0];
		if (!estimator.EstimateModelNonminimal(points, sample, inliers.size(), &models))
			return maxS;
	} else
	{
		int *sample = new int[inlier_limit];
		vector<int> pool = inliers;
		for (int i = 0; i < inlier_limit; ++i)
		{
			int idx = round((pool.size() - 1) * (float)rand() / RAND_MAX);
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
			vector<int> pool = inliers;
			for (int i = 0; i < inlier_limit; ++i)
			{
				int idx = round((pool.size() - 1) * (float)rand() / RAND_MAX);
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
bool GCRANSAC<ModelEstimator, Model>::FullLocalOptimization(const Mat &points,
	vector<int> &so_far_the_best_inliers,
	Model &so_far_the_best_model,
	Score &so_far_the_best_score,
	int &max_iteration,
	const ModelEstimator &estimator,
	const int trial_number,
	const float probability)
{
	Score S, maxS = { 0,0 };
	Model best_model;
	vector<int> best_inliers;

	++lo_number;

	if (so_far_the_best_score.I < 16) {
		return false;
	}

	int sample_number = MIN(14, so_far_the_best_inliers.size() / 2);

	GetScore(points, so_far_the_best_model, estimator, threshold, so_far_the_best_inliers, true);
	vector<int> pool = so_far_the_best_inliers;
	
	int *sample = new int[sample_number];
	vector<int> inliers;
	vector<Model> models;
	for (int i = 0; i < trial_number; ++i)
	{
		inliers.resize(0);
		models.resize(0);
		Sample(points, pool, neighbours, sample_number, sample);

		if (!estimator.EstimateModelNonminimal(points, sample, sample_number, &models))
			continue;

		if (models.size() == 0 || models[0].descriptor.rows != 3)
			continue;

		S = OneStepLocalOptimization(points, threshold, 4 * threshold, use_inlier_limit ? estimator.InlierLimit() : INT_MAX, models[0], estimator, inliers, only_one_lo ? 1 : 4);

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
		so_far_the_best_score.I = so_far_the_best_inliers.size();
	}
}


template <class ModelEstimator, class Model>
bool GCRANSAC<ModelEstimator, Model>::LocalOptimizationWithGraphCut(const Mat &points, 
	vector<int> &so_far_the_best_inliers,
	Model &so_far_the_best_model,
	Score &so_far_the_best_score,
	int &max_iteration, 
	const ModelEstimator &estimator, 
	const int trial_number,
	const float probability)
{
	Score maxS = so_far_the_best_score;
	Model best_model = so_far_the_best_model;
	vector<int> best_inliers;

	vector<int> inliers;
	bool there_is_change;
	vector<Model> models;
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

		int used_points = MIN(inlier_limit, inliers.size());
		int *sample = new int[used_points];

		for (int trial = 0; trial < trial_number; ++trial)
		{
			if (used_points < inliers.size())
			{
				for (int i = 0; i < used_points; ++i)
				{
					int idx = round((inliers.size() - 1) * static_cast<float>(rand()) / RAND_MAX);
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
					maxS.I = inliers.size();
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
Score GCRANSAC<ModelEstimator, Model>::GetScore(const Mat &points, const Model &model, const ModelEstimator &estimator, const float threshold, vector<int> &inliers, bool store_inliers)
{	
	Score s = { 0,0 };
	float sqrThr = 2 * threshold * threshold;
	if (store_inliers)
		inliers.resize(0);
	
#if 1 || USE_CONCURRENCY
	int process_number = 8; 
	int step_size = points.rows / process_number;

	vector<vector<int>> process_inliers;
	if (store_inliers)
		process_inliers.resize(process_number);

	vector<Score> process_scores(process_number, {0,0});

	concurrency::parallel_for(0, process_number, [&](int process)
	{
		if (store_inliers)
			process_inliers[process].reserve(step_size);
		const int start_idx = process * step_size;
		const int end_idx = MIN(points.rows - 1, (process + 1) * step_size);
		float dist;

		for (int i = start_idx; i < end_idx; ++i)
		{
			dist = estimator.Error(points.row(i), model);
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
void GCRANSAC<ModelEstimator, Model>::Labeling(const Mat &points, 
	int neighbor_number, 
	const vector<vector<DMatch>> &neighbors,
	Model &model,
	ModelEstimator estimator,
	float lambda,
	float threshold,
	vector<int> &inliers,
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
		float distance = estimator.Error(points.row(i), model);
		float energy = exp(-(distance*distance) / sqr_thr);

		e->add_term1(i, energy, 0);
	}

	if (lambda > 0)
	{
		for (int i = 0; i < points.rows; ++i)
		{
			float distance1 = estimator.Error(points.row(i), model);
			float energy1 = exp(-(distance1*distance1) / sqr_thr);

			for (int j = 1; j < neighbors[i].size(); ++j)
			{
				int n_idx = neighbors[i][j].trainIdx;

				float distance2 = estimator.Error(points.row(n_idx), model);
				float energy2 = exp(-(distance2*distance2) / sqr_thr);
				
				const float e00 = 0.5 * (energy1 + energy2);
				const float e01 = 1;
				const float e10 = 1;
				const float e11 = 1 - 0.5 * (energy1 + energy2);

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