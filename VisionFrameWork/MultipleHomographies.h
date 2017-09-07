#pragma once

#include <thread>
#include "RobustMethods.h"

using namespace std;
using namespace cv;
using namespace concurrency;

template<typename T>
class MultipleHomographies
{
public:
	MultipleHomographies() {
		_concurentThreadsSupported = std::thread::hardware_concurrency();

		_outlierRatio = 0.3;
		_confidence = 0.95;
	}

	~MultipleHomographies() {}

	enum Method { SequentialRANSAC, Sequential3PTRANSAC, MultiRANSAC };

	void Detect(Method method, const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, T threshold, int stability_cut, Mat F = Mat());
	void DrawClusters(Mat * const img1, Mat * const img2, const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints);

protected:
	vector<vector<int>> _clusterPoints;
	vector<Mat> _clusterHomographies;
	T _confidence;
	T _outlierRatio;

	int _concurentThreadsSupported;

	void Detect_RANSAC(const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, const vector<int> * const possible_points, T threshold, Mat &H, vector<int> &inliers);
	void Detect_SequentialRANSAC(const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, T threshold, int stability_cut);

	void Detect_RANSAC_3PT(const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, const vector<int> * const possible_points, T threshold, Mat &H, vector<int> &inliers, Mat F);
	void Detect_SequentialRANSAC_3PT(const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, T threshold, int stability_cut, Mat F);

	void Detect_MultiRansac(const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints, T threshold);

	int GetClusterNumber() { return _clusterPoints.size(); }
};

template<typename T>
void MultipleHomographies<T>::Detect(Method method, 
	const vector<Point_<T>> * const srcPoints, 
	const vector<Point_<T>> * const dstPoints,
	T threshold,
	int stability_cut,
	Mat F)
{
	_clusterPoints.resize(srcPoints->size());

	switch (method)
	{
	case Method::SequentialRANSAC:
		Detect_SequentialRANSAC(srcPoints, dstPoints, threshold, stability_cut);
		break;
	case Method::Sequential3PTRANSAC:
		Detect_SequentialRANSAC_3PT(srcPoints, dstPoints, threshold, stability_cut, F);
		break;
	case Method::MultiRANSAC:
		Detect_MultiRansac(srcPoints, dstPoints, threshold);
		break;
	default:
		break;
	}
}

/*
MultiRANSAC, 4-point homography estimation technique (DLT)
*/
template<typename T>
void MultipleHomographies<T>::Detect_MultiRansac(const vector<Point_<T>> * const srcPoints, 
	const vector<Point_<T>> * const dstPoints, 
	T threshold)
{
	int N = srcPoints->size();
	int maxIterations = 1000;
	int upperBound = 10 * maxIterations;
	int inlierNumber = 0;
	int model_number = 4;
	int sample_num = 4;
	T sqrThreshold = threshold * threshold;

	int bestInlierNumber = -1;
	vector<Mat> bestHomographies(model_number);
	vector<vector<int>> bestInliers(model_number);
	
	vector<vector<int>> inliers;
	int ** samples = new int*[model_number];
	for (int i = 0; i < model_number; ++i)
		samples[i] = new int[sample_num];

	vector<Mat> homographies(model_number);
	vector<vector<int>> currInliers(model_number);
	
	// Iterate
	for (int it = 0; it < maxIterations && it < upperBound; ++it)
	{
		// Select samples for all the models
		vector<int> possible_indices(N);
		for (int i = 0; i < N; ++i)
			possible_indices[i] = i;

		vector<vector<Point_<T>>> pts1(model_number);
		vector<vector<Point_<T>>> pts2(model_number);

		bool error = false;
		for (int m = 0; m < model_number; ++m)
		{
			pts1[m] = vector<Point_<T>>(sample_num);
			pts2[m] = vector<Point_<T>>(sample_num);

			for (int s = 0; s < sample_num; ++s)
			{
				int idx = round((possible_indices.size() - 1) * (rand() / (T)RAND_MAX));
				samples[m][s] = possible_indices[idx];
				pts1[m][s] = srcPoints->at(samples[m][s]);
				pts2[m][s] = dstPoints->at(samples[m][s]);

				possible_indices.erase(possible_indices.begin() + idx);

				if (possible_indices.size() == 0 && (m < model_number - 1 || s < sample_num - 1))
				{
					error = true;
					break;
				}
			}
		}

		if (error)
		{
			fprintf(stderr,
				"No samples could be selected!\n");
			break;
		}

		// Compute the current model parameters
		for (int m = 0; m < model_number; ++m)
			homographies[m] = findHomography(pts1[m], pts2[m]);

		// Select the inliers that are within threshold_ from the model
		inlierNumber = 0;
		vector<bool> usabilityMask(N, false);
		Mat pt1, pt2, diff;
		for (int m = 0; m < model_number; ++m)
		{
			currInliers[m].resize(N);
			int currentInlierNumber = 0;
			for (int pt = 0; pt < N; ++pt)
			{
				if (usabilityMask[pt])
					continue;

				pt1 = (Mat_<T>(3, 1) << srcPoints->at(pt).x, srcPoints->at(pt).y, 1);
				pt2 = (Mat_<T>(3, 1) << dstPoints->at(pt).x, dstPoints->at(pt).y, 1);
				pt1 = homographies[m] * pt1;
				pt1 = pt1 / pt1.at<T>(2);

				diff = pt2 - pt1;

				if ((diff.at<T>(0)*diff.at<T>(0) + diff.at<T>(1)*diff.at<T>(1)) < sqrThreshold)
				{
					usabilityMask[pt] = true;
					currInliers[m][currentInlierNumber] = pt;
					++currentInlierNumber;
				}
			}
			currInliers[m].resize(currentInlierNumber);
			inlierNumber = inlierNumber + currentInlierNumber;
		}

		// Better match ?
		if (inlierNumber > bestInlierNumber)
		{
			bestInlierNumber = inlierNumber;
			bestInliers = currInliers;
			bestHomographies = homographies;
			
			//MultiRansac preparation for probability computation
			int multiIndicesNumber = 0;
			for (int multiIter = 0; multiIter < bestInliers.size(); multiIter++)
				multiIndicesNumber += bestInliers.at(multiIter).size();

			int multiSelectionSize = sample_num * model_number;

			cout << multiIndicesNumber << endl;
			cout << multiSelectionSize << endl;

			// Compute the k parameter (k=log(z)/log(1-w^n))
			/*T w = static_cast<T> (bestInlierNumber) / static_cast<T> (multiIndicesNumber);
			cout << w << endl;
			T p_no_outliers = 1.0 - pow(w, static_cast<T> (multiSelectionSize));
			cout << p_no_outliers << endl;
			p_no_outliers =
				(std::max) (std::numeric_limits<T>::epsilon(), p_no_outliers);
			cout << p_no_outliers << endl;
			// Avoid division by -Inf
			p_no_outliers =
				(std::min) (1.0 - std::numeric_limits<T>::epsilon(), p_no_outliers);
			// Avoid division by 0.
			maxIterations = MAX(10, log(1.0 - _outlierRatio) / log(p_no_outliers));*/
		}

		fprintf(stdout,
			"[sm::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n",
			it, maxIterations, inlierNumber, bestInlierNumber);

		/*if (debug_verbosity_level > 1)
			fprintf(stdout,
			"[sm::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n",
			iterations_, k, n_inliers_count, n_best_inliers_count);
		if (iterations_ > max_iterations_)
		{
			if (debug_verbosity_level > 0)
				fprintf(stdout,
				"[sm::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n");
			break;
		}*/
	}

	_clusterPoints = bestInliers;
	_clusterHomographies = bestHomographies;

	/*if (debug_verbosity_level > 0)
	{
		size_t multiModelSize = 0;
		for (size_t modelIter = 0; modelIter < model_.size(); modelIter++)
			multiModelSize += model_[modelIter].size();
		fprintf(stdout,
			"[sm::RandomSampleConsensus::computeModel] Model: %zu size, %d inliers.\n",
			model_.size(), n_best_inliers_count);
	}

	if (model_.empty())
	{
		inliers_.clear();
		return (false);
	}

	// Get the set of inliers that correspond to the best model found so far
	sac_model_->selectWithinDistance(model_coefficients_, threshold_, inliers_);*/

}

/*
Sequential RANSAC, 4-point homography estimation technique (DLT)
*/
template<typename T>
void MultipleHomographies<T>::Detect_SequentialRANSAC(const vector<Point_<T>> * const srcPoints,
	const vector<Point_<T>> * const dstPoints,
	T threshold,
	int stability_cut)
{
	Mat H;
	vector<int> inliers;
	vector<int> possible_points(srcPoints->size());
	for (int i = 0; i < possible_points.size(); ++i)
		possible_points[i] = i;

	bool converge = false;
	int stability = 0;
	while (possible_points.size() > 4 && !converge)
	{
		Detect_RANSAC(srcPoints, dstPoints, &possible_points, threshold, H, inliers);

		for (int i = 0; i < inliers.size(); ++i)
		{
			vector<int>::iterator it = std::find(possible_points.begin(), possible_points.end(), inliers[i]);
			possible_points.erase(it);
		}

		if (inliers.size() >= stability_cut)
		{
			_clusterPoints.push_back(inliers);
			_clusterHomographies.push_back(H);
			stability = 0;
		}
		else
			++stability;

		if (inliers.size() == 0 || stability > 10)
			converge = true;

		cout << "Inlier number = " << inliers.size() << endl;
		cout << "Remaining points = " << possible_points.size() << endl;
	}
}

template<typename T>
void MultipleHomographies<T>::Detect_RANSAC(const vector<Point_<T>> * const srcPoints,
	const vector<Point_<T>> * const dstPoints,
	const vector<int> * const possible_points,
	T threshold,
	Mat &H,
	vector<int> &inliers)
{
	int sample_num = 4;

	// Calculate iteration number
	T w = pow(_outlierRatio, sample_num);
	T p = _confidence;

	int iteration_num = log(1 - p) / log(1 - w);
	cout << "Iteration number = " << iteration_num << endl;

	T sqrThreshold = threshold * threshold;

	vector<int> bestInliers(_concurentThreadsSupported, -1);
	vector<Mat> bestHomographies(_concurentThreadsSupported);
	concurrency::parallel_for(0, _concurentThreadsSupported, [&](int proc)
	{
		vector<int> possible_indices(possible_points->size());
		for (int i = 0; i < possible_indices.size(); ++i)
			possible_indices[i] = possible_points->at(i);

		vector<Point_<T>> pts1(sample_num);
		vector<Point_<T>> pts2(sample_num);
		vector<int> samples(sample_num);

		for (int it = 0; it < iteration_num / _concurentThreadsSupported; ++it)
		{
			// Select points
			for (int i = 0; i < sample_num; ++i)
			{
				int idx = round((possible_indices.size() - 1) * (rand() / (T)RAND_MAX));
				samples[i] = possible_indices[idx];
				pts1[i] = srcPoints->at(samples[i]);
				pts2[i] = dstPoints->at(samples[i]);

				possible_indices.erase(possible_indices.begin() + idx);
			}

			// Calculate homography
			Mat H = findHomography(pts1, pts2);

			// Count inliers
			Mat pt1, pt2, diff;
			int inlierNumber = 0;
			for (int i = 0; i < possible_points->size(); ++i)
			{
				int idx = possible_points->at(i);
				pt1 = (Mat_<T>(3, 1) << srcPoints->at(idx).x, srcPoints->at(idx).y, 1);
				pt2 = (Mat_<T>(3, 1) << dstPoints->at(idx).x, dstPoints->at(idx).y, 1);
				pt1 = H * pt1;
				pt1 = pt1 / pt1.at<T>(2);

				diff = pt2 - pt1;

				if ((diff.at<T>(0)*diff.at<T>(0) + diff.at<T>(1)*diff.at<T>(1)) < sqrThreshold)
					++inlierNumber;
			}

			if (inlierNumber > bestInliers[proc])
			{
				bestInliers[proc] = inlierNumber;
				bestHomographies[proc] = H;
			}

			// Put back points
			possible_indices.resize(possible_indices.size() + sample_num);
			for (int i = 0; i < sample_num; ++i)
				possible_indices[possible_indices.size() - 1 - i] = samples[i];
		}
	});

	// Select best homography
	int bestIdx = 0;
	for (int proc = 1; proc < _concurentThreadsSupported; ++proc)
		if (bestInliers[proc] > bestInliers[bestIdx])
			bestIdx = proc;

	// Select inliers
	inliers.resize(bestInliers[bestIdx]);
	Mat pt1, pt2, diff;
	int inlNumber = 0;
	H = bestHomographies[bestIdx];
	for (int i = 0; i < possible_points->size(); ++i)
	{
		int idx = possible_points->at(i);
		pt1 = (Mat_<T>(3, 1) << srcPoints->at(idx).x, srcPoints->at(idx).y, 1);
		pt2 = (Mat_<T>(3, 1) << dstPoints->at(idx).x, dstPoints->at(idx).y, 1);
		pt1 = H * pt1;
		pt1 = pt1 / pt1.at<T>(2);

		diff = pt2 - pt1;

		if ((diff.at<T>(0)*diff.at<T>(0) + diff.at<T>(1)*diff.at<T>(1)) < sqrThreshold)
			inliers[inlNumber++] = idx;
	}
}

/*
	Sequential RANSAC, 3-point homography estimation technique (3PT)
*/
template<typename T>
void MultipleHomographies<T>::Detect_SequentialRANSAC_3PT(const vector<Point_<T>> * const srcPoints,
	const vector<Point_<T>> * const dstPoints,
	T threshold,
	int stability_cut,
	Mat F)
{
	Mat H;
	vector<int> inliers;
	vector<int> possible_points(srcPoints->size());
	for (int i = 0; i < possible_points.size(); ++i)
		possible_points[i] = i;

	bool converge = false;
	int stability = 0;
	while (possible_points.size() > 3 && !converge)
	{
		Detect_RANSAC_3PT(srcPoints, dstPoints, &possible_points, threshold, H, inliers, F);

		for (int i = 0; i < inliers.size(); ++i)
		{
			vector<int>::iterator it = std::find(possible_points.begin(), possible_points.end(), inliers[i]);
			possible_points.erase(it);
		}

		if (inliers.size() >= stability_cut)
		{
			_clusterPoints.push_back(inliers);
			_clusterHomographies.push_back(H);
			stability = 0;
		}
		else
			++stability;

		if (inliers.size() == 0 || stability > 10)
			converge = true;

		cout << "Inlier number = " << inliers.size() << endl;
		cout << "Remaining points = " << possible_points.size() << endl;
	}
}

template<typename T>
void MultipleHomographies<T>::Detect_RANSAC_3PT(const vector<Point_<T>> * const srcPoints,
	const vector<Point_<T>> * const dstPoints,
	const vector<int> * const possible_points,
	T threshold,
	Mat &H,
	vector<int> &inliers,
	Mat F)
{
	int sample_num = 3;

	// Calculate iteration number
	T w = pow(_outlierRatio, sample_num);
	T p = _confidence;

	int iteration_num = log(1 - p) / log(1 - w);
	cout << "Iteration number = " << iteration_num << endl;

	T sqrThreshold = threshold * threshold;

	vector<int> bestInliers(_concurentThreadsSupported, -1);
	vector<Mat> bestHomographies(_concurentThreadsSupported);
	concurrency::parallel_for(0, _concurentThreadsSupported, [&](int proc)
	{
		vector<int> possible_indices(possible_points->size());
		for (int i = 0; i < possible_indices.size(); ++i)
			possible_indices[i] = possible_points->at(i);

		vector<Point_<T>> pts1(sample_num);
		vector<Point_<T>> pts2(sample_num);
		vector<int> samples(sample_num);

		for (int it = 0; it < iteration_num / _concurentThreadsSupported; ++it)
		{
			// Select points
			for (int i = 0; i < sample_num; ++i)
			{
				int idx = round((possible_indices.size() - 1) * (rand() / (T)RAND_MAX));
				samples[i] = possible_indices[idx];
				pts1[i] = srcPoints->at(samples[i]);
				pts2[i] = dstPoints->at(samples[i]);

				possible_indices.erase(possible_indices.begin() + idx);
			}

			// Calculate homography
			Mat H;
			vision::GetHomography3PT<T>(F, pts1, pts2, H);

			// Count inliers
			Mat pt1, pt2, diff;
			int inlierNumber = 0;
			for (int i = 0; i < possible_points->size(); ++i)
			{
				int idx = possible_points->at(i);
				pt1 = (Mat_<T>(3, 1) << srcPoints->at(idx).x, srcPoints->at(idx).y, 1);
				pt2 = (Mat_<T>(3, 1) << dstPoints->at(idx).x, dstPoints->at(idx).y, 1);
				pt1 = H * pt1;
				pt1 = pt1 / pt1.at<T>(2);

				diff = pt2 - pt1;

				if ((diff.at<T>(0)*diff.at<T>(0) + diff.at<T>(1)*diff.at<T>(1)) < sqrThreshold)
					++inlierNumber;
			}

			if (inlierNumber > bestInliers[proc])
			{
				bestInliers[proc] = inlierNumber;
				bestHomographies[proc] = H;
			}

			// Put back points
			possible_indices.resize(possible_indices.size() + sample_num);
			for (int i = 0; i < sample_num; ++i)
				possible_indices[possible_indices.size() - 1 - i] = samples[i];
		}
	});

	// Select best homography
	int bestIdx = 0;
	for (int proc = 1; proc < _concurentThreadsSupported; ++proc)
		if (bestInliers[proc] > bestInliers[bestIdx])
			bestIdx = proc;

	// Select inliers
	inliers.resize(bestInliers[bestIdx]);
	Mat pt1, pt2, diff;
	int inlNumber = 0;
	H = bestHomographies[bestIdx];
	for (int i = 0; i < possible_points->size(); ++i)
	{
		int idx = possible_points->at(i);
		pt1 = (Mat_<T>(3, 1) << srcPoints->at(idx).x, srcPoints->at(idx).y, 1);
		pt2 = (Mat_<T>(3, 1) << dstPoints->at(idx).x, dstPoints->at(idx).y, 1);
		pt1 = H * pt1;
		pt1 = pt1 / pt1.at<T>(2);

		diff = pt2 - pt1;

		if ((diff.at<T>(0)*diff.at<T>(0) + diff.at<T>(1)*diff.at<T>(1)) < sqrThreshold)
			inliers[inlNumber++] = idx;
	}
}

template<typename T>
void MultipleHomographies<T>::DrawClusters(Mat * const img1, Mat * const img2, const vector<Point_<T>> * const srcPoints, const vector<Point_<T>> * const dstPoints)
{
	for (int i = 0; i < _clusterPoints.size(); ++i)
	{
		Scalar color(T(rand()) / RAND_MAX * 255, T(rand()) / RAND_MAX * 255, T(rand()) / RAND_MAX * 255);

		for (int j = 0; j < _clusterPoints[i].size(); ++j)
		{
			circle(*img1, srcPoints->at(_clusterPoints[i][j]), 3, color, 2);
			circle(*img2, dstPoints->at(_clusterPoints[i][j]), 3, color, 2);
		}
	}
}