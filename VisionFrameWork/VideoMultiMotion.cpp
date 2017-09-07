#include "stdafx.h"
#include "VideoMultiMotion.h"


VideoMultiMotion::VideoMultiMotion() : log_to_console(LOG)
{
}


VideoMultiMotion::~VideoMultiMotion()
{
}

bool VideoMultiMotion::Process(Mat _points, double _lambda, double _mean_shift_thr, double _assigning_thr, double _complexity_beta)
{
	if (log_to_console)
		printf("VideoMultiMotion Started.\n");
	points = _points;
	lambda = _lambda;
	mean_shift_thr = _mean_shift_thr;
	assigning_thr = _assigning_thr;
	complexity_beta = _complexity_beta;

	return Process();
}

bool VideoMultiMotion::Process()
{
	labeling.resize(points.cols, 0);
	planes = new vector<Mat>();
	tempPlanes = new vector<Mat>();

	if (log_to_console)
		printf("Generate hypothesises\n");
	clock_t time = clock();
	GenerateHypothesises();
	if (log_to_console)
		printf("%d hypothesises generated (%d ms).\n", planes->size(), clock() - time);

	if (log_to_console)
		printf("Detecting multiple motions\n");
	time = clock();
	MultipleMotionDetection();
	if (log_to_console)
		printf("%d motions detected (%d ms).\n", planes->size(), clock() - time);

	//if (log_to_console)
	//	printf("Remove week cluster. From %d clusters ", planes->size());
	//int removedClusters = 0;
	//time = clock();

	/*for (int i = fundamentalMatrices->size() - 1; i >= 0; --i)
	{
		if (inlierNumber[i] < FM_ESTIMATION_METHOD)
		{
			++removedClusters;
			for (int j = 0; j < labeling.size(); ++j)
				if (labeling[j] == i)
					labeling[j] = -1;
		}
	}
	cluster_number = fundamentalMatrices->size() - removedClusters;*/

	/*if (log_to_console)
		printf("%d are kept (%d ms)\n", fundamentalMatrices->size() - removedClusters, clock() - time);
	return fundamentalMatrices->size() > 0;*/
	return planes->size() > 1;
}

void VideoMultiMotion::MultipleMotionDetection()
{
	// Determine neighbourhood
	Mat pointVectors = (Mat_<float>(points.cols, points.rows));
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)points.cols, [&](int i)
#else
	for (int i = 0; i < srcPoints.size(); ++i)
#endif
	{
		pointVectors.at<float>(i, 0) = points.at<double>(0, i);
		pointVectors.at<float>(i, 1) = points.at<double>(1, i);
		pointVectors.at<float>(i, 2) = points.at<double>(2, i);
		pointVectors.at<float>(i, 3) = points.at<double>(3, i);
		pointVectors.at<float>(i, 4) = points.at<double>(4, i);
	}
#if USE_CONCURRENCY
	);
#endif

	FlannBasedMatcher flann;
	flann.knnMatch(pointVectors, pointVectors, neighbours, 6);
	//flann.radiusMatch(pointVectors, pointVectors, neighbours, 50);

	int itNum = 0;
	labeling.resize(points.cols, -1);
	double lastEnergy = INT_MAX;
	vector<double> lastTwoEnergies(2, DBL_MAX);
	int not_changed_number = 0;

	while (itNum++ < MAX_ITERATION_NUMBER)
	{
		bool changed = false;
		MergingStep(changed, itNum == 1);

		if (changed)
			not_changed_number = 0;
		else
			++not_changed_number;

		if (planes->size() == 1)
			break;
		else if (planes->size() == 0)
		{
			if (log_to_console)
				printf("No motion found!\n");
			break;
		}

		double energy;
		LabelingStep(energy);

		if (log_to_console)
			printf("Iteration %d.   Number of clusters = %d   Energy = %f\n", itNum, planes->size(), energy);

		if (!changed && (abs(lastEnergy - energy) < CONVERGENCE_THRESHOLD || lastTwoEnergies[0] == energy))
		{
			if (log_to_console)
				printf("Number of clusters = %d   Energy = %f\n", planes->size(), energy);
			final_energy = energy;
			break;
		}

		lastTwoEnergies[0] = lastTwoEnergies[1];
		lastTwoEnergies[1] = energy;

		lastEnergy = energy;
	}

	iterationNumber = itNum - 1;
}

void VideoMultiMotion::MergingStep(bool &changed, bool isFirstCall)
{
	Mat featureVectors(planes->size(), 25, CV_64F);
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)planes->size(), [&](int i)
#else
	for (int i = 0; i < fundamentalMatrices->size(); ++i)
#endif
	{
		Mat P = planes->at(i);

		for (int r = 0; r < 5; ++r)
			for (int c = 0; c < 5; ++c)
				featureVectors.at<double>(i, r*5 + c) = P.at<double>(r, c);

		/*featureVectors.at<double>(i, 0) = P.at<double>(0);
		featureVectors.at<double>(i, 1) = P.at<double>(1);
		featureVectors.at<double>(i, 2) = P.at<double>(2);
		featureVectors.at<double>(i, 3) = P.at<double>(3);
		featureVectors.at<double>(i, 4) = P.at<double>(4);
		featureVectors.at<double>(i, 5) = P.at<double>(5);*/
	}
#if USE_CONCURRENCY
	);
#endif
		
	MeanShiftClustering<double> msc;
	Mat rClusters;
	vector<vector<int>> clusterPoints;
	bool result = msc.Cluster(featureVectors, mean_shift_thr, rClusters, clusterPoints);

	if (!result)
		return;
	
	tempPlanes->clear();
	tempPlanes->reserve(rClusters.rows);
	
	if (isFirstCall)
	{
		for (int i = 0; i < rClusters.rows; ++i)
		{
			//if (clusterPoints[i].size() < 1)
			//	continue;

			Mat P = (Mat_<double>(5, 5) << rClusters.at<double>(i, 0), rClusters.at<double>(i, 1), rClusters.at<double>(i, 2), rClusters.at<double>(i, 3), rClusters.at<double>(i, 4),
				rClusters.at<double>(i, 5), rClusters.at<double>(i, 6), rClusters.at<double>(i, 7), rClusters.at<double>(i, 8), rClusters.at<double>(i, 9), 
				rClusters.at<double>(i, 10), rClusters.at<double>(i, 11), rClusters.at<double>(i, 12), rClusters.at<double>(i, 13), rClusters.at<double>(i, 14), 
				rClusters.at<double>(i, 15), rClusters.at<double>(i, 16), rClusters.at<double>(i, 17), rClusters.at<double>(i, 18), rClusters.at<double>(i, 19), 
				rClusters.at<double>(i, 20), rClusters.at<double>(i, 21), rClusters.at<double>(i, 22), rClusters.at<double>(i, 23), rClusters.at<double>(i, 24));
			tempPlanes->push_back(P);
		}
	}
	else
	{
		for (int i = 0; i < rClusters.rows; ++i)
		{
			Mat P = (Mat_<double>(5, 5) << rClusters.at<double>(i, 0), rClusters.at<double>(i, 1), rClusters.at<double>(i, 2), rClusters.at<double>(i, 3), rClusters.at<double>(i, 4),
				rClusters.at<double>(i, 5), rClusters.at<double>(i, 6), rClusters.at<double>(i, 7), rClusters.at<double>(i, 8), rClusters.at<double>(i, 9),
				rClusters.at<double>(i, 10), rClusters.at<double>(i, 11), rClusters.at<double>(i, 12), rClusters.at<double>(i, 13), rClusters.at<double>(i, 14),
				rClusters.at<double>(i, 15), rClusters.at<double>(i, 16), rClusters.at<double>(i, 17), rClusters.at<double>(i, 18), rClusters.at<double>(i, 19),
				rClusters.at<double>(i, 20), rClusters.at<double>(i, 21), rClusters.at<double>(i, 22), rClusters.at<double>(i, 23), rClusters.at<double>(i, 24));
			tempPlanes->push_back(P);
		}
	}

	bool thereIsChange = tempPlanes->size() != planes->size();
	if (thereIsChange)
	{
		planes = tempPlanes;
	}
}

inline double dataEnergyMotionEmpty(int p, int l, void *data)
{
	return 0;
}

inline double dataEnergyMotion(int p, int l, void *data)
{
	VideoMultiMotion::EnergyDataStruct *myData = (VideoMultiMotion::EnergyDataStruct *)data;

	Mat P = myData->planes->at(l);
	Mat pt = myData->points->col(p);

	Mat distMat = P * pt;

	double dist = 0;
	for (int i = 0; i < pt.rows; ++i)
		dist += distMat.at<double>(i) * distMat.at<double>(i);
	/*for (int i = 0; i < pt.rows; ++i)
		dist += pt.at<double>(i) * P.at<double>(i);
	dist += P.at<double>(pt.rows);*/

	double energy = abs(dist);
	double lambda = myData->energy_lambda;

	myData->minEnergyPerPoint[p] = MIN(myData->minEnergyPerPoint[p], energy);
	return (1.0 / lambda) * energy;
}

inline double DetermineNeighboringWeight(int p1, int p2, int l1, int l2, VideoMultiMotion::EnergyDataStruct *data)
{
	if (l1 == l2)
		return 0;
	return 1;

	/*const Mat &F1 = data->fundamentals->at(l1);
	const Mat &F2 = data->fundamentals->at(l2);

	double *pt11 = new double[2] {data->srcPoints->at(p1).x, data->srcPoints->at(p1).y};
	double *pt12 = new double[2] {data->dstPoints->at(p1).x, data->dstPoints->at(p1).y};
	double *pt21 = new double[2] {data->srcPoints->at(p2).x, data->srcPoints->at(p2).y};
	double *pt22 = new double[2] {data->dstPoints->at(p2).x, data->dstPoints->at(p2).y};
	
	//double d11 = SampsonDistance(pt11, pt12, F1);
	double d12 = SampsonDistance(pt11, pt12, F2);
	double d21 = SampsonDistance(pt21, pt22, F1);
	//double d22 = SampsonDistance(pt21, pt22, F2);

	delete pt11;
	delete pt12;
	delete pt21;
	delete pt22;

	double d1 = exp(-d12*d12 / 4.0);
	double d2 = exp(-d21*d21 / 4.0);

	return 1 - 0.5 * (d1 + d2);*/
}

inline double smoothnessEnergyMotion(int p1, int p2, int l1, int l2, void *data)
{
	VideoMultiMotion::EnergyDataStruct *myData = (VideoMultiMotion::EnergyDataStruct *)data;

	double weight = DetermineNeighboringWeight(p1, p2, l1, l2, myData);

	double lambda = myData->energy_lambda;
	return l1 != l2 ? lambda * weight : 0;
}

void  VideoMultiMotion::LabelingStep(double &energy)
{
	energy = 0;

	int *result = new int[points.cols];   // stores result of optimization
	int iteration_number = 0;

	// set up the needed data to pass to function for the data costs
	GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(points.cols, planes->size());
	EnergyDataStruct toFn(&points, planes, lambda);
	gc->setDataCost(&dataEnergyMotion, &toFn);
	gc->setSmoothCost(&smoothnessEnergyMotion, &toFn);
	gc->setLabelCost(complexity_beta);

	// Set neighbourhood
	for (int i = 0; i < neighbours.size(); ++i)
	{
		for (int j = 0; j < neighbours[i].size(); ++j)
		{
			int idx = neighbours[i][j].trainIdx;
			if (idx != i)
				gc->setNeighbors(i, idx);
		}
	}

	energy = gc->expansion(iteration_number, 1000);
	
	vector<Mat> pts(planes->size());

	inlierNumber = vector<int>(planes->size(), 0);
	for (int i = 0; i < points.cols; ++i)
	{
		int l = gc->whatLabel(i);
		Mat P = planes->at(l);
		Mat pt = points.col(i);
		Mat distMat = P * pt;

		double dist = 0;
		for (int i = 0; i < pt.rows; ++i)
			dist += distMat.at<double>(i) * distMat.at<double>(i);

		if (dist > 2.5 * assigning_thr)
		{
			gc->setLabel(i, -1);
			continue;
		}
		++inlierNumber[l];
	}

	for (int i = 0; i < planes->size(); ++i)
	{
		pts[i] = Mat_<double>(5, inlierNumber[i]);
	}

	vector<int> currIdx(planes->size(), 0);
	for (int i = 0; i < points.cols; ++i)
	{
		int l = gc->whatLabel(i);
		labeling[i] = l;

		if (l == -1)
			continue;

		pts[l].at<double>(0, currIdx[l]) = points.at<double>(0, i);
		pts[l].at<double>(1, currIdx[l]) = points.at<double>(1, i);
		pts[l].at<double>(2, currIdx[l]) = points.at<double>(2, i);
		pts[l].at<double>(3, currIdx[l]) = points.at<double>(3, i);
		pts[l].at<double>(4, currIdx[l]) = points.at<double>(4, i);
		++currIdx[l];
	}

	for (int i = pts.size() - 1; i >= 0; --i)
	{
		if (pts[i].cols < 5)
		{
			if (pts[i].cols < 1)
			{
				planes->erase(planes->begin() + i);
				for (int j = 0; j < labeling.size(); ++j)
					if (labeling[j] == i)
						labeling[j] = -1;
			}
			continue;
		}

		Mat plane;
		FitPlane(pts[i], plane);
	}

	delete gc;
	delete[] result;
}

void VideoMultiMotion::GenerateHypothesises()
{
	int N = HYPOTHESIS_NUMBER * points.cols;
	int M = 4;

	planes->resize(N);

#if USE_CONCURRENCY
	concurrency::parallel_for(0, N, [&](int i)
#else
	for (int i = 0; i < N; ++i)
#endif
	{
		// Generate minimal random subset
		vector<int> mss(M);
		Mat pts(points.rows, M, CV_64F);
		bool okay = false;
		while (!okay)
		{
			for (int j = 0; j < M; ++j)
			{
				while (1)
				{
					bool hasIt = false;
					int idx = (points.cols - 1) * (rand() / (double)RAND_MAX);
					for (int k = 0; k < j; ++k)
					{
						if (mss[k] == idx)
						{
							hasIt = true;
							break;
						}
					}

					if (!hasIt)
					{
						points.col(idx).copyTo(pts.col(j));
						mss[j] = idx;
						break;
					}
				}
			}

			// Fit plane
			Mat plane;
			FitPlane(pts, plane);
			planes->at(i) = plane;

			okay = true;
		}
	}
#if USE_CONCURRENCY
	);
#endif
}

void VideoMultiMotion::FitPlane(Mat pts, Mat &plane)
{
	Mat result;
	GramSmithOrth(pts, result);

	plane = Mat::eye(pts.rows, pts.rows, CV_64F) - result * result.t();


	/*Mat massPoint = Mat::zeros(pts.rows, 1, CV_64F);
	for (int i = 0; i < pts.cols; ++i)
		massPoint += pts.col(i) / pts.cols;

	double avgDist = 0;
	double avgRatio = 0;

	for (unsigned int i = 0; i < pts.cols; ++i)
	{
		pts.at<double>(0, i) -= massPoint.at<double>(0);
		pts.at<double>(1, i) -= massPoint.at<double>(1);
		pts.at<double>(2, i) -= massPoint.at<double>(2);
		pts.at<double>(3, i) -= massPoint.at<double>(3);
		pts.at<double>(4, i) -= massPoint.at<double>(4);
		avgDist += norm(pts.col(i) - massPoint);
	}

	avgDist = avgDist / pts.cols;
	avgRatio = sqrt(2) / avgDist;

	for (unsigned int i = 0; i < pts.cols; ++i)
	{
		pts.at<double>(0, i) = pts.at<double>(0, i) * avgRatio;
		pts.at<double>(1, i) = pts.at<double>(1, i) * avgRatio;
		pts.at<double>(2, i) = pts.at<double>(2, i) * avgRatio;
		pts.at<double>(3, i) = pts.at<double>(3, i) * avgRatio;
		pts.at<double>(4, i) = pts.at<double>(4, i) * avgRatio;
	}

	Mat evals, evecs;
	Mat PtP = pts * pts.t();
	//Mat PtP = pts.t() * pts;
	eigen(PtP, evals, evecs);
	
	plane = Mat(pts.rows + 1, 1, CV_64F);

	Mat n = (Mat_<double>(pts.rows, 1) << evecs.row(evecs.rows - 1).at<double>(0),
		evecs.row(evecs.rows - 1).at<double>(1), 
		evecs.row(evecs.rows - 1).at<double>(2),
		evecs.row(evecs.rows - 1).at<double>(3),
		evecs.row(evecs.rows - 1).at<double>(4));
	n = n / norm(n);

	plane.at<double>(0) = n.at<double>(0);
	plane.at<double>(1) = n.at<double>(1);
	plane.at<double>(2) = n.at<double>(2);
	plane.at<double>(3) = n.at<double>(3);
	plane.at<double>(4) = n.at<double>(4);

	double w = -massPoint.dot(n);
	plane.at<double>(5) = w;*/
}

void VideoMultiMotion::GramSmithOrth(Mat const &points, Mat &result)
{
	int K = points.rows;
	int D = points.cols;

	Mat I = Mat::eye(K, K, CV_64F);

	result = Mat::zeros(K, D, CV_64F);
	
	Mat col = points.col(0) / norm(points.col(0));
	col.copyTo(result.col(0));

	for (int i = 1; i < D; ++i)
	{
		Mat newcol = (I - result*result.t()) * points.col(i);
		newcol = newcol / norm(newcol);
		newcol.copyTo(result.col(i));
	}
}