#include "stdafx.h"
#include "MultiHAF.h"
#include "MeanShiftClustering.h"
#include "MeanShift.h"
#include "GCoptimization.h"
#include "highgui.h"

MultiHAF::MultiHAF(double _thr_fund_mat, double _thr_hom, double _locality, double _lambda, double _beta) :
	threshold_fundamental_matrix(_thr_fund_mat),
	threshold_homography(_thr_hom),
	locality_lambda(_locality),
	energy_lambda(_lambda),
	complexity_beta(_beta),
	affine_threshold(DEFAULT_AFFINE_THRESHOLD),
	lineness_threshold(DEFAULT_LINENESS_THRESHOLD),
	log_to_console(LOG_TO_CONSOLE)
{
}

MultiHAF::~MultiHAF()
{
	Release();
}

void MultiHAF::Release()
{
	clusterHomographies.resize(0);
	image1.release();
	image2.release();
	F.release();
	inlierIndices.resize(0);
	homographies.resize(0);
	srcPointsOriginal.resize(0);
	dstPointsOriginal.resize(0);
	affinesOriginal.resize(0);
	srcPoints.resize(0);
	dstPoints.resize(0);
	affines.resize(0);
}

bool MultiHAF::Process(Mat _image1, Mat _image2, vector<Point2d> _srcPoints, vector<Point2d> _dstPoints, vector<Mat> _affines)
{
	printf("MultiHAF Started.\n");
	srcPointsOriginal = _srcPoints;
	dstPointsOriginal = _dstPoints;
	affinesOriginal = _affines;

	return Process(_image1, _image2);
}

bool MultiHAF::Process(Mat _image1, Mat _image2)
{
	if (_image1.empty() || _image2.empty())
	{
		printf("Error: Images are not valid!\n");
		return false;
	}

	image1 = _image1;
	image2 = _image2;

	if (srcPointsOriginal.size() < 8 || 
		dstPointsOriginal.size() != srcPointsOriginal.size() || 
		affinesOriginal.size() != srcPointsOriginal.size())
	{
		// Detect feature points
		// TODO
	}

	GetFundamentalMatrixAndRefineData();
	
	if (degenerate_case)
	{
		HandleDegenerateCase();
	}
	else
	{
		ComputeLocalHomographies();
		//GetHomographyDiversity();

		/*clusterHomographies.resize(homographies.size());
		for (int i = 0; i < homographies.size(); ++i)
			clusterHomographies[i] = homographies[i];
			*/
		EstablishStablePointSets();
		ClusterMergingAndLabeling();
		//HomographyCompatibilityCheck();
	}

	return true;
}

void MultiHAF::HomographyCompatibilityCheck()
{
	vector<vector<Point2d>> srcPointsPerCluster(clusterHomographies.size());
	vector<vector<Point2d>> dstPointsPerCluster(clusterHomographies.size());

#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)labeling.size(), [&](int i)
#else
	for (int i = 0; i < labeling.size(); ++i)
#endif
	{
		int l = labeling[i];
		if (l > -1)
		{
			srcPointsPerCluster[l].push_back(srcPointsOriginal[i]);
			dstPointsPerCluster[l].push_back(dstPointsOriginal[i]);
		}
	}
#if USE_CONCURRENCY
	);
#endif

	F = F / F.at<double>(2, 2);

#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)clusterHomographies.size(), [&](int i)
#else
	for (int i = 0; i < clusterHomographies.size(); ++i)
#endif
	{
		bool remove = false;
		if (srcPointsPerCluster[i].size() > 4)
		{
			Mat H = findHomography(srcPointsPerCluster[i], dstPointsPerCluster[i]);
			H = H / H.at<double>(2, 2);

			// Compute compability with F
			double comp = norm(H.t() * F + F.t() * H);
			printf("Compatibility of the %d-th homography = %f\n", i, comp);

			remove = comp > 1.0;
		}
		else
			remove = true;

		if (remove)
		{
			for (int j = 0; j < labeling.size(); ++j)
				if (labeling[j] == i)
					labeling[j] = -1;
		}			
	}
#if USE_CONCURRENCY
	);
#endif
}

void MultiHAF::ClusterMergingAndLabeling()
{
	// Determine neighbourhood
	Mat pointVectors = (Mat_<float>(srcPointsOriginal.size(), 4));
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)srcPointsOriginal.size(), [&](int i)
#else
	for (int i = 0; i < srcPointsOriginal.size(); ++i)
#endif
	{
		pointVectors.at<float>(i, 0) = srcPointsOriginal.at(i).x;
		pointVectors.at<float>(i, 1) = srcPointsOriginal.at(i).y;
		pointVectors.at<float>(i, 2) = dstPointsOriginal.at(i).x;
		pointVectors.at<float>(i, 3) = dstPointsOriginal.at(i).y;
	}
#if USE_CONCURRENCY
		);
#endif

	FlannBasedMatcher flann;
	flann.radiusMatch(pointVectors, pointVectors, neighbours, 1.0 / locality_lambda);

	// Second mean shift round
	int itNum = 0;
	labeling.resize(srcPointsOriginal.size(), -1);
	double lastEnergy = INT_MAX;
	int not_changed_number = 0;

	while (itNum++ < MAX_ITERATION_NUMBER)
	{
		bool changed = false;
		MergingStep(changed);

		if (changed)
			not_changed_number = 0;
		else
			++not_changed_number;

		if (clusterHomographies.size() == 1)
		{
			labeling.resize(srcPointsOriginal.size(), -1);
			ComputeInliersOfHomography(0);
			break;
		}
		else if (clusterHomographies.size() == 0)
		{
			printf("No homography found!\n");
			break;
		}
		
		double energy;
		LabelingStep(energy);

		if (log_to_console)
			printf("Iteration %d.   Number of clusters = %d   Energy = %f\n", itNum, clusterHomographies.size(), energy);

		/*Mat im1 = image1.clone();
		Mat im2 = image2.clone();
		DrawClusters(im1, im2);
		imshow("Image 1", im1);
		imshow("Image 2", im2);
		waitKey(0);*/

		if ((!changed && abs(lastEnergy - energy) < CONVERGENCE_THRESHOLD)/* || not_changed_number > 1*/)
		{
			finalEnergy = energy;
			printf("Number of clusters = %d   Energy = %f\n", clusterHomographies.size(), energy);
			break;
		}

		lastEnergy = energy;
		
		//RefinementStep();
	}

	iterationNumber = itNum - 1;
}

void MultiHAF::DrawClusters(Mat &img1, Mat &img2, int size, int minimumPointNumber)
{
	vector<Scalar> colors(clusterHomographies.size() + 1);

	colors[0] = Scalar(0, 0, 0);
	for (int i = 1; i < clusterHomographies.size() + 1; ++i)
		colors[i] = Scalar(255 * (float)rand() / RAND_MAX, 255 * (float)rand() / RAND_MAX, 255 * (float)rand() / RAND_MAX);

	colors[1] = Scalar(255, 0, 0);
	colors[2] = Scalar(255, 255, 255);
	colors[3] = Scalar(0, 0, 255);
	colors[4] = Scalar(255, 255, 0);
	colors[5] = Scalar(255, 0, 255);
	colors[6] = Scalar(0, 255, 255);
	colors[7] = Scalar(0, 255, 0);
	colors[8] = Scalar(127, 255, 0);
	colors[9] = Scalar(0, 255, 127);
	colors[10] = Scalar(127, 127, 0);
	colors[11] = Scalar(63, 255, 127);

	vector<int> inlierNumbers(clusterHomographies.size(), 0);
	for (int i = 0; i < labeling.size(); ++i)
	{
		if (labeling[i] == -1)
			continue;
		++inlierNumbers[labeling[i]];
	}

	for (int i = 0; i < labeling.size(); ++i)
	{
		if (labeling[i] == -1 || inlierNumbers[labeling[i]] < minimumPointNumber)
			continue;

		circle(img1, srcPointsOriginal[i], size, colors[labeling[i] + 1], 2 * size);
		circle(img2, dstPointsOriginal[i], size, colors[labeling[i] + 1], 2 * size);
	}
}

void MultiHAF::MergingStep(bool &changed)
{
	Mat featureVectors(clusterHomographies.size(), 6, CV_64F);
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)clusterHomographies.size(), [&](int i)
#else
	for (int i = 0; i < clusterHomographies.size(); ++i)
#endif
	{
		Mat H = clusterHomographies[i];

		Mat pt1 = (Mat_<double>(3, 1) << 0, 0, 1);
		pt1 = H * pt1;
		pt1 = pt1 / pt1.at<double>(2);

		Mat pt2 = (Mat_<double>(3, 1) << 1, 0, 1);
		pt2 = H * pt2;
		pt2 = pt2 / pt2.at<double>(2);

		Mat pt3 = (Mat_<double>(3, 1) << 0, 1, 1);
		pt3 = H * pt3;
		pt3 = pt3 / pt3.at<double>(2);

		featureVectors.at<double>(i, 0) = pt1.at<double>(0);
		featureVectors.at<double>(i, 1) = pt1.at<double>(1);
		featureVectors.at<double>(i, 2) = pt2.at<double>(0);
		featureVectors.at<double>(i, 3) = pt2.at<double>(1);
		featureVectors.at<double>(i, 4) = pt3.at<double>(0);
		featureVectors.at<double>(i, 5) = pt3.at<double>(1);
	}
#if USE_CONCURRENCY
	);
#endif

	MeanShiftClustering<double> msc;
	Mat resultingClusters;
	vector<vector<int>> clusterPoints;
	msc.Cluster(featureVectors, threshold_homography, resultingClusters, clusterPoints);

	vector<Mat> homographies;
	homographies.reserve(clusterHomographies.size());
	for (int i = 0; i < resultingClusters.rows; ++i)
	{
		Mat pts1 = (Mat_<double>(3, 2) << 0, 0,
			1, 0,
			0, 1);

		Mat pts2 = (Mat_<double>(3, 2) << resultingClusters.at<double>(i, 0), resultingClusters.at<double>(i, 1),
			resultingClusters.at<double>(i, 2), resultingClusters.at<double>(i, 3),
			resultingClusters.at<double>(i, 4), resultingClusters.at<double>(i, 5));

		Mat H;
		vision::GetHomography<double>(pts1, pts2, H, vision::HomographyEstimator::THREE_POINT, noArray(), F);
			
		vector<int> inliers;
		for (int j = 0; j < srcPoints.size(); ++j)
		{
			Mat pt1 = (Mat_<double>(3, 1) << srcPoints.at(j).x, srcPoints.at(j).y, 1);
			pt1 = H * pt1;
			pt1 = pt1 / pt1.at<double>(2);

			Mat pt2 = (Mat_<double>(3, 1) << dstPoints.at(j).x, dstPoints.at(j).y, 1);

			if (norm(pt1 - pt2) < threshold_homography)
			{
				inliers.push_back(j);
			}
		}

		Mat lineParams = Mat_<double>(inliers.size(), 3);
		for (int j = 0; j < inliers.size(); ++j)
		{
			int idx = inliers[j];
			lineParams.at<double>(j, 0) = srcPoints.at(idx).x;
			lineParams.at<double>(j, 1) = srcPoints.at(idx).y;
			lineParams.at<double>(j, 2) = 1;
		}

		Mat eValues, eVectors;

		lineParams = lineParams.t() * lineParams;
		eigen(lineParams, eValues, eVectors);

		if (eValues.at<double>(2) < lineness_threshold || inliers.size() < 3)
			continue;

		homographies.push_back(H);
	}

	bool thereIsChange = homographies.size() != clusterHomographies.size();
	if (thereIsChange)
	{
		clusterHomographies = homographies;
	}
}

inline string keyFromPoints(Point2d p1, Point2d p2)
{
	return to_string((int)p1.x) + to_string((int)p1.y) + to_string((int)p2.x) + to_string((int)p2.y);
}

inline double computeLambda(const map<string, double> const * homographySpreads, string key)
{
	//Vec3d div = diversityImage->at<Vec3d>((int)p1.y, (int)p1.x);

	int maxDiv = 30;

	std::map<string, double>::const_iterator it = homographySpreads->find(key);
	double spread = 0;
	if (it != homographySpreads->end())
		spread = it->second;

	//cout << spread << endl;

	double lambda = 0.3 + 0.3 * (MIN(maxDiv, spread) / maxDiv);
	//cout << lambda << endl;
	return lambda;

	if (spread < 1)
		return 0.4;
	if (spread < 50)
		return 0.6;
	if (spread < 500)
		return 0.5;
	if (spread < 1000)
		return 0.4;
	return 0.3;
}

inline double computeThreshold(const map<string, double> const * homographySpreads, string key)
{
	//Vec3d div = diversityImage->at<Vec3d>((int)p1.y, (int)p1.x);

	int maxDiv = 30;

	std::map<string, double>::const_iterator it = homographySpreads->find(key);
	double spread = 0;
	if (it != homographySpreads->end())
		spread = it->second;

	//cout << spread << endl;
	
	if (spread < 1)
		return 2.3;
	if (spread < 50)
		return 2.6;
	if (spread < 500)
		return 3.0;
	if (spread < 1000)
		return 6.0;
	return 2.4;
}

inline double dataEnergy(int p, int l, void *data)
{
	MultiHAF::EnergyDataStruct *myData = (MultiHAF::EnergyDataStruct *)data;
	
	Mat H = myData->homographies->at(l); 
	Mat pt1 = (Mat_<double>(3, 1) << myData->srcPoints->at(p).x, myData->srcPoints->at(p).y, 1);
	Mat pt2 = (Mat_<double>(3, 1) << myData->dstPoints->at(p).x, myData->dstPoints->at(p).y, 1);
	
	Mat pt = H * pt1;
	pt = pt / pt.at<double>(2);

	Mat diff = pt - pt2;
	double energy = norm(diff);

	double lambda;// = computeLambda(myData->homographySpreads, keyFromPoints(myData->srcPoints->at(p), myData->dstPoints->at(p)));
	lambda = myData->energy_lambda;

	myData->minEnergyPerPoint[p] = MIN(myData->minEnergyPerPoint[p], norm(diff));
	return (1.0 / lambda) * energy;
}

inline double smoothnessEnergy(int p1, int p2, int l1, int l2, void *data)
{
	MultiHAF::EnergyDataStruct *myData = (MultiHAF::EnergyDataStruct *)data;
	//double lambda1 = computeLambda(myData->homographySpreads, keyFromPoints(myData->srcPoints->at(p1), myData->dstPoints->at(p1)));
	//double lambda2 = computeLambda(myData->homographySpreads, keyFromPoints(myData->srcPoints->at(p2), myData->dstPoints->at(p2)));

	double lambda;// = 0.5 * (lambda1 + lambda2);
	lambda = myData->energy_lambda;

	return l1 != l2 ? lambda : 0;
}

inline double labelEnergy(int l)
{
	return 1;
}

void MultiHAF::LabelingStep(double &energy)
{
	energy = 0;
	double sqr_threshold = threshold_homography * threshold_homography;

	int *result = new int[srcPointsOriginal.size()];   // stores result of optimization
	int iteration_number = 0;
	pointNumberPerCluster.resize(clusterHomographies.size(), 0);
		
	// set up the needed data to pass to function for the data costs
	GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(srcPointsOriginal.size(), clusterHomographies.size());
	EnergyDataStruct toFn(&srcPointsOriginal, &dstPointsOriginal, &clusterHomographies, energy_lambda, &homographySpreads);
	gc->setDataCost(&dataEnergy, &toFn);
	gc->setSmoothCost(&smoothnessEnergy, &toFn);
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

	// Compute energies
	EnergyDataStruct toFn3(&srcPointsOriginal, &dstPointsOriginal, &clusterHomographies, 0.001, &homographySpreads);
	gc->setLabelCost(0.0);
	gc->setDataCost(&dataEnergy, &toFn3);
	gc->setSmoothCost(&smoothnessEnergy, &toFn3);
	finalDataEnergy = gc->compute_energy();

	gc->setLabelCost(complexity_beta);
	finalComplexityEnergy = gc->compute_energy();

	EnergyDataStruct toFn2(&srcPointsOriginal, &dstPointsOriginal, &clusterHomographies, 1.0, &homographySpreads);
	gc->setDataCost(&dataEnergy, &toFn2);
	gc->setSmoothCost(&smoothnessEnergy, &toFn2);
	gc->setLabelCost(0.0);
	finalNeighbouringEnergy = gc->compute_energy();

	//gcOpt = gc;

	vector<Mat> pts1(clusterHomographies.size());
	vector<Mat> pts2(clusterHomographies.size());
	vector<Mat> tmpAffines(clusterHomographies.size());
	vector<int> inlierNumber(clusterHomographies.size(), 0);
	for (int i = 0; i < srcPointsOriginal.size(); ++i)
	{
		if (toFn.minEnergyPerPoint[i] > 3 * threshold_homography)
		{
			gc->setLabel(i, -1);
			continue;
		}
		int l = gc->whatLabel(i);
		++inlierNumber[l];
		++pointNumberPerCluster[l];
	}

	for (int i = 0; i < clusterHomographies.size(); ++i)
	{
		pts1[i] = Mat_<double>(inlierNumber[i], 2);
		pts2[i] = Mat_<double>(inlierNumber[i], 2);
		tmpAffines[i] = Mat_<double>(inlierNumber[i], 4);
	}

	vector<int> currIdx(clusterHomographies.size(), 0);
	for (int i = 0; i < srcPointsOriginal.size(); ++i)
	{
		int l = gc->whatLabel(i);
		labeling[i] = l;

		if (l == -1)
			continue;

		pts1[l].at<double>(currIdx[l], 0) = srcPointsOriginal.at(i).x;
		pts1[l].at<double>(currIdx[l], 1) = srcPointsOriginal.at(i).y;
		pts2[l].at<double>(currIdx[l], 0) = dstPointsOriginal.at(i).x;
		pts2[l].at<double>(currIdx[l], 1) = dstPointsOriginal.at(i).y;

		tmpAffines[l].at<double>(currIdx[l], 0) = affinesOriginal.at(i).at<double>(0, 0);
		tmpAffines[l].at<double>(currIdx[l], 1) = affinesOriginal.at(i).at<double>(0, 1);
		tmpAffines[l].at<double>(currIdx[l], 2) = affinesOriginal.at(i).at<double>(1, 0);
		tmpAffines[l].at<double>(currIdx[l], 3) = affinesOriginal.at(i).at<double>(1, 1);

		++currIdx[l];
	}

	for (int i = 0; i < pts1.size(); ++i)
	{
		if (pts1[i].rows == 0)
			continue;
		vision::GetHomographyHAF<double>(tmpAffines[i], F, pts1[i], pts2[i], clusterHomographies[i], false, false);
	}
		
	delete gc;
	delete[] result;
}

void MultiHAF::RefinementStep()
{
	vector<Mat> pts1(clusterHomographies.size());
	vector<Mat> pts2(clusterHomographies.size());
	vector<Mat> affines(clusterHomographies.size());
	vector<int> currentIdx(clusterHomographies.size(), 0);

	for (int i = 0; i < pointNumberPerCluster.size(); ++i)
	{
		pts1[i] = Mat(pointNumberPerCluster[i], 2, CV_64F);
		pts2[i] = Mat(pointNumberPerCluster[i], 2, CV_64F);
		affines[i] = Mat(pointNumberPerCluster[i], 4, CV_64F);
	}

	for (int i = 0; i < labeling.size(); ++i)
	{
		if (labeling[i] == -1)
			continue;

		int l = labeling[i];
		int idx = currentIdx[l];
		
		pts1[l].at<double>(idx, 0) = srcPointsOriginal[i].x;
		pts1[l].at<double>(idx, 1) = srcPointsOriginal[i].y;
		pts2[l].at<double>(idx, 0) = dstPointsOriginal[i].x;
		pts2[l].at<double>(idx, 1) = dstPointsOriginal[i].y;

		affines[l].at<double>(idx, 0) = affinesOriginal[i].at<double>(0, 0);
		affines[l].at<double>(idx, 1) = affinesOriginal[i].at<double>(0, 1);
		affines[l].at<double>(idx, 2) = affinesOriginal[i].at<double>(1, 0);
		affines[l].at<double>(idx, 3) = affinesOriginal[i].at<double>(1, 1);
		
		++currentIdx[l];
	}

	for (int i = 0; i < clusterHomographies.size(); ++i)
	{
		if (pointNumberPerCluster[i] < 1)
			continue;
		vision::GetHomography<double>(pts1[i], pts2[i], clusterHomographies[i], vision::HomographyEstimator::HAF, affines[i], F);
	}
}

void MultiHAF::EstablishStablePointSets()
{
	// Compute feature vector for each homography
	Mat featureVectors(srcPoints.size(), 10, CV_64F);

#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)srcPoints.size(), [&](int i)
#else
	for (int i = 0; i < srcPoints.size(); ++i)
#endif
	{
		Mat H = homographies[i];

		Mat pt1 = (Mat_<double>(3, 1) << 0, 0, 1);
		pt1 = H * pt1;
		pt1 = pt1 / pt1.at<double>(2);

		Mat pt2 = (Mat_<double>(3, 1) << 1, 0, 1);
		pt2 = H * pt2;
		pt2 = pt2 / pt2.at<double>(2);

		Mat pt3 = (Mat_<double>(3, 1) << 0, 1, 1);
		pt3 = H * pt3;
		pt3 = pt3 / pt3.at<double>(2);

		featureVectors.at<double>(i, 0) = pt1.at<double>(0);
		featureVectors.at<double>(i, 1) = pt2.at<double>(0);
		featureVectors.at<double>(i, 2) = pt3.at<double>(0);
		featureVectors.at<double>(i, 3) = pt1.at<double>(1);
		featureVectors.at<double>(i, 4) = pt2.at<double>(1);
		featureVectors.at<double>(i, 5) = pt3.at<double>(1);
		featureVectors.at<double>(i, 6) = srcPoints[i].x * locality_lambda;
		featureVectors.at<double>(i, 7) = srcPoints[i].y * locality_lambda;
		featureVectors.at<double>(i, 8) = dstPoints[i].x * locality_lambda;
		featureVectors.at<double>(i, 9) = dstPoints[i].y * locality_lambda;

	}
#if USE_CONCURRENCY
	);
#endif
	
	Mat clusters;
	vector<vector<int>> clusterPoints;
	MeanShiftClustering<double> msc;
	msc.Cluster(featureVectors, threshold_homography, clusters, clusterPoints);

	// Filter the obtained clusters and apply LSQ homography fitting
	for (int i = 0; i < clusterPoints.size(); ++i)
	{
		if (clusterPoints[i].size() < 3)
			continue;

		// Compute the centroid and fit line to the points
		Mat pts1 = Mat_<double>(clusterPoints[i].size(), 2);
		Mat pts2 = Mat_<double>(clusterPoints[i].size(), 2);
		Mat clusterAffines = Mat_<double>(clusterPoints[i].size(), 4);
		for (int j = 0; j < clusterPoints[i].size(); ++j)
		{
			int idx = clusterPoints[i][j];

			pts1.at<double>(j, 0) = srcPoints[idx].x;
			pts1.at<double>(j, 1) = srcPoints[idx].y;
			pts2.at<double>(j, 0) = dstPoints[idx].x;
			pts2.at<double>(j, 1) = dstPoints[idx].y;
			clusterAffines.row(j) = (Mat_<double>(1, 4) << affines[idx].at<double>(0, 0), affines[idx].at<double>(0, 1),
				affines[idx].at<double>(1, 0), affines[idx].at<double>(1, 1));
		}

		// Fit homography in LSQ sense to the points
		Mat H;
		vision::GetHomography<double>(pts1, pts2, H, vision::HomographyEstimator::THREE_POINT, clusterAffines, F);

		clusterHomographies.push_back(H);

		pts1.release();
		pts2.release();
		clusterAffines.release();
	}

	featureVectors.release();
	clusters.release();
	clusterPoints.resize(0);

	if (log_to_console)
		printf("Number of stable, local clusters = %d\n", clusterHomographies.size());
}

void MultiHAF::ComputeLocalHomographies()
{
	homographies.resize(affines.size());

#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)srcPoints.size(), [&](int i)
#else
	for (int i = 0; i < srcPoints.size(); ++i)
#endif
	{
		// Add to class containers
		Mat pt1 = (Mat_<double>(1, 2) << srcPoints[i].x, srcPoints[i].y);
		Mat pt2 = (Mat_<double>(1, 2) << dstPoints[i].x, dstPoints[i].y);

		Mat A;
		A = (Mat_<double>(1, 4) << affines[i].at<double>(0, 0), affines[i].at<double>(0, 1), affines[i].at<double>(1, 0), affines[i].at<double>(1, 1));

		vision::GetHomographyHAF<double>(A, F, pt1, pt2, homographies[i]);
	}
#if USE_CONCURRENCY
	);
#endif
}

void MultiHAF::HandleDegenerateCase()
{
	printf("Handle degenerate case. Apply RANSAC homography estimation to the point correspondences\n");
	
	vector<uchar> mask;
	Mat H = findHomography(srcPointsOriginal, dstPointsOriginal, CV_RANSAC, threshold_homography, mask);

	labeling.resize(srcPointsOriginal.size());
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)srcPointsOriginal.size(), [&](int i)
#else
	for (int i = 0; i < srcPointsOriginal.size(); ++i)
#endif
	{
		labeling[i] = mask[i] ? 0 : -1;
	}
#if USE_CONCURRENCY
	);
#endif

	clusterHomographies.push_back(H);
}

void MultiHAF::ComputeInliersOfHomography(int idx)
{
#if USE_CONCURRENCY
	concurrency::parallel_for(0, (int)srcPointsOriginal.size(), [&](int i)
#else
	for (int i = 0; i < srcPointsOriginal.size(); ++i)
#endif
	{
		Mat pt1 = (Mat_<double>(3,1) << srcPointsOriginal[i].x, srcPointsOriginal[i].y, 1);
		Mat pt2 = (Mat_<double>(3,1) << dstPointsOriginal[i].x, dstPointsOriginal[i].y, 1);
		Mat H = clusterHomographies[idx];

		pt1 = H * pt1;
		pt1 = pt1 / pt1.at<double>(2);

		if (norm(pt1 - pt2) < threshold_homography)
			labeling[i] = idx;
	}
#if USE_CONCURRENCY
	);
#endif
}

void MultiHAF::GetFundamentalMatrixAndRefineData()
{
	vector<uchar> mask;
	F = findFundamentalMat(srcPointsOriginal, dstPointsOriginal, CV_FM_RANSAC, threshold_fundamental_matrix, 0.99, mask);

	// Return 
	if (norm(F) < 1e-5)
	{
		degenerate_case = true;
		printf("Degenerate case, the fundamental matrix cannot be estimated.\n");
		return;
	}

	degenerate_case = false;

	for (int i = 0; i < srcPointsOriginal.size(); ++i)
	{
		if (mask[i])
		{
			Mat a = (Mat)srcPointsOriginal[i];
			Mat b = (Mat)dstPointsOriginal[i];

			// Apply Hartley & Sturm optimization to the original point locations
			Mat c, d;
			bool error = !vision::OptimalTriangulation<double>(F, a, b, c, d);
			if (!error)
			{
				Mat A = (Mat_<double>(2, 2) << affinesOriginal[i].at<double>(0, 0), affinesOriginal[i].at<double>(0, 1),
					affinesOriginal[i].at<double>(1, 0), affinesOriginal[i].at<double>(1, 1));

				// Remove poor quality affine transformations
				double scaleError, angularError, distanceError;
				vision::GetAffineConsistency(A, F, c, d, scaleError, angularError, distanceError);
				if (distanceError > 1.0)
					continue;

				// Get optimal affine transformation
				Mat optA;
				vision::GetOptimalAffineTransformation<double>(A, F, c, d, optA);
				
				affines.push_back(optA);
				srcPoints.push_back(Point2d(c.at<double>(0), c.at<double>(1)));
				dstPoints.push_back(Point2d(d.at<double>(0), d.at<double>(1)));
			}
		}
	}
	
	printf("%d points kept from the initial %d after filtering.\n", srcPoints.size(), srcPointsOriginal.size());

	if (srcPoints.size() < 8)
	{
		degenerate_case = true;
		printf("Degenerate case, not enough points remained.\n");
		return;
	}
}
