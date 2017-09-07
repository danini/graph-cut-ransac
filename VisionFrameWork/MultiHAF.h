#pragma once

#include <vector>
#include <map>
#include <opencv2\highgui\highgui.hpp>

#define DEFAULT_THRESHOLD_FUNDAMENTAL_MATRIX	4.9
#define DEFAULT_THRESHOLD_HOMOGRAPHY			6.0
#define DEFAULT_LOCALITY						0.02
#define DEFAULT_LAMBDA							3.0
#define DEFAULT_BETA							150.0

#define DEFAULT_AFFINE_THRESHOLD				1.0
#define DEFAULT_LINENESS_THRESHOLD				0.005
#define MAX_ITERATION_NUMBER					100
#define CONVERGENCE_THRESHOLD					1e-5

#define USE_CONCURRENCY							TRUE
#define LOG_TO_CONSOLE							0

using namespace std;

class MultiHAF
{
public:
	struct EnergyDataStruct
	{
		const std::vector<Point2d> * const srcPoints;
		const std::vector<Point2d> * const dstPoints;
		const std::vector<Mat> * const homographies;
		const double energy_lambda;
		const map<string, double> * const homographySpreads;

		vector<double> minEnergyPerPoint;

		EnergyDataStruct(const vector<Point2d> * const p1,
			const vector<Point2d> * const p2, 
			const vector<Mat> * const hs,
			const double lambda,
			const map<string, double> * const _homographySpreads) :
			srcPoints(p1),
			dstPoints(p2),
			homographies(hs),
			energy_lambda(lambda),
			homographySpreads(_homographySpreads),
			minEnergyPerPoint(vector<double>(p1->size(), INT_MAX))
		{
		}
	};

	MultiHAF(double _thr_fund_mat = DEFAULT_THRESHOLD_FUNDAMENTAL_MATRIX, 
		double _thr_hom = DEFAULT_THRESHOLD_HOMOGRAPHY,
		double _locality = DEFAULT_LOCALITY,
		double _lambda = DEFAULT_LAMBDA,
		double _beta = DEFAULT_BETA);

	~MultiHAF();
	void Release();

	bool Process(Mat _image1, Mat _image2, vector<Point2d> _srcPoints, vector<Point2d> _dstPoints, vector<Mat> _affines);
	bool Process(Mat _image1, Mat _image2);

	int GetLabel(int idx) { return labeling[idx]; }
	void GetLabels(vector<int> &_labeling) { _labeling = labeling; }
	int GetPointNumber() { return labeling.size(); }
	int GetClusterNumber() { return clusterHomographies.size(); }
	int GetIterationNumber() { return iterationNumber; }

	void DrawClusters(Mat &img1, Mat &img2, int size, int minimumPointNumber);
	void HomographyCompatibilityCheck();

	double GetEnergy() { return finalEnergy; }
	double GetDataEnergy() { return finalDataEnergy; }
	double GetNeighbouringEnergy() { return finalNeighbouringEnergy; }
	double GetComplexityEnergy() { return finalComplexityEnergy; }
protected:

	Mat image1, image2;
	vector<Point2d> srcPointsOriginal, dstPointsOriginal;
	vector<Point2d> srcPoints, dstPoints;
	vector<Mat> affinesOriginal, affines, homographies;
	vector<int> inlierIndices;
	Mat F;
	bool log_to_console;
	bool degenerate_case;
	double finalEnergy, finalDataEnergy, finalNeighbouringEnergy, finalComplexityEnergy;

	map<string, double> homographySpreads;

	vector<vector<DMatch>> neighbours;
	
	vector<Mat> clusterHomographies;
	vector<int> labeling;
	vector<int> pointNumberPerCluster;

	double threshold_fundamental_matrix;
	double threshold_homography;
	double locality_lambda;
	double energy_lambda;
	double complexity_beta;
	double affine_threshold;
	double lineness_threshold;

	int iterationNumber;
	
	void GetFundamentalMatrixAndRefineData();
	void ComputeLocalHomographies();
	void EstablishStablePointSets();
	void ClusterMergingAndLabeling();

	void MergingStep(bool &changed);
	void LabelingStep(double &energy);
	void RefinementStep();

	void HandleDegenerateCase();
	void ComputeInliersOfHomography(int idx);

	void GetHomographyDiversity();
};


