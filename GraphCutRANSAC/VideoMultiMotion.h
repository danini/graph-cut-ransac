#pragma once

#include "MeanShiftClustering.h"
#include "MeanShift.h"
#include "GCoptimization.h"
#include "highgui.h"

#define LOG										1
#define MAX_ITERATION_NUMBER					50
#define CONVERGENCE_THRESHOLD					1e-5
#define DEFAULT_THRESHOLD						0.0030
#define DEFAULT_ASSIGN_THRESHOLD				3.0000
#define DEFAULT_LOCALITY						0.02
#define DEFAULT_LAMBDA							0.8
#define DEFAULT_BETA							10.0

#define FM_ESTIMATION_METHOD					8
#define HYPOTHESIS_NUMBER						1

class VideoMultiMotion
{
public:
	struct EnergyDataStruct
	{
		const Mat * const points;
		const std::vector<Mat> * const planes;
		const double energy_lambda;

		vector<double> minEnergyPerPoint;

		EnergyDataStruct(const Mat * const p,
			const vector<Mat> * const pls,
			const double lambda) :
			points(p),
			planes(pls),
			energy_lambda(lambda),
			minEnergyPerPoint(vector<double>(p->cols, INT_MAX))
		{
		}
	};

	VideoMultiMotion();
	~VideoMultiMotion();

	bool Process(Mat _points, double _lambda = DEFAULT_LAMBDA, double _mean_shift_thr = DEFAULT_THRESHOLD, double _assigning_thr = DEFAULT_ASSIGN_THRESHOLD, double _complexity_beta = DEFAULT_BETA);
	bool Process();

	void GetLabels(vector<int> &_labeling) { _labeling = labeling; }
	int GetLabelNumber() { return planes->size();  }
	int GetClusterNumber() { return cluster_number; }
	double GetEnergy() { return final_energy;  }

	double GetDataEnergy() { return finalDataEnergy; }
	double GetNeighbouringEnergy() { return finalNeighbouringEnergy; }
	double GetComplexityEnergy() { return finalComplexityEnergy; }
	
protected:
	bool log_to_console;

	int cluster_number;
	Mat points;
	vector<int> labeling;
	vector<vector<DMatch>> neighbours;
	int iterationNumber;
	vector<int> inlierNumber;
	double final_energy;
	double finalDataEnergy, finalNeighbouringEnergy, finalComplexityEnergy;
	
	double lambda, mean_shift_thr, assigning_thr, complexity_beta;

	vector<Mat> *planes;
	vector<Mat> *tempPlanes;

	void GenerateHypothesises();
	void MultipleMotionDetection();
	void MergingStep(bool &changed, bool isFirstCall);
	void LabelingStep(double &energy);

	double SampsonDistance(Mat const &pt1, Mat const &pt2, Mat const &F);

	void FitPlane(Mat points, Mat &plane);
	void GramSmithOrth(Mat const &points, Mat &result);
};

