#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include "estimator.h"
#include "prosac.h"

using namespace theia;

struct Line2D
{
	int mss1, mss2;
	cv::Mat descriptor;
	Line2D() {}
	Line2D(const Line2D& other)
	{
		mss1 = other.mss1;
		mss2 = other.mss2;
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class LineEstimator : public Estimator < cv::Mat, Line2D >
{
protected:

public:
	LineEstimator() {}
	~LineEstimator() {}

	int SampleSize() const {
		return 2;
	}

	int InlierLimit() const {
		return 14;
	}

	bool EstimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<Line2D>* models) const
	{
		Line2D model;

		if (sample_number < 2)
			return false;
		
		cv::Mat A(sample_number, 3, CV_64F);
		int idx;
		cv::Mat mass_point = cv::Mat::zeros(1, 2, CV_32F);
		for (auto i = 0; i < sample_number; ++i)
		{
			idx = sample[i];
			mass_point = mass_point + data.row(idx);

			A.at<double>(i, 0) = (double)data.at<float>(idx, 0);
			A.at<double>(i, 1) = (double)data.at<float>(idx, 1);
			A.at<double>(i, 2) = 1;
		}
		mass_point = mass_point * (1.0 / sample_number);

		cv::Mat AtA = A.t() * A;
		cv::Mat eValues, eVectors;
		eigen(AtA, eValues, eVectors);

		cv::Mat line = eVectors.row(2);
		line.convertTo(line, CV_32F);
		
		float length = sqrt(line.at<float>(0) * line.at<float>(0) + line.at<float>(1) * line.at<float>(1));
		line.at<float>(0) /= length;
		line.at<float>(1) /= length;

		line.at<float>(2) = -(line.at<float>(0) * mass_point.at<float>(0) + line.at<float>(1) * mass_point.at<float>(1));

		model.descriptor = line.t();
		models->push_back(model);

		if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
			return false;
		return true;
	}

	bool EstimateModel(const cv::Mat& data,
		const int *sample, 
		std::vector<Line2D>* models) const
	{
		Line2D model;

		if (sample[0] == sample[1])
			return false;

		// model calculation 
		int M = SampleSize();

		cv::Mat pt1 = data.row(sample[0]);
		cv::Mat pt2 = data.row(sample[1]);
				
		cv::Mat v = pt2 - pt1;
		v = v / norm(v);
		cv::Mat n = (cv::Mat_<float>(2, 1) << -v.at<float>(1), v.at<float>(0));
		float c = -(n.at<float>(0) * pt2.at<float>(0) + n.at<float>(1) * pt2.at<float>(1));

		model.mss1 = sample[0];
		model.mss2 = sample[1];
		model.descriptor = (cv::Mat_<float>(3, 1) << n.at<float>(0), n.at<float>(1), c);
		models->push_back(model);
		if (model.descriptor.rows == 0 || model.descriptor.cols == 0)
			return false;
		return true;
	}

	double Error(const cv::Mat& point, const Line2D& model) const
	{
		return (double)abs(point.at<float>(0) * model.descriptor.at<float>(0) + point.at<float>(1) * model.descriptor.at<float>(1) + model.descriptor.at<float>(2));
	}

	float Error(const cv::Mat& point, const cv::Mat& descriptor) const
	{
		return abs(point.at<float>(0) * descriptor.at<float>(0) + point.at<float>(1) * descriptor.at<float>(1) + descriptor.at<float>(2));
	}
};