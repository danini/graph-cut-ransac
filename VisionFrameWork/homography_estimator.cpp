#include "stdafx.h"

#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <cv.h>

#include "estimator.h"
#include "prosac.h"

using namespace std;
using namespace theia;
using namespace cv;

struct Homography
{
	Mat descriptor;
	Homography() {}
	Homography(const Homography& other)
	{
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class RobustHomographyEstimator : public Estimator < Mat, Homography >
{
protected:

public:
	RobustHomographyEstimator() {}
	~RobustHomographyEstimator() {}

	int SampleSize() const {
		return 4;
	}

	int InlierLimit() const {
		return 28;
	}

	bool EstimateModel(const Mat& data,
		const int *sample,
		vector<Homography>* models) const
	{
		// model calculation 
		int M = SampleSize();
		
		vector<Point2d> pts1(M);
		vector<Point2d> pts2(M);

		for (int i = 0; i < M; ++i)
		{
			pts1[i].x = (double)data.at<float>(sample[i], 0);
			pts1[i].y = (double)data.at<float>(sample[i], 1);
			pts2[i].x = (double)data.at<float>(sample[i], 3);
			pts2[i].y = (double)data.at<float>(sample[i], 4);
		}

		Mat H = findHomography(pts1, pts2);
		H.convertTo(H, CV_32F);

		if (H.cols == 0)
			return false;

		Homography model;
		model.descriptor = H;
		models->push_back(model);
		return true;
	}

	bool EstimateModelNonminimal(const Mat& data,
		const int *sample,
		int sample_number,
		vector<Homography>* models) const
	{

		if (sample_number < SampleSize())
			return false;

		// model calculation 
		int M = sample_number;

		vector<Point2d> pts1(M);
		vector<Point2d> pts2(M);

		for (int i = 0; i < M; ++i)
		{
			pts1[i].x = (double)data.at<float>(sample[i], 0);
			pts1[i].y = (double)data.at<float>(sample[i], 1);
			pts2[i].x = (double)data.at<float>(sample[i], 3);
			pts2[i].y = (double)data.at<float>(sample[i], 4);
		}

		Mat H = findHomography(pts1, pts2, NULL, 0);
		H.convertTo(H, CV_32F);

		if (H.cols == 0)
		{
			H = findHomography(pts1, pts2, NULL, CV_LMEDS);
			H.convertTo(H, CV_32F);

			if (H.cols == 0)
				return false;
		}

		Homography model;
		model.descriptor = H;
		models->push_back(model);
		return true;
	}

	double Error(const Mat& point, const Homography& model) const
	{
		const float* s = (float *)point.data;
		const float* p = (float *)model.descriptor.data;

		const float x1 = *s;
		const float y1 = *(s + 1);
		const float x2 = *(s + 3);
		const float y2 = *(s + 4);

		const float t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const float t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const float t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const float d1 = x2 - (t1 / t3);
		const float d2 = y2 - (t2 / t3);

		return d1*d1 + d2*d2;
	}

	float Error(const Mat& point, const Mat& descriptor) const
	{
		const float* s = (float *)point.data;
		const float* p = (float *)descriptor.data;

		const float x1 = *s;
		const float y1 = *(s + 1);
		const float x2 = *(s + 3);
		const float y2 = *(s + 4);

		const float t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const float t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const float t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const float d1 = x2 - (t1 / t3);
		const float d2 = y2 - (t2 / t3);

		return d1*d1 + d2*d2;
	}

	bool Algorithm_4_point(const Mat& data,
		const int *sample,
		int sample_number,
		vector<Homography>* models) const
	{
		/*float a[8 * 9];
		Mat A(8, 9, CV_32F, a);
		
		for (int i = 0; i < 4; ++i)
		{
			float x1 = data.at<float>(sample[i], 0);
			float y1 = data.at<float>(sample[i], 1);
			float x2 = data.at<float>(sample[i], 3);
			float y2 = data.at<float>(sample[i], 4);

			int r = i * 9;
			a[r + 0] = x1;
			a[r + 1] = y1;
			a[r + 2] = 1;
			a[r + 3] = 1;
			a[r + 4] = 1;
			a[r + 5] = 1;
			a[r + 6] = 1;
			a[r + 7] = 1;

			a[r + 0] = x1;
			a[r + 1] = y1;
			a[r + 2] = 1;

			a[r + 0] = x1;
			a[r + 1] = y1;
			a[r + 2] = 1;

			a[r + 0] = x1;
			a[r + 1] = y1;
			a[r + 2] = 1;
		}*/

		return true;
	}

};