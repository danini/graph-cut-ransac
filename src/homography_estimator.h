#include "estimator.h"

#include <opencv2/calib3d/calib3d.hpp>

struct Homography
{
	cv::Mat descriptor;
	Homography() {}
	Homography(const Homography& other)
	{
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class RobustHomographyEstimator : public theia::Estimator < cv::Mat, Homography >
{
protected:

public:
	RobustHomographyEstimator() {}
	~RobustHomographyEstimator() {}

	size_t sampleSize() const {
		return 4;
	}

	size_t inlierLimit() const {
		return 28;
	}

	bool estimateModel(const cv::Mat& data,
		const int *sample,
		std::vector<Homography>* models) const
	{
		// model calculation 
		int M = sampleSize();
		
		std::vector<cv::Point2d> pts1(M);
		std::vector<cv::Point2d> pts2(M);

		for (auto i = 0; i < M; ++i)
		{
			pts1[i].x = (double)data.at<double>(sample[i], 0);
			pts1[i].y = (double)data.at<double>(sample[i], 1);
			pts2[i].x = (double)data.at<double>(sample[i], 3);
			pts2[i].y = (double)data.at<double>(sample[i], 4);
		}

		cv::Mat H = cv::findHomography(pts1, pts2);
		H.convertTo(H, CV_64F);

		if (H.cols == 0)
			return false;

		Homography model;
		model.descriptor = H;
		models->emplace_back(model);
		return true;
	}

	bool estimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<Homography>* models) const
	{

		if (sample_number < sampleSize())
			return false;

		// model calculation 
		int M = sample_number;

		std::vector<cv::Point2d> pts1(M);
		std::vector<cv::Point2d> pts2(M);

		for (auto i = 0; i < M; ++i)
		{
			pts1[i].x = (double)data.at<double>(sample[i], 0);
			pts1[i].y = (double)data.at<double>(sample[i], 1);
			pts2[i].x = (double)data.at<double>(sample[i], 3);
			pts2[i].y = (double)data.at<double>(sample[i], 4);
		}

		cv::Mat H = cv::findHomography(pts1, pts2, NULL, 0);
		H.convertTo(H, CV_64F);

		if (H.cols == 0)
		{
			H = cv::findHomography(pts1, pts2, NULL, CV_LMEDS);
			H.convertTo(H, CV_64F);

			if (H.cols == 0)
				return false;
		}

		Homography model;
		model.descriptor = H;
		models->emplace_back(model);
		return true;
	}

	double residual(const cv::Mat& point, const Homography& model) const
	{
		const double* s = (double *)point.data;
		const double* p = (double *)model.descriptor.data;

		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 3);
		const double y2 = *(s + 4);

		const double t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double d1 = x2 - (t1 / t3);
		const double d2 = y2 - (t2 / t3);

		return d1*d1 + d2*d2;
	}

	double residual(const cv::Mat& point, const cv::Mat& descriptor) const
	{
		const double* s = (double *)point.data;
		const double* p = (double *)descriptor.data;

		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 3);
		const double y2 = *(s + 4);

		const double t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double d1 = x2 - (t1 / t3);
		const double d2 = y2 - (t2 / t3);

		return d1*d1 + d2*d2;
	}

	bool Algorithm_4_point(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<Homography>* models) const
	{
		/*double a[8 * 9];
		cv::Mat A(8, 9, CV_64F, a);
		
		for (auto i = 0; i < 4; ++i)
		{
			double x1 = data.at<double>(sample[i], 0);
			double y1 = data.at<double>(sample[i], 1);
			double x2 = data.at<double>(sample[i], 3);
			double y2 = data.at<double>(sample[i], 4);

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