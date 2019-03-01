#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include "estimator.h"
#include "prosac.h"

using namespace theia;

struct FundamentalMatrix
{
	cv::Mat descriptor;
	std::vector<int> mss;

	FundamentalMatrix() {}
	FundamentalMatrix(const FundamentalMatrix& other)
	{
		descriptor = other.descriptor.clone();
		mss = other.mss;
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class FundamentalMatrixEstimator : public Estimator < cv::Mat, FundamentalMatrix >
{
protected:

public:
	FundamentalMatrixEstimator() {}
	~FundamentalMatrixEstimator() {}

	int SampleSize() const { 
		return 7;
	}

	int InlierLimit() const {
		return 49;
	}

	bool EstimateModel(const cv::Mat& data,
		const int *sample,
		std::vector<FundamentalMatrix>* models) const
	{
		// model calculation 
		int M = SampleSize();

		Algorithm_7_point(data, sample, M, models);
		if (models->size() == 0)
			return false;
		return true;
	}

	bool EstimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<FundamentalMatrix>* models) const
	{

		// model calculation 
		int M = sample_number;

		if (sample_number < 8)
			return false;

		Algorithm_8_point(data, sample, sample_number, models);
		return true;
	}

	double Error(const cv::Mat& point, const FundamentalMatrix& model) const
	{
		const float* s = (float *)point.data;
		const float x1 = *s;
		const float y1 = *(s + 1);
		const float x2 = *(s + 2);
		const float y2 = *(s + 3);

		const float* p = (float *)model.descriptor.data;

		const float f11 = *(p);
		const float f12 = *(p + 1);
		const float f13 = *(p + 2);
		const float f21 = *(p + 3);
		const float f22 = *(p + 4);
		const float f23 = *(p + 5);
		const float f31 = *(p + 6);
		const float f32 = *(p + 7);
		const float f33 = *(p + 8);

		const float l1 = f11* x2 + f21 * y2 + f31;
		const float l2 = f12 * x2 + f22 * y2 + f32;
		const float l3 = f13 * x2 + f23 * y2 + f33;

		const float t1 = f11 * x1 + f12 * y1 + f13;
		const float t2 = f21 * x1 + f22 * y1 + f23;
		const float t3 = f31 * x1 + f32 * y1 + f33;

		const float a1 = l1 * x1 + l2 * y1 + l3;
		const float a2 = sqrt(l1 * l1 + l2 * l2);

		const float b1 = t1 * x2 + t2 * y2 + t3;
		const float b2 = sqrt(t1 * t1 + t2 * t2);

		const float d = l1 * x2 + l2 * y2 + l3;
		const float d1 = a1 / a2;
		const float d2 = b1 / b2;

		return (double)abs(0.5 * (d1 + d2));
	}

	float Error(const cv::Mat& point, const cv::Mat& descriptor) const
	{
		const float* s = (float *)point.data;
		const float x1 = *s;
		const float y1 = *(s + 1);
		const float x2 = *(s + 2);
		const float y2 = *(s + 3);

		const float* p = (float *)descriptor.data;

		const float f11 = *(p);
		const float f12 = *(p + 1);
		const float f13 = *(p + 2);
		const float f21 = *(p + 3);
		const float f22 = *(p + 4);
		const float f23 = *(p + 5);
		const float f31 = *(p + 6);
		const float f32 = *(p + 7);
		const float f33 = *(p + 8);

		const float l1 = f11 * x2 + f21 * y2 + f31;
		const float l2 = f12 * x2 + f22 * y2 + f32;
		const float l3 = f13 * x2 + f23 * y2 + f33;

		const float t1 = f11 * x1 + f12 * y1 + f13;
		const float t2 = f21 * x1 + f22 * y1 + f23;
		const float t3 = f31 * x1 + f32 * y1 + f33;

		const float a1 = l1 * x1 + l2 * y1 + l3;
		const float a2 = sqrt(l1 * l1 + l2 * l2);

		const float b1 = t1 * x2 + t2 * y1 + t3;
		const float b2 = sqrt(t1 * t1 + t2 * t2);

		const float d1 = a1 / a2;
		const float d2 = b1 / b2;

		return abs(0.5f * (d1 + d2));
	}

	bool Algorithm_8_point(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<FundamentalMatrix>* models) const
	{
		float f[9];
		cv::Mat evals(1, 9, CV_32F), evecs(9, 9, CV_32F);
		cv::Mat A(sample_number, 9, CV_32F);
		cv::Mat F(3, 3, CV_32F, f);
		int i;

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < sample_number; i++)
		{
			float x0 = data.at<float>(sample[i], 0), y0 = data.at<float>(sample[i], 1);
			float x1 = data.at<float>(sample[i], 2), y1 = data.at<float>(sample[i], 3);

			A.at<float>(i, 0) = x1*x0;
			A.at<float>(i, 1) = x1*y0;
			A.at<float>(i, 2) = x1;
			A.at<float>(i, 3) = y1*x0;
			A.at<float>(i, 4) = y1*y0;
			A.at<float>(i, 5) = y1;
			A.at<float>(i, 6) = x0;
			A.at<float>(i, 7) = y0;
			A.at<float>(i, 8) = 1;
		}

		// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		// the solution is linear subspace of dimensionality 2.
		// => use the last two singular std::vectors as a basis of the space
		// (according to SVD properties)
		cv::Mat cov = A.t() * A;
		eigen(cov, evals, evecs);

		for (i = 0; i < 9; ++i)
			f[i] = evecs.at<float>(8, i);

		Model model;
		model.descriptor = F;
		models->push_back(model);
		return true;
	}

	bool Algorithm_7_point(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<FundamentalMatrix>* models) const
	{

		float a[7 * 9], v[9 * 9], c[4], r[3];
		float *f1, *f2;
		float t0, t1, t2;
		cv::Mat evals, evecs(9, 9, CV_32F, v);
		cv::Mat A(7, 9, CV_32F, a);
		cv::Mat coeffs(1, 4, CV_32F, c);
		cv::Mat roots(1, 3, CV_32F, r);
		int i, k, n;

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < 7; i++)
		{
			float x0 = data.at<float>(sample[i], 0), y0 = data.at<float>(sample[i], 1);
			float x1 = data.at<float>(sample[i], 2), y1 = data.at<float>(sample[i], 3);

			a[i * 9 + 0] = x1*x0;
			a[i * 9 + 1] = x1*y0;
			a[i * 9 + 2] = x1;
			a[i * 9 + 3] = y1*x0;
			a[i * 9 + 4] = y1*y0;
			a[i * 9 + 5] = y1;
			a[i * 9 + 6] = x0;
			a[i * 9 + 7] = y0;
			a[i * 9 + 8] = 1;
		}

		// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		// the solution is linear subspace of dimensionality 2.
		// => use the last two singular std::vectors as a basis of the space
		// (according to SVD properties)
		cv::Mat cov = A.t() * A;
		eigen(cov, evals, evecs);
		f1 = v + 7 * 9;
		f2 = v + 8 * 9;

		// f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
		// as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
		// so f ~ lambda*f1 + (1 - lambda)*f2.
		// use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
		// it will be a cubic equation.
		// find c - polynomial coefficients.
		for (i = 0; i < 9; i++)
			f1[i] -= f2[i];

		t0 = f2[4] * f2[8] - f2[5] * f2[7];
		t1 = f2[3] * f2[8] - f2[5] * f2[6];
		t2 = f2[3] * f2[7] - f2[4] * f2[6];

		c[3] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2;

		c[2] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -
			f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
			f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
			f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
			f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
			f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
			f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);

		t0 = f1[4] * f1[8] - f1[5] * f1[7];
		t1 = f1[3] * f1[8] - f1[5] * f1[6];
		t2 = f1[3] * f1[7] - f1[4] * f1[6];

		c[1] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
			f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
			f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
			f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
			f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
			f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
			f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);

		c[0] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;

		// solve the cubic equation; there can be 1 to 3 roots ...
		n = solveCubic(coeffs, roots);

		if (n < 1 || n > 3)
			return false;

		for (k = 0; k < n; k++)
		{
			float f[9];
			cv::Mat F(3, 3, CV_32F, f);

			// for each root form the fundamental matrix
			float lambda = r[k], mu = 1.;
			float s = f1[8] * r[k] + f2[8];

			// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
			if (fabs(s) > DBL_EPSILON)
			{
				mu = 1.0f / s;
				lambda *= mu;

				for (auto i = 0; i < 8; ++i)
					f[i] = f1[i] * lambda + f2[i] * mu;
				f[8] = 1.0;

				/* orient. constr. */
				if (!all_ori_valid(&F, data, sample, sample_number)) {
					continue;
				}

				Model model;
				model.descriptor = F;
				model.mss.resize(sample_number);
				for (auto i = 0; i < sample_number; ++i)
					model.mss[i] = sample[i];

				models->push_back(model);
			}
		}

		return true;
	}

	/************** oriented constraints ******************/
	void epipole(cv::Mat &ec, const cv::Mat *F) const
	{
		ec = F->row(0).cross(F->row(2));

		for (auto i = 0; i < 3; i++)
			if ((ec.at<float>(i) > 1.9984e-15) || (ec.at<float>(i) < -1.9984e-15)) return;
		ec = F->row(1).cross(F->row(2));
	}

	float getorisig(const cv::Mat *F, const cv::Mat *ec, const cv::Mat &u) const
	{
		float s1, s2;		
		s1 = F->at<float>(0,0) * u.at<float>(3) + F->at<float>(1,0) * u.at<float>(4) + F->at<float>(2,0);
		s2 = ec->at<float>(1) - ec->at<float>(2) * u.at<float>(1);
		return(s1 * s2);
	}

	int all_ori_valid(const cv::Mat *F, const cv::Mat &data, const int *sample, int N) const
	{
		cv::Mat ec;
		float sig, sig1;
		int i;
		epipole(ec, F);
		sig1 = getorisig(F, &ec, data.row(sample[0]));
		for (i = 1; i < N; i++)
		{
			sig = getorisig(F, &ec, data.row(sample[i]));
			if (sig1 * sig < 0) return 0;
		}
		return 1;
	}
};