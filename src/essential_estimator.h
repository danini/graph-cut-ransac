#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include "five_point/5point.h"
#include "estimator.h"

struct EssentialMatrix
{
	cv::Mat descriptor;
	cv::Mat F;
	std::vector<int> mss;

	EssentialMatrix() {}
	EssentialMatrix(const EssentialMatrix& other)
	{
		descriptor = other.descriptor.clone();
		F = other.F.clone();
		mss = other.mss;
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class EssentialMatrixEstimator : public theia::Estimator < cv::Mat, EssentialMatrix >
{
protected:

public:
	cv::Mat K1i, K2ti;

	EssentialMatrixEstimator() {}
	~EssentialMatrixEstimator() {}

	size_t sampleSize() const {
		return 5;
	}

	size_t inlierLimit() const {
		return 7 * sampleSize();
	}

	bool estimateModel(const cv::Mat& data,
		const int *sample,
		std::vector<EssentialMatrix>* models) const
	{
		// model calculation 
		int M = sampleSize();

		//Algorithm_8_point(data, sample, M, models);
		Algorithm_5_point(data, sample, M, models);
		if (models->size() == 0)
			return false;
		return true;
	}

	bool estimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<EssentialMatrix>* models) const
	{

		// model calculation 
		int M = sample_number;

		if (sample_number < 7)
			return false;

		if (sample_number == 7)
			solverSevenPoint(data, sample, M, models);
		else
			solverEightPoint(data, sample, sample_number, models);

		if (models->size() == 0)
			return false;

		for (auto i = 0; i < models->size(); ++i)
			if (models->at(i).F.rows != 3 || models->at(i).descriptor.rows != 3)
				return false;

		return true;
	}

	double residual(const cv::Mat& point, const EssentialMatrix& model) const
	{
		const double* s = (double *)point.data;
		const double x1 = *(s + 6);
		const double y1 = *(s + 7);
		const double x2 = *(s + 9);
		const double y2 = *(s + 10);
				
		const double* p = (double *)model.F.data;

		const double l1 = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
		const double l2 = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
		const double l3 = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);

		const double t1 = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double a1 = l1 * x1 + l2 * y1 + l3;
		const double a2 = sqrt(l1 * l1 + l2 * l2);

		const double b1 = t1 * x2 + t2 * y2 + t3;
		const double b2 = sqrt(t1 * t1 + t2 * t2);

		const double d = l1 * x2 + l2 * y2 + l3;
		const double d1 = a1 / a2;
		const double d2 = b1 / b2;

		return (double)abs(0.5 * (d1 + d2));
	}

	double residual(const cv::Mat& point, const cv::Mat& descriptor) const
	{
		const double* s = (double *)point.data;
		const double x1 = *(s + 6 - 6);
		const double y1 = *(s + 7 - 6);
		const double x2 = *(s + 9 - 6);
		const double y2 = *(s + 10 - 6);

		const double* p = (double *)descriptor.data;

		const double l1 = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
		const double l2 = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
		const double l3 = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);

		const double t1 = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double a1 = l1 * x1 + l2 * y1 + l3;
		const double a2 = sqrt(l1 * l1 + l2 * l2);

		const double b1 = t1 * x2 + t2 * y1 + t3;
		const double b2 = sqrt(t1 * t1 + t2 * t2);

		const double d1 = a1 / a2;
		const double d2 = b1 / b2;

		return abs(0.5f * (d1 + d2));
	}

	bool Algorithm_5_point(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<EssentialMatrix>* models) const
	{
		cv::Mat E, P;
		std::vector<cv::Point2d> pts1(5), pts2(5);

		for (auto i = 0; i < 5; i++)
		{
			double x0 = data.at<double>(sample[i], 0), y0 = data.at<double>(sample[i], 1);
			double x1 = data.at<double>(sample[i], 3), y1 = data.at<double>(sample[i], 4);

			pts1[i].x = static_cast<double>(x0);
			pts1[i].y = static_cast<double>(y0);
			pts2[i].x = static_cast<double>(x1);
			pts2[i].y = static_cast<double>(y1);
		}

		Solve5PointEssential(pts1, pts2, E, P);
		E.convertTo(E, CV_64F);

		if (E.rows != 3)
			return false;

		Model model;
		model.descriptor = E;
		model.F = K2ti * E * K1i;

		models->emplace_back(model);
		return true;
	}

	bool solverEightPoint(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<EssentialMatrix>* models) const
	{
		double f[9];
		cv::Mat evals(1, 9, CV_64F), evecs(9, 9, CV_64F);
		cv::Mat A(sample_number, 9, CV_64F);
		cv::Mat E(3, 3, CV_64F, f);
		int i;


		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < sample_number; i++)
		{
			/*if (sample[i] >= data.rows)
				return false;*/

			//double x0 = data.at<double>(sample[i], 0), y0 = data.at<double>(sample[i], 1);
			//double x1 = data.at<double>(sample[i], 3), y1 = data.at<double>(sample[i], 4);

			//std::cout << sample[i] << endl;
			int idx = sample[i] * 12;
			const double *data_ptr = ((double *)data.data + idx);
			double x0 = *(data_ptr), y0 = *(data_ptr + 1);
			double x1 = *(data_ptr + 3), y1 = *(data_ptr + 4);

			A.at<double>(i, 0) = x1*x0;
			A.at<double>(i, 1) = x1*y0;
			A.at<double>(i, 2) = x1;
			A.at<double>(i, 3) = y1*x0;
			A.at<double>(i, 4) = y1*y0;
			A.at<double>(i, 5) = y1;
			A.at<double>(i, 6) = x0;
			A.at<double>(i, 7) = y0;
			A.at<double>(i, 8) = 1;
		}

		// A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
		// the solution is linear subspace of dimensionality 2.
		// => use the last two singular std::vectors as a basis of the space
		// (according to SVD properties)
		cv::Mat cov = A.t() * A;
		eigen(cov, evals, evecs);

		for (i = 0; i < 9; ++i)
			f[i] = evecs.at<double>(8, i);
		//f[8] = 1.0;

		//std::cout << F << endl;

		Model model;
		model.descriptor = E;
		model.F = K2ti * E * K1i;
		models->emplace_back(model);
		return true;
	}

	bool solverSevenPoint(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<EssentialMatrix>* models) const
	{

		double a[7 * 9], v[9 * 9], c[4], r[3];
		double *f1, *f2;
		double t0, t1, t2;
		cv::Mat evals, evecs(9, 9, CV_64F, v);
		cv::Mat A(7, 9, CV_64F, a);
		cv::Mat coeffs(1, 4, CV_64F, c);
		cv::Mat roots(1, 3, CV_64F, r);
		int i, k, n;

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		for (i = 0; i < 7; i++)
		{
			double x0 = data.at<double>(sample[i], 0), y0 = data.at<double>(sample[i], 1);
			double x1 = data.at<double>(sample[i], 3), y1 = data.at<double>(sample[i], 4);

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
			double f[9];
			cv::Mat E(3, 3, CV_64F, f);

			// for each root form the fundamental matrix
			double lambda = r[k], mu = 1.;
			double s = f1[8] * r[k] + f2[8];

			// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
			if (fabs(s) > DBL_EPSILON)
			{
				mu = 1.0f / s;
				lambda *= mu;

				for (auto i = 0; i < 8; ++i)
					f[i] = f1[i] * lambda + f2[i] * mu;
				f[8] = 1.0;

				/* orient. constr. */
				if (!all_ori_valid(&E, data, sample, sample_number)) {
					continue;
				}

				Model model;
				model.descriptor = E;
				model.F = K2ti * E * K1i;
				model.mss.resize(sample_number);
				for (auto i = 0; i < sample_number; ++i)
					model.mss[i] = sample[i];

				models->emplace_back(model);
			}
		}

		return true;
	}

	/************** oriented constraints ******************/
	void epipole(cv::Mat &ec, const cv::Mat *F) const
	{
		ec = F->row(0).cross(F->row(2));

		for (auto i = 0; i < 3; i++)
			if ((ec.at<double>(i) > 1.9984e-15) || (ec.at<double>(i) < -1.9984e-15)) return;
		ec = F->row(1).cross(F->row(2));
	}

	double getorisig(const cv::Mat *F, const cv::Mat *ec, const cv::Mat &u) const
	{
		double s1, s2;

		s1 = F->at<double>(0) * u.at<double>(3) + F->at<double>(3) * u.at<double>(4) + F->at<double>(6) * u.at<double>(5);
		s2 = ec->at<double>(1) * u.at<double>(2) - ec->at<double>(2) * u.at<double>(1);
		return(s1 * s2);
	}

	int all_ori_valid(const cv::Mat *F, const cv::Mat &data, const int *sample, int N) const
	{
		cv::Mat ec;
		double sig, sig1;
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