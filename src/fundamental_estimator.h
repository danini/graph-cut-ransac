#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include "estimator.h"

struct FundamentalMatrix
{
	cv::Mat descriptor;

	FundamentalMatrix() {}
	FundamentalMatrix(const FundamentalMatrix& other)
	{
		descriptor = other.descriptor.clone();
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class FundamentalMatrixEstimator : public theia::Estimator < cv::Mat, FundamentalMatrix >
{
protected:

public:
	FundamentalMatrixEstimator() {}
	~FundamentalMatrixEstimator() {}

	size_t sampleSize() const {
		return 7;
	}

	size_t inlierLimit() const {
		return 49;
	}

	bool estimateModel(const cv::Mat& data,
		const int *sample,
		std::vector<FundamentalMatrix>* models) const
	{
		// Model calculation by the seven point algorithm
		constexpr size_t M = 7;

		solverSevenPoint(data, sample, M, models);
		if (models->size() == 0)
			return false;
		return true;
	}

	double squaredSampsonDistance(const cv::Mat& point,
		const cv::Mat& descriptor) const
	{
		const double* p = reinterpret_cast<double *>(descriptor.data);
		const double* s = reinterpret_cast<double *>(point.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		double rxc = *(p)* x2 + *(p + 3) * y2 + *(p + 6);
		double ryc = *(p + 1) * x2 + *(p + 4) * y2 + *(p + 7);
		double rwc = *(p + 2) * x2 + *(p + 5) * y2 + *(p + 8);
		double r = (x1 * rxc + y1 * ryc + rwc);
		double rx = *(p)* x1 + *(p + 1) * y1 + *(p + 2);
		double ry = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);

		return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry);
	}

	double sampsonDistance(const cv::Mat& point,
		const cv::Mat& descriptor) const
	{
		return sqrt(squaredSampsonDistance(point, descriptor));
	}

	double symmetricEpipolarDistance(const cv::Mat& point,
		const cv::Mat& descriptor) const
	{
		const double* s = reinterpret_cast<double *>(point.data);
		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double* p = reinterpret_cast<double *>(descriptor.data);

		const double f11 = *(p);
		const double f12 = *(p + 1);
		const double f13 = *(p + 2);
		const double f21 = *(p + 3);
		const double f22 = *(p + 4);
		const double f23 = *(p + 5);
		const double f31 = *(p + 6);
		const double f32 = *(p + 7);
		const double f33 = *(p + 8);

		const double l1 = f11 * x2 + f21 * y2 + f31;
		const double l2 = f12 * x2 + f22 * y2 + f32;
		const double l3 = f13 * x2 + f23 * y2 + f33;

		const double t1 = f11 * x1 + f12 * y1 + f13;
		const double t2 = f21 * x1 + f22 * y1 + f23;
		const double t3 = f31 * x1 + f32 * y1 + f33;

		const double a1 = l1 * x1 + l2 * y1 + l3;
		const double a2 = sqrt(l1 * l1 + l2 * l2);

		const double b1 = t1 * x2 + t2 * y1 + t3;
		const double b2 = sqrt(t1 * t1 + t2 * t2);

		const double d1 = a1 / a2;
		const double d2 = b1 / b2;

		return abs(0.5 * (d1 + d2));
	}

	double squaredResidual(const cv::Mat& point,
		const FundamentalMatrix& model) const
	{
		return squaredSampsonDistance(point, model.descriptor);
	}

	double squaredResidual(const cv::Mat& point,
		const cv::Mat& descriptor) const
	{
		return squaredSampsonDistance(point, descriptor);
	}

	double residual(const cv::Mat& point, 
		const FundamentalMatrix& model) const
	{
		return residual(point, model.descriptor);
	}

	double residual(const cv::Mat& point, 
		const cv::Mat& descriptor) const
	{
		return sampsonDistance(point, descriptor);
	}

	// Validate the model by checking the number of inlier with symmetric epipolar distance
	// instead of Sampson distance. In general, Sampson distance is more accurate but less
	// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
	// every so-far-the-best model is checked if it has enough inlier with symmetric
	// epipolar distance as well. 
	bool isValidModel(const FundamentalMatrix& model,
		const cv::Mat& data,
		const std::vector<int> &inliers,
		const double threshold) const
	{
		size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
		const cv::Mat &descriptor = model.descriptor; // The decriptor of the current model
		static const size_t M = sampleSize(); // Size of a minimal sample

		// Iterate through the inliers determined by Sampson distance
		for (const auto &idx : inliers)
			// Calculate the residual using symmetric epipolar distance and check if
			// it is smaller than the threshold.
			if (symmetricEpipolarDistance(data.row(idx), descriptor) < threshold)
				// Increase the inlier number and terminate if enough inliers have been found.
				if (++inlier_number > M)
					return true;
		// If the algorithm has not terminated earlier, there are not enough inliers.
		return false;
	}
		
	bool estimateModelNonminimal(
		const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<FundamentalMatrix>* models) const
	{
		// model calculation 
		const size_t M = sample_number;

		if (M < 8)
			return false;

		cv::Mat normalized_points(M, data.cols, data.type()); // The normalized point coordinates
		cv::Mat T1, T2; // The normalizing transformations in the 1st and 2nd images

		// Normalize the point coordinates to achieve numerical stability when
		// applying the least-squares model fitting.
		if (!normalizePoints(data, // The data points
			sample, // The points to which the model will be fit
			M, // The number of points
			normalized_points, // The normalized point coordinates
			T1, // The normalizing transformation in the first image
			T2)) // The normalizing transformation in the second image
			return false;
		
		// The eight point fundamental matrix fitting algorithm
		solverEightPoint(normalized_points,
			nullptr,
			M, 
			models);

		// Denormalizing the estimated fundamental matrices
		const cv::Mat T2_transpose = T2.t();
		for (auto &model : *models)
			model.descriptor = T2_transpose * model.descriptor * T1;
		return true;
	}

	bool normalizePoints(
		const cv::Mat& data, // The data points
		const int *sample, // The points to which the model will be fit
		size_t sample_number,// The number of points
		cv::Mat &normalized_points, // The normalized point coordinates
		cv::Mat &T1, // The normalizing transformation in the first image
		cv::Mat &T2) const // The normalizing transformation in the second image
	{
		const size_t cols = data.cols;
		double *normalized_points_ptr = reinterpret_cast<double *>(normalized_points.data);
		const double *points_ptr = reinterpret_cast<double *>(data.data);
				
		double mass_point_src[2], // Mass point in the first image
			mass_point_dst[2]; // Mass point in the second image
		// Initializing the mass point coordinates
		mass_point_src[0] = 
			mass_point_src[1] = 
			mass_point_dst[0] = 
			mass_point_dst[1] = 
			0.0f;

		// Calculating the mass points in both images
		for (size_t i = 0; i < sample_number; ++i)
		{
			// Get pointer of the current point
			const double *d_idx = points_ptr + cols * sample[i];

			// Add the coordinates to that of the mass points
			mass_point_src[0] += *(d_idx);
			mass_point_src[1] += *(d_idx + 1);
			mass_point_dst[0] += *(d_idx + 2);
			mass_point_dst[1] += *(d_idx + 3);
		}

		// Get the average
		mass_point_src[0] /= sample_number;
		mass_point_src[1] /= sample_number;
		mass_point_dst[0] /= sample_number;
		mass_point_dst[1] /= sample_number;

		// Get the mean distance from the mass points
		double average_distance_src = 0.0, 
			average_distance_dst = 0.0;
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			const double dx1 = mass_point_src[0] - x1;
			const double dy1 = mass_point_src[1] - y1;
			const double dx2 = mass_point_dst[0] - x2;
			const double dy2 = mass_point_dst[1] - y2;

			average_distance_src += sqrt(dx1 * dx1 + dy1 * dy1); 
			average_distance_dst += sqrt(dx2 * dx2 + dy2 * dy2);
		}

		average_distance_src /= sample_number;
		average_distance_dst /= sample_number;

		// Calculate the sqrt(2) / MeanDistance ratios
		static const double sqrt_2 = sqrt(2);
		const double ratio_src = sqrt_2 / average_distance_src;
		const double ratio_dst = sqrt_2 / average_distance_dst;

		// Compute the normalized coordinates
		for (size_t i = 0; i < sample_number; ++i)
		{
			const double *d_idx = points_ptr + cols * sample[i];

			const double x1 = *(d_idx);
			const double y1 = *(d_idx + 1);
			const double x2 = *(d_idx + 2);
			const double y2 = *(d_idx + 3);

			*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
			*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
			*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
			*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;
		}

		// Creating the normalizing transformations
		T1 = (cv::Mat_<double>(3, 3) << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1);

		T2 = (cv::Mat_<double>(3, 3) << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1);
		return true;
	}

	bool solverEightPoint(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<FundamentalMatrix>* models) const
	{
		if (sample == nullptr)
			sample_number = data.rows;

		double f[9];
		cv::Mat evals(1, 9, CV_64F), evecs(9, 9, CV_64F);
		cv::Mat A(sample_number, 9, CV_64F);
		cv::Mat F(3, 3, CV_64F, f);

		// form a linear system: i-th row of A(=a) represents
		// the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
		double x0, y0, x1, y1;
		for (size_t i = 0; i < sample_number; i++)
		{
			if (sample == nullptr)
			{
				x0 = data.at<double>(i, 0);
				y0 = data.at<double>(i, 1);
				x1 = data.at<double>(i, 2);
				y1 = data.at<double>(i, 3);
			}
			else
			{
				x0 = data.at<double>(sample[i], 0);
				y0 = data.at<double>(sample[i], 1);
				x1 = data.at<double>(sample[i], 2);
				y1 = data.at<double>(sample[i], 3);
			}

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

		for (size_t i = 0; i < 9; ++i)
			f[i] = evecs.at<double>(8, i);

		Model model;
		model.descriptor = F;
		models->push_back(model);
		return true;
	}

	bool solverSevenPoint(const cv::Mat& data,
		const int *sample,
		int sample_number,
		std::vector<FundamentalMatrix>* models) const
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
			const int sample_idx = sample[i];

			double x0 = data.at<double>(sample_idx, 0),
				y0 = data.at<double>(sample_idx, 1),
				x1 = data.at<double>(sample_idx, 2), 
				y1 = data.at<double>(sample_idx, 3);

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
			cv::Mat F(3, 3, CV_64F, f);

			// for each root form the fundamental matrix
			double lambda = r[k], mu = 1.;
			double s = f1[8] * r[k] + f2[8];

			// normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
			if (fabs(s) > std::numeric_limits<double>::epsilon())
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
				models->push_back(model);
			}
		}

		return true;
	}

	/************** oriented epipolar constraints ******************/
	void epipole(cv::Mat &ec, 
		const cv::Mat *F) const
	{
		ec = F->row(0).cross(F->row(2));

		for (auto i = 0; i < 3; i++)
			if ((ec.at<double>(i) > 1.9984e-15) || 
				(ec.at<double>(i) < -1.9984e-15)) 
				return;
		ec = F->row(1).cross(F->row(2));
	}

	double getorisig(const cv::Mat *F, 
		const cv::Mat *ec, 
		const cv::Mat &u) const
	{
		double s1, s2;		
		s1 = F->at<double>(0,0) * u.at<double>(2) + F->at<double>(1,0) * u.at<double>(3) + F->at<double>(2,0);
		s2 = ec->at<double>(1) - ec->at<double>(2) * u.at<double>(1);
		return s1 * s2;
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