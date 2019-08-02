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
		return 7 * sampleSize();
	}

	bool estimateModel(
		const cv::Mat& data,
		const int *sample,
		std::vector<Homography>* models) const
	{
		static const size_t M = sampleSize();
		solverFourPoint(data,
			sample,
			M,
			models);
		return true;
	}

	bool estimateModelNonminimal(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<Homography>* models) const
	{
		if (sample_number < sampleSize())
			return false;

		cv::Mat normalized_points(sample_number, data.cols, data.type()); // The normalized point coordinates
		cv::Mat T1, T2; // The normalizing transformations in the 1st and 2nd images
		T1 = T2 = cv::Mat::zeros(3, 3, data.type());

		// Normalize the point coordinates to achieve numerical stability when
		// applying the least-squares model fitting.
		if (!normalizePoints(data, // The data points
			sample, // The points to which the model will be fit
			sample_number, // The number of points
			normalized_points, // The normalized point coordinates
			T1, // The normalizing transformation in the first image
			T2)) // The normalizing transformation in the second image
			return false;

		// The four point fundamental matrix fitting algorithm
		solverFourPoint(normalized_points,
			nullptr,
			sample_number,
			models);

		// Denormalizing the estimated fundamental matrices
		for (auto &model : *models)
			model.descriptor = T2.inv() * model.descriptor * T1;
		return true;
	}

	double residual(const cv::Mat& point, 
		const Homography& model) const
	{
		return residual(point, model.descriptor);
	}

	double residual(const cv::Mat& point, 
		const cv::Mat& descriptor) const
	{
		const double* s = reinterpret_cast<double *>(point.data);
		const double* p = reinterpret_cast<double *>(descriptor.data);

		const double x1 = *s;
		const double y1 = *(s + 1);
		const double x2 = *(s + 2);
		const double y2 = *(s + 3);

		const double t1 = *p * x1 + *(p + 1) * y1 + *(p + 2);
		const double t2 = *(p + 3) * x1 + *(p + 4) * y1 + *(p + 5);
		const double t3 = *(p + 6) * x1 + *(p + 7) * y1 + *(p + 8);

		const double d1 = x2 - (t1 / t3);
		const double d2 = y2 - (t2 / t3);

		return sqrt(d1*d1 + d2*d2);
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
			if (sample[i] >= data.rows)
				return false;

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

		T1 = (cv::Mat_<double>(3, 3) << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1);

		T2 = (cv::Mat_<double>(3, 3) << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1);
		return true;
	}

	bool solverFourPoint(
		const cv::Mat& data_,
		const int *sample_,
		const size_t sample_number_,
		std::vector<Homography>* models_) const
	{
		constexpr size_t equation_number = 2;
		cv::Mat A(equation_number * sample_number_, 8, CV_64F);
		cv::Mat inhomogeneous(equation_number * sample_number_, 1, CV_64F);

		constexpr size_t columns = 4;
		double *A_ptr = reinterpret_cast<double *>(A.data);
		double *b_ptr = reinterpret_cast<double *>(inhomogeneous.data);
		const double *data_ptr = reinterpret_cast<double *>(data_.data);
		
		for (auto i = 0; i < sample_number_; ++i)
		{
			const double *point_ptr = sample_ == nullptr ?
				data_ptr + i * columns :
				data_ptr + sample_[i] * columns;

			double x1 = point_ptr[0],
				y1 = point_ptr[1],
				x2 = point_ptr[2],
				y2 = point_ptr[3];

			(*A_ptr++) = -x1;
			(*A_ptr++) = -y1;
			(*A_ptr++) = -1;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = x2 * x1;
			(*A_ptr++) = x2 * y1;
			(*b_ptr++) = -x2;

			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = 0;
			(*A_ptr++) = -x1;
			(*A_ptr++) = -y1;
			(*A_ptr++) = -1;
			(*A_ptr++) = y2 * x1;
			(*A_ptr++) = y2 * y1;
			(*b_ptr++) = -y2;
		}

		cv::Mat h;
		cv::solve(A, inhomogeneous, h, cv::DECOMP_SVD);
		h.push_back(1.0);
		cv::Mat H(3, 3, CV_64F, h.data);

		Homography model;
		model.descriptor = H;
		models_->emplace_back(model);
		return true;
	}

};