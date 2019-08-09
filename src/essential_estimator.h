#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>

#include <unsupported/Eigen/Polynomials>
#include <Eigen/Eigen>

#include "estimator.h"
#include "model.h"

class EssentialMatrix : public Model
{
public:
	EssentialMatrix() :
		Model(Eigen::MatrixXd(3, 3)) 
	{}
	EssentialMatrix(const EssentialMatrix& other)
	{
		descriptor = other.descriptor;
	}
};

// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
class EssentialMatrixEstimator : public theia::Estimator < cv::Mat, Model >
{
protected:
	const Eigen::Matrix3d intrinsics_src,
		intrinsics_dst,
		intrinsics_src_inverse,
		intrinsics_dst_inverse_transpose;

public:
	EssentialMatrixEstimator(Eigen::Matrix3d intrinsics_src_,
		Eigen::Matrix3d intrinsics_dst_) :
		intrinsics_src(intrinsics_src_),
		intrinsics_dst(intrinsics_dst_),
		intrinsics_src_inverse(intrinsics_src_.inverse()),
		intrinsics_dst_inverse_transpose(intrinsics_dst_.inverse().transpose())
	{}
	~EssentialMatrixEstimator() {}

	size_t sampleSize() const {
		return 5;
	}

	size_t inlierLimit() const {
		return 7 * sampleSize();
	}

	bool estimateModel(const cv::Mat& data,
		const int *sample,
		std::vector<Model>* models) const
	{
		// Model calculation by the seven point algorithm
		constexpr size_t sample_size = 5;

		return solverSteweniusFivePoint(data, 
			sample, 
			sample_size, 
			models);
	}

	inline double squaredSampsonDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		const double* point_ptr = reinterpret_cast<double *>(point.data);
		const double x1 = point_ptr[0];
		const double y1 = point_ptr[1];
		const double x2 = point_ptr[2];
		const double y2 = point_ptr[3];

		const double f11 = descriptor(0, 0);
		const double f12 = descriptor(0, 1);
		const double f13 = descriptor(0, 2);
		const double f21 = descriptor(1, 0);
		const double f22 = descriptor(1, 1);
		const double f23 = descriptor(1, 2);
		const double f31 = descriptor(2, 0);
		const double f32 = descriptor(2, 1);
		const double f33 = descriptor(2, 2);

		double rxc = f11 * x2 + f21 * y2 + f31;
		double ryc = f12 * x2 + f22 * y2 + f32;
		double rwc = f13 * x2 + f23 * y2 + f33;
		double r = (x1 * rxc + y1 * ryc + rwc);
		double rx = f11 * x1 + f12 * y1 + f13;
		double ry = f21 * x1 + f22 * y1 + f23;

		return r * r / (rxc * rxc + ryc * ryc + rx * rx + ry * ry);
	}

	inline double sampsonDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		return sqrt(squaredSampsonDistance(point, descriptor));
	}

	inline double symmetricEpipolarDistance(const cv::Mat& point,
		const Eigen::Matrix3d& descriptor) const
	{
		const double* point_ptr = reinterpret_cast<double *>(point.data);
		const double x1 = point_ptr[0];
		const double y1 = point_ptr[1];
		const double x2 = point_ptr[2];
		const double y2 = point_ptr[3];

		const double f11 = descriptor(0, 0);
		const double f12 = descriptor(0, 1);
		const double f13 = descriptor(0, 2);
		const double f21 = descriptor(1, 0);
		const double f22 = descriptor(1, 1);
		const double f23 = descriptor(1, 2);
		const double f31 = descriptor(2, 0);
		const double f32 = descriptor(2, 1);
		const double f33 = descriptor(2, 2);

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
		const Model& model) const
	{
		return squaredSampsonDistance(point, model.descriptor);
	}

	inline double squaredResidual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		return squaredSampsonDistance(point, descriptor);
	}

	double residual(const cv::Mat& point,
		const Model& model) const
	{
		return residual(point, model.descriptor);
	}

	inline double residual(const cv::Mat& point,
		const Eigen::MatrixXd& descriptor) const
	{
		return sampsonDistance(point, descriptor);
	}

	// Validate the model by checking the number of inlier with symmetric epipolar distance
	// instead of Sampson distance. In general, Sampson distance is more accurate but less
	// robust to degenerate solutions than the symmetric epipolar distance. Therefore,
	// every so-far-the-best model is checked if it has enough inlier with symmetric
	// epipolar distance as well. 
	bool isValidModel(const Model& model,
		const cv::Mat& data,
		const std::vector<int> &inliers,
		const double threshold) const
	{
		size_t inlier_number = 0; // Number of inlier if using symmetric epipolar distance
		const Eigen::Matrix3d &descriptor = model.descriptor; // The decriptor of the current model
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
		std::vector<Model>* models) const
	{
		// model calculation 
		const size_t M = sample_number;

		if (M < 5)
			return false;

		cv::Mat normalized_points(M, data.cols, data.type()); // The normalized point coordinates
		Eigen::Matrix3d T1, T2; // The normalizing transformations in the 1st and 2nd images

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
		if (!solverSteweniusFivePoint(normalized_points,
			nullptr,
			M,
			models))
			return false;

		// Denormalizing the estimated fundamental matrices
		const Eigen::Matrix3d T2_transpose = T2.transpose();
		for (auto &model : *models)
			model.descriptor = T2_transpose * model.descriptor * T1;
		return true;
	}

	inline bool normalizePoints(
		const cv::Mat& data, // The data points
		const int *sample, // The points to which the model will be fit
		size_t sample_number,// The number of points
		cv::Mat &normalized_points, // The normalized point coordinates
		Eigen::Matrix3d &T1, // The normalizing transformation in the first image
		Eigen::Matrix3d &T2) const // The normalizing transformation in the second image
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
			0.0;

		// Calculating the mass points in both images
		for (size_t i = 0; i < sample_number; ++i)
		{
			// Get pointer of the current point
			const double *d_idx = points_ptr + cols * sample[i];

			// Add the coordinates to that of the mass points
			mass_point_src[0] += d_idx[0];
			mass_point_src[1] += d_idx[1];
			mass_point_dst[0] += d_idx[2];
			mass_point_dst[1] += d_idx[3];
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

			const double x1 = d_idx[0];
			const double y1 = d_idx[1];
			const double x2 = d_idx[2];
			const double y2 = d_idx[3];

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

			const double x1 = d_idx[0];
			const double y1 = d_idx[1];
			const double x2 = d_idx[2];
			const double y2 = d_idx[3];

			*normalized_points_ptr++ = (x1 - mass_point_src[0]) * ratio_src;
			*normalized_points_ptr++ = (y1 - mass_point_src[1]) * ratio_src;
			*normalized_points_ptr++ = (x2 - mass_point_dst[0]) * ratio_dst;
			*normalized_points_ptr++ = (y2 - mass_point_dst[1]) * ratio_dst;
		}

		// Creating the normalizing transformations
		T1 << ratio_src, 0, -ratio_src * mass_point_src[0],
			0, ratio_src, -ratio_src * mass_point_src[1],
			0, 0, 1;

		T2 << ratio_dst, 0, -ratio_dst * mass_point_dst[0],
			0, ratio_dst, -ratio_dst * mass_point_dst[1],
			0, 0, 1;
		return true;
	}

	inline bool solverSteweniusFivePoint(const cv::Mat& data,
		const int *sample,
		size_t sample_number,
		std::vector<Model>* models) const
	{
		if (sample == nullptr)
			sample_number = data.rows;

		Eigen::MatrixXd coefficients(sample_number, 9);
		const double *data_ptr = reinterpret_cast<double *>(data.data);
		const int cols = data.cols;

		// Step 1. Create the nx9 matrix containing epipolar constraints.
		//   Essential matrix is a linear combination of the 4 vectors spanning the null space of this
		//   matrix.
		double x0, y0, x1, y1;
		for (size_t i = 0; i < sample_number; i++)
		{
			int offset;
			if (sample == nullptr)
				offset = cols * i;
			else
				offset = cols * sample[i];

			x0 = data_ptr[offset];
			y0 = data_ptr[offset + 1];
			x1 = data_ptr[offset + 2];
			y1 = data_ptr[offset + 3];

			coefficients.row(i) << 
				x1 * x0, y1 * x0, x0, x1 * y0, y1 * y0, y0, x1, y1, 1.0;
		}

		// Extract the null space from a minimal sampling (using LU) or non-minimal sampling (using SVD).
		Eigen::Matrix<double, 9, 4> nullSpace;

		if (sample_number == 5) {
			const Eigen::FullPivLU<Eigen::MatrixXd> lu(coefficients);
			if (lu.dimensionOfKernel() != 4) {
				return false;
			}
			nullSpace = lu.kernel();
		}
		else {
			const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
				coefficients.transpose() * coefficients, Eigen::ComputeFullV);
			nullSpace = svd.matrixV().rightCols<4>();
		}

		const Eigen::Matrix<double, 1, 4> nullSpaceMatrix[3][3] = {
			{nullSpace.row(0), nullSpace.row(3), nullSpace.row(6)},
			{nullSpace.row(1), nullSpace.row(4), nullSpace.row(7)},
			{nullSpace.row(2), nullSpace.row(5), nullSpace.row(8)} };

		// Step 2. Expansion of the epipolar constraints on the determinant and trace.
		const Eigen::Matrix<double, 10, 20> constraintMatrix = buildConstraintMatrix(nullSpaceMatrix);

		// Step 3. Eliminate part of the matrix to isolate polynomials in z.
		Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> c_lu(constraintMatrix.block<10, 10>(0, 0));
		const Eigen::Matrix<double, 10, 10> eliminatedMatrix = c_lu.solve(constraintMatrix.block<10, 10>(0, 10));

		Eigen::Matrix<double, 10, 10> actionMatrix = Eigen::Matrix<double, 10, 10>::Zero();
		actionMatrix.block<3, 10>(0, 0) = eliminatedMatrix.block<3, 10>(0, 0);
		actionMatrix.row(3) = eliminatedMatrix.row(4);
		actionMatrix.row(4) = eliminatedMatrix.row(5);
		actionMatrix.row(5) = eliminatedMatrix.row(7);
		actionMatrix(6, 0) = -1.0;
		actionMatrix(7, 1) = -1.0;
		actionMatrix(8, 3) = -1.0;
		actionMatrix(9, 6) = -1.0;

		Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigensolver(actionMatrix);
		const Eigen::VectorXcd& eigenvalues = eigensolver.eigenvalues();

		// Now that we have x, y, and z we need to substitute them back into the null space to get a valid
		// essential matrix solution.
		for (size_t i = 0; i < 10; i++) {
			// Only consider real solutions.
			if (eigenvalues(i).imag() != 0) {
				continue;
			}
			Eigen::Matrix3d E_dst_src;
			Eigen::Map<Eigen::Matrix<double, 9, 1>>(E_dst_src.data()) =
				nullSpace * eigensolver.eigenvectors().col(i).tail<4>().real();

			/* Orientation constraint */
			if (!all_ori_valid(E_dst_src,
				data,
				sample,
				sample_number)) {
				continue;
			}

			EssentialMatrix model;
			model.descriptor = E_dst_src;
			models->push_back(model);
		}

		return models->size() > 0;
	}

	inline Eigen::Matrix<double, 1, 10> multiplyDegOnePoly(
		const Eigen::RowVector4d& a,
		const Eigen::RowVector4d& b) const;

	inline Eigen::Matrix<double, 1, 20> multiplyDegTwoDegOnePoly(
		const Eigen::Matrix<double, 1, 10>& a,
		const Eigen::RowVector4d& b) const;

	inline Eigen::Matrix<double, 10, 20> buildConstraintMatrix(
		const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

	inline Eigen::Matrix<double, 9, 20> getTraceConstraint(
		const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

	inline Eigen::Matrix<double, 1, 10>
		computeEETranspose(const Eigen::Matrix<double, 1, 4> nullSpace[3][3], int i, int j) const;

	inline Eigen::Matrix<double, 1, 20> getDeterminantConstraint(
		const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const;

	/************** oriented epipolar constraints ******************/
	inline void epipole(Eigen::Vector3d &ec,
		const Eigen::Matrix3d &F) const
	{
		ec = F.row(0).cross(F.row(2));

		for (auto i = 0; i < 3; i++)
			if ((ec(i) > 1.9984e-15) ||
				(ec(i) < -1.9984e-15))
				return;
		ec = F.row(1).cross(F.row(2));
	}

	inline double getorisig(const Eigen::Matrix3d &F,
		const Eigen::Vector3d &ec,
		const cv::Mat &u) const
	{
		double s1, s2;
		s1 = F(0, 0) * u.at<double>(2) + F(1, 0) * u.at<double>(3) + F(2, 0);
		s2 = ec(1) - ec(2) * u.at<double>(1);
		return s1 * s2;
	}

	inline int all_ori_valid(const Eigen::Matrix3d &F,
		const cv::Mat &data,
		const int *sample,
		int N) const
	{
		Eigen::Vector3d ec;
		double sig, sig1;
		int i;
		epipole(ec, F);

		if (sample == nullptr)
		{
			sig1 = getorisig(F, ec, data.row(0));
			for (i = 1; i < N; i++)
			{
				sig = getorisig(F, ec, data.row(i));
				if (sig1 * sig < 0) return 0;
			}
		}
		else
		{
			sig1 = getorisig(F, ec, data.row(sample[0]));
			for (i = 1; i < N; i++)
			{
				sig = getorisig(F, ec, data.row(sample[i]));
				if (sig1 * sig < 0) return 0;
			}
		}
		return 1;
	}
};

// Multiply two degree one polynomials of variables x, y, z.
// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
// Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
inline Eigen::Matrix<double, 1, 10> EssentialMatrixEstimator::multiplyDegOnePoly(
	const Eigen::RowVector4d& a,
	const Eigen::RowVector4d& b) const {
	Eigen::Matrix<double, 1, 10> output;
	// x^2
	output(0) = a(0) * b(0);
	// xy
	output(1) = a(0) * b(1) + a(1) * b(0);
	// y^2
	output(2) = a(1) * b(1);
	// xz
	output(3) = a(0) * b(2) + a(2) * b(0);
	// yz
	output(4) = a(1) * b(2) + a(2) * b(1);
	// z^2
	output(5) = a(2) * b(2);
	// x
	output(6) = a(0) * b(3) + a(3) * b(0);
	// y
	output(7) = a(1) * b(3) + a(3) * b(1);
	// z
	output(8) = a(2) * b(3) + a(3) * b(2);
	// 1
	output(9) = a(3) * b(3);
	return output;
}

// Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
// x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
inline Eigen::Matrix<double, 1, 20> EssentialMatrixEstimator::multiplyDegTwoDegOnePoly(
	const Eigen::Matrix<double, 1, 10>& a,
	const Eigen::RowVector4d& b) const {
	Eigen::Matrix<double, 1, 20> output;
	// x^3
	output(0) = a(0) * b(0);
	// x^2y
	output(1) = a(0) * b(1) + a(1) * b(0);
	// xy^2
	output(2) = a(1) * b(1) + a(2) * b(0);
	// y^3
	output(3) = a(2) * b(1);
	// x^2z
	output(4) = a(0) * b(2) + a(3) * b(0);
	// xyz
	output(5) = a(1) * b(2) + a(3) * b(1) + a(4) * b(0);
	// y^2z
	output(6) = a(2) * b(2) + a(4) * b(1);
	// xz^2
	output(7) = a(3) * b(2) + a(5) * b(0);
	// yz^2
	output(8) = a(4) * b(2) + a(5) * b(1);
	// z^3
	output(9) = a(5) * b(2);
	// x^2
	output(10) = a(0) * b(3) + a(6) * b(0);
	// xy
	output(11) = a(1) * b(3) + a(6) * b(1) + a(7) * b(0);
	// y^2
	output(12) = a(2) * b(3) + a(7) * b(1);
	// xz
	output(13) = a(3) * b(3) + a(6) * b(2) + a(8) * b(0);
	// yz
	output(14) = a(4) * b(3) + a(7) * b(2) + a(8) * b(1);
	// z^2
	output(15) = a(5) * b(3) + a(8) * b(2);
	// x
	output(16) = a(6) * b(3) + a(9) * b(0);
	// y
	output(17) = a(7) * b(3) + a(9) * b(1);
	// z
	output(18) = a(8) * b(3) + a(9) * b(2);
	// 1
	output(19) = a(9) * b(3);
	return output;
}

inline Eigen::Matrix<double, 1, 20> EssentialMatrixEstimator::getDeterminantConstraint(
	const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
	// Singularity constraint.
	return multiplyDegTwoDegOnePoly(
		multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][2]) -
		multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][1]),
		nullSpace[2][0]) +
		multiplyDegTwoDegOnePoly(
			multiplyDegOnePoly(nullSpace[0][2], nullSpace[1][0]) -
			multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][2]),
			nullSpace[2][1]) +
		multiplyDegTwoDegOnePoly(
			multiplyDegOnePoly(nullSpace[0][0], nullSpace[1][1]) -
			multiplyDegOnePoly(nullSpace[0][1], nullSpace[1][0]),
			nullSpace[2][2]);
}

// Shorthand for multiplying the Essential matrix with its transpose.
inline Eigen::Matrix<double, 1, 10> EssentialMatrixEstimator::computeEETranspose(
	const Eigen::Matrix<double, 1, 4> nullSpace[3][3],
	int i,
	int j) const {
	return multiplyDegOnePoly(nullSpace[i][0], nullSpace[j][0]) +
		multiplyDegOnePoly(nullSpace[i][1], nullSpace[j][1]) +
		multiplyDegOnePoly(nullSpace[i][2], nullSpace[j][2]);
}

// Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
inline Eigen::Matrix<double, 9, 20> EssentialMatrixEstimator::getTraceConstraint(
	const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
	Eigen::Matrix<double, 9, 20> traceConstraint;

	// Compute EEt.
	Eigen::Matrix<double, 1, 10> eet[3][3];
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			eet[i][j] = 2 * computeEETranspose(nullSpace, i, j);
		}
	}

	// Compute the trace.
	const Eigen::Matrix<double, 1, 10> trace = eet[0][0] + eet[1][1] + eet[2][2];

	// Multiply EEt with E.
	for (auto i = 0; i < 3; i++) {
		for (auto j = 0; j < 3; j++) {
			traceConstraint.row(3 * i + j) = multiplyDegTwoDegOnePoly(eet[i][0], nullSpace[0][j]) +
				multiplyDegTwoDegOnePoly(eet[i][1], nullSpace[1][j]) +
				multiplyDegTwoDegOnePoly(eet[i][2], nullSpace[2][j]) -
				0.5 * multiplyDegTwoDegOnePoly(trace, nullSpace[i][j]);
		}
	}

	return traceConstraint;
}

inline Eigen::Matrix<double, 10, 20> EssentialMatrixEstimator::buildConstraintMatrix(
	const Eigen::Matrix<double, 1, 4> nullSpace[3][3]) const {
	Eigen::Matrix<double, 10, 20> constraintMatrix;
	constraintMatrix.block<9, 20>(0, 0) = getTraceConstraint(nullSpace);
	constraintMatrix.row(9) = getDeterminantConstraint(nullSpace);
	return constraintMatrix;
}