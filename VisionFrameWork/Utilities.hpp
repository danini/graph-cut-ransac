#pragma once

#ifndef __UTILITIES__
#define __UTILITIES__
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/calib3d/calib3d_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif

#pragma region File Utils

#define BASE_T double
typedef std::vector<Point_<BASE_T>> PointVec;
typedef std::vector<PointVec>       PointVecMap;
typedef std::vector<cv::Mat_<BASE_T>>   VectorOfMatrices;
typedef std::vector<std::vector<cv::Mat_<BASE_T>>>   VectorOfVectorOfMatrices;

inline
  void FscanfOrDie(FILE *fptr, const char *format, double *value) {
    /*int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      std::cerr << "Invalid UW data file." << std::endl;
    }*/
  }

inline
  void FscanfOrDie(FILE *fptr, const char *format, int *value) {
    /*int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      std::cerr << "Invalid UW data file." << std::endl;
    }*/
  }
/*
inline
void ReadVector(FILE *fptr, std::vector<double>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		FscanfOrDie(fptr, "%lf", &vec[i]);
}*/

inline 
void ReadVector(FILE *fptr, std::vector<int>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		FscanfOrDie(fptr, "%d", &vec[i]);
}
/*
inline 
void ReadVector(FILE *fptr, std::vector<float>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		FscanfOrDie(fptr, "%g", &vec[i]);
}*/

inline 
void ReadVector(FILE *fptr, std::vector<double>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		FscanfOrDie(fptr, "%lf", &vec[i]);
}
/*
inline 
void ReadVector(FILE *fptr, std::vector<Point_<float>>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i) {
		FscanfOrDie(fptr, "%g", &(vec[i].x));
		FscanfOrDie(fptr, "%g", &(vec[i].y));
	}
}*/

inline 
void ReadVector(FILE *fptr, std::vector<Point_<double>>& vec) {
	int vec_size;
	FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i) {
		FscanfOrDie(fptr, "%lf", &(vec[i].x));
		FscanfOrDie(fptr, "%lf", &(vec[i].y));
	}
}

inline 
void WriteVector(FILE *fptr, const std::vector<BASE_T>& vec) {
	if(vec.size() == 0) {
		fprintf(fptr, "%d\n", (int)vec.size());
	} else {
		fprintf(fptr, "%d ", (int)vec.size());

		fprintf(fptr, "%g", vec[0]);
		for (unsigned i = 1; i<vec.size(); ++i) {
			fprintf(fptr, " %g", vec[i]);
		}
		fprintf(fptr, "\n");
	}
}

inline 
void WriteVector(FILE *fptr, const std::vector<int>& vec) {
	if(vec.size() == 0) {
		fprintf(fptr, "%d\n", (int)vec.size());
	} else {
		fprintf(fptr, "%d ", (int)vec.size());

		fprintf(fptr, "%d", vec[0]);
		for (unsigned i = 1; i<vec.size(); ++i) {
			fprintf(fptr, " %d", vec[i]);
		}
		fprintf(fptr, "\n");
	}
}

inline 
void WriteVector(FILE *fptr, const std::vector<Point_<BASE_T>>& vec) {
	if(vec.size() == 0) {
		fprintf(fptr, "%d\n", (int)vec.size());
	} else {
		fprintf(fptr, "%d ", (int)vec.size());

		fprintf(fptr, "%.16g %.16g", vec[0].x, vec[0].y);
		for (unsigned i = 1; i<vec.size(); ++i) {
			fprintf(fptr, " %.16g %.16g", vec[i].x, vec[i].y);
		}
		fprintf(fptr, "\n");
	}
}

/// MATRICES
/*
inline 
void ReadMatrix(FILE *fptr, cv::Mat_<float>& mat) {
	int dimRows, dimCols;
	FscanfOrDie(fptr, "%d", &dimRows);
	FscanfOrDie(fptr, "%d", &dimCols);

	mat = cv::Mat(dimRows,dimCols, cv::DataType<float>::type, cvScalar(0.));
	for (int i = 0; i<dimRows; ++i)
		for (int j = 0; j<dimCols; ++j) {
			float val;
			FscanfOrDie(fptr, "%g", &val);
			mat.at<float>(i,j) = val;
		}
}*/

inline 
void ReadMatrix(FILE *fptr, cv::Mat_<double>& mat) {
	int dimRows, dimCols;
	FscanfOrDie(fptr, "%d", &dimRows);
	FscanfOrDie(fptr, "%d", &dimCols);

	mat = cv::Mat(dimRows,dimCols, cv::DataType<double>::type, cvScalar(0.));
	for (int i = 0; i<dimRows; ++i)
		for (int j = 0; j<dimCols; ++j) {
			double val;
			FscanfOrDie(fptr, "%lf", &val);
			mat.at<double>(i,j) = val;
		}
}

inline 
void WriteMatrix(FILE *fptr, const cv::Mat_<double>& mat) {
	fprintf(fptr, "%d %d", (int)mat.rows, (int)mat.cols);

	for (int i = 0; i<mat.rows; ++i)
		for (int j = 0; j<mat.cols; ++j) {
			double val = mat.at<double>(i,j);
			fprintf(fptr, " %g", val);
		}
		fprintf(fptr, "\n");
}
/*
inline 
void WriteMatrix(FILE *fptr, const cv::Mat_<float>& mat) {
	fprintf(fptr, "%d %d", (int)mat.rows, (int)mat.cols);

	for (int i = 0; i<mat.rows; ++i)
		for (int j = 0; j<mat.cols; ++j) {
			float val = mat.at<float>(i,j);
			fprintf(fptr, " %g", val);
		}
		fprintf(fptr, "\n");
}*/

/// VECTORS OF MATRICES
inline 
void ReadVector(FILE *fptr, VectorOfMatrices& vec) {
	int vec_size; FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		ReadMatrix(fptr, vec[i]);
}

inline 
void ReadVector(FILE *fptr, VectorOfVectorOfMatrices& vec) {
	int vec_size; FscanfOrDie(fptr, "%d", &vec_size);
	vec.resize(vec_size);
	for (int i = 0; i<vec_size; ++i)
		ReadVector(fptr, vec[i]);
}

inline 
void WriteVector(FILE *fptr, const VectorOfMatrices& vec) {
	if(vec.size() > 0)
		fprintf(fptr, "%d ", (int)vec.size());
	else
		fprintf(fptr, "%d\n", (int)vec.size());
	for (unsigned i = 0; i<vec.size(); ++i) WriteMatrix(fptr, vec[i]);
}

inline 
void WriteVector(FILE *fptr, const VectorOfVectorOfMatrices& vec) {
	if(vec.size() > 0)
		fprintf(fptr, "%d ", (int)vec.size());
	else
		fprintf(fptr, "%d\n", (int)vec.size());
	for (unsigned i = 0; i<vec.size(); ++i) WriteVector(fptr, vec[i]);
}

#pragma endregion

#pragma region Image / Correlation Functions

// UNUSED
/*
inline float znccf( const cv::Mat &img1, const cv::Mat &img2 )
{
	int M = img1.cols;
	int N = img1.rows;
	
	const float* img_1 = img1.ptr<float>();
	const float* img_2 = img2.ptr<float>();
	unsigned MN = M * N;
		
	float img1_avg = 0;
	float img2_avg = 0;
		
	for( unsigned i = 0; i<MN; ++i )
	{
		img1_avg += img_1[i];
		img2_avg += img_2[i];
	}
	
	img1_avg /= float(MN);
	img2_avg /= float(MN);
	

	float sum_img1_img2 = 0;
	float sum_img1_2 = 0;
	float sum_img2_2 = 0;

	for( unsigned i = 0; i<MN; ++i )
	{
		auto a = (img_1[i]-img1_avg);
		auto b = (img_2[i]-img2_avg);
		sum_img1_img2 += a*b;  
		sum_img1_2    += a*a;
		sum_img2_2    += b*b;
	}
	
	return sum_img1_img2/sqrt(sum_img1_2*sum_img2_2);
}

template <unsigned CHANNELS, typename T>
inline void _mean( const cv::Mat &img, T avg[CHANNELS] )
{
	int M = img.cols;
	int N = img.rows;
	
	uchar* img_1 = img.data;
	unsigned MN = M * N;
	
	for(unsigned i = 0; i<CHANNELS; ++i)
		avg[i] = 0;

	for(unsigned i = 0; i<MN; ++i)
		for(unsigned j = 0; j<CHANNELS; ++j)
			avg[j] += img_1[CHANNELS*i+j];
	
	for(unsigned i = 0; i<CHANNELS; ++i)
		avg[i] /= (T)MN;
}

template <unsigned CHANNELS, typename T>
inline void _sum2( const cv::Mat &img, const T avg[CHANNELS], T sum_2[CHANNELS] )
{
	int M = img.cols;
	int N = img.rows;
	
	uchar* img_1 = img.data;
	unsigned MN = M * N;

	for(unsigned i = 0; i<MN; ++i)
		for(unsigned j = 0; j < CHANNELS; ++j) {
			auto a = (img_1[3*i+j]-avg[j]);
			sum_2[j]    += a*a;
		}
}

template <unsigned CHANNELS, typename T>
inline T zncc( const cv::Mat &img1, cv::Mat &img2, const T img2_avg[CHANNELS], const T sum_img2_2[CHANNELS] )
{
	int M = img1.cols;
	int N = img1.rows;
	
	uchar* img_1 = img1.data;
	uchar* img_2 = img2.data;
	unsigned MN = M * N;
		
	T img1_avg[CHANNELS];
	_mean<CHANNELS>(img1, img1_avg);

	T sum_img1_img2[CHANNELS];
	T sum_img1_2[CHANNELS];

	for(unsigned i = 0; i<CHANNELS; ++i) {
		sum_img1_2[i] = 0;
		sum_img1_img2[i] = 0;
	}

	for(unsigned i = 0; i<MN; ++i)
		for(unsigned j = 0; j < CHANNELS; ++j) {
			auto a = (img_1[3*i+j]-img1_avg[j]);
			auto b = (img_2[3*i+j]-img2_avg[j]);
			sum_img1_img2[j] += a*b;
			sum_img1_2[j]    += a*a;
		}

	T result = 0;
	for(unsigned j = 0; j < CHANNELS; ++j)
		result += sum_img1_img2[j]/sqrt(sum_img1_2[j]*sum_img2_2[j]);

	return result;
}*/

template <typename T, unsigned CHANNELS>
inline T zncc( const cv::Mat &img1, const cv::Mat &img2 )
{
	int M = img1.cols;
	int N = img1.rows;
	
	const uchar* img_1 = img1.data;
	const uchar* img_2 = img2.data;
	unsigned MN = M * N;
		
	T img1_avg[CHANNELS];
	T img2_avg[CHANNELS];
	for( unsigned j = 0; j < CHANNELS; ++j ) {
		img1_avg[j] = 0;
		img2_avg[j] = 0;
	}

	for( unsigned i = 0; i<MN; ++i )
		for( unsigned j = 0; j < CHANNELS; ++j ) {
			img1_avg[j] += img_1[CHANNELS*i+j];
			img2_avg[j] += img_2[CHANNELS*i+j];
		}
	
	for( unsigned i = 0; i<CHANNELS; ++i ) {
		img1_avg[i] /= (T)MN;
		img2_avg[i] /= (T)MN;
	}

	T sum_img1_img2 = 0;
	T sum_img1_2 = 0;
	T sum_img2_2 = 0;
	
	for( unsigned i = 0; i<MN; ++i )
	{
		for(unsigned j = 0; j < CHANNELS; ++j) {
			auto a = (img_1[3*i+j]-img1_avg[j]);
			auto b = (img_2[3*i+j]-img2_avg[j]);
			sum_img1_img2 += a*b;
			sum_img1_2    += a*a;
			sum_img2_2    += b*b;
		}
	}
	
	return sum_img1_img2/sqrt(sum_img1_2*sum_img2_2);
}
/*
static double* weights;
*/
inline
void init_sqrtGaussianWeights(unsigned M, unsigned N, double egypersigma, double* w)
{
	const unsigned MN = M * N;
	const double denom = -2*egypersigma*egypersigma; // TODO hopp itt 0.5-nek kellene lennie 2 helyett!!

	for( unsigned i = 0; i<MN; ++i )
	{
		double c1 = (i%M) - M/2.0;
		double c2 = int(i/M) - N/2.0;
		w[i] = sqrt(exp(denom * (c1*c1 + c2*c2)));
	}
}

// Weighted Zero-mean Normalized Cross Correlation
template <typename T, unsigned CHANNELS>
inline T wzncc( const cv::Mat &img1, const cv::Mat &img2, const double* weights )
{
	unsigned M = img1.cols;
	unsigned N = img1.rows;

	/*if (weights == NULL) { //hehe not safe
		weights = new double[M * N];
		init_sqrtGaussianWeights(M,N, weights);
	}*/

	const uchar* img_1 = img1.data;
	const uchar* img_2 = img2.data;
	unsigned MN = M * N;
		
	T img1_avg[CHANNELS];
	T img2_avg[CHANNELS];
	for( unsigned j = 0; j < CHANNELS; ++j ) {
		img1_avg[j] = 0;
		img2_avg[j] = 0;
	}

	for( unsigned i = 0; i<MN; ++i )
		for( unsigned j = 0; j < CHANNELS; ++j ) {
			img1_avg[j] += img_1[CHANNELS*i+j];
			img2_avg[j] += img_2[CHANNELS*i+j];
		}
	
	for( unsigned i = 0; i<CHANNELS; ++i ) {
		img1_avg[i] /= (T)MN;
		img2_avg[i] /= (T)MN;
	}

	T sum_img1_img2 = 0;
	T sum_img1_2 = 0;
	T sum_img2_2 = 0;

	for( unsigned i = 0; i<MN; ++i )
	{
		for(unsigned j = 0; j < CHANNELS; ++j) {
			auto a = weights[i]*(img_1[3*i+j]-img1_avg[j]);
			auto b = weights[i]*(img_2[3*i+j]-img2_avg[j]);
			sum_img1_img2 += a*b;  
			sum_img1_2    += a*a;
			sum_img2_2    += b*b;
		}
	}
	
	return sum_img1_img2/sqrt(sum_img1_2*sum_img2_2);
}

#pragma endregion

#pragma region Geometric and Epipolar

inline Vec2d rectmin(const Vec2d &a, const Vec2d &b) {
	return Vec2d(std::min(a[0], b[0]), std::min(a[1], b[1]));
}

inline Vec2d rectmax(const Vec2d &a, const Vec2d &b) {
	return Vec2d(std::max(a[0], b[0]), std::max(a[1], b[1]));
}

inline Vec2d vecabs(const Vec2d &a) {
	return Vec2d(std::abs(a[0]), std::abs(a[1]));
}

inline double TripleProduct(cv::Mat a, cv::Mat b, cv::Mat c) { return a.dot(b.cross(c)); }

inline
	cv::Mat FromSphericalNormal(double u, double v)
{
	double sin0 = sin(u);
	return (cv::Mat_<double>(3,1,CV_64F) << sin0*cos(v), sin0*sin(v), cos(u));
}

inline
	cv::Mat FromSphericalNormal4(double u, double v)
{
	double sin0 = sin(u);
	return (cv::Mat_<double>(4,1,CV_64F) << sin0*cos(v), sin0*sin(v), cos(u), 0);
}

inline
	cv::Vec2d ToSphericalNormal(const cv::Vec3d nvec)
{
	double len = sqrt(nvec.dot(nvec));
	return cv::Vec2d(acos(nvec[2] / len), atan2(nvec[1], nvec[0]), 1.0);
}

inline
	cv::Mat GetRotationFromLookat(const cv::Mat &Eye, const cv::Mat &LookAt, const cv::Mat &Up)
{
	cv::Mat up = (1.0/sqrt(Up.dot(Up))) * Up;
#if !RIGHTHANDED
	cv::Mat forward = Eye - LookAt;
#else
	cv::Mat forward = LookAt - Eye;
#endif
	forward = (1.0/sqrt(forward.dot(forward))) * forward;

	cv::Mat side = up.cross(forward); side = (1.0/sqrt(side.dot(side))) * side;
	up = forward.cross(side);  up = (1.0/sqrt(up.dot(up))) * up;

	cv::Mat rotation = cv::Mat(3, 3, cv::DataType<double>::type);
	rotation.col(0) = side * 1.0;
	rotation.col(1) = up * 1.0;
	rotation.col(2) = forward * 1.0;
	return rotation.t();
}

inline
	void decomposeEssentialMat( InputArray _E, OutputArray _R1, OutputArray _R2, OutputArray _t )
{
	Mat E = _E.getMat().reshape(1, 3);
	CV_Assert(E.cols == 3 && E.rows == 3);
	Mat D, U, Vt;
	SVD::compute(E, D, U, Vt);
	if (determinant(U) < 0) U *= -1.;
	if (determinant(Vt) < 0) Vt *= -1.;
	Mat W = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
	W.convertTo(W, E.type());
	Mat R1, R2, t;
	R1 = U * W * Vt;
	R2 = U * W.t() * Vt;
	t = U.col(2) * 1.0;
	R1.copyTo(_R1);
	R2.copyTo(_R2);
	t.copyTo(_t);
}

/*inline
	int recoverPose( InputArray E, const vector<Point2d> &_points1, const vector<Point2d> &_points2, OutputArray _R,
	OutputArray _t, Mat K1, Mat K2, OutputArray _mask = noArray())
{
	Mat K1i = K1.inv();
	Mat K2i = K2.inv();
	Mat apoints1 = Mat(_points1);
	Mat apoints2 = Mat(_points2);
	Mat points1, points2;
	apoints1.copyTo(points1);
	apoints2.copyTo(points2);

	int npoints = points1.checkVector(2);
	CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints && points1.type() == points2.type());
	if (points1.channels() > 1)
	{
		points1 = points1.reshape(1, npoints);
		points2 = points2.reshape(1, npoints);
	}
	points1.convertTo(points1, CV_64F);
	points2.convertTo(points2, CV_64F);

	points1 = points1.t();
	points2 = points2.t();
	Mat mucika = Mat::ones(1, npoints, CV_64F);
	points1.push_back(mucika);
	points2.push_back(mucika);
	points1 = K1i * points1;
	points2 = K2i * points2;
	points1.pop_back(1);
	points2.pop_back(1);

	Mat R1, R2, t;
	decomposeEssentialMat(E, R1, R2, t);
	Mat P0 = Mat::eye(3, 4, R1.type());
	Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
	P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
	P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
	P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
	P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;
	// Do the cheirality check.
	// Notice here a threshold dist is used to filter
	// out far away points (i.e. infinite points) since
	// there depth may vary between postive and negtive.
	double dist = 50.0;
	Mat Q, Q1, Q2, Q3, Q4;
	triangulatePoints(P0, P1, points1, points2, Q);
	Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	Q1 = Q.clone();
	mask1 = (Q.row(2) < dist) & mask1;
	Q = P1 * Q;
	mask1 = (Q.row(2) > 0) & mask1;
	mask1 = (Q.row(2) < dist) & mask1;
	triangulatePoints(P0, P2, points1, points2, Q);
	Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	Q2 = Q.clone();
	mask2 = (Q.row(2) < dist) & mask2;
	Q = P2 * Q;
	mask2 = (Q.row(2) > 0) & mask2;
	mask2 = (Q.row(2) < dist) & mask2;
	triangulatePoints(P0, P3, points1, points2, Q);
	Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	Q3 = Q.clone();
	mask3 = (Q.row(2) < dist) & mask3;
	Q = P3 * Q;
	mask3 = (Q.row(2) > 0) & mask3;
	mask3 = (Q.row(2) < dist) & mask3;
	triangulatePoints(P0, P4, points1, points2, Q);
	Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
	Q.row(0) /= Q.row(3);
	Q.row(1) /= Q.row(3);
	Q.row(2) /= Q.row(3);
	Q.row(3) /= Q.row(3);
	Q4 = Q.clone();
	mask4 = (Q.row(2) < dist) & mask4;
	Q = P4 * Q;
	mask4 = (Q.row(2) > 0) & mask4;
	mask4 = (Q.row(2) < dist) & mask4;
	mask1 = mask1.t();
	mask2 = mask2.t();
	mask3 = mask3.t();
	mask4 = mask4.t();

	// If _mask is given, then use it to filter outliers.
	if (!_mask.empty())
	{
		Mat mask = _mask.getMat();
		CV_Assert(mask.size() == mask1.size());
		bitwise_and(mask, mask1, mask1);
		bitwise_and(mask, mask2, mask2);
		bitwise_and(mask, mask3, mask3);
		bitwise_and(mask, mask4, mask4);
	}
	if (_mask.empty() && _mask.needed())
	{
		_mask.create(mask1.size(), CV_8U);
	}
	CV_Assert(_R.needed() && _t.needed());
	_R.create(3, 3, R1.type());
	_t.create(3, 1, t.type());
	int good1 = countNonZero(mask1);
	int good2 = countNonZero(mask2);
	int good3 = countNonZero(mask3);
	int good4 = countNonZero(mask4);
	if (good1 >= good2 && good1 >= good3 && good1 >= good4)
	{
		R1.copyTo(_R);
		t.copyTo(_t);
		if (_mask.needed()) mask1.copyTo(_mask);
		return good1;
	}
	else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
	{
		R2.copyTo(_R);
		t.copyTo(_t);
		if (_mask.needed()) mask2.copyTo(_mask);
		return good2;
	}
	else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
	{
		t = -t;
		R1.copyTo(_R);
		t.copyTo(_t);
		if (_mask.needed()) mask3.copyTo(_mask);
		return good3;
	}
	else
	{
		t = -t;
		R2.copyTo(_R);
		t.copyTo(_t);
		if (_mask.needed()) mask4.copyTo(_mask);
		return good4;
	}
}  */
#pragma endregion

namespace cv
{

	int RANSACUpdateNumIters( double p, double ep, int modelPoints, int maxIters );

	class LMSolver : public Algorithm
	{
	public:
		class Callback
		{
		public:
			virtual ~Callback() {}
			virtual bool compute(InputArray param, OutputArray err, OutputArray J) const = 0;
		};

		virtual void setCallback(const Ptr<LMSolver::Callback>& cb) = 0;
		virtual int run(InputOutputArray _param0) const = 0;
	};

	//Ptr<LMSolver> createLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters);

	class PointSetRegistrator : public Algorithm
	{
	public:
		class Callback
		{
		public:
			virtual ~Callback() {}
			virtual int runKernel(InputArray m1, InputArray m2, OutputArray model) const = 0;
			virtual void computeError(InputArray m1, InputArray m2, InputArray model, OutputArray err) const = 0;
			virtual bool checkSubset(InputArray, InputArray, int) const { return true; }
		};

		virtual void setCallback(const Ptr<PointSetRegistrator::Callback>& cb) = 0;
		virtual bool run(InputArray m1, InputArray m2, OutputArray model, OutputArray mask) const = 0;
	};

	Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& cb,
		int modelPoints, double threshold,
		double confidence=0.99, int maxIters=1000 );

	Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& cb,
		int modelPoints, double confidence=0.99, int maxIters=1000 );

	template<typename T> inline int compressElems( T* ptr, const uchar* mask, int mstep, int count )
	{
		int i, j;
		for( i = j = 0; i < count; i++ )
			if( mask[i*mstep] )
			{
				if( i > j )
					ptr[j] = ptr[i];
				j++;
			}
			return j;
	}

	class LMSolverImpl : public LMSolver
	{
	public:
		LMSolverImpl() : maxIters(100) { init(); }
		LMSolverImpl(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }

		void init()
		{
			epsx = epsf = FLT_EPSILON;
			printInterval = 0;
		}

		int run(InputOutputArray _param0) const
		{
			Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
			int ptype = param0.type();

			CV_Assert( (param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
			CV_Assert( cb );

			int lx = param0.rows + param0.cols - 1;
			param0.convertTo(x, CV_64F);

			if( x.cols != 1 )
				transpose(x, x);

			if( !cb->compute(x, r, J) )
				return -1;
			double S = norm(r, NORM_L2SQR);
			int nfJ = 2;

			mulTransposed(J, A, true);
			gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

			Mat D = A.diag().clone();

			const double Rlo = 0.25, Rhi = 0.75;
			double lambda = 1, lc = 0.75;
			int i, iter = 0;

			if( printInterval != 0 )
			{
				printf("************************************************************************************\n");
				printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
				printf("************************************************************************************\n");
			}

			for( ;; )
			{
				CV_Assert( A.type() == CV_64F && A.rows == lx );
				A.copyTo(Ap);
				for( i = 0; i < lx; i++ )
					Ap.at<double>(i, i) += lambda*D.at<double>(i);
				solve(Ap, v, d, DECOMP_EIG);
				subtract(x, d, xd);
				if( !cb->compute(xd, rd, noArray()) )
					return -1;
				nfJ++;
				double Sd = norm(rd, NORM_L2SQR);
				gemm(A, d, -1, v, 2, temp_d);
				double dS = d.dot(temp_d);
				double R = (S - Sd)/(fabs(dS) > DBL_EPSILON ? dS : 1);

				if( R > Rhi )
				{
					lambda *= 0.5;
					if( lambda < lc )
						lambda = 0;
				}
				else if( R < Rlo )
				{
					// find new nu if R too low
					double t = d.dot(v);
					double nu = (Sd - S)/(fabs(t) > DBL_EPSILON ? t : 1) + 2;
					nu = std::min(std::max(nu, 2.), 10.);
					if( lambda == 0 )
					{
						invert(A, Ap, DECOMP_EIG);
						double maxval = DBL_EPSILON;
						for( i = 0; i < lx; i++ )
							maxval = std::max(maxval, std::abs(Ap.at<double>(i,i)));
						lambda = lc = 1./maxval;
						nu *= 0.5;
					}
					lambda *= nu;
				}

				if( Sd < S )
				{
					nfJ++;
					S = Sd;
					std::swap(x, xd);
					if( !cb->compute(x, r, J) )
						return -1;
					mulTransposed(J, A, true);
					gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
				}

				iter++;
				bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

				if( printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed) )
				{
					printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
						(proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
				}

				if(!proceed)
					break;
			}

			if( param0.size != x.size )
				transpose(x, x);

			x.convertTo(param0, ptype);
			if( iter == maxIters )
				iter = -iter;

			return iter;
		}

		void setCallback(const Ptr<LMSolver::Callback>& _cb) { cb = _cb; }

		Ptr<LMSolver::Callback> cb;

		double epsx;
		double epsf;
		int maxIters;
		int printInterval;
	};
	/*
	template<typename T, typename A1, typename A2>
	Ptr<T> makePtr(const A1& a1, const A2& a2)
	{
	return Ptr<T>(new T(a1, a2));
	} 

	Ptr<LMSolver> createLMSolver(const Ptr<LMSolver::Callback>& cb, int maxIters)
	{
	return makePtr<LMSolverImpl>(cb, maxIters);
	} */

};

#endif
