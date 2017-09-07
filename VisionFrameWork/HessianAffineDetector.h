#pragma once

#include <iostream>
#include <fstream>

#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"

using namespace cv;
using namespace std;

struct HessianAffineParams
{
	float threshold;
	int   max_iter;
	float desc_factor;
	int   patch_size;
	bool  verbose;
	HessianAffineParams()
	{
		threshold = 16.0f / 3.0f;
		max_iter = 16;
		desc_factor = 3.0f*sqrt(3.0f);
		patch_size = 41;
		verbose = false;
	}
};

struct Keypoint
{
	float x, y, s;
	float a11, a12, a21, a22;
	float response;
	int type;
	unsigned char desc[128];
};

struct HessianAffineDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
public:
	const Mat image;
	SIFTDescriptor sift;
	vector<Keypoint> keys;
public:

	HessianAffineDetector(const Mat &image, const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) :
		HessianDetector(par),
		AffineShape(ap),
		image(image),
		sift(sp)
	{
		this->setHessianKeypointCallback(this);
		this->setAffineShapeCallback(this);
	}

	void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
	{
		findAffineShape(blur, x, y, s, pixelDistance, type, response);
	}

	void onAffineShapeFound(
		const Mat &blur, float x, float y, float s, float pixelDistance,
		float a11, float a12,
		float a21, float a22,
		int type, float response, int iters)
	{
		// convert shape into a up is up frame
		rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

		// now sample the patch
		if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
		{
			// compute SIFT
			sift.computeSiftDescriptor(this->patch);
			// store the keypoint
			keys.push_back(Keypoint());
			Keypoint &k = keys.back();
			k.x = x; k.y = y; k.s = s; k.a11 = a11; k.a12 = a12; k.a21 = a21; k.a22 = a22; k.response = response; k.type = type;
			for (int i = 0; i<128; i++)
				k.desc[i] = (unsigned char)sift.vec[i];
			// debugging stuff
			if (0)
			{
				cout << "x: " << x << ", y: " << y
					<< ", s: " << s << ", pd: " << pixelDistance
					<< ", a11: " << a11 << ", a12: " << a12 << ", a21: " << a21 << ", a22: " << a22
					<< ", t: " << type << ", r: " << response << endl;
				//for (size_t i = 0; i<sift.vec.size(); i++)
				//	cout << " " << sift.vec[i];
				cout << endl;
			}
		}
	}

	void getKeypoints(vector<KeyPoint> &points, vector<Mat> &affines, Mat &descriptors)
	{
		points.resize(keys.size());
		affines.resize(keys.size());
		descriptors = Mat(keys.size(), 128, CV_64F);

		for (size_t i = 0; i < keys.size(); i++)
		{
			Keypoint &k = keys[i];

			points[i].size = AffineShape::par.mrSize * k.s;
			points[i].pt.x = k.x;
			points[i].pt.y = k.y;
			affines[i] = (Mat_<double>(2, 2) << (double)k.a11, (double)k.a12, (double)k.a21, (double)k.a22);

			for (size_t j = 0; j < 128; j++)
				descriptors.at<double>(i, j) = (double)k.desc[j];
		}
	}

	void exportKeypoints(ostream &out)
	{
		out << 128 << endl;
		out << keys.size() << endl;
		for (size_t i = 0; i<keys.size(); i++)
		{
			Keypoint &k = keys[i];

			float sc = AffineShape::par.mrSize * k.s;
			Mat A = (Mat_<float>(2, 2) << k.a11, k.a12, k.a21, k.a22);
			SVD svd(A, SVD::FULL_UV);

			float *d = (float *)svd.w.data;
			d[0] = 1.0f / (d[0] * d[0] * sc*sc);
			d[1] = 1.0f / (d[1] * d[1] * sc*sc);

			A = svd.u * Mat::diag(svd.w) * svd.u.t();

			out << k.x << " " << k.y << " " << A.at<float>(0, 0) << " " << A.at<float>(0, 1) << " " << A.at<float>(1, 1);
			for (size_t i = 0; i<128; i++)
				out << " " << int(k.desc[i]);
			out << endl;
		}
	}
};

