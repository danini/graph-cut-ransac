#include "stdafx.h"
#include "EllipticASiftDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>
#include <cv.h>

using namespace cv;

EllipticASiftDetector::EllipticASiftDetector()
{
}

EllipticASiftDetector::ParallelOp::ParallelOp(const Mat& _img, std::vector<std::vector<EllipticKeyPoint>> &kps, std::vector<Mat> &dsps) {
	img = _img;
	keypoints_array = &kps[0];
	descriptors_array = &dsps[0];
}

void EllipticASiftDetector::ParallelOp::affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai, Mat& A) const 
{
	int h = img.rows;
	int w = img.cols;

	mask = Mat(h, w, CV_8UC1, Scalar(255)); 

	A = Mat::eye(2,3, CV_64F);

	if(phi != 0.0)
	{
		phi *= M_PI/180.;
		double s = sin(phi);
		double c = cos(phi);

		A = (Mat_<float>(2,2) << c, -s, s, c);

		Mat corners = (Mat_<float>(4,2) << 0, 0, w, 0, w, h, 0, h);
		Mat tcorners = corners*A.t();
		Mat tcorners_x, tcorners_y;
		tcorners.col(0).copyTo(tcorners_x);
		tcorners.col(1).copyTo(tcorners_y);
		std::vector<Mat> channels;
		channels.push_back(tcorners_x);
		channels.push_back(tcorners_y);
		merge(channels, tcorners);

		Rect rect = boundingRect(tcorners);
		A =  (Mat_<float>(2,3) << c, -s, -rect.x, s, c, -rect.y);

		warpAffine(img, img, A, Size(rect.width, rect.height), INTER_LINEAR, BORDER_REPLICATE);
	}
	if(tilt != 1.0)
	{
		double s = 0.8*sqrt(tilt*tilt-1);
		GaussianBlur(img, img, Size(0,0), s, 0.01);
		resize(img, img, Size(0,0), 1.0/tilt, 1.0, INTER_NEAREST);
		A.row(0) = A.row(0)/tilt;
	}
	if(tilt != 1.0 || phi != 0.0)
	{
		h = img.rows;
		w = img.cols;
		warpAffine(mask, mask, A, Size(w, h), INTER_NEAREST);
	}
	invertAffineTransform(A, Ai);
}

void EllipticASiftDetector::ParallelOp::operator()( const cv::Range &r ) const {

	for (register int tl = r.start; tl != r.end; ++tl) {

		std::vector<EllipticKeyPoint>& keypoints0 = keypoints_array[tl-1];
		Mat& descriptors0 = descriptors_array[tl-1];
		double t = pow(2, 0.5*tl);

		for(double /*TODO: used to be int, i changed dis(Ivan)*/ phi = 0; phi < 180; phi += 72.0/t)
		{
			std::vector<EllipticKeyPoint> ekps;
			std::vector<KeyPoint> kps;
			Mat desc;

			Mat timg, mask, Ai, A;
			img.copyTo(timg);

			affineSkew(t, phi, timg, mask, Ai, A);
			
			if (method == "CUSTOM")
			{
				doTracking(timg, kps, desc, "SIFT");

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);

				/*doTracking(timg, kps, desc, "SURF");

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);*/

				doTracking(timg, kps, desc, "ORB");

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			}
			else
			{
				doTracking(timg, kps, desc, method);

				ekps.resize(kps.size());
				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			}

			//Mat_<double> _A = A.colRange(0,2);
			//Mat_<double> _Ai = Ai.colRange(0,2);
			
			/*{
				SiftFeatureDetector detector;
				SiftDescriptorExtractor extractor;
				
				detector.detect(timg, kps, mask);
				extractor.compute(timg, kps, desc);
				ekps.resize(kps.size());

				for(unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);		
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());
				descriptors0.push_back(desc);
			}*/

			/*{
				SurfFeatureDetector detector(350, 16);
				SurfDescriptorExtractor extractor;


				detector.detect(timg, kps, mask);
				extractor.compute(timg, kps, desc);
				ekps.resize(kps.size());

				for (unsigned int i = 0; i < kps.size(); i++)
				{
					ekps[i] = EllipticKeyPoint(kps[i], Ai);
				}
				keypoints0.insert(keypoints0.end(), ekps.begin(), ekps.end());

				Mat cols(desc.rows, desc.cols, desc.type(), cvScalar(0.));
				cv::hconcat(desc, cols, desc);
				descriptors0.push_back(desc);
			}*/
		}
	}
}

void EllipticASiftDetector::ParallelOp::doTracking(Mat const &img, std::vector<KeyPoint>& keypoints, Mat& descriptors, string method) const
{
	/*
	"FAST" – FastFeatureDetector
	"STAR" – StarFeatureDetector
	"SIFT" – SIFT (nonfree module)
	"SURF" – SURF (nonfree module)
	"ORB" – ORB
	"BRISK" – BRISK
	"MSER" – MSER
	"GFTT" – GoodFeaturesToTrackDetector
	"HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
	"Dense" – DenseFeatureDetector
	"SimpleBlob" – SimpleBlobDetector
	*/
	if (method == "SIFT")
	{
		cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
		f2d->detect(img, keypoints);
		f2d->compute(img, keypoints, descriptors);
	} else if (method == "SURF")
	{
		cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
		f2d->detect(img, keypoints);
		f2d->compute(img, keypoints, descriptors);
	}
	else if (method == "ORB")
	{
		cv::Ptr<ORB> orb = ORB::create();
		orb->detect(img, keypoints);
		orb->compute(img, keypoints, descriptors);
	}
	else if (method == "AKAZE")
	{
		cv::Ptr<AKAZE> akaze = AKAZE::create();
		akaze->detect(img, keypoints);
		akaze->compute(img, keypoints, descriptors);
	}
	else if (method == "BRISK")
	{
		cv::Ptr<BRISK> brisk = BRISK::create();
		brisk->detect(img, keypoints);
		brisk->compute(img, keypoints, descriptors);
	}
	else if (method == "KAZE")
	{
		cv::Ptr<KAZE> brisk = KAZE::create();
		brisk->detect(img, keypoints);
		brisk->compute(img, keypoints, descriptors);
	}
}

void EllipticASiftDetector::detectAndCompute(const Mat& img, std::vector< EllipticKeyPoint >& keypoints, Mat& descriptors, string method)
{
	_method = method;

	auto keypoints_array = std::vector<std::vector<EllipticKeyPoint>>(5);
	auto descriptors_array = std::vector<Mat>(5);

	auto sum = EllipticASiftDetector::ParallelOp(img, keypoints_array, descriptors_array);
	sum.method = _method;
	parallel_for_(Range(1, 6), sum); // non-inclusive end: 6 (elements: 1,2,3,4,5)

	// Merge!
	keypoints.clear();
	descriptors = Mat(0, 128, CV_64F);

	for(auto tl = 0; tl < 5; tl++) {
		keypoints.insert(keypoints.end(), keypoints_array[tl].begin(), keypoints_array[tl].end());
		descriptors.push_back(descriptors_array[tl]);
	}
}  