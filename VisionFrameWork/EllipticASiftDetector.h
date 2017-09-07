#pragma once

#include "stdafx.h"
#include "EllipticKeyPoint.h"

class EllipticASiftDetector
{
public:
  EllipticASiftDetector();
 
  void detectAndCompute(const Mat& img, std::vector< EllipticKeyPoint >& keypoints, Mat& descriptors, string method = "SIFT");

protected:
  struct ParallelOp : public cv::ParallelLoopBody {
    ParallelOp(const Mat& _img, std::vector<std::vector<EllipticKeyPoint>> &kps, std::vector<Mat> &dsps);
    void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai, Mat& A) const;
	void operator()(const cv::Range &r) const;
	void doTracking(Mat const &img, std::vector<KeyPoint>& keypoints, Mat& descriptors, string method) const;

    Mat img;
    std::vector<EllipticKeyPoint>* keypoints_array;
	Mat* descriptors_array;
	string method;
  };


  string _method;
};
