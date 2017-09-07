#pragma once

#define _USE_MATH_DEFINES
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
 
using namespace cv;
 
class ASiftDetector
{
public:
  ASiftDetector();
 
  void detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors);

protected:
  struct ParallelOp : public cv::ParallelLoopBody {
    ParallelOp(const Mat& _img, std::vector<std::vector<KeyPoint>> &kps, std::vector<Mat> &dsps);
    void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai) const;
    void operator()( const cv::Range &r ) const;

    Mat img;
    std::vector<KeyPoint>* keypoints_array;
    Mat* descriptors_array;
  };
  
};