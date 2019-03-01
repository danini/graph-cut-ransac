// VisionFrameWork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <functional>
#include <algorithm>
#include "GCRANSAC.h"
#include <ppl.h>
#include <ctime>
#include "line_estimator.cpp"
#include "essential_estimator.cpp"
#include "fundamental_estimator.cpp"
#include "homography_estimator.cpp"

#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

struct stat info;

void test_fundamental_matrix_fitting(std::string source_path_,
	std::string destination_path_,
	std::string out_correspondence_path_,
	std::string in_correspondence_path_,
	std::string output_match_image_path_,
	const float confidence_,
	const float inlier_outlier_threshold_,
	const float spatial_coherence_weight_,
	const float neighborhood_size_,
	const int fps_);

void draw_line(cv::Mat &descriptor_, cv::Mat &image_);

void draw_matches(cv::Mat points_, 
	std::vector<int> inliers_, 
	cv::Mat image1_, 
	cv::Mat image2_, 
	cv::Mat &out_image_);

bool save_points_to_file(const cv::Mat &points_, 
	const char* file_, 
	std::vector<int> *inliers_ = NULL);

bool load_points_from_file(cv::Mat &points_, 
	const char* file_);

void detect_features(std::string name_, 
	cv::Mat image1_, 
	cv::Mat image2_, 
	cv::Mat &points_);

void projection_matrices_from_essential_matrix(const cv::Mat &essential_,
	cv::Mat &projection_1_, 
	cv::Mat &projection_2_, 
	cv::Mat &projection_3_, 
	cv::Mat &projection_4_);

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	std::string task = "head";

	// Create the task directory of doesn't exist
	std::string dir = "results/" + task;

	if (stat(dir.c_str(), &info) != 0)
		if (_mkdir(dir.c_str()) != 0)
		{
			std::cerr << "Error while creating a new folder in 'results'\n";
			return -1;
		}

	std::string srcImagePath = "data/" + task + "/" + task + "1.jpg";
	std::string dstImagePath = "data/" + task + "/" + task + "2.jpg";
	std::string input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
	std::string output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
	std::string output_matchImagePath = "results/" + task + "/matches_" + task + ".png";

	const float confidence = 0.99f;
	const int fps = -1;
	const float inlier_outlier_threshold = 2.00f;
	const float spatial_coherence_weight = 0.14f;
	const float neighborhood_size = 20.0f;

	test_fundamental_matrix_fitting(srcImagePath,
		dstImagePath,
		input_correspondence_path,
		output_correspondence_path,
		output_matchImagePath,
		confidence,
		inlier_outlier_threshold,
		spatial_coherence_weight,
		neighborhood_size,
		fps);

	return 0;
} 

void test_fundamental_matrix_fitting(std::string source_path_,
	std::string destination_path_,
	std::string out_correspondence_path_,
	std::string in_correspondence_path_,
	std::string output_match_image_path_,
	const float confidence_,
	const float inlier_outlier_threshold_,
	const float spatial_coherence_weight_,
	const float neighborhood_size_,
	const int fps_)
{
	std::vector<std::string> tests(0);

	int iteration_number = 0;

	// Read the images
	cv::Mat image1 = cv::imread(source_path_);
	cv::Mat image2 = cv::imread(destination_path_);

	// Detect keypoints using SIFT 
	cv::Mat points;
	detect_features(in_correspondence_path_, image1, image2, points);

	// Apply Graph Cut RANSAC
	FundamentalMatrixEstimator estimator;
	std::vector<int> inliers;
	FundamentalMatrix model;

	GCRANSAC<FundamentalMatrixEstimator, FundamentalMatrix> gcransac;
	gcransac.set_fps(fps_); // Set the desired FPS (-1 means no limit)

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	gcransac.run(points,
		estimator,
		model,
		inliers,
		iteration_number,
		inlier_outlier_threshold_,
		spatial_coherence_weight_,
		neighborhood_size_,
		1.0f - confidence_,
		true,
		true);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);

	// Write statistics
	printf("Elapsed time = %f secs\n", elapsed_seconds.count());
	printf("Inlier number = %d\n", static_cast<int>(inliers.size()));
	printf("LO Steps = %d\n", gcransac.get_lo_number());
	printf("GC Steps = %d\n", gcransac.get_gc_number());
	printf("Iteration number = %d\n", iteration_number);

	// Save data
	cv::Mat match_image;
	draw_matches(points, 
		inliers, 
		image1, 
		image2, 
		match_image);

	imwrite(output_match_image_path_, match_image); // Save the matched image_
	save_points_to_file(points, out_correspondence_path_.c_str(), &inliers); // Save the inliers_ into file_
}

void draw_line(cv::Mat &descriptor_, 
	cv::Mat &image_)
{
	cv::Point2f pt1(0, -descriptor_.at<float>(2) / descriptor_.at<float>(1));
	cv::Point2f pt2(static_cast<float>(image_.cols), -(image_.cols * descriptor_.at<float>(0) + descriptor_.at<float>(2)) / descriptor_.at<float>(1));
	cv::line(image_, pt1, pt2, cv::Scalar(0, 255, 0), 2);
}

void draw_matches(cv::Mat points_, 
	std::vector<int> inliers_, 
	cv::Mat image1_, 
	cv::Mat image2_, 
	cv::Mat &out_image_)
{
	float rotation_angle = 0;
	bool horizontal = true;

	if (image1_.cols < image1_.rows)
	{
		rotation_angle = 90;
	}

	int counter = 0;
	int size = 10;

	if (horizontal)
	{
		out_image_ = cv::Mat(image1_.rows, 2 * image1_.cols, image1_.type()); // Your final image_

		cv::Mat roiImgResult_Left = out_image_(cv::Rect(0, 0, image1_.cols, image1_.rows)); //Img1 will be on the left part
		cv::Mat roiImgResult_Right = out_image_(cv::Rect(image1_.cols, 0, image2_.cols, image2_.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		cv::Mat roiImg1 = image1_(cv::Rect(0, 0, image1_.cols, image1_.rows));
		cv::Mat roiImg2 = image2_(cv::Rect(0, 0, image2_.cols, image2_.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

		for (auto i = 0; i < inliers_.size(); ++i)
		{
			int idx = inliers_[i];
			cv::Point2d pt1((double)points_.at<float>(idx, 0), (double)points_.at<float>(idx, 1));
			cv::Point2d pt2(image2_.cols + (double)points_.at<float>(idx, 2), (double)points_.at<float>(idx, 3));

			cv::Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);

			cv::circle(out_image_, pt1, size, color, static_cast<int>(size * 0.4f));
			cv::circle(out_image_, pt2, size, color, static_cast<int>(size * 0.4f));
			cv::line(out_image_, pt1, pt2, color, 2);
		}
	}
	else
	{
		out_image_ = cv::Mat(2 * image1_.rows, image1_.cols, image1_.type()); // Your final image_

		cv::Mat roiImgResult_Left = out_image_(cv::Rect(0, 0, image1_.cols, image1_.rows)); //Img1 will be on the left part
		cv::Mat roiImgResult_Right = out_image_(cv::Rect(0, image1_.rows, image2_.cols, image2_.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		cv::Mat roiImg1 = image1_(cv::Rect(0, 0, image1_.cols, image1_.rows));
		cv::Mat roiImg2 = image2_(cv::Rect(0, 0, image2_.cols, image2_.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

		for (auto i = 0; i < inliers_.size(); ++i)
		{
			int idx = inliers_[i];
			cv::Point2d pt1((double)points_.at<float>(idx, 0), (double)points_.at<float>(idx, 1));
			cv::Point2d pt2(image2_.cols + (double)points_.at<float>(idx, 2), (double)points_.at<float>(idx, 3));

			cv::Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);
			cv::circle(out_image_, pt1, size, color, static_cast<int>(size * 0.4));
			cv::circle(out_image_, pt2, size, color, static_cast<int>(size * 0.4));
			cv::line(out_image_, pt1, pt2, color, 2);
		}
	}

	cv::imshow("Image Out", out_image_);
	cv::waitKey(0);
}

void detect_features(std::string scene_name_, 
	cv::Mat image1_, 
	cv::Mat image2_, 
	cv::Mat &points_)
{
	if (load_points_from_file(points_, scene_name_.c_str()))
	{
		printf("Match number: %d\n", points_.rows);
		return;
	}

	printf("Detect SIFT features\n");
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	
	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
	detector->detect(image1_, keypoints1);
	detector->compute(image1_, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", static_cast<int>(keypoints1.size()));

	detector->detect(image2_, keypoints2);
	detector->compute(image2_, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", static_cast<int>(keypoints2.size()));

	std::vector<std::vector< cv::DMatch >> matches_vector;
	cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(32));
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);
	
	std::vector<std::tuple<float, cv::Point2f, cv::Point2f>> correspondences;
	for (auto match : matches_vector)
	{
		if (match.size() == 2 && match[0].distance < match[1].distance * 0.8)
		{
			auto& kp1 = keypoints1[match[0].queryIdx];
			auto& kp2 = keypoints2[match[0].trainIdx];
			correspondences.push_back(std::make_tuple<float, cv::Point2f, cv::Point2f>(match[0].distance / match[1].distance, (cv::Point2f)kp1.pt, (cv::Point2f)kp2.pt));
		}
	}
	
	// Sort the points for PROSAC
	std::sort(correspondences.begin(), correspondences.end(), [](const std::tuple<float, cv::Point2f, cv::Point2f>& correspondence_1_, 
		const std::tuple<float, cv::Point2f, cv::Point2f>& correspondence_2_) -> bool
	{
		return std::get<0>(correspondence_1_) < std::get<0>(correspondence_2_);
	});

	points_ = cv::Mat(static_cast<int>(correspondences.size()), 4, CV_32F);
	float *points_ptr = reinterpret_cast<float*>(points_.data);

	for (auto[distance_ratio, point_1, point_2] : correspondences)
	{
		*(points_ptr++) = point_1.x;
		*(points_ptr++) = point_1.y;
		*(points_ptr++) = point_2.x;
		*(points_ptr++) = point_2.y;
	}

	save_points_to_file(points_, scene_name_.c_str());
	printf("Match number: %d\n", static_cast<int>(points_.rows));
}

bool load_points_from_file(cv::Mat &points, const char* file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;
	
	int N;
	std::string line;
	int line_idx = 0;
	float *points_ptr = NULL;

	while (getline(infile, line))
	{
		if (line_idx++ == 0)
		{
			N = atoi(line.c_str());
			points = cv::Mat(N, 4, CV_32F);
			points_ptr = reinterpret_cast<float*>(points.data);
			continue;
		}

		std::istringstream split(line);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
		split >> *(points_ptr++);
	}

	infile.close();
	return true;
}

bool save_points_to_file(const cv::Mat &points, const char* file, std::vector<int> *inliers)
{
	std::ofstream outfile(file, std::ios::out);
	
	float *points_ptr = reinterpret_cast<float*>(points.data);
	const int M = points.cols;
	
	if (inliers == NULL)
	{
		outfile << points.rows << std::endl;
		for (auto i = 0; i < points.rows; ++i)
		{
			for (auto j = 0; j < M; ++j)
				outfile << *(points_ptr++) << " ";
			outfile << std::endl;
		}
	}
	else
	{
		outfile << inliers->size() << std::endl;
		for (auto i = 0; i < inliers->size(); ++i)
		{
			const int offset = inliers->at(i) * M;
			for (auto j = 0; j < M; ++j)
				outfile << *(points_ptr + offset + j) << " ";
			outfile << std::endl;
		}
	}

	outfile.close();

	return true;
}