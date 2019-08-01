#pragma once

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

/*
	Function declaration
*/
void drawLine(
	cv::Mat &descriptor_, 
	cv::Mat &image_);

void drawMatches(
	cv::Mat points_,
	std::vector<int> inliers_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &out_image_);

bool savePointsToFile(
	const cv::Mat &points_,
	const char* file_,
	const std::vector<int> *inliers_ = NULL);

bool loadPointsFromFile(
	cv::Mat &points_,
	const char* file_);

void detectFeatures(
	std::string name_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &points_);

/*
	Function definition
*/

void drawLine(cv::Mat &descriptor_,
	cv::Mat &image_)
{
	cv::Point2d pt1(0, -descriptor_.at<double>(2) / descriptor_.at<double>(1));
	cv::Point2d pt2(static_cast<double>(image_.cols), -(image_.cols * descriptor_.at<double>(0) + descriptor_.at<double>(2)) / descriptor_.at<double>(1));
	cv::line(image_, pt1, pt2, cv::Scalar(0, 255, 0), 2);
}

void drawMatches(cv::Mat points_,
	std::vector<int> inliers_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &out_image_)
{
	double rotation_angle = 0;
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
			cv::Point2d pt1((double)points_.at<double>(idx, 0), (double)points_.at<double>(idx, 1));
			cv::Point2d pt2(image2_.cols + (double)points_.at<double>(idx, 2), (double)points_.at<double>(idx, 3));

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
			cv::Point2d pt1((double)points_.at<double>(idx, 0), (double)points_.at<double>(idx, 1));
			cv::Point2d pt2(image2_.cols + (double)points_.at<double>(idx, 2), (double)points_.at<double>(idx, 3));

			cv::Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);
			cv::circle(out_image_, pt1, size, color, static_cast<int>(size * 0.4));
			cv::circle(out_image_, pt2, size, color, static_cast<int>(size * 0.4));
			cv::line(out_image_, pt1, pt2, color, 2);
		}
	}

	cv::imshow("Image Out", out_image_);
	cv::waitKey(0);
}

void detectFeatures(std::string scene_name_,
	cv::Mat image1_,
	cv::Mat image2_,
	cv::Mat &points_)
{
	if (loadPointsFromFile(points_,
		scene_name_.c_str()))
	{
		printf("Match number: %d\n", points_.rows);
		return;
	}

	printf("Detect SIFT features\n");
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;

	cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
	detector->detect(image1_, keypoints1);
	detector->compute(image1_, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", static_cast<int>(keypoints1.size()));

	detector->detect(image2_, keypoints2);
	detector->compute(image2_, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", static_cast<int>(keypoints2.size()));

	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector< std::vector< cv::DMatch >> matches_vector;
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

	std::vector<std::tuple<double, cv::Point2d, cv::Point2d>> correspondences;
	for (auto match : matches_vector)
	{
		if (match.size() == 2 && match[0].distance < match[1].distance * 0.8)
		{
			auto& kp1 = keypoints1[match[0].queryIdx];
			auto& kp2 = keypoints2[match[0].trainIdx];
			correspondences.push_back(std::make_tuple<double, cv::Point2d, cv::Point2d>(match[0].distance / match[1].distance, (cv::Point2d)kp1.pt, (cv::Point2d)kp2.pt));
		}
	}

	// Sort the points for PROSAC
	std::sort(correspondences.begin(), correspondences.end(), [](const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_1_,
		const std::tuple<double, cv::Point2d, cv::Point2d>& correspondence_2_) -> bool
	{
		return std::get<0>(correspondence_1_) < std::get<0>(correspondence_2_);
	});

	points_ = cv::Mat(static_cast<int>(correspondences.size()), 4, CV_64F);
	double *points_ptr = reinterpret_cast<double*>(points_.data);

	for (auto[distance_ratio, point_1, point_2] : correspondences)
	{
		*(points_ptr++) = point_1.x;
		*(points_ptr++) = point_1.y;
		*(points_ptr++) = point_2.x;
		*(points_ptr++) = point_2.y;
	}

	savePointsToFile(points_, scene_name_.c_str());
	printf("Match number: %d\n", static_cast<int>(points_.rows));
}

bool loadPointsFromFile(cv::Mat &points,
	const char* file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;

	int N;
	std::string line;
	int line_idx = 0;
	double *points_ptr = NULL;

	while (getline(infile, line))
	{
		if (line_idx++ == 0)
		{
			N = atoi(line.c_str());
			points = cv::Mat(N, 4, CV_64F);
			points_ptr = reinterpret_cast<double*>(points.data);
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

bool savePointsToFile(const cv::Mat &points, const char* file, const std::vector<int> *inliers)
{
	std::ofstream outfile(file, std::ios::out);

	double *points_ptr = reinterpret_cast<double*>(points.data);
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