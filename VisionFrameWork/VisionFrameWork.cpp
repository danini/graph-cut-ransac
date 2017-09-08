// VisionFrameWork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <cv.h>
#include <opencv2\highgui\highgui.hpp>
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

enum TEST { LINE_2D, FUNDAMENTAL_MATRIX, HOMOGRAPHY, ESSENTIAL_MATRIX };

void TestFundamentalMatrix(std::string srcPath,
	std::string dstPath,
	std::string out_corrPath,
	std::string in_corrPath,
	std::string output_srcImagePath,
	std::string output_dstImagePath,
	const float probability,
	const float threshold,
	const float lambda,
	const float neighborhood_size,
	const int fps);

void DrawLine(cv::Mat &descriptor, cv::Mat &image);
void DrawMatches(cv::Mat points, std::vector<int> inliers, cv::Mat image1, cv::Mat image2, cv::Mat &out_image);

bool SavePointsToFile(const cv::Mat &points, const char* file);
bool LoadPointsFromFile(cv::Mat &points, const char* file);
void DetectFeatures(std::string name, cv::Mat image1, cv::Mat image2, cv::Mat &points);
void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4);

int main(int argc, const char* argv[])
{
	srand(time(NULL));
	TEST test_type = TEST::FUNDAMENTAL_MATRIX;
	std::string task = "head";

	// Create the task directory of doesn't exist
	std::string dir = "results/" + task;

	if (stat(dir.c_str(), &info) != 0)
		if (_mkdir(dir.c_str()) != 0)
		{
			std::cerr << "Error while creating a new folder in 'results'\n";
			return -1;
		}

	std::string srcImagePath = "data/head/head1.jpg";
	std::string dstImagePath = "data/head/head2.jpg";
	std::string input_correspondence_path = "results/" + task + "/" + task + "_points_with_no_annotation.txt";
	std::string output_correspondence_path = "results/" + task + "/result_" + task + ".txt";
	std::string output_srcImagePath = "results/" + task + "/out_" + task + "1.png";
	std::string output_dstImagePath = "results/" + task + "/out_" + task + "2.png";

	const float probability = 0.99;
	const int fps = -1;
	const float threshold = 2.00;
	const float lambda = 0.14;
	const float neighborhood_size = 20.0;

	switch (test_type)
	{
	case LINE_2D:
		//TestLine2D();
		break;
	case FUNDAMENTAL_MATRIX:
		TestFundamentalMatrix(srcImagePath,
			dstImagePath,
			input_correspondence_path,
			output_correspondence_path,
			output_srcImagePath,
			output_dstImagePath,
			probability,
			threshold,
			lambda,
			neighborhood_size,
			fps);
		break;
	case HOMOGRAPHY:
		//TestHomography();
		break;
	default:
		break;
	}

	return 0;
} 

void TestFundamentalMatrix(std::string srcPath,
	std::string dstPath,
	std::string out_corrPath,
	std::string in_corrPath,
	std::string output_srcImagePath,
	std::string output_dstImagePath,
	const float probability,
	const float threshold,
	const float lambda,
	const float neighborhood_size,
	const int fps)
{
	std::vector<std::string> tests(0);

	/*tests.push_back("corr");
	tests.push_back("booksh");
	tests.push_back("box");
	tests.push_back("castle");
	tests.push_back("graff");
	tests.push_back("head");
	tests.push_back("kampa");
	tests.push_back("leafs");
	tests.push_back("plant");
	tests.push_back("rotunda");
	tests.push_back("shout");
	tests.push_back("valbonne");
	tests.push_back("wall");
	tests.push_back("wash");
	tests.push_back("zoom");
	tests.push_back("Kyoto");

	tests.push_back("barrsmith");
	tests.push_back("bonhall");
	tests.push_back("bonython");
	tests.push_back("boxesandbooks");
	tests.push_back("elderhalla");
	tests.push_back("elderhallb");
	tests.push_back("glasscasea");
	tests.push_back("glasscaseb");
	tests.push_back("hartley");
	tests.push_back("johnssona");
	tests.push_back("johnssonb");
	tests.push_back("ladysymon");
	tests.push_back("library");
	tests.push_back("napiera");
	tests.push_back("napierb");
	tests.push_back("nese");
	tests.push_back("oldclassicswing");
	tests.push_back("physics");
	tests.push_back("sene");
	tests.push_back("stairs");
	tests.push_back("unihouse");
	tests.push_back("unionhouse");

	// strechamvs dataset
	tests.push_back("Brussels");
	tests.push_back("Dresden");
	tests.push_back("Leuven1");
	tests.push_back("Leuven2");

	// middlebury dataset
	tests.push_back("dino1");
	tests.push_back("dino2");
	tests.push_back("temple1");
	tests.push_back("temple2");*/

	int iteration_number = 0;

	// Read the images
	cv::Mat image1 = cv::imread(srcPath);
	cv::Mat image2 = cv::imread(dstPath);

	// Detect keypoints using SIFT 
	cv::Mat points;
	DetectFeatures(in_corrPath, image1, image2, points);

	// Apply Graph Cut RANSAC
	FundamentalMatrixEstimator estimator;
	std::vector<int> inliers;
	FundamentalMatrix model;

	GCRANSAC<FundamentalMatrixEstimator, FundamentalMatrix> gcransac;
	gcransac.SetFPS(fps); // Set the desired FPS (-1 means no limit)

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	gcransac.Run(points, 
		estimator, 
		model, 
		inliers,
		iteration_number,
		threshold, 
		lambda, 
		neighborhood_size, 
		1.0 - probability, 
		true, 
		true);
	end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::time_t end_time = std::chrono::system_clock::to_time_t(end);
	
	cv::Mat out_image;
	DrawMatches(points, 
		inliers, 
		image1, 
		image2, 
		out_image);

	imwrite("gc_matches.png", out_image);

	std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

	//imwrite("gc_" + test + "_1.png", img1); 
	//imwrite("gc_" + test + "_2.png", img2);

	/*stats[0].geometric_error += GetGeometricErrorF(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2);
	stats[0].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
	stats[0].iteration_number += iteration_number;
	stats[0].lo_steps += gcransac.GetLONumber();
	stats[0].gc_steps += gcransac.GetGCNumber();*/
}

void DrawLine(cv::Mat &descriptor, cv::Mat &image)
{
	cv::Point2f pt1(0, -descriptor.at<float>(2) / descriptor.at<float>(1));
	cv::Point2f pt2(image.cols, -(image.cols * descriptor.at<float>(0) + descriptor.at<float>(2)) / descriptor.at<float>(1));
	cv::line(image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
}

void DrawMatches(cv::Mat points, std::vector<int> inliers, cv::Mat image1, cv::Mat image2, cv::Mat &out_image)
{
	float rotation_angle = 0;
	bool horizontal = true;

	if (image1.cols < image1.rows)
	{
		rotation_angle = 90;
	}

	int counter = 0;
	int size = 10;

	if (horizontal)
	{
		out_image = cv::Mat(image1.rows, 2 * image1.cols, image1.type()); // Your final image

		cv::Mat roiImgResult_Left = out_image(cv::Rect(0, 0, image1.cols, image1.rows)); //Img1 will be on the left part
		cv::Mat roiImgResult_Right = out_image(cv::Rect(image1.cols, 0, image2.cols, image2.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		cv::Mat roiImg1 = image1(cv::Rect(0, 0, image1.cols, image1.rows));
		cv::Mat roiImg2 = image2(cv::Rect(0, 0, image2.cols, image2.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

		for (int i = 0; i < inliers.size(); ++i)
		{
			int idx = inliers[i];
			cv::Point2d pt1((double)points.at<float>(idx, 0), (double)points.at<float>(idx, 1));
			cv::Point2d pt2(image2.cols + (double)points.at<float>(idx, 2), (double)points.at<float>(idx, 3));

			cv::Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);

			cv::circle(out_image, pt1, size, color, size * 0.4);
			cv::circle(out_image, pt2, size, color, size * 0.4);
			cv::line(out_image, pt1, pt2, color, 2);
		}
	}
	else
	{
		out_image = cv::Mat(2 * image1.rows, image1.cols, image1.type()); // Your final image

		cv::Mat roiImgResult_Left = out_image(cv::Rect(0, 0, image1.cols, image1.rows)); //Img1 will be on the left part
		cv::Mat roiImgResult_Right = out_image(cv::Rect(0, image1.rows, image2.cols, image2.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		cv::Mat roiImg1 = image1(cv::Rect(0, 0, image1.cols, image1.rows));
		cv::Mat roiImg2 = image2(cv::Rect(0, 0, image2.cols, image2.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

		for (int i = 0; i < inliers.size(); ++i)
		{
			int idx = inliers[i];
			cv::Point2d pt1((double)points.at<float>(idx, 0), (double)points.at<float>(idx, 1));
			cv::Point2d pt2(image2.cols + (double)points.at<float>(idx, 2), (double)points.at<float>(idx, 3));

			cv::Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);
			cv::circle(out_image, pt1, size, color, size * 0.4);
			cv::circle(out_image, pt2, size, color, size * 0.4);
			cv::line(out_image, pt1, pt2, color, 2);
		}
	}

	cv::imshow("Image Out", out_image);
	cv::waitKey(0);
}

void DetectFeatures(std::string name, cv::Mat image1, cv::Mat image2, cv::Mat &points)
{
	if (LoadPointsFromFile(points, name.c_str()))
	{
		printf("Match number: %d\n", points.rows);
		return;
	}

	printf("Detect SIFT features\n");
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;

	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
	detector->detect(image1, keypoints1);
	detector->compute(image1, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", keypoints1.size());

	detector->detect(image2, keypoints2);
	detector->compute(image2, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", keypoints2.size());

	std::vector<std::vector< cv::DMatch >> matches_vector;
	cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(32));
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

	std::vector<cv::Point2f> src_points, dst_points;
	for (auto m : matches_vector)
	{
		if (m.size() == 2 && m[0].distance < m[1].distance * 0.7)
		{
			auto& kp1 = keypoints1[m[0].queryIdx];
			auto& kp2 = keypoints2[m[0].trainIdx];
			src_points.push_back(kp1.pt);
			dst_points.push_back(kp2.pt);
		}
	}

	points = cv::Mat(src_points.size(), 4, CV_32F);
	float *points_ptr = reinterpret_cast<float*>(points.data);

	for (int i = 0; i < src_points.size(); ++i)
	{
		*(points_ptr++) = src_points[i].x;
		*(points_ptr++) = src_points[i].y;
		*(points_ptr++) = dst_points[i].x;
		*(points_ptr++) = dst_points[i].y;
	}

	SavePointsToFile(points, name.c_str());
	printf("Match number: %d\n", dst_points.size());
}

bool LoadPointsFromFile(cv::Mat &points, const char* file)
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

bool SavePointsToFile(const cv::Mat &points, const char* file)
{
	std::ofstream outfile(file, std::ios::out);
	
	float *points_ptr = reinterpret_cast<float*>(points.data);
	outfile << points.rows << std::endl;
	for (auto i = 0; i < points.rows; ++i)
	{
		outfile << *(points_ptr++) << " ";
		outfile << *(points_ptr++) << " ";
		outfile << *(points_ptr++) << " ";
		outfile << *(points_ptr++) << " ";
		outfile << std::endl;
	}

	outfile.close();

	return true;
}