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
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2\xfeatures2d\nonfree.hpp>

#include "VisionLibrary.h"

using namespace concurrency;
using namespace std;
using namespace cv;
using namespace vision;

enum TEST { LINE_2D, FUNDAMENTAL_MATRIX, HOMOGRAPHY, ESSENTIAL_MATRIX };

struct LineStats
{
	int iteration_number;
	float angular_error;
	float distance_error;
	float processing_time;
	float non_all_inlier_cases;
	int lo_steps;
	int gc_steps;

	void normalize(int repetation)
	{
		iteration_number = iteration_number / repetation;
		angular_error = angular_error / repetation;
		distance_error = distance_error / repetation;
		processing_time = processing_time / repetation;
		non_all_inlier_cases = non_all_inlier_cases / repetation;
		lo_steps = lo_steps / repetation;
		gc_steps = gc_steps / repetation;
	}

	string print(char separator)
	{
		return to_string(iteration_number) + separator +
			to_string(angular_error) + separator +
			to_string(distance_error) + separator +
			to_string(processing_time) + separator +
			to_string(non_all_inlier_cases) + separator +
			to_string(lo_steps) + separator +
			to_string(gc_steps) + separator;
	}
};

struct Stats
{
	int iteration_number;
	float geometric_error;
	float inlier_ratio;
	float processing_time;
	int lo_steps;
	int gc_steps;
	float all_inlier_sample;
	float not_all_inlier_sample;

	void normalize(int repetation)
	{
		iteration_number = iteration_number / repetation;
		geometric_error = geometric_error / repetation;
		inlier_ratio = inlier_ratio / repetation;
		processing_time = processing_time / repetation;
		lo_steps = round((float)lo_steps / repetation);
		gc_steps = round((float)gc_steps / repetation);
		all_inlier_sample = ((float)all_inlier_sample / repetation);
		not_all_inlier_sample = ((float)not_all_inlier_sample / repetation);
	}

	string print(char separator)
	{
		return to_string(iteration_number) + separator +
			to_string(geometric_error) + separator +
			to_string(inlier_ratio) + separator +
			to_string(processing_time) + separator +
			to_string(lo_steps) + separator +
			to_string(gc_steps) + separator +
			to_string(all_inlier_sample) + separator +
			to_string(not_all_inlier_sample) + separator;
	}
};

struct EssentialStats
{
	int iteration_number;
	float translation_error;
	float rotation_error;
	float inlier_ratio;
	float processing_time;
	int lo_steps;
	int gc_steps;
	float all_inlier_sample;
	float not_all_inlier_sample;

	void normalize(int repetation)
	{
		iteration_number = iteration_number / repetation;
		translation_error = translation_error / repetation;
		rotation_error = rotation_error / repetation;
		inlier_ratio = inlier_ratio / repetation;
		processing_time = processing_time / repetation;
		lo_steps = round((float)lo_steps / repetation);
		gc_steps = round((float)gc_steps / repetation);
		all_inlier_sample = ((float)all_inlier_sample / repetation);
		not_all_inlier_sample = ((float)not_all_inlier_sample / repetation);
	}

	string print(char separator)
	{
		return to_string(iteration_number) + separator +
			to_string(translation_error) + separator +
			to_string(rotation_error) + separator +
			to_string(inlier_ratio) + separator +
			to_string(processing_time) + separator +
			to_string(lo_steps) + separator +
			to_string(gc_steps) + separator +
			to_string(all_inlier_sample) + separator +
			to_string(not_all_inlier_sample) + separator;
	}
};

void TestLine2D();
void TestFundamentalMatrix();
void TestEssentialMatrix();
void TestHomography();

void ReadAnnotatedPoints(string filename, Mat &points, vector<int> &labels);
void LoadMatrix(string filename, Mat &F);
void DrawLine(Mat &descriptor, Mat &image);
void DrawMatches(Mat points, vector<int> labeling, Mat image1, Mat image2, Mat &out_image);

double GetGeometricErrorF(Mat F, vector<Point2d> gt_inlier_pts1, vector<Point2d> gt_inlier_pts);
double GetGeometricErrorH(Mat H, vector<Point2d> gt_inlier_pts1, vector<Point2d> gt_inlier_pts);

int DesiredIterationNumber(int inlier_number, int point_number, int sample_size, float probability);

void GetFundamentalFromPerspective(Mat P1, Mat P2, Mat &F);
double GetEssentialError(Mat E, Mat K1, Mat K2, Mat P1, Mat P2, float &err_t, float &err_R);
bool SavePointsToFile(vector<Point2f> &src_points, vector<Point2f> &dst_points, const char* file);
bool LoadPointsFromFile(vector<Point2f> &src_points, vector<Point2f> &dst_points, const char* file);
void DetectFeatures(string name, Mat image1, Mat image2, vector<Point2f> &src_points, vector<Point2f> &dst_points);
bool LoadProjMatrix(Mat &P, string file);
void TransformPointsWithIntrinsics(vector<Point2f> const &srcPointsIn, vector<Point2f> const &dstPointsIn, Mat K1, Mat K2, vector<Point2f> &srcPointsOut, vector<Point2f> &dstPointsOut);
void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4);

int main(int argc, const char* argv[])
{
	//srand(time(NULL));
	srand(0);
	TEST test_type = TEST::ESSENTIAL_MATRIX;

	switch (test_type)
	{
	case LINE_2D:
		for (int i = 0; i < 100; ++i)
			TestLine2D();
		break;
	case FUNDAMENTAL_MATRIX:
		TestFundamentalMatrix();
		break;
	case ESSENTIAL_MATRIX:
		TestEssentialMatrix();
		break;
	case HOMOGRAPHY:
		TestHomography();
		break;
	default:
		break;
	}

	//while (1);

	return 0;
} 

void TestFundamentalMatrix()
{
	vector<string> tests(0);
	// CMP dataset
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
	tests.push_back("Kyoto");*/

	// AdelaideRMF dataset
	//tests.push_back("barrsmith");
	//tests.push_back("bonhall");
	//tests.push_back("bonython");
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
	/*tests.push_back("Brussels");
	tests.push_back("Dresden");
	tests.push_back("Leuven1");
	tests.push_back("Leuven2");

	// middlebury dataset
	tests.push_back("dino1");
	tests.push_back("dino2");
	tests.push_back("temple1");
	tests.push_back("temple2");*/

	ofstream file("results.csv");
	file.close();
	file = ofstream("results_geom.csv");
	file.close();

	for each (string test in tests)
	{
		ofstream file("results/comparison_" + test + ".csv");
		file << "Threshold;Lambda;GT Err.;Iter. Num;Geom. Err.;Inl. Rat.;Time;LO;GC;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;LO;GC;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;LO;GC;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;LO;GC;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;LO;GC\n";
		file.close();
	}


	int repetation_number = 500;
	float probability = 0.05;
	int fps = 60;

	// Iterate through the tests
	//for (float threshold = 0.31; threshold < 0.32; threshold += 0.01)
	float threshold = 0.31;
	{
		//float lambda = 0.00;
		for (float lambda = 0.14; lambda < 0.15; lambda += 0.01)
		{
			double gt_error = 0;

			int test_idx = -1;
			for each (string test in tests)
			{
				++test_idx;
				float mean_misclassification_error = 0.0f;
				float mean_geom_error = 0.0f;
				float mean_inliers = 0.0f;
				float mean_iteration_num = 0.0f;
				float mean_time = 0.0f;

				vector<Stats> stats(5, { 0, 0, 0, 0 });
				for (int rep = 0; rep < repetation_number; ++rep)
				{
					printf("Iteration %d.\n", rep);
					Mat image1 = imread("data/fundamental_matrix/" + test + "A.png");
					Mat image2 = imread("data/fundamental_matrix/" + test + "B.png");
					if (image1.cols == 0)
					{
						image1 = imread("data/fundamental_matrix/" + test + "A.jpg");
						image2 = imread("data/fundamental_matrix/" + test + "B.jpg");
					}

					Mat points;
					vector<int> labels;
					ReadAnnotatedPoints("data/fundamental_matrix/" + test + "_pts.txt", points, labels);
					//LoadMatrix("data/fundamental_matrix/" + test + "_model.txt", F);

					//F = F / F.at<float>(2, 2);

					Mat F(3, 3, CV_64F);
					vector<Point2d> gt_inlier_pts1, gt_inlier_pts2;
					for (int i = 0; i < points.rows; ++i)
					{
						Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
						Point2d pt2((double)points.at<float>(i, 3), (double)points.at<float>(i, 4));
						if (labels[i] == 0)
						{
							//circle(image1, pt1, 4, Scalar(255, 255, 255), 4);
							//circle(image2, pt2, 4, Scalar(255, 255, 255), 4);
						}
						else
						{
							gt_inlier_pts1.push_back(pt1);
							gt_inlier_pts2.push_back(pt2);
							//circle(image1, pt1, 4, Scalar(255, 0, 255), 4);
							//circle(image2, pt2, 4, Scalar(255, 0, 255), 4);
						}
					}

					/*imshow("Image1", image1);
					imshow("Image2", image2);
					waitKey(0);*/

					F = findFundamentalMat(gt_inlier_pts1, gt_inlier_pts2, CV_FM_7POINT);
					if (F.rows > 3)
					{
						int best_inl_num = 0;
						for (int i = 0; i < 3; ++i)
						{
							int inl_num = 0;
							Mat tempF = (Mat_<double>(3, 3) << F.at<double>(i * 3, 0), F.at<double>(i * 3, 1), F.at<double>(i * 3, 2),
								F.at<double>(i * 3 + 1, 0), F.at<double>(i * 3 + 1, 1), F.at<double>(i * 3 + 1, 2),
								F.at<double>(i * 3 + 2, 0), F.at<double>(i * 3 + 2, 1), F.at<double>(i * 3 + 2, 2));
							tempF = tempF / tempF.at<double>(2, 2);
							tempF.convertTo(tempF, CV_32F);

							for (int j = 0; j < points.rows; ++j)
							{
								Mat pt1 = (Mat_<float>(3, 1) << points.at<float>(j, 0), points.at<float>(j, 1), 1);
								Mat pt2 = (Mat_<float>(1, 3) << points.at<float>(j, 3), points.at<float>(j, 4), 1);

								float err = vision::SymmetricEpipolarError<float>(pt1, pt2, tempF);
								if (err < 2)
									++inl_num;
							}

							if (inl_num > best_inl_num)
							{
								best_inl_num = inl_num;
								F = tempF.clone();
								F = F / F.at<float>(2, 2);
							}
						}


					}
					else
					{
						F = F / F.at<double>(2, 2);
						F.convertTo(F, CV_32F);
					}


					gt_error = GetGeometricErrorF(F, gt_inlier_pts1, gt_inlier_pts2);

					/*
					Apply Graph Cut-based LO-RANSAC
					*/
					FundamentalMatrixEstimator estimator;
					vector<int> inliers;
					FundamentalMatrix modelGC, modelLO, modelLOP, modelLOC, modelPlain;

					GCRANSAC<FundamentalMatrixEstimator, FundamentalMatrix> gcransac(GCRANSAC<FundamentalMatrixEstimator, FundamentalMatrix>::FundamentalMatrix);
					gcransac.SetFPS(fps);

					int iteration_number = 0;

					std::chrono::time_point<std::chrono::system_clock> start, end;
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelGC, inliers, threshold, lambda, 20, probability, iteration_number, true, true);
					end = std::chrono::system_clock::now();

					std::chrono::duration<double> elapsed_seconds = end - start;
					std::time_t end_time = std::chrono::system_clock::to_time_t(end);

					std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
					stats[0].processing_time += (float)elapsed_seconds.count();

					vector<int> obtained_labeling(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					Mat img1 = image1.clone();
					Mat img2 = image2.clone();

					int errors = 0;
					int found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;

						/*Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
						Point2d pt2((double)points.at<float>(i, 3), (double)points.at<float>(i, 4));
						if (obtained_labeling[i] == 1)
						{
						circle(img1, pt1, 2, Scalar(0, 0, 255), 2);
						circle(img2, pt2, 2, Scalar(0, 0, 255), 2);
						}
						else
						{
						circle(img1, pt1, 2, Scalar(0, 0, 0), 2);
						circle(img2, pt2, 2, Scalar(0, 0, 0), 2);
						}*/
					}

					/*Mat out_image;
					DrawMatches(points, obtained_labeling, image1, image2, out_image);

					imwrite("gc_" + test + "_matches.png", out_image);*/
					//imwrite("gc_" + test + "_1.png", img1); 
					//imwrite("gc_" + test + "_2.png", img2);

					stats[0].geometric_error += GetGeometricErrorF(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[0].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[0].iteration_number += iteration_number;
					stats[0].lo_steps += gcransac.GetLONumber();
					stats[0].gc_steps += gcransac.GetGCNumber();

					bool is_all_inlier_sample = true;
					for (int i = 0; i < modelGC.mss.size(); ++i)
						if (labels[modelGC.mss[i]] == 0)
						{
							is_all_inlier_sample = false;
							break;
						}

					if (GetGeometricErrorF(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2) < 1.2 * gt_error)
					{
						stats[0].all_inlier_sample += is_all_inlier_sample ? 1 : 0;
						stats[0].not_all_inlier_sample += is_all_inlier_sample ? 0 : 1;
					}

					//cout << iteration_number << " ";
					//cout << GetGeometricErrorF(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2) << " ";

					//if (lambda > 0.0) 
					//	continue;

					/*
					Apply Full LO-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLO, inliers, threshold, 0.0, 20, probability, iteration_number, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[1].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[1].geometric_error += GetGeometricErrorF(modelLO.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[1].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[1].iteration_number += iteration_number;
					stats[1].lo_steps += gcransac.GetLONumber();
					stats[1].gc_steps += 0;

					is_all_inlier_sample = true;
					for (int i = 0; i < modelLO.mss.size(); ++i)
						if (labels[modelLO.mss[i]] == 0)
						{
							is_all_inlier_sample = false;
							break;
						}


					if (GetGeometricErrorF(modelLO.descriptor, gt_inlier_pts1, gt_inlier_pts2) < 1.2 * gt_error)
					{
						stats[1].all_inlier_sample += is_all_inlier_sample ? 1 : 0;
						stats[1].not_all_inlier_sample += is_all_inlier_sample ? 0 : 1;
					}
					//cout << iteration_number << " ";
					//cout << GetGeometricErrorF(modelLO.descriptor, gt_inlier_pts1, gt_inlier_pts2) << "\n";

					/*
					Apply LO+-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOP, inliers, threshold, 0.0, 20, probability, iteration_number, false, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[2].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[2].geometric_error += GetGeometricErrorF(modelLOP.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[2].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[2].iteration_number += iteration_number;
					stats[2].lo_steps += gcransac.GetLONumber();
					stats[2].gc_steps += 0;

					is_all_inlier_sample = true;
					for (int i = 0; i < modelLOP.mss.size(); ++i)
						if (labels[modelLOP.mss[i]] == 0)
						{
							is_all_inlier_sample = false;
							break;
						}

					if (GetGeometricErrorF(modelLOP.descriptor, gt_inlier_pts1, gt_inlier_pts2) < 1.2 * gt_error)
					{
						stats[2].all_inlier_sample += is_all_inlier_sample ? 1 : 0;
						stats[2].not_all_inlier_sample += is_all_inlier_sample ? 0 : 1;
					}

					/*
					Apply LO'-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOC, inliers, threshold, 0.0, 20, probability, iteration_number, false, true, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[3].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[3].geometric_error += GetGeometricErrorF(modelLOC.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[3].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[3].iteration_number += iteration_number;
					stats[3].lo_steps += gcransac.GetLONumber();
					stats[3].gc_steps += 0;

					is_all_inlier_sample = true;
					for (int i = 0; i < modelLOC.mss.size(); ++i)
						if (labels[modelLOC.mss[i]] == 0)
						{
							is_all_inlier_sample = false;
							break;
						}

					if (GetGeometricErrorF(modelLOC.descriptor, gt_inlier_pts1, gt_inlier_pts2) < 1.2 * gt_error)
					{
						stats[3].all_inlier_sample += is_all_inlier_sample ? 1 : 0;
						stats[3].not_all_inlier_sample += is_all_inlier_sample ? 0 : 1;
					}

					/*
					Apply Plain RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelPlain, inliers, threshold, 0.0, 20, probability, iteration_number, false, false, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[4].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[4].geometric_error += GetGeometricErrorF(modelPlain.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[4].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[4].iteration_number += iteration_number;
					stats[4].lo_steps += 0;
					stats[4].gc_steps += 0;

					is_all_inlier_sample = true;
					for (int i = 0; i < modelPlain.mss.size(); ++i)
						if (labels[modelPlain.mss[i]] == 0)
						{
							is_all_inlier_sample = false;
							break;
						}

					if (GetGeometricErrorF(modelPlain.descriptor, gt_inlier_pts1, gt_inlier_pts2) < 1.2 * gt_error)
					{
						stats[4].all_inlier_sample += is_all_inlier_sample ? 1 : 0;
						stats[4].not_all_inlier_sample += is_all_inlier_sample ? 0 : 1;
					}

					image1.release();
					image2.release();
					F.release();
				}

				stats[0].normalize(repetation_number);
				stats[1].normalize(repetation_number);
				stats[2].normalize(repetation_number);
				stats[3].normalize(repetation_number);
				stats[4].normalize(repetation_number);

				ofstream file("results/comparison_" + test + ".csv", ios::app);
				file << threshold << ";" << lambda << ";" << gt_error << ";";
				file << stats[0].print(';') << ";";
				file << stats[1].print(';') << ";";
				file << stats[2].print(';') << ";";
				file << stats[3].print(';') << ";";
				file << stats[4].print(';') << endl;

				file.close();

				/*cout << stats[0].processing_time << " ";
				cout << stats[1].processing_time << " ";
				cout << stats[2].processing_time << " ";
				cout << stats[3].processing_time << endl;*/

				/*ofstream file("results/results_" + test + ".csv", ios::app);
				file << threshold << ";" << lambda << ";" << mean_misclassification_error << ";" << mean_geom_error << ";" << mean_iteration_num << ";" << mean_inliers << ";" << mean_time << endl;
				file.close();*/

				file = ofstream("results.csv", ios::app);
				file << test << ";";
				file << stats[0].print(';') << ";";
				file << stats[1].print(';') << ";";
				file << stats[2].print(';') << ";";
				file << stats[3].print(';') << ";";
				file << stats[4].print(';') << endl;
				file.close();
			}

			break;
			/*file = ofstream("results.csv", ios::app);
			file << "\n";
			file.close();

			file = ofstream("results_geom.csv", ios::app);
			file << "\n";
			file.close();*/
		}
	}
}

void TestEssentialMatrix()
{
	vector<string> tests(0);
	// CMP dataset
	vector<string> folders;
	folders.push_back("fountainp11");
	folders.push_back("entryp10");
	folders.push_back("herzjesusp8");
	folders.push_back("castlep19");
	folders.push_back("castlep30");
	folders.push_back("herzjesusp25");

	vector<int> images;
	images.push_back(11);
	images.push_back(10);
	images.push_back(8);
	images.push_back(19);
	images.push_back(30);
	images.push_back(25);

	vector<string> extensions;
	extensions.push_back(".jpg");
	extensions.push_back(".png");
	extensions.push_back(".png");
	extensions.push_back(".png");
	extensions.push_back(".png");
	extensions.push_back(".jpg");
		
	int repetation_number = 10;
	float probability = 0.05;
	int fps = 30;

	// Load essential matrices from file
	if (false)
	{
		printf("Load essential matrices from file.\n");

		ifstream essential_file("estimated_matrices_gc.csv");
		ofstream err_file("errors_E_gc.csv");
		err_file.close();
		
		string line;
		vector<EssentialStats> stats(5, { 0, 0, 0, 0, 0, 0 });

		int rep = 0;
		int test = 0;
		int image_num_1 = 0;
		int image_num_2 = 1;
		while (getline(essential_file, line))
		{
			// Load image data
			string test1;
			if (image_num_1 < 10)
				test1 = "0";
			test1 = test1 + "00" + to_string(image_num_1);

			string test2;
			if (image_num_2 < 10)
				test2 = "0";
			test2 = test2 + "00" + to_string(image_num_2);

			Mat P1(3, 4, CV_32F);
			LoadProjMatrix(P1, "data/" + folders[test] + "/" + test1 + ".P");
			Mat P2(3, 4, CV_32F);
			LoadProjMatrix(P2, "data/" + folders[test] + "/" + test2 + ".P");

			Mat K1, R1, t1;
			decomposeProjectionMatrix(P1, K1, R1, t1);
			Mat K2, R2, t2;
			decomposeProjectionMatrix(P2, K2, R2, t2);

			// Load estimated data
			++rep;
			std::string token;
			std::istringstream split(line);

			Mat E(3, 3, CV_32F);

			for (int method = 0; method < 5; ++method)
			{
				float *E_ptr = reinterpret_cast<float*>(E.data);
				for (int i = 0; i < 9; i++)
				{
					std::getline(split, token, ';');
					*(E_ptr++) = static_cast<float>(atof(token.c_str()));
				}

				float err_t, err_R;
				GetEssentialError(E, K1, K2, P1, P2, err_t, err_R);

				stats[method].rotation_error += err_R;
				stats[method].translation_error += err_t;
			}

			if (rep == repetation_number)
			{
				ofstream err_file("errors_E.csv", fstream::app);
				printf("%d %d | ", image_num_1, image_num_2);
				for (int method = 0; method < 5; ++method)
				{
					stats[method].normalize(repetation_number);
					printf("%f %f | ", stats[method].rotation_error, stats[method].translation_error);

					err_file << stats[method].rotation_error << ";" << stats[method].translation_error << ";";
				}
				err_file << endl;
				err_file.close();
				printf("\n");

				stats = vector<EssentialStats>(5, { 0, 0, 0, 0, 0, 0 });

				rep = 0;
				++image_num_2;
				if (image_num_2 >= MIN(images[test], image_num_1 + 7))
				{
					image_num_1++;
					image_num_2 = image_num_1 + 1;

					if (image_num_1 >= images[test] - 1)
					{
						image_num_1 = 0;
						image_num_2 = 1;
						++test;
					}
				}
			}
		}

		essential_file.close();
		return;
	}

	for (int s_idx = 0; s_idx < folders.size(); ++s_idx)
	{
		string folder = folders[s_idx];
		string img_extension = extensions[s_idx];
		int image_num = images[s_idx];

		for (int i = 0; i < image_num - 1; ++i)
		{
			string test1;
			if (i < 10)
				test1 = "0";
			test1 = test1 + "00" + to_string(i);

			vector<EssentialStats> stats(5, { 0, 0, 0, 0, 0, 0 });

			Mat image1 = imread("data/" + folder + "/" + test1 + img_extension);
			//resize(image1, image1, Size(image1.cols / 3, image1.rows / 3));
			Mat P1(3, 4, CV_32F);
			LoadProjMatrix(P1, "data/" + folder + "/" + test1 + ".P");

			Mat K1, R1, t1;
			decomposeProjectionMatrix(P1, K1, R1, t1);

			for (int j = i + 1; j < MIN(image_num, i + 7); ++j)
			{
				string test2;
				if (j < 10)
					test2 = "0";
				test2 = test2 + "00" + to_string(j);

				printf("Match %d. with the %d.\n", i + 1, j + 1);

				Mat image2 = imread("data/" + folder + "/" + test2 + img_extension);
				//resize(image2, image2, Size(image2.cols / 3, image2.rows / 3));
				Mat P2(3, 4, CV_32F);
				LoadProjMatrix(P2, "data/" + folder + "/" + test2 + ".P");

				Mat K2, R2, t2;
				decomposeProjectionMatrix(P2, K2, R2, t2);
				
				// Detect and match features
				vector<Point2f> src_points, dst_points;
				DetectFeatures("data/" + folder + "/" + to_string(i) + "_" + to_string(j) + ".pts", image1, image2, src_points, dst_points);
				
				vector<Point2f> src_points_norm, dst_points_norm;
				TransformPointsWithIntrinsics(src_points, dst_points, K1, K2, src_points_norm, dst_points_norm);
				float scale = 1.0f / ((K1.at<float>(0, 0) + K1.at<float>(1, 1) + K2.at<float>(0, 0) + K2.at<float>(1, 1)) / 4.0f);

				Mat points(src_points_norm.size(), 12, CV_32F);  
				float *points_ptr = (float*)points.data;
				for (int k = 0; k < points.rows; ++k)
				{
					*(points_ptr++) = src_points_norm[k].x;
					*(points_ptr++) = src_points_norm[k].y;
					*(points_ptr++) = 1;
					*(points_ptr++) = dst_points_norm[k].x;
					*(points_ptr++) = dst_points_norm[k].y;
					*(points_ptr++) = 1;
					*(points_ptr++) = src_points[k].x;
					*(points_ptr++) = src_points[k].y;
					*(points_ptr++) = 1;
					*(points_ptr++) = dst_points[k].x;
					*(points_ptr++) = dst_points[k].y;
					*(points_ptr++) = 1;
				}

				float mean_misclassification_error = 0.0f;
				float mean_geom_error = 0.0f;
				float mean_inliers = 0.0f;
				float mean_iteration_num = 0.0f;
				float mean_time = 0.0f;

				float threshold = 0.31;
				float lambda = 0.14;

				for (int rep = 0; rep < repetation_number; ++rep)
				{
					/*
						Apply Graph Cut-based LO-RANSAC
					*/
					EssentialMatrixEstimator estimator;
					estimator.K1i = K1.inv();
					estimator.K2ti = K2.t().inv();

					vector<int> inliers;
					EssentialMatrix modelGC, modelLO, modelLOP, modelLOC, modelPlain;

					GCRANSAC<EssentialMatrixEstimator, EssentialMatrix> gcransac(GCRANSAC<EssentialMatrixEstimator, EssentialMatrix>::FundamentalMatrix);
					gcransac.SetFPS(fps);

					int iteration_number = 0;

					std::chrono::time_point<std::chrono::system_clock> start, end;
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelGC, inliers, threshold, lambda, 10 * scale, probability, iteration_number, true, true, true);
					end = std::chrono::system_clock::now();

					std::chrono::duration<double> elapsed_seconds = end - start;
					stats[0].processing_time += (float)elapsed_seconds.count();

					float err_t, err_R;

					GetEssentialError(modelGC.descriptor, K1, K2, P1, P2, err_t, err_R);
					stats[0].translation_error += err_t;
					stats[0].rotation_error += err_R;
					stats[0].inlier_ratio += 1;
					stats[0].iteration_number += iteration_number;
					stats[0].lo_steps += gcransac.GetLONumber();
					stats[0].gc_steps += gcransac.GetGCNumber();

					std::cout << "[GC-RSC] Elapsed time: " << stats[0].processing_time / (rep + 1) << "s | Inlier number: " << inliers.size() << " | Translation Error: " << stats[0].translation_error / (rep + 1) << " | Rotation Error: " << stats[0].rotation_error / (rep + 1) << "\n";

					/*ofstream e_file("estimated_matrices_gc.csv", fstream::app);
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelGC.descriptor.data + i) << ";";
					e_file << endl;
					e_file.close();*/

					/*for (int i = 0; i < inliers.size(); ++i)
					{
						Point2f pt1(points.at<float>(inliers[i], 0), points.at<float>(inliers[i], 1));
						Point2f pt2(points.at<float>(inliers[i], 3), points.at<float>(inliers[i], 4));
						circle(image1, src_points[inliers[i]], 4, Scalar(255, 0, 255), 4);
						circle(image2, dst_points[inliers[i]], 4, Scalar(255, 0, 255), 4);
					}

					imshow("Image1", image1);
					imshow("Image2", image2);
					waitKey(0);*/

					/*
					Apply Graph Cut-based LO-RANSAC
					*/
					gcransac.SetFPS(fps);

					iteration_number = 0;

					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLO, inliers, threshold, 0, 20, probability, iteration_number, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					stats[1].processing_time += (float)elapsed_seconds.count();

					GetEssentialError(modelLO.descriptor, K1, K2, P1, P2, err_t, err_R);
					stats[1].translation_error += err_t;
					stats[1].rotation_error += err_R;
					stats[1].inlier_ratio += 1;
					stats[1].iteration_number += iteration_number;
					stats[1].lo_steps += gcransac.GetLONumber();
					stats[1].gc_steps += gcransac.GetGCNumber();

					std::cout << "[LO-RSC] Elapsed time: " << stats[1].processing_time / (rep + 1) << "s | Inlier number: " << inliers.size() << " | Translation Error: " << stats[1].translation_error / (rep + 1) << " | Rotation Error: " << stats[1].rotation_error / (rep + 1) << "\n";

					/**/
					gcransac.SetFPS(fps);

					iteration_number = 0;

					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOP, inliers, threshold, 0, 20, probability, iteration_number, false, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					stats[2].processing_time += (float)elapsed_seconds.count();
					GetEssentialError(modelLOP.descriptor, K1, K2, P1, P2, err_t, err_R);
					stats[2].translation_error += err_t;
					stats[2].rotation_error += err_R;
					stats[2].inlier_ratio += 1;
					stats[2].iteration_number += iteration_number;
					stats[2].lo_steps += gcransac.GetLONumber();
					stats[2].gc_steps += gcransac.GetGCNumber();

					std::cout << "[LO+-RSC] Elapsed time: " << stats[2].processing_time / (rep + 1) << "s | Inlier number: " << inliers.size() << " | Translation Error: " << stats[2].translation_error / (rep + 1) << " | Rotation Error: " << stats[2].rotation_error / (rep + 1) << "\n";

					/**/
					gcransac.SetFPS(fps);

					iteration_number = 0;

					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOC, inliers, threshold, 0, 20, probability, iteration_number, false, true, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					stats[3].processing_time += (float)elapsed_seconds.count();
					GetEssentialError(modelLOC.descriptor, K1, K2, P1, P2, err_t, err_R);
					stats[3].translation_error += err_t;
					stats[3].rotation_error += err_R;
					stats[3].inlier_ratio += 1;
					stats[3].iteration_number += iteration_number;
					stats[3].lo_steps += gcransac.GetLONumber();
					stats[3].gc_steps += gcransac.GetGCNumber();

					std::cout << "[LO'-RSC] Elapsed time: " << stats[3].processing_time / (rep + 1) << "s | Inlier number: " << inliers.size() << " | Translation Error: " << stats[3].translation_error / (rep + 1) << " | Rotation Error: " << stats[3].rotation_error / (rep + 1) << "\n";

					/**/
					gcransac.SetFPS(fps);

					iteration_number = 0;

					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelPlain, inliers, threshold, 0, 20, probability, iteration_number, false, false, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					stats[4].processing_time += (float)elapsed_seconds.count();
					GetEssentialError(modelPlain.descriptor, K1, K2, P1, P2, err_t, err_R);
					stats[4].translation_error += err_t;
					stats[4].rotation_error += err_R;
					stats[4].inlier_ratio += 1;
					stats[4].iteration_number += iteration_number;
					stats[4].lo_steps += gcransac.GetLONumber();
					stats[4].gc_steps += gcransac.GetGCNumber();

					std::cout << "[PL-RSC] Elapsed time: " << stats[4].processing_time / (rep + 1) << "s | Inlier number: " << inliers.size() << " | Translation Error: " << stats[4].translation_error / (rep + 1) << " | Rotation Error: " << stats[4].rotation_error / (rep + 1) << "\n";

					std::cout << "------------\n";

					// Write estimated models to file
					ofstream e_file("estimated_matrices_30_fps.csv", fstream::app);
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelGC.descriptor.data + i) << ";";
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelLO.descriptor.data + i) << ";";
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelLOC.descriptor.data + i) << ";";
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelLOP.descriptor.data + i) << ";";
					for (int i = 0; i < 9; ++i)
						e_file << *((float *)modelPlain.descriptor.data + i) << ";";
					e_file << endl;
					e_file.close();
				}

				stats[0].normalize(repetation_number);
				stats[1].normalize(repetation_number);
				stats[2].normalize(repetation_number);
				stats[3].normalize(repetation_number);
				stats[4].normalize(repetation_number);

				ofstream file = ofstream("results_E_" + folder + "_30_fps.csv", ios::app);
				file << threshold << ";";
				file << lambda << ";";
				file << i << ";";
				file << j << ";";
				file << stats[0].print(';') << ";";
				file << stats[1].print(';') << ";";
				file << stats[2].print(';') << ";";
				file << stats[3].print(';') << ";";
				file << stats[4].print(';') << endl;
				file.close();
			}
		}
	}

	while (1);

}

void TestHomography()
{
	vector<string> tests(0);
	// CMP dataset
	tests.push_back("LePoint1");
	tests.push_back("LePoint2"); 
	tests.push_back("LePoint3"); 
	tests.push_back("graf"); 
	tests.push_back("ExtremeZoom"); 
	tests.push_back("city"); 
	tests.push_back("CapitalRegion"); 
	tests.push_back("BruggeTower"); 
	tests.push_back("BruggeSquare"); 
	tests.push_back("BostonLib"); 
	tests.push_back("boat"); 
	tests.push_back("adam"); 
	tests.push_back("WhiteBoard"); 
	tests.push_back("Eiffel"); 
	tests.push_back("Brussels"); 
	tests.push_back("Boston");

	// extremeview dataset
	tests.push_back("extremeview/vin");
	tests.push_back("extremeview/grand");
	tests.push_back("extremeview/dum");
	tests.push_back("extremeview/there");
	tests.push_back("extremeview/adam");
	tests.push_back("extremeview/cafe");
	tests.push_back("extremeview/cat");
	tests.push_back("extremeview/face");
	tests.push_back("extremeview/fox");
	tests.push_back("extremeview/girl");
	tests.push_back("extremeview/graf");
	tests.push_back("extremeview/index");
	tests.push_back("extremeview/mag");
	tests.push_back("extremeview/pkk");
	tests.push_back("extremeview/shop");

	ofstream file("results_hom.csv");
	file.close();
	file = ofstream("results_geom_hom.csv");
	file.close();

	float probability = 0.01;
	int repetation_number = 1;

	for each (string test in tests)
	{
		ofstream file("results/comparison_" + test + "_hom.csv");
		file << "Threshold;Lambda;GT Err.;Iter. Num;Geom. Err.;Inl. Rat.;Time;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time;;";
		file << "Iter. Num;Geom. Err.;Inl. Rat.;Time\n";
		file.close();
	}

	// Iterate through the tests
	// Fundamental matrix est. thr. = 0.31
	// Homgoraphy est. thr. = 0.31

	float sphere_radius = 20;
	float threshold = 0.31;
	//for (float threshold = 0.31; threshold < 0.33; threshold += 0.10)
	//for (sphere_radius = 10.0; sphere_radius < 100; sphere_radius += 10)
	{
		//float lambda = 0.00;
		for (float lambda = 0.14; lambda < 0.15; lambda += 0.01)
		{
			double gt_error = 0;

			int test_idx = -1;
			for each (string test in tests)
			{
				++test_idx;
				float mean_misclassification_error = 0.0f;
				float mean_geom_error = 0.0f;
				float mean_inliers = 0.0f;
				float mean_iteration_num = 0.0f;
				float mean_time = 0.0f;

				vector<Stats> stats(5, { 0, 0, 0, 0, 0, 0 });
				for (int rep = 0; rep < repetation_number; ++rep)
				{
					printf("Iteration %d.\n", rep);
					Mat image1 = imread("data/homography/" + test + "A.png");
					Mat image2 = imread("data/homography/" + test + "B.png");
					if (image1.cols == 0)
					{
						image1 = imread("data/homography/" + test + "A.jpg");
						image2 = imread("data/homography/" + test + "B.jpg");
					}

					Mat points;
					vector<int> labels;
					ReadAnnotatedPoints("data/homography/" + test + "_pts.txt", points, labels);

					Mat H(3, 3, CV_64F);
					vector<Point2d> gt_inlier_pts1, gt_inlier_pts2;
					for (int i = 0; i < points.rows; ++i)
					{
						Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
						Point2d pt2((double)points.at<float>(i, 3), (double)points.at<float>(i, 4));
						if (labels[i] == 0)
						{
							//circle(image1, pt1, 4, Scalar(255, 255, 255), 4);
						//	circle(image2, pt2, 4, Scalar(255, 255, 255), 4);
						}
						else
						{
							gt_inlier_pts1.push_back(pt1);
							gt_inlier_pts2.push_back(pt2);
							//circle(image1, pt1, 4, Scalar(255, 0, 255), 4);
							//circle(image2, pt2, 4, Scalar(255, 0, 255), 4);
						}
					}

					/*imshow("Image1", image1);
					imshow("Image2", image2);
					waitKey(0);*/

					H = findHomography(gt_inlier_pts1, gt_inlier_pts2);
					H = H / H.at<double>(2, 2);
					H.convertTo(H, CV_32F);

					gt_error = GetGeometricErrorH(H, gt_inlier_pts1, gt_inlier_pts2);

					/*
					Apply Graph Cut-based LO-RANSAC
					*/
					RobustHomographyEstimator estimator;
					vector<int> inliers;
					Homography modelGC, modelLO, modelLOP, modelLOC, modelRAN;

					GCRANSAC<RobustHomographyEstimator, Homography> gcransac(GCRANSAC<RobustHomographyEstimator, Homography>::Homography);
					//gcransac.SetFPS(60);

					int iteration_number = 0;

					std::chrono::time_point<std::chrono::system_clock> start, end;
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelGC, inliers, threshold, lambda, sphere_radius, probability, iteration_number, true);
					end = std::chrono::system_clock::now();

					std::chrono::duration<double> elapsed_seconds = end - start;
					std::time_t end_time = std::chrono::system_clock::to_time_t(end);

					std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
					stats[0].processing_time += (float)elapsed_seconds.count();

					vector<int> obtained_labeling(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					int errors = 0;
					int found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					/*for (int i = 0; i < points.rows; ++i)
					{
						Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
						Point2d pt2((double)points.at<float>(i, 3), (double)points.at<float>(i, 4));
						if (obtained_labeling[i] == 1)
						{
							circle(image1, pt1, 4, Scalar(0, 255, 255), 4);
							circle(image2, pt2, 4, Scalar(0, 255, 255), 4);
						}
					}

					imshow("Image1", image1);
					imshow("Image2", image2);
					waitKey(0);*/


					/*Mat out_image;
					DrawMatches(points, obtained_labeling, image1, image2, out_image);

					imwrite(test + "_matches.png", out_image);
					continue;*/

					stats[0].geometric_error += GetGeometricErrorH(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[0].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[0].iteration_number += iteration_number;
					stats[0].lo_steps += gcransac.GetLONumber();
					stats[0].gc_steps += gcransac.GetGCNumber();

					//cout << (float)found_inliers / gt_inlier_pts1.size() << " ";
					//cout << GetGeometricErrorF(modelGC.descriptor, gt_inlier_pts1, gt_inlier_pts2) << " ";

					//if (lambda > 0.0)
					//	continue;

					/*
					Apply Full LO-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLO, inliers, threshold, 0.0, 0, probability, iteration_number, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[1].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[1].geometric_error += GetGeometricErrorH(modelLO.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[1].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[1].iteration_number += iteration_number;
					stats[1].lo_steps += gcransac.GetLONumber();
					stats[1].gc_steps += 0;

					//cout << iteration_number << " ";
					//cout << GetGeometricErrorF(modelLO.descriptor, gt_inlier_pts1, gt_inlier_pts2) << "\n";

					/*
					Apply LO+-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOP, inliers, threshold, 0.0, 0, probability, iteration_number, false, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[2].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[2].geometric_error += GetGeometricErrorH(modelLOP.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[2].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[2].iteration_number += iteration_number;
					stats[2].lo_steps += gcransac.GetLONumber();
					stats[2].gc_steps += 0;

					/*
					Apply LO'-RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelLOC, inliers, threshold, 0.0, 0, probability, iteration_number, false, true, true);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[3].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[3].geometric_error += GetGeometricErrorH(modelLOC.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[3].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[3].iteration_number += iteration_number;
					stats[3].lo_steps += gcransac.GetLONumber();
					stats[3].gc_steps += 0;

					/*
					RANSAC
					*/
					iteration_number = 0;

					inliers.resize(0);
					start = std::chrono::system_clock::now();
					gcransac.Run(points, estimator, modelRAN, inliers, threshold, 0.0, 0, probability, iteration_number, false, false, false, false);
					end = std::chrono::system_clock::now();

					elapsed_seconds = end - start;
					end_time = std::chrono::system_clock::to_time_t(end);

					stats[4].processing_time += (float)elapsed_seconds.count();

					obtained_labeling = vector<int>(points.rows, 0);
					for (int i = 0; i < inliers.size(); ++i)
						obtained_labeling[inliers[i]] = 1;

					errors = 0;
					found_inliers = 0;
					for (int i = 0; i < obtained_labeling.size(); ++i)
					{
						if (obtained_labeling[i] != labels[i])
							++errors;
						if (labels[i] == 1 && obtained_labeling[i] == 1)
							++found_inliers;
					}

					stats[4].geometric_error += GetGeometricErrorH(modelRAN.descriptor, gt_inlier_pts1, gt_inlier_pts2);
					stats[4].inlier_ratio += (float)found_inliers / gt_inlier_pts1.size();
					stats[4].iteration_number += iteration_number;
					stats[4].lo_steps += 0;
					stats[4].gc_steps += 0;


					image1.release();
					image2.release();
					H.release();
				}

				stats[0].normalize(repetation_number);
				stats[1].normalize(repetation_number);
				stats[2].normalize(repetation_number);
				stats[3].normalize(repetation_number);
				stats[4].normalize(repetation_number);

				ofstream file("results/comparison_" + test + "_hom.csv", ios::app);
				file << threshold << ";" << lambda << ";" << gt_error << ";";
				file << stats[0].print(';') << ";";
				file << stats[1].print(';') << ";";
				file << stats[2].print(';') << ";";
				file << stats[3].print(';') << ";";
				file << stats[4].print(';') << endl;

				file.close();

				file = ofstream("results_hom.csv", ios::app);
				file << test << ";";
				file << stats[0].print(';') << ";";
				file << stats[1].print(';') << ";";
				file << stats[2].print(';') << ";";
				file << stats[3].print(';') << ";";
				file << stats[4].print(';') << endl;
				file.close();

				/*ofstream file("results/results_" + test + ".csv", ios::app);
				file << threshold << ";" << lambda << ";" << mean_misclassification_error << ";" << mean_geom_error << ";" << mean_iteration_num << ";" << mean_inliers << ";" << mean_time << endl;
				file.close();

				file = ofstream("results_geom.csv", ios::app);
				file << mean_geom_error << ";";
				file.close();*/
			}

			break;
			/*file = ofstream("results.csv", ios::app);
			file << "\n";
			file.close();

			file = ofstream("results_geom.csv", ios::app);
			file << "\n";
			file.close();*/
		}
	}
}

double GetGeometricErrorF(Mat F, vector<Point2d> gt_inlier_pts1, vector<Point2d> gt_inlier_pts2)
{
	float geom_error = 0;
	float gt_geom_error = 0;
	for (int i = 0; i < gt_inlier_pts1.size(); ++i)
	{
		Mat pt1 = (Mat_<float>(3, 1) << (float)gt_inlier_pts1[i].x, (float)gt_inlier_pts1[i].y, 1);
		Mat pt2 = (Mat_<float>(1, 3) << (float)gt_inlier_pts2[i].x, (float)gt_inlier_pts2[i].y, 1);

		geom_error += vision::SampsonError<float>(pt1, pt2, F);
		//	gt_geom_error += vision::SymmetricEpipolarError<float>(pt1, pt2, F);

		//geom_error += vision::SampsonError<float>(pt1, pt2, F);
		//gt_geom_error += vision::SampsonError<float>(pt1, pt2, gtF);
	}

	geom_error = sqrt(geom_error) / gt_inlier_pts1.size();
	return geom_error;
}

double GetGeometricErrorH(Mat H, vector<Point2d> gt_inlier_pts1, vector<Point2d> gt_inlier_pts2)
{
	float geom_error = 0;
	float gt_geom_error = 0;
	int bad_pts = 0;
	for (int i = 0; i < gt_inlier_pts1.size(); ++i)
	{
		Mat pt1 = (Mat_<float>(3, 1) << (float)gt_inlier_pts1[i].x, (float)gt_inlier_pts1[i].y, 1);
		Mat pt2 = (Mat_<float>(3, 1) << (float)gt_inlier_pts2[i].x, (float)gt_inlier_pts2[i].y, 1);

		pt1 = H * pt1;
		if (abs(pt1.at<float>(2)) < 1e-8)
		{
			++bad_pts;
			continue;
		}

		pt1 = pt1 / pt1.at<float>(2);

		geom_error += norm(pt1 - pt2);
	}
	
	geom_error = sqrt(geom_error) / (gt_inlier_pts1.size() - bad_pts);

	return geom_error;
}

void TestLine2D()
{
	//ofstream file("line_results.csv");
	//file.close();

	float probability = 0.05;
	int repetation_number = 1000;
	
	bool dashed = true;
	float knot_sizes = 10;
	int knot_number = 10;

	int point_number = 50;
	float line_threshold = 0.5;

	for (float outlier_ratio = 1.0; outlier_ratio <= 200; outlier_ratio += 5)
	{
		for (float noise = 0.0; noise < 10.0; noise += 1.0)
		{
			vector<LineStats> stats(5, { 0,0,0,0,0,0,0 });

			for (int rep = 0; rep < repetation_number; ++rep)
			{
				cout << "Iteration number = " << rep << endl;
				/*
					Generate scene
				*/
				Mat image(600, 600, CV_8UC3, Scalar(255, 255, 255));
				Mat image_gt(600, 600, CV_8UC3, Scalar(255, 255, 255));
				Mat points(point_number + outlier_ratio * point_number, 2, CV_32F);

				// Generate random line
				float alpha = M_PI * (float)rand() / RAND_MAX;
				//cout << alpha << endl;
				Point2d lineNormal(sin(alpha), cos(alpha));
				lineNormal = lineNormal / norm(lineNormal);
				Point2d lineTangent(-lineNormal.y, lineNormal.x);

				Point2d center(150, 150); // image.cols * (double)rand() / RAND_MAX, image.rows * (double)rand() / RAND_MAX);
				float c = -(lineNormal.x * center.x + lineNormal.y * center.y);

				float gt_dist_from_center = abs(lineNormal.x * 300 + lineNormal.y * 300 + c);
				Mat gt_model_desc = (Mat_<float>(3, 1) << lineNormal.x, lineNormal.y, c);

				// Add outliers
				for (int i = 0; i < outlier_ratio * point_number; ++i)
				{
					points.at<float>(point_number + i, 0) = image.cols * (float)rand() / RAND_MAX;
					points.at<float>(point_number + i, 1) = image.rows * (float)rand() / RAND_MAX;

					circle(image, (Point2d)points.row(point_number + i), 2, Scalar(0, 0, 0), 2);
				}

				// Sample the line at random locations
				if (knot_number == 1)
				{
					for (int i = 0; i < point_number; ++i)
					{
						if (lineNormal.x > lineNormal.y)
						{
							points.at<float>(i, 1) = image.rows * (float)rand() / RAND_MAX;
							points.at<float>(i, 0) = -(lineNormal.y * points.at<float>(i, 1) + c) / lineNormal.x;
						}
						else
						{
							points.at<float>(i, 0) = image.cols * (float)rand() / RAND_MAX;
							points.at<float>(i, 1) = -(lineNormal.x * points.at<float>(i, 0) + c) / lineNormal.y;
						}

						points.at<float>(i, 0) = points.at<float>(i, 0) + noise * (float)rand() / RAND_MAX;
						points.at<float>(i, 1) = points.at<float>(i, 1) + noise * (float)rand() / RAND_MAX;

						circle(image, (Point2d)points.row(i), 2, Scalar(0, 0, 255), 2);
					}
				}
				else
				{
					for (int kn = 0; kn < knot_number; ++kn)
					{
						Point2f knot;
						if (lineNormal.x > lineNormal.y)
						{
							knot.y = image.rows * (float)rand() / RAND_MAX;
							knot.x = -(lineNormal.y * knot.y + c) / lineNormal.x;
						}
						else
						{
							knot.x = image.cols * (float)rand() / RAND_MAX;
							knot.y = -(lineNormal.x * knot.x + c) / lineNormal.y;
						}

						int pt_per_knot = point_number / knot_number;
						for (int i = 0; i < pt_per_knot; ++i)
						{
							float dist = knot_sizes * (float)rand() / RAND_MAX;
							points.at<float>(kn * pt_per_knot + i, 0) = (float)knot.x + (float)lineTangent.x * dist + noise * (float)rand() / RAND_MAX;
							points.at<float>(kn * pt_per_knot + i, 1) = (float)knot.y + (float)lineTangent.y * dist + noise * (float)rand() / RAND_MAX;

							circle(image, (Point2d)points.row(kn * pt_per_knot + i), 2, Scalar(0, 0, 255), 2);
						}
					}
				}
				
				/*image_gt = image.clone();*/

				Point2d pt1(0, -c / lineNormal.y);
				Point2d pt2(image.cols, -((image.cols * lineNormal.x) + c) / lineNormal.y);
				//line(image_gt, pt1, pt2, Scalar(255, 0, 255), 2);*/

				//imwrite("results/unrealistic_gt.png", image_gt);

				/*
					Apply Graph Cut-based LO-RANSAC
				*/
				LineEstimator estimator;
				vector<int> inliers;
				Line2D model;

				GCRANSAC<LineEstimator, Line2D> gcransac(GCRANSAC<LineEstimator, Line2D>::Line2d);
				//gcransac.SetFPS(10);

				int iteration_number;

				std::chrono::time_point<std::chrono::system_clock> start, end;
				start = std::chrono::system_clock::now();
				gcransac.Run(points, estimator, model, inliers, line_threshold, 0.14f, 10, probability, iteration_number, true);
				end = std::chrono::system_clock::now();

				std::chrono::duration<double> elapsed_seconds = end - start;

				std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

				for (int i = 0; i < inliers.size(); ++i)
				{
					//circle(image, (Point2d)points.row(inliers[i]), 1, Scalar(0, 255, 0), 2);
				}

				Point2f normal_GC(model.descriptor.at<float>(0), model.descriptor.at<float>(1));
				normal_GC = normal_GC / norm(normal_GC);
				float angular_error = MIN(acos(normal_GC.dot(lineNormal)), acos(normal_GC.dot(-lineNormal))) / M_PI * 180.0f;

				if (angular_error != angular_error)
					angular_error = 0.0;

				float dist_from_center = abs(normal_GC.x * 300 + normal_GC.y * 300 + model.descriptor.at<float>(2));

				stats[0].processing_time += (float)elapsed_seconds.count();
				stats[0].angular_error += angular_error; //angular_error > 5 ? 1 : 0;
				stats[0].distance_error += abs(gt_dist_from_center - dist_from_center);
				stats[0].iteration_number += iteration_number;
				stats[0].lo_steps += gcransac.GetLONumber();
				stats[0].gc_steps += gcransac.GetGCNumber();

				if (estimator.Error(points.row(model.mss1), gt_model_desc) > MAX(1, noise) ||
					estimator.Error(points.row(model.mss2), gt_model_desc) > MAX(1, noise))
					++stats[0].non_all_inlier_cases;

				Mat gc_result = model.descriptor.clone();

				/*imwrite("dashed_line.png", image);
				imshow("Image", image);
				waitKey(0);

				continue;
				*/
				/* */
				iteration_number = 0;
				start = std::chrono::system_clock::now();
				gcransac.Run(points, estimator, model, inliers, line_threshold, 0.14f, 20, probability, iteration_number, false, false);
				end = std::chrono::system_clock::now();
				elapsed_seconds = end - start;

				Point2f normal_LO(model.descriptor.at<float>(0), model.descriptor.at<float>(1));
				normal_LO = normal_LO / norm(normal_LO);

				angular_error = MIN(acos(normal_LO.dot(lineNormal)), acos(normal_LO.dot(-lineNormal))) / M_PI * 180.0f;

				if (angular_error != angular_error)
					angular_error = 0.0;

				dist_from_center = abs(normal_LO.x * 300 + normal_LO.y * 300 + model.descriptor.at<float>(2));

				stats[1].processing_time += (float)elapsed_seconds.count();
				stats[1].angular_error += angular_error; //angular_error > 5 ? 1 : 0;
				stats[1].distance_error += abs(gt_dist_from_center - dist_from_center);
				stats[1].iteration_number += iteration_number;
				stats[1].lo_steps += gcransac.GetLONumber();
				stats[1].gc_steps += 0;

				if (estimator.Error(points.row(model.mss1), gt_model_desc) > MAX(1, noise) ||
					estimator.Error(points.row(model.mss2), gt_model_desc) > MAX(1, noise))
					++stats[1].non_all_inlier_cases;

				pt1 = Point2d(0, -model.descriptor.at<float>(2) / model.descriptor.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * model.descriptor.at<float>(0) + model.descriptor.at<float>(2)) / model.descriptor.at<float>(1));
				line(image, pt1, pt2, Scalar(0, 128, 0), 2);

				//imshow("Image", image);

				/* */
				iteration_number = 0;
				start = std::chrono::system_clock::now();
				gcransac.Run(points, estimator, model, inliers, line_threshold, 0.14f, 20, probability, iteration_number, false, true);
				end = std::chrono::system_clock::now();
				elapsed_seconds = end - start;

				Point2f normal_LOP(model.descriptor.at<float>(0), model.descriptor.at<float>(1));
				normal_LOP = normal_LOP / norm(normal_LOP);

				angular_error = MIN(acos(normal_LOP.dot(lineNormal)), acos(normal_LOP.dot(-lineNormal))) / M_PI * 180.0f;

				if (angular_error != angular_error)
					angular_error = 0.0;

				dist_from_center = abs(normal_LOP.x * 300 + normal_LOP.y * 300 + model.descriptor.at<float>(2));

				stats[2].processing_time += (float)elapsed_seconds.count();
				stats[2].angular_error += angular_error; //angular_error > 5 ? 1 : 0;
				stats[2].distance_error += abs(gt_dist_from_center - dist_from_center);
				stats[2].iteration_number += iteration_number;
				stats[2].lo_steps += gcransac.GetLONumber();
				stats[2].gc_steps += 0;


				if (estimator.Error(points.row(model.mss1), gt_model_desc) > MAX(1, noise) ||
					estimator.Error(points.row(model.mss2), gt_model_desc) > MAX(1, noise))
					++stats[2].non_all_inlier_cases;

				pt1 = Point2d(0, -model.descriptor.at<float>(2) / model.descriptor.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * model.descriptor.at<float>(0) + model.descriptor.at<float>(2)) / model.descriptor.at<float>(1));
				line(image, pt1, pt2, Scalar(0, 0, 255), 2);

				//imshow("Image", image);

				/* */
				iteration_number = 0;
				start = std::chrono::system_clock::now();
				gcransac.Run(points, estimator, model, inliers, line_threshold, 0.14f, 20, probability, iteration_number, false, true, true);
				end = std::chrono::system_clock::now();
				elapsed_seconds = end - start;

				Point2f normal_LOC(model.descriptor.at<float>(0), model.descriptor.at<float>(1));
				normal_LOC = normal_LOC / norm(normal_LOC);

				angular_error = MIN(acos(normal_LOC.dot(lineNormal)), acos(normal_LOC.dot(-lineNormal))) / M_PI * 180.0f;

				if (angular_error != angular_error)
					angular_error = 0.0;

				dist_from_center = abs(normal_LOC.x * 300 + normal_LOC.y * 300 + model.descriptor.at<float>(2));

				stats[3].processing_time += (float)elapsed_seconds.count();
				stats[3].angular_error += angular_error; //angular_error > 5 ? 1 : 0;
				stats[3].distance_error += abs(gt_dist_from_center - dist_from_center);
				stats[3].iteration_number += iteration_number;
				stats[3].lo_steps += gcransac.GetLONumber();
				stats[3].gc_steps += 0;

				if (estimator.Error(points.row(model.mss1), gt_model_desc) > MAX(1, noise) ||
					estimator.Error(points.row(model.mss2), gt_model_desc) > MAX(1, noise))
					++stats[3].non_all_inlier_cases;

				pt1 = Point2d(0, -model.descriptor.at<float>(2) / model.descriptor.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * model.descriptor.at<float>(0) + model.descriptor.at<float>(2)) / model.descriptor.at<float>(1));
				line(image, pt1, pt2, Scalar(191, 191, 0), 2);


				pt1 = Point2d(0, -gc_result.at<float>(2) / gc_result.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * gc_result.at<float>(0) + gc_result.at<float>(2)) / gc_result.at<float>(1));
				line(image, pt1, pt2, Scalar(255, 0, 0), 2);

				/* */
				iteration_number = 0;
				start = std::chrono::system_clock::now();
				gcransac.Run(points, estimator, model, inliers, line_threshold, 0.14f, 20, probability, iteration_number, false, false, false, false);
				end = std::chrono::system_clock::now();
				elapsed_seconds = end - start;

				Point2f normal_PLAIN(model.descriptor.at<float>(0), model.descriptor.at<float>(1));
				normal_PLAIN = normal_PLAIN / norm(normal_PLAIN);

				angular_error = MIN(acos(normal_PLAIN.dot(lineNormal)), acos(normal_PLAIN.dot(-lineNormal))) / M_PI * 180.0f;

				if (angular_error != angular_error)
					angular_error = 0.0;

				dist_from_center = abs(normal_PLAIN.x * 300 + normal_PLAIN.y * 300 + model.descriptor.at<float>(2));

				stats[4].processing_time += (float)elapsed_seconds.count();
				stats[4].angular_error += angular_error; //angular_error > 5 ? 1 : 0;
				stats[4].distance_error += abs(gt_dist_from_center - dist_from_center);
				stats[4].iteration_number += iteration_number;
				stats[4].lo_steps += gcransac.GetLONumber();
				stats[4].gc_steps += 0;

				if (estimator.Error(points.row(model.mss1), gt_model_desc) > MAX(1, noise) ||
					estimator.Error(points.row(model.mss2), gt_model_desc) > MAX(1, noise))
					++stats[4].non_all_inlier_cases;

				pt1 = Point2d(0, -model.descriptor.at<float>(2) / model.descriptor.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * model.descriptor.at<float>(0) + model.descriptor.at<float>(2)) / model.descriptor.at<float>(1));
				line(image, pt1, pt2, Scalar(191, 191, 0), 2);


				pt1 = Point2d(0, -gc_result.at<float>(2) / gc_result.at<float>(1));
				pt2 = Point2d(image.cols, -(image.cols * gc_result.at<float>(0) + gc_result.at<float>(2)) / gc_result.at<float>(1));
				line(image, pt1, pt2, Scalar(255, 0, 0), 2);

				//imshow("Image", image);
				//imwrite("results/unrealistic.png", image);
				//waitKey(0);
				cout << stats[0].angular_error << " " << stats[1].angular_error << endl;
			}

			stats[0].normalize(repetation_number);
			stats[1].normalize(repetation_number);
			stats[2].normalize(repetation_number);
			stats[3].normalize(repetation_number);
			stats[4].normalize(repetation_number);


			ofstream file("line_results.csv", ios::app);
			file << outlier_ratio << ";" << noise << ";";
			file << stats[0].print(';') << ";";
			file << stats[1].print(';') << ";";
			file << stats[2].print(';') << ";";
			file << stats[3].print(';') << ";";
			file << stats[4].print(';') << endl;
			file.close();
		}
	}

	/*DrawLine(model.descriptor, image);

	imshow("Image", image);
	waitKey(0);*/



	/*RansacParameters params;
	//params.min_inlier_ratio = 0.0001;
	params.error_thresh = 10.0;

	Prosac<LineEstimator> prosac_fundamental(params, lineEstimator);
	prosac_fundamental.Initialize();
	RansacSummary summary;

	Line2D obtained_line;
	prosac_fundamental.Estimate(points, &obtained_line, &summary);

	Point2d pt1(0, -obtained_line.decriptor.at<double>(2) / obtained_line.decriptor.at<double>(1));
	Point2d pt2(image.cols, -(image.cols * obtained_line.decriptor.at<double>(0) + obtained_line.decriptor.at<double>(2)) / obtained_line.decriptor.at<double>(1));
	line(image, pt1, pt2, Scalar(0, 255, 0), 2);

	imshow("Resulting Image", image);
	waitKey(0);*/
}

void DrawLine(Mat &descriptor, Mat &image)
{
	Point2f pt1(0, -descriptor.at<float>(2) / descriptor.at<float>(1));
	Point2f pt2(image.cols, -(image.cols * descriptor.at<float>(0) + descriptor.at<float>(2)) / descriptor.at<float>(1));
	line(image, pt1, pt2, Scalar(0, 255, 0), 2);
}

void LoadMatrix(string filename, Mat &F)
{
	ifstream file(filename);

	for (int r = 0; r < F.rows; ++r)
		for (int c = 0; c < F.cols; ++c)
			file >> F.at<float>(r, c);
	file.close();
}

void ReadAnnotatedPoints(string filename, Mat &points, vector<int> &labels)
{
	ifstream file(filename);

	double x1, y1, x2, y2, a, s;
	string str;

	vector<Point2d> pts1;
	vector<Point2d> pts2;
	if (filename.find("extremeview") != std::string::npos) // For extremeview dataset
	{
		while (file >> x1 >> y1 >> x2 >> y2 >> s >> s >> str >> str  >> a)
		{
			pts1.push_back(Point2d(x1, y1));
			pts2.push_back(Point2d(x2, y2));
			labels.push_back(a > 0 ? 1 : 0);
		}
	}
	else
	{
		while (file >> x1 >> y1 >> s >> x2 >> y2 >> s >> a)
		{
			pts1.push_back(Point2d(x1, y1));
			pts2.push_back(Point2d(x2, y2));
			labels.push_back(a > 0 ? 1 : 0);
		}
	}

	file.close();

	points = Mat(pts1.size(), 6, CV_32F);
	for (int i = 0; i < pts1.size(); ++i)
	{
		points.at<float>(i, 0) = pts1[i].x;
		points.at<float>(i, 1) = pts1[i].y;
		points.at<float>(i, 2) = 1;
		points.at<float>(i, 3) = pts2[i].x;
		points.at<float>(i, 4) = pts2[i].y;
		points.at<float>(i, 5) = 1;
	}
}

void DrawMatches(Mat points, vector<int> labeling, Mat image1, Mat image2, Mat &out_image)
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
		out_image = Mat(image1.rows, 2 * image1.cols, image1.type()); // Your final image

		Mat roiImgResult_Left = out_image(Rect(0, 0, image1.cols, image1.rows)); //Img1 will be on the left part
		Mat roiImgResult_Right = out_image(Rect(image1.cols, 0, image2.cols, image2.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		Mat roiImg1 = image1(Rect(0, 0, image1.cols, image1.rows));
		Mat roiImg2 = image2(Rect(0, 0, image2.cols, image2.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult

		for (int i = 0; i < points.rows; ++i)
		{
			if (counter++ % 4 != 0)
				continue;

			Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
			Point2d pt2(image2.cols + (double)points.at<float>(i, 3), (double)points.at<float>(i, 4));

			Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);
			if (labeling[i] == 0)
			{
				color = Scalar(0, 0, 0);
				line(out_image, pt1 - Point2d(size, size), pt1 + Point2d(size, size), color, 2);
				line(out_image, pt1 - Point2d(size, -size), pt1 + Point2d(size, -size), color, 2);

				line(out_image, pt2 - Point2d(size, size), pt2 + Point2d(size, size), color, 2);
				line(out_image, pt2 - Point2d(size, -size), pt2 + Point2d(size, -size), color, 2);
			}
			else
			{

				circle(out_image, pt1, size, color, size * 0.4);
				circle(out_image, pt2, size, color, size * 0.4);
			}
			line(out_image, pt1, pt2, color, 2);
		}
	}
	else
	{
		out_image = Mat(2 * image1.rows, image1.cols, image1.type()); // Your final image

		Mat roiImgResult_Left = out_image(Rect(0, 0, image1.cols, image1.rows)); //Img1 will be on the left part
		Mat roiImgResult_Right = out_image(Rect(0, image1.rows, image2.cols, image2.rows)); //Img2 will be on the right part, we shift the roi of img1.cols on the right

		Mat roiImg1 = image1(Rect(0, 0, image1.cols, image1.rows));
		Mat roiImg2 = image2(Rect(0, 0, image2.cols, image2.rows));

		roiImg1.copyTo(roiImgResult_Left); //Img1 will be on the left of imgResult
		roiImg2.copyTo(roiImgResult_Right); //Img2 will be on the right of imgResult
		
		for (int i = 0; i < points.rows; ++i)
		{
			//if (counter++ % 4 != 0)
			//	continue;

			Point2d pt1((double)points.at<float>(i, 0), (double)points.at<float>(i, 1));
			Point2d pt2((double)points.at<float>(i, 3), image2.rows + (double)points.at<float>(i, 4));

			Scalar color(255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX, 255 * (double)rand() / RAND_MAX);
			if (labeling[i] == 0)
			{
				color = Scalar(0, 0, 0);
				line(out_image, pt1 - Point2d(size, size), pt1 + Point2d(size, size), color, 2);
				line(out_image, pt1 - Point2d(size, -size), pt1 + Point2d(size, -size), color, 2);

				line(out_image, pt2 - Point2d(size, size), pt2 + Point2d(size, size), color, 2);
				line(out_image, pt2 - Point2d(size, -size), pt2 + Point2d(size, -size), color, 2);
			}
			else
			{

				circle(out_image, pt1, size, color, size * 0.4);
				circle(out_image, pt2, size, color, size * 0.4);
			}
			line(out_image, pt1, pt2, color, 2);
		}
	}

	imshow("Image Out", out_image);
	waitKey(0);
}

int DesiredIterationNumber(int inlier_number, int point_number, int sample_size, float probability)
{
	float q = pow((float)inlier_number / point_number, sample_size);

	float iter = log(probability) / log(1 - q);
	if (iter < 0)
		return INT_MAX;
	return (int)iter + 1;
}

void DetectFeatures(string name, Mat image1, Mat image2, vector<Point2f> &src_points, vector<Point2f> &dst_points)
{
	if (LoadPointsFromFile(src_points, dst_points, name.c_str()))
	{
		printf("Match number: %d\n", dst_points.size());
		return;
	}

	printf("Detect SURF features\n");
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;

	cv::Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create();
	detector->detect(image1, keypoints1);
	detector->compute(image1, keypoints1, descriptors1);
	printf("Features found in the first image: %d\n", keypoints1.size());

	detector->detect(image2, keypoints2);
	detector->compute(image2, keypoints2, descriptors2);
	printf("Features found in the second image: %d\n", keypoints2.size());

	std::vector<std::vector< cv::DMatch >> matches_vector;
	cv::FlannBasedMatcher matcher(new cv::flann::KDTreeIndexParams(5), new cv::flann::SearchParams(32));
	matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

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

	SavePointsToFile(src_points, dst_points, name.c_str());

	printf("Match number: %d\n", dst_points.size());
}

bool LoadPointsFromFile(vector<Point2f> &src_points, vector<Point2f> &dst_points, const char* file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;

	float x1, y1, x2, y2;

	string line;
	while (getline(infile, line))
	{
		std::istringstream split(line);
		split >> x1 >> y1 >> x2 >> y2;

		src_points.push_back(Point2f(x1, y1));
		dst_points.push_back(Point2f(x2, y2));
	}

	infile.close();
	return true;
}

bool LoadProjMatrix(Mat &P, string file)
{
	std::ifstream infile(file);

	if (!infile.is_open())
		return false;
	
	float *P_ptr = (float *)P.data;
	while (infile >> *(P_ptr++));
	infile.close();
	return true;
}

bool SavePointsToFile(vector<Point2f> &src_points, vector<Point2f> &dst_points, const char* file)
{
	std::ofstream outfile(file, ios::out);

	for (auto i = 0; i < src_points.size(); ++i)
	{
		outfile << src_points[i].x << " " << src_points[i].y << " ";
		outfile << dst_points[i].x << " " << dst_points[i].y << " ";
		outfile << endl;
	}

	outfile.close();

	return true;
}


void TransformPointsWithIntrinsics(vector<Point2f> const &srcPointsIn, vector<Point2f> const &dstPointsIn, Mat K1, Mat K2, vector<Point2f> &srcPointsOut, vector<Point2f> &dstPointsOut)
{
	int N = srcPointsIn.size();

	Mat K1inv = K1.inv();
	Mat K2invt = K2.inv().t();

	srcPointsOut.resize(N);
	dstPointsOut.resize(N);

	for (int i = 0; i < N; ++i)
	{
		Mat pt1 = (Mat_<float>(3, 1) << srcPointsIn[i].x, srcPointsIn[i].y, 1);
		Mat pt2 = (Mat_<float>(1, 3) << dstPointsIn[i].x, dstPointsIn[i].y, 1);
		pt1 = K1inv * pt1;
		pt2 = (pt2 * K2invt);

		srcPointsOut[i].x = pt1.at<float>(0);
		srcPointsOut[i].y = pt1.at<float>(1);
		dstPointsOut[i].x = pt2.at<float>(0);
		dstPointsOut[i].y = pt2.at<float>(1);
	}
}

void ProjectionsFromEssential(const cv::Mat &E, cv::Mat &P1, cv::Mat &P2, cv::Mat &P3, cv::Mat &P4)
{
	// Assumes input E is a rank 2 matrix, with equal singular values
	cv::SVD svd(E);

	cv::Mat VT = svd.vt;
	cv::Mat V = svd.vt.t();
	cv::Mat U = svd.u;
	cv::Mat W = cv::Mat::zeros(3, 3, CV_32F);

	// Find rotation, translation
	W.at<float>(0, 1) = -1.0;
	W.at<float>(1, 0) = 1.0;
	W.at<float>(2, 2) = 1.0;

	// P1, P2, P3, P4
	P1 = cv::Mat::eye(3, 4, CV_32F);
	P2 = cv::Mat::eye(3, 4, CV_32F);
	P3 = cv::Mat::eye(3, 4, CV_32F);
	P4 = cv::Mat::eye(3, 4, CV_32F);

	// Rotation
	P1(cv::Range(0, 3), cv::Range(0, 3)) = U*W*VT;
	P2(cv::Range(0, 3), cv::Range(0, 3)) = U*W*VT;
	P3(cv::Range(0, 3), cv::Range(0, 3)) = U*W.t()*VT;
	P4(cv::Range(0, 3), cv::Range(0, 3)) = U*W.t()*VT;

	// Translation
	P1(cv::Range::all(), cv::Range(3, 4)) = U(cv::Range::all(), cv::Range(2, 3)) * 1;
	P2(cv::Range::all(), cv::Range(3, 4)) = -U(cv::Range::all(), cv::Range(2, 3));
	P3(cv::Range::all(), cv::Range(3, 4)) = U(cv::Range::all(), cv::Range(2, 3)) * 1;
	P4(cv::Range::all(), cv::Range(3, 4)) = -U(cv::Range::all(), cv::Range(2, 3));
}

double GetEssentialError(Mat E, Mat K1, Mat K2, Mat P1, Mat P2, float &err_t, float &err_R)
{
	if (E.rows != 3)
		return 0;
	
	Mat F_gt;
	GetFundamentalFromPerspective(P1, P2, F_gt);
	F_gt = F_gt / F_gt.at<float>(2, 2);

	Mat E_gt = K2.t() * F_gt * K1;
	  
	Mat R1_gt, R2_gt, t_gt;
	cv::decomposeEssentialMat(E_gt, R1_gt, R2_gt, t_gt);

	Mat R1, R2, t;
	cv::decomposeEssentialMat(E, R1, R2, t);
	
	// Get translation error
	err_t = FLT_MAX;

	t_gt = t_gt / norm(t_gt);
	t = t / norm(t);

	err_t = MIN(err_t, acos(t_gt.dot(t)));
	err_t = MIN(err_t, acos(t_gt.dot(-t)));
	err_t = MIN(err_t, acos((-t_gt).dot(t)));
	err_t = MIN(err_t, acos((-t_gt).dot(-t)));

	err_t = MIN(err_t, acos((R1_gt * t_gt).dot(R1 * t)));
	err_t = MIN(err_t, acos((R1_gt * t_gt).dot(-R1 * t)));
	err_t = MIN(err_t, acos((-R1_gt * t_gt).dot(R1 * t)));
	err_t = MIN(err_t, acos((-R1_gt * t_gt).dot(-R1 * t)));

	err_t = MIN(err_t, acos((R1_gt * t_gt).dot(R2 * t)));
	err_t = MIN(err_t, acos((R1_gt * t_gt).dot(-R2 * t)));
	err_t = MIN(err_t, acos((-R1_gt * t_gt).dot(R2 * t)));
	err_t = MIN(err_t, acos((-R1_gt * t_gt).dot(-R2 * t)));

	err_t = MIN(err_t, acos((R2_gt * t_gt).dot(R2 * t)));
	err_t = MIN(err_t, acos((R2_gt * t_gt).dot(-R2 * t)));
	err_t = MIN(err_t, acos((-R2_gt * t_gt).dot(R2 * t)));
	err_t = MIN(err_t, acos((-R2_gt * t_gt).dot(-R2 * t)));

	err_t = MIN(err_t, acos((R2_gt * t_gt).dot(R1 * t)));
	err_t = MIN(err_t, acos((R2_gt * t_gt).dot(-R1 * t)));
	err_t = MIN(err_t, acos((-R2_gt * t_gt).dot(R1 * t)));
	err_t = MIN(err_t, acos((-R2_gt * t_gt).dot(-R1 * t)));

	// Get rotation error
	err_R = FLT_MAX;

	Mat tpt = (Mat_<float>(3, 1) << 1, 1, 1);
	err_R = MIN(err_R, norm(R1_gt * tpt - R1 * tpt));
	err_R = MIN(err_R, norm(R1_gt * tpt - R2 * tpt));
	err_R = MIN(err_R, norm(R2_gt * tpt - R1 * tpt));
	err_R = MIN(err_R, norm(R2_gt * tpt - R2 * tpt));
	err_R = MIN(err_R, norm(R1_gt.inv() * tpt - R1.inv() * tpt));
	err_R = MIN(err_R, norm(R1_gt.inv() * tpt - R2.inv() * tpt));
	err_R = MIN(err_R, norm(R2_gt.inv() * tpt - R1.inv() * tpt));
	err_R = MIN(err_R, norm(R2_gt.inv() * tpt - R2.inv() * tpt));

	return 0;
}

void GetFundamentalFromPerspective(Mat P1, Mat P2, Mat &F)
{
	Mat e1;
	Mat mtm = P1.t() * P1;
	Mat evalues;
	Mat evectors;
	eigen(mtm, evalues, evectors);
	e1 = evectors.row(evectors.rows - 1);
	e1 = e1.t();

	Mat e2 = P2 * e1;

	Mat e2x = (Mat_<float>(3, 3) << 0, -e2.at<float>(2), e2.at<float>(1),
		e2.at<float>(2), 0, -e2.at<float>(0),
		-e2.at<float>(1), e2.at<float>(0), 0);

	F = e2x * P2 * P1.inv(DECOMP_SVD);
}
