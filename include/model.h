#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <Eigen/Eigen>
#include "estimator.h"

class Model
{
public:
	std::vector<int> inliers; // Inliers of the current model
	Eigen::MatrixXd descriptor; // The descriptor of the current model
	const theia::Estimator<cv::Mat, Model> * estimator; // The pointer of the estimator which obtained the current model
	bool finalizable; // Is the model finalizable
	double probability; // The probability of the model being a good one
	double preference_vector_length;
	cv::Mat preference_vector;
	double tanimoto_distance, eucledian_distance;

	void setEstimator(const theia::Estimator<cv::Mat, Model> * estimator_)
	{
		estimator = estimator_;
	}

	Model(const Eigen::MatrixXd &descriptor_) : 
		descriptor(descriptor_),
		finalizable(true)
	{

	}

	Model() : finalizable(true)
	{

	}
};

class FundamentalMatrix : public Model
{
public:
	FundamentalMatrix() :
		Model(Eigen::MatrixXd(3, 3))
	{}
	FundamentalMatrix(const FundamentalMatrix& other)
	{
		descriptor = other.descriptor;
	}
};

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

class Homography : public Model
{
public:
	Homography() :
		Model(Eigen::MatrixXd(3, 3))
	{}

	Homography(const Homography& other)
	{
		descriptor = other.descriptor;
	}
};