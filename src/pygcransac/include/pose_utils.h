// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
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
#include <opencv2/core/eigen.hpp>
#include <Eigen/Eigen>

class CameraIntrinsics
{
protected:
	Eigen::Matrix3d intrinsicMatrix;
	double focalLengthX,
		focalLengthY,
		imageWidth,
		imageHeight,
		principalPointX,
		principalPointY;

public:
	CameraIntrinsics(
		const double &kFocalLengthX_,
		const double &kFocalLengthY_,
		const double &kPrincipalPointX_,
		const double &kPrincipalPointY_,
		const double &kImageWidth_,
		const double &kImageHeight_) : 
		focalLengthX(kFocalLengthX_),
		focalLengthY(kFocalLengthY_),
		principalPointX(kPrincipalPointX_),
		principalPointY(kPrincipalPointY_),
		imageWidth(kImageWidth_),
		imageHeight(kImageHeight_)
	{
		intrinsicMatrix << kFocalLengthX_, 0, kPrincipalPointX_,
			0, kFocalLengthY_, kPrincipalPointY_,
			0, 0, 1;
	}
	
	CameraIntrinsics(
		const double &kFocalLength_,
		const double &kPrincipalPointX_,
		const double &kPrincipalPointY_) : 
		CameraIntrinsics(
			kFocalLength_,
			kFocalLength_,
			kPrincipalPointX_,
			kPrincipalPointY_,
			kPrincipalPointX_ * 2,
			kPrincipalPointY_ * 2)
	{
		
	}

	CameraIntrinsics() : 
		CameraIntrinsics(1672, 1672, 960, 540, 2 * 960, 2 * 540)
	{

	}

	const Eigen::Matrix3d &matrix() const
	{
		return intrinsicMatrix;
	}
};

class CameraPose
{
protected:
	Eigen::Matrix3d rotationValue;
	Eigen::Vector3d positionValue;
	Eigen::Vector3d translationValue;

public:	
	CameraPose(const Eigen::Matrix3d &kRotation_,
		const Eigen::Vector3d &kPosition_) :
			rotationValue(kRotation_),
			positionValue(kPosition_),
			translationValue(-kRotation_ * kPosition_)
	{
	}

	CameraPose(const Eigen::Quaterniond &kRotationQuaternion_,
		const Eigen::Vector3d &kPosition_) :
			CameraPose(kRotationQuaternion_.toRotationMatrix(),
				kPosition_)
	{
	}

	CameraPose() :
			CameraPose(Eigen::Matrix3d::Identity(),
				Eigen::Vector3d::Zero())
	{
	}

	const Eigen::Matrix3d &rotation() const
	{
		return rotationValue;
	}

	const Eigen::Vector3d &position() const
	{
		return positionValue;
	}

	const Eigen::Vector3d &translation() const
	{
		return translationValue;
	}

	Eigen::Matrix<double, 3, 4> projectionMatrix() const
	{
		Eigen::Matrix<double, 3, 4> P;
		P << rotationValue, translationValue;
		return P;
	}

};

double rotationError(
	const Eigen::Matrix3d& kRotation1_,
	const Eigen::Matrix3d& kRotation2_)
{
	constexpr double kRadianToDegree = 180 / M_PI;
	Eigen::Matrix3d R2R1 =
		kRotation2_ * kRotation1_.transpose();

	const double kCosineAngle =
		std::max(std::min(1.0, 0.5 * (R2R1.trace() - 1.0)), -1.0);
	const double kAngle = std::acos(kCosineAngle);
	return kRadianToDegree * kAngle;
}

double translationError(
	const Eigen::Vector3d& kTranslation1_,
	const Eigen::Vector3d& kTranslation2_)
{
	constexpr double kRadianToDegree = 180 / M_PI;
	Eigen::Vector3d t1 = kTranslation1_.normalized();
	Eigen::Vector3d t2 = kTranslation2_.normalized();

	const double kCosineAngle = t1.dot(t2);
	const double kAngle = std::acos(std::max(std::min(kCosineAngle, 1.0), -1.0));
	return kAngle * kRadianToDegree;
}

void calculateRelativePoseError(
	const Eigen::Matrix3d &kEssentialMatrix_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const Eigen::Matrix3d &kRelativeRotation_,
	const Eigen::Vector3d &kRelativeTranslation_,
	double& rotationError_,
	double& translationError_)
{
	// Converting the essential matrix to OpenCV format
	cv::Mat essentialMatrixCV;
	cv::eigen2cv(kEssentialMatrix_, essentialMatrixCV);

	// Decomposing the essential matrix
	cv::Mat rotation1CV, rotation2CV, translationCV;
	cv::decomposeEssentialMat(essentialMatrixCV, rotation1CV, rotation2CV, translationCV);

	// Converting the decomposed rotations and translation to Eigen format
	Eigen::Matrix3d rotation1, rotation2;
	Eigen::Vector3d translation;

	cv::cv2eigen(rotation1CV, rotation1);
	cv::cv2eigen(rotation2CV, rotation2);
	cv::cv2eigen(translationCV, translation);

	// Calculating the pose error
	rotationError_ =
		MIN(rotationError_, 
		MIN(rotationError(kRelativeRotation_, rotation1), 
		rotationError(kRelativeRotation_, rotation2)));
	translationError_ =
		MIN(translationError_, 
		MIN(translationError(kRelativeTranslation_, translation), 
		translationError(kRelativeTranslation_, -translation)));
}

Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d& cross_vec) 
{
	Eigen::Matrix3d cross;
	cross << 0.0, -cross_vec.z(), cross_vec.y(),
		cross_vec.z(), 0.0, -cross_vec.x(),
		-cross_vec.y(), cross_vec.x(), 0.0;
	return cross;
}

// x = (R2 * X + t2)
// x = (R2 * R1^t * X + t2)
// x = (R2 * (R1^t * X - t1) + t2)
// x = R2 * R1^t * X - R2 * t1 + t2
void essentialMatrixFromTwoProjectionMatrices(
	const Eigen::Matrix<double, 3, 4>& projection_1_,
	const Eigen::Matrix<double, 3, 4>& projection_2_,
	Eigen::Matrix3d* essential_matrix) 
{
	// Create the Ematrix from the poses.
	const Eigen::Matrix3d R1 = projection_1_.leftCols<3>();
	const Eigen::Matrix3d R2 = projection_2_.leftCols<3>();
	const Eigen::Vector3d t1 = projection_1_.rightCols<1>();
	const Eigen::Vector3d t2 = projection_2_.rightCols<1>();

	// Pos1 = -R1^t * t1.
	// Pos2 = -R2^t * t2.
	// t = R1 * (pos2 - pos1).
	// t = R1 * (-R2^t * t2 + R1^t * t1)
	// t = t1 - R1 * R2^t * t2;

	// Relative transformation between to cameras.
	const Eigen::Matrix3d relative_rotation = R1 * R2.transpose();
	const Eigen::Vector3d translation = (t1 - relative_rotation * t2).normalized();
	*essential_matrix = crossProductMatrix(translation) * relative_rotation;
}

// Given either a fundamental or essential matrix and two corresponding images
// points such that ematrix * point2 produces a line in the first image,
// this method finds corrected image points such that
// corrected_point1^t * ematrix * corrected_point2 = 0.
void findOptimalImagePoints(
	const Eigen::Matrix3d& ematrix,
	const Eigen::Vector2d& point1,
	const Eigen::Vector2d& point2,
	Eigen::Vector2d &corrected_point1,
	Eigen::Vector2d &corrected_point2) 
{
	const Eigen::Vector3d point1_homog = point1.homogeneous();
	const Eigen::Vector3d point2_homog = point2.homogeneous();

	// A helper matrix to isolate certain coordinates.
	Eigen::Matrix<double, 2, 3> s_matrix;
	s_matrix << 1, 0, 0, 0, 1, 0;

	const Eigen::Matrix2d e_submatrix = ematrix.topLeftCorner<2, 2>();

	// The epipolar line from one image point in the other image.
	Eigen::Vector2d epipolar_line1 = s_matrix * ematrix * point2_homog;
	Eigen::Vector2d epipolar_line2 = s_matrix * ematrix.transpose() * point1_homog;

	const double a = epipolar_line1.transpose() * e_submatrix * epipolar_line2;
	const double b =
		(epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm()) / 2.0;
	const double c = point1_homog.transpose() * ematrix * point2_homog;

	const double d = sqrt(b * b - a * c);

	double lambda = c / (b + d);
	epipolar_line1 -= e_submatrix * lambda * epipolar_line1;
	epipolar_line2 -= e_submatrix.transpose() * lambda * epipolar_line2;

	lambda *=
		(2.0 * d) / (epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm());

	corrected_point1 =
		(point1_homog - s_matrix.transpose() * lambda * epipolar_line1)
			.hnormalized();
	corrected_point2 =
		(point2_homog - s_matrix.transpose() * lambda * epipolar_line2)
			.hnormalized();
}

bool linearTriangulation(
	const Eigen::Matrix<double, 3, 4>& projection_1_,
	const Eigen::Matrix<double, 3, 4>& projection_2_,
	const cv::Mat& point_,
	Eigen::Vector4d& triangulated_point_) 
{
	Eigen::Matrix4d design_matrix;
	design_matrix.row(0) = point_.at<double>(0) * projection_1_.row(2) - projection_1_.row(0);
	design_matrix.row(1) = point_.at<double>(1) * projection_1_.row(2) - projection_1_.row(1);
	design_matrix.row(2) = point_.at<double>(2) * projection_2_.row(2) - projection_2_.row(0);
	design_matrix.row(3) = point_.at<double>(3) * projection_2_.row(2) - projection_2_.row(1);

	// Extract nullspace.
	triangulated_point_ = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	return true;
}

bool linearTriangulation(
	const Eigen::Matrix<double, 3, 4>& projection_1_,
	const Eigen::Matrix<double, 3, 4>& projection_2_,
	const Eigen::Vector2d point1_,
	const Eigen::Vector2d point2_,
	Eigen::Vector4d& triangulated_point_) 
{
	Eigen::Matrix4d design_matrix;
	design_matrix.row(0) = point1_(0) * projection_1_.row(2) - projection_1_.row(0);
	design_matrix.row(1) = point1_(1) * projection_1_.row(2) - projection_1_.row(1);
	design_matrix.row(2) = point2_(0) * projection_2_.row(2) - projection_2_.row(0);
	design_matrix.row(3) = point2_(1) * projection_2_.row(2) - projection_2_.row(1);

	// Extract nullspace.
	triangulated_point_ = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	return true;
}

// Triangulates 2 posed views
bool optimalTriangulation(
	const Eigen::Matrix<double, 3, 4>& projection_1_,
	const Eigen::Matrix<double, 3, 4>& projection_2_,
	const cv::Mat& point_,
	Eigen::Vector4d& triangulated_point_) 
{
	Eigen::Vector2d point1, point2;
	point1 << point_.at<double>(0), point_.at<double>(1);
	point2 << point_.at<double>(2), point_.at<double>(3);

	Eigen::Matrix3d ematrix;
	essentialMatrixFromTwoProjectionMatrices(projection_1_, projection_2_, &ematrix);

	Eigen::Vector2d corrected_point1, corrected_point2;
	findOptimalImagePoints(
		ematrix, point1, point2, corrected_point1, corrected_point2);

	// Now the two points are guaranteed to intersect. We can use the DLT method
	// since it is easy to construct.
	return linearTriangulation(
		projection_1_, projection_2_, 
		corrected_point1, corrected_point2, 
		triangulated_point_);
}