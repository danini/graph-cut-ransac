// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "solver_engine.h"
#include "perspective_n_point_estimator.h"
#include "solver_p3p.h"
#include "solver_dls_pnp.h"
#include <iostream>

namespace gcransac
{
	namespace estimator
	{
		namespace solver
		{
			// This is the estimator class for estimating a homography matrix between two images. A model estimation method and error calculation method are implemented
			class PnPBundleAdjustment : public SolverEngine
			{
			protected:
				// The options for the bundle adjustment
				pose_lib::BundleOptions bundle_options;

			public:
				PnPBundleAdjustment(
					const pose_lib::BundleOptions::LossType &loss_type_ = pose_lib::BundleOptions::LossType::TRUNCATED,
					const size_t &maximum_iterations_ = 25)
				{
					bundle_options.loss_type = loss_type_;
					bundle_options.max_iterations = maximum_iterations_;
				}

				~PnPBundleAdjustment()
				{
				}

				pose_lib::BundleOptions& getMutableOptions()
				{
					return bundle_options;
				}

				const pose_lib::BundleOptions& getOptions() const
				{
					return bundle_options;
				}

				// Determines if there is a chance of returning multiple models
				// the function 'estimateModel' is applied.
				static constexpr bool returnMultipleModels()
				{
					return maximumSolutions() > 1;
				}

				// The maximum number of solutions returned by the estimator
				static constexpr size_t maximumSolutions()
				{
					return 4;
				}

				// The minimum number of points required for the estimation
				static constexpr size_t sampleSize()
				{
					return 4;
				}

				// It returns true/false depending on if the solver needs the gravity direction
				// for the model estimation. 
				static constexpr bool needsGravity()
				{
					return false;
				}

				// Estimate the model parameters from the given point sample
				// using weighted fitting if possible.
				OLGA_INLINE bool estimateModel(
					const cv::Mat& data_, // The set of data points
					const size_t *sample_, // The sample used for the estimation
					size_t sample_number_, // The size of the sample
					std::vector<Model> &models_, // The estimated model parameters
					const double *weights_ = nullptr) const; // The weight for each point
			};
			
			OLGA_INLINE bool PnPBundleAdjustment::estimateModel(
				const cv::Mat& data_,
				const size_t *sample_,
				size_t sample_number_,
				std::vector<Model> &models_,
				const double *weights_) const
			{
				// Check if we have enough points for the bundle adjustment
				if (sample_number_ < sampleSize())
					return false;

				// If no sample is provided use all points
				if (sample_ == nullptr)
					sample_number_ = data_.rows;

				// If there is no initial model provided estimate one
				std::vector<Model> temp_models;
				if (models_.size() == 0)
				{
					cv::Mat inlier_image_points(sample_number_, 2, CV_64F),
						inlier_object_points(sample_number_, 3, CV_64F);
							
					Eigen::Matrix3d rotation;
					Eigen::Vector3d translation;

					for (size_t i = 0; i < sample_number_; ++i)
					{
						const size_t idx = 
							sample_ == nullptr ? i : sample_[i];
						inlier_image_points.at<double>(i, 0) = data_.at<double>(idx, 0);
						inlier_image_points.at<double>(i, 1) = data_.at<double>(idx, 1);
						inlier_object_points.at<double>(i, 0) = data_.at<double>(idx, 2);
						inlier_object_points.at<double>(i, 1) = data_.at<double>(idx, 3);
						inlier_object_points.at<double>(i, 2) = data_.at<double>(idx, 4);
					}

					cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
						cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format
					cv::Mat cv_rodrigues;

					// Applying numerical optimization to the estimated pose parameters
					cv::solvePnP(inlier_object_points, // The object points
						inlier_image_points, // The image points
						cv::Mat::eye(3, 3, CV_64F), // The camera's intrinsic parameters 
						cv::Mat(), // An empty vector since the radial distortion is not known
						cv_rodrigues, // The initial rotation
						cv_translation, // The initial translation
						false, // Use the initial values
						cv::SOLVEPNP_EPNP); // Apply numerical refinement
			
					// Convert the rotation vector back to a rotation matrix
					cv::Rodrigues(cv_rodrigues, cv_rotation);

					// Transpose the rotation matrix back
					cv_rotation = cv_rotation.t();

					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << rotation, translation;
					temp_models.emplace_back(model);

					/*estimator::solver::DLSPnP solver;
					solver.estimateModel(data_, // All point correspondences
						sample_, // The sample, i.e., indices of points to be used
						sample_number_, // The size of the sample
						temp_models, // The estimated model parameters
						weights_); // The weights used for the estimation*/

					/*size_t temp_sample[3];
					for (size_t rep = 0; rep < 20; ++rep)
					{
						temp_sample[0] = round(static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (sample_number_ - 1));
						temp_sample[1] = round(static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (sample_number_ - 1));
						temp_sample[2] = round(static_cast<double>(rand()) / static_cast<double>(RAND_MAX) * (sample_number_ - 1));

						temp_sample[0] = sample_[temp_sample[0]];
						temp_sample[1] = sample_[temp_sample[1]];
						temp_sample[2] = sample_[temp_sample[2]];

						estimator::solver::P3PSolver solver;
						solver.estimateModel(data_, // All point correspondences
							temp_sample, // The sample, i.e., indices of points to be used
							3, // The size of the sample
							temp_models, // The estimated model parameters
							weights_); // The weights used for the estimation
					}*/
				} else
					temp_models = models_;
				models_.clear();

				// Iterating through the possible models.
				// This is 1 if the eight-point solver is used.
				// Otherwise, it is up to 3. 
				for (auto& temp_model : temp_models)
				{					
					const Eigen::Matrix3d &R = temp_model.descriptor.block<3, 3>(0, 0);
					const Eigen::Vector3d &t = temp_model.descriptor.rightCols<1>();

					pose_lib::CameraPose pose(R, t);	

					// Iterating through the possible poses and optimizing each
					// Apply bundle adjustment
					pose_lib::refine_pnp(
						data_, // All point correspondences
						sample_, // The sample, i.e., indices of points to be used
						sample_number_, // The size of the sample
						&pose, // The optimized pose
						bundle_options, // The bundle adjustment options
						weights_); // The weights for the weighted LSQ fitting
				
					Model model;
					model.descriptor.resize(3, 4);
					model.descriptor << pose.R(), pose.t;
					models_.emplace_back(model);
				}

				return models_.size();
			}
		}
	}
}