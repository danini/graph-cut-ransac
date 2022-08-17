#include <vector>	
#include <map>
#include <thread>
#include <queue>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>
#include "utils.h"
#include "pose_utils.h"
#include "data_utils.h"

#include "GCRANSAC.h"

#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/single_point_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/napsac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/rigid_transformation_estimator.h"

#include "preemption/preemption_sprt.h"

#include "inlier_selectors/empty_inlier_selector.h"
#include "inlier_selectors/space_partitioning_ransac.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_rigid_transformation_svd.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_p3p.h"
#include "estimators/solver_acp1p.h"
#include "estimators/solver_acp1p_cayley.h"
#include "estimators/solver_dls_pnp.h"
#include "estimators/solver_epnp_lm.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif 

void loadOrientedPoints(
	const std::string &kPath_,
	std::vector<double> &values_);

void loadImagePairs(
	const std::string &kPath_,
	std::vector<std::pair<std::string, std::string>> &imagePairs_);

template <size_t _HeaderLines>
void loadPoses(
	const std::string &kPath_,
	std::map<std::string, CameraPose> &poses_);

void loadCalibrations(
	const std::string &kPath_,
	std::map<std::string, CameraIntrinsics> &calibrations_);

void processPair(
	const std::string &kScene_,
	const std::string &kDatasetPath_,
	const std::string &k3DDatabasePath_,
	const std::string &kCorrespondencePath_,
	const std::string &kSourceImageName_,
	const std::string &kDestinationImageName_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const CameraIntrinsics &kSourceIntrinsics_,
	const CameraIntrinsics &kDestinationIntrinsics_,
	const cv::Mat &kPoint3d_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_,
	const double &kNormalAreaSize_,
	const double &kPointAssignmentThreshold_);

template <size_t _Method>
void selectOrientedPoints(
	const std::string &kSourceImageName_,
	const std::string &kDestinationImageName_,
	const std::string &kDatabaseName_,
	const double &kSNNThreshold_,
	const cv::Mat &kCorrespondences_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const CameraIntrinsics &kSourceIntrinsics_,
	const CameraIntrinsics &kDestinationIntrinsics_,
	const cv::Mat &kPoints3d_,
	const double &kThreshold_,
	std::vector<int> &assignment);
	
template<typename _MinimalSolver>
void relativePoseEstimation(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_);

template<typename _MinimalSolver>
void absolutePoseEstimation(
	const std::string &kScene_,
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kPoint3d_,
	const std::vector<int> &assignment2D3D_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_,
	const size_t &kNormalAreaSize_,
	const size_t &kSpatialWeight_,
	const bool kNormalDirection_);

using namespace gcransac;

std::mutex savingMutex;
const size_t kCoreNumber = 10;

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	// Parameters
	constexpr size_t kFeatureNumber = 8000,
		kNormalAreaSize = 100,
		kCoreNumber = 18;

	constexpr double kSNNThreshold = 0.85,	
		kPointAssignmentThreshold = 0.5,
		kInlierOutlierThreshold = 1.5;

	const std::vector<std::string> scenes = 
		{ "StMarysChurch", "OldHospital", "KingsCollege", "ShopFacade" }; 

	for (const auto &scene : scenes)
	{
		for (size_t kNormalAreaSize : { 200 })
		{
			const std::string datasetPath = "/media/hdd3tb/datasets/absolute_pose/" + scene + "/",
				filenamePoints = "points3D_" + std::to_string(kNormalAreaSize) + ".txt",
				filenameCalibrations = "model_train/calibrations.txt",
				filenamePairs = scene + "_top20.txt",
				filename3DAssignment = "point3d_assignments_8000_upright.h5",
				filenameCorrs = scene + "_ACs8000_upright.h5";
			std::vector<std::string> filenamesPose = { scene + "/dataset_train.txt", 
				scene + "/dataset_test.txt" }; 

			// Load the 3D point cloud
			printf("Loading 3D oriented point cloud...\n");
			constexpr size_t kOrientedPointDimensions = 2 * 3;
			std::vector<double> point3DData;
			loadOrientedPoints(datasetPath + filenamePoints,
				point3DData);
			const size_t k3DPointNumber = point3DData.size() / kOrientedPointDimensions;
			cv::Mat points3d(k3DPointNumber, 
				kOrientedPointDimensions, 
				CV_64F, 
				&point3DData[0]);

			// Load the image pairs to be tested
			printf("Loading image pairs...\n");
			std::vector<std::pair<std::string, std::string>> imagePairs;
			loadImagePairs(datasetPath + filenamePairs,
				imagePairs);

			std::queue<size_t> processingQueue;
			for (size_t pairIdx = 0; pairIdx < imagePairs.size(); ++pairIdx)
				processingQueue.emplace(pairIdx);
			std::mutex queueMutex;

			// Load the calibration matrices
			printf("Loading calibrations...\n");
			std::map<std::string, CameraIntrinsics> calibrations;
			loadCalibrations(datasetPath + filenameCalibrations,
				calibrations);

			// Load the camera poses
			printf("Loading camera poses...\n");
			std::map<std::string, CameraPose> poses;
			for (const auto &filename : filenamesPose)
				loadPoses<3>(
					datasetPath + filename,
					poses);

			printf("Processing image pairs...\n");
#pragma omp parallel for num_threads(kCoreNumber)
			for (int coreIdx = 0; coreIdx < kCoreNumber; ++coreIdx)
			//for (int imagePairIdx = 0; imagePairIdx < imagePairs.size(); ++imagePairIdx)
			//for (const auto &[sourceImageName, destinationImageName] : imagePairs)
			{
				while (!processingQueue.empty())
				{
					queueMutex.lock();
					if (processingQueue.empty())
						break;
					size_t imagePairIdx = processingQueue.front();
					processingQueue.pop();
					queueMutex.unlock();

					const auto &[sourceImageName, destinationImageName] = imagePairs[imagePairIdx];

					if (poses.find(sourceImageName) == std::end(poses))
					{
						fprintf(stderr, "The pose of image '%s' is unknown.\n", sourceImageName.c_str());	
						continue;
					} 
					
					if (poses.find(destinationImageName) == std::end(poses))
					{
						fprintf(stderr, "The pose of image '%s' is unknown.\n", destinationImageName.c_str());	
						continue;
					} 
					
					if (calibrations.find(sourceImageName) == std::end(calibrations))
					{
						calibrations[sourceImageName] = 
							CameraIntrinsics(1660, 960, 540);
						//fprintf(stderr, "The intrinsic parameters of image '%s' is unknown.\n", sourceImageName.c_str());	
						//continue;
					} 
					
					if (calibrations.find(destinationImageName) == std::end(calibrations))
					{
						calibrations[destinationImageName] = 
							CameraIntrinsics(1660, 960, 540);
						//fprintf(stderr, "The intrinsic parameters of image '%s' is unknown.\n", destinationImageName.c_str());	
						//continue;
					}

					for (double kSNNThreshold : { /*0.85, 0.90, 0.95,*/ 0.90, 0.95, 1.0 })
						for (double kInlierOutlierThreshold : { 4.5 })
							processPair(
								scene,
								datasetPath,
								datasetPath + filename3DAssignment,
								datasetPath + filenameCorrs,
								sourceImageName,
								destinationImageName,
								poses[sourceImageName],
								poses[destinationImageName],
								calibrations[sourceImageName],
								calibrations[destinationImageName],
								points3d,
								kSNNThreshold,
								kInlierOutlierThreshold,
								kNormalAreaSize,
								kPointAssignmentThreshold);
				}
			}
		}
	}
	return 0;
}

void processPair(
	const std::string &kScene_,
	const std::string &kDatasetPath_,
	const std::string &k3DDatabasePath_,
	const std::string &kCorrespondencePath_,
	const std::string &kSourceImageName_,
	const std::string &kDestinationImageName_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const CameraIntrinsics &kSourceIntrinsics_,
	const CameraIntrinsics &kDestinationIntrinsics_,
	const cv::Mat &kPoints3d_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_,
	const double &kNormalAreaSize_,
	const double &kPointAssignmentThreshold_)
{
	printf("Process pair '%s' and '%s'\n", 
		kSourceImageName_.c_str(), 
		kDestinationImageName_.c_str());

	// Read affine correspondences from the database
	cv::Mat correspondences;	
	datareading::readData(kCorrespondencePath_,
		kSourceImageName_ + "-" + kDestinationImageName_,
		correspondences);
	const size_t kPointNumber = correspondences.rows;

	// Filter the correspondences by the SNN ratio
	std::vector<size_t> filteredIndices;
	filteredIndices.reserve(correspondences.rows);
	for (size_t pointIdx = 0; pointIdx < correspondences.rows; ++pointIdx)
		if (correspondences.at<double>(pointIdx, 8) < kSNNThreshold_)
			filteredIndices.emplace_back(pointIdx);
	cv::Mat filteredCorrespondences(filteredIndices.size(), correspondences.cols, correspondences.type());
	for (size_t idx = 0; idx < filteredIndices.size(); ++idx)
		correspondences.row(filteredIndices[idx]).copyTo(filteredCorrespondences.row(idx));


	// Normalize the correspondences
	cv::Mat normalizedCorrespondences(filteredCorrespondences.size(), filteredCorrespondences.type());	
	gcransac::utils::normalizeCorrespondences(
		filteredCorrespondences,
		kSourceIntrinsics_.matrix(),
		kDestinationIntrinsics_.matrix(),
		normalizedCorrespondences);

	// Calculate the threshold normalizer
	const double kThresholdNormalizer = 
		0.25 * (kSourceIntrinsics_.matrix()(0, 0) + kSourceIntrinsics_.matrix()(1, 1) + kDestinationIntrinsics_.matrix()(0, 0) + kDestinationIntrinsics_.matrix()(1, 1));
		

	// Select a 3D oriented point for each correspondence
	std::vector<int> pointAssignment;
	selectOrientedPoints<2>(
		kSourceImageName_,
		kDestinationImageName_,
		k3DDatabasePath_,
		kSNNThreshold_,
		normalizedCorrespondences,
		kSourcePose_,
		kDestinationPose_,
		kSourceIntrinsics_,
		kDestinationIntrinsics_,
		kPoints3d_,
		kInlierOutlierThreshold_ / kThresholdNormalizer,
		pointAssignment);

	// Relative pose estimation from point correspondences
	/*relativePoseEstimation<gcransac::estimator::solver::EssentialMatrixFivePointNisterSolver>(
		filteredCorrespondences,
		normalizedCorrespondences,
		kSourcePose_,
		kDestinationPose_,
		kSourceIntrinsics_.matrix(),
		kDestinationIntrinsics_.matrix(),
		kSNNThreshold_,
		kInlierOutlierThreshold_);

	// Relative pose estimation from affine correspondences
	relativePoseEstimation<gcransac::estimator::solver::EssentialMatrixTwoAffineSolver>(
		filteredCorrespondences,
		normalizedCorrespondences,
		kSourcePose_,
		kDestinationPose_,
		kSourceIntrinsics_.matrix(),
		kDestinationIntrinsics_.matrix(),
		kSNNThreshold_,
		kInlierOutlierThreshold_);*/

	// Absolute pose by P3P
	if (kSNNThreshold_ == 0.9)
		absolutePoseEstimation<gcransac::estimator::solver::P3PSolver>(
			kScene_,
			filteredCorrespondences,
			normalizedCorrespondences,
			kPoints3d_,
			pointAssignment, 
			kSourcePose_,
			kDestinationPose_,
			kSourceIntrinsics_.matrix(),
			kDestinationIntrinsics_.matrix(),
			kSNNThreshold_,
			kInlierOutlierThreshold_,
			10,
			0.0,
			0);

	// Absolute pose by affine correspondences
	absolutePoseEstimation<gcransac::estimator::solver::ACP1PCayleySolver>(
		kScene_,
		filteredCorrespondences,
		normalizedCorrespondences,
		kPoints3d_,
		pointAssignment, 
		kSourcePose_,
		kDestinationPose_,
		kSourceIntrinsics_.matrix(),
		kDestinationIntrinsics_.matrix(),
		kSNNThreshold_,
		kInlierOutlierThreshold_,
		kNormalAreaSize_,
		0.0,
		0);

	/*absolutePoseEstimation<gcransac::estimator::solver::ACP1PCayleySolver>(
		kScene_,
		filteredCorrespondences,
		normalizedCorrespondences,
		kPoints3d_,
		pointAssignment, 
		kSourcePose_,
		kDestinationPose_,
		kSourceIntrinsics_.matrix(),
		kDestinationIntrinsics_.matrix(),
		kSNNThreshold_,
		kInlierOutlierThreshold_,
		kNormalAreaSize_,
		0.4,
		0);*/
}

template<typename _MinimalSolver>
void absolutePoseEstimation(
	const std::string &kScene_,
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kPoint3d_,
	const std::vector<int> &assignment2D3D_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_,
	const size_t &kNormalAreaSize_,
	const size_t &kSpatialWeight_,
	const bool kNormalDirection_)
{
	// Calculate the threshold normalizer
	const double kThresholdNormalizer = 
		0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));

	cv::Mat dataPoints;
	const size_t kPointNumber = kNormalizedCorrespondences_.rows;
	size_t max_iteration_number = 1000,
		min_iteration_number = 1000,
		lo_number = 50;

	// Count how many 2D-3D assignments are found
	std::vector<size_t> validAssignments;
	validAssignments.reserve(kPointNumber);
	for (size_t pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
		if (assignment2D3D_[pointIdx] > -1)
			validAssignments.emplace_back(pointIdx);
	const size_t kValidAssignmentNumber = validAssignments.size();
	
	// The default estimator for PnP fitting
	typedef estimator::PerspectiveNPointEstimator<_MinimalSolver, // The solver used for fitting a model to a minimal sample
		estimator::solver::DLSPnP> // The solver used for fitting a model to a non-minimal sample
		PnPEstimator;

	Pose6D model; // The estimated model parameters

	// Initialize the samplers
	// The main sampler is used for sampling in the main RANSAC loop
	typedef sampler::Sampler<cv::Mat, size_t> AbstractSampler;
	std::unique_ptr<AbstractSampler> main_sampler;
		
	const Eigen::Matrix3d kInverseSourceRotation = 
		kSourcePose_.rotation().transpose();
	const Eigen::Matrix3d kInverseDestinationRotation = 
		kDestinationPose_.rotation().transpose();
	const Eigen::Vector3d &kDestinationPosition = 
		kDestinationPose_.position();

	const Eigen::Matrix<double, 3, 4> kProjectionSource = kSourcePose_.projectionMatrix(),
		kProjectionDestination = kDestinationPose_.projectionMatrix();

	cv::Mat transformedPoints3d(kPoint3d_.size(), kPoint3d_.type());
	for (size_t idx = 0; idx < kPoint3d_.rows; ++idx)
	{
		Eigen::Vector3d p, n;
		p << 
			kPoint3d_.at<double>(idx, 0), 
			kPoint3d_.at<double>(idx, 1), 
			kPoint3d_.at<double>(idx, 2);
		n << 
			kPoint3d_.at<double>(idx, 3), 
			kPoint3d_.at<double>(idx, 4), 
			kPoint3d_.at<double>(idx, 5);

		p = kSourcePose_.rotation() * p + kSourcePose_.translation();
		n = kSourcePose_.rotation() * n;

		// Invert the normal direction if it does not point towards the camera
		Eigen::Vector3d ppn = p + n,
			pmn = p - n;

		if (kNormalDirection_)
		{
			if (ppn.squaredNorm() > pmn.squaredNorm())
				n *= -1;
		} else
		{
			if (ppn.squaredNorm() < pmn.squaredNorm())
				n *= -1;
		}

		transformedPoints3d.at<double>(idx, 0) = p(0);
		transformedPoints3d.at<double>(idx, 1) = p(1);
		transformedPoints3d.at<double>(idx, 2) = p(2);
		
		transformedPoints3d.at<double>(idx, 3) = n(0);
		transformedPoints3d.at<double>(idx, 4) = n(1);
		transformedPoints3d.at<double>(idx, 5) = n(2);
	}

	// Doing standard PnP
	cv::Mat pointsForNeighborhood(kValidAssignmentNumber, 5, kPoint3d_.type());
	if constexpr (std::is_same<_MinimalSolver, gcransac::estimator::solver::P3PSolver>())
	{
		// Initialize the data points
		dataPoints.create(kValidAssignmentNumber, 5, CV_64F);
		
		for (size_t idx = 0; idx < kValidAssignmentNumber; ++idx)
		{
			const size_t &pointIdx = validAssignments[idx];
			const int assignment = assignment2D3D_[pointIdx];			
			double depth = 
				transformedPoints3d.at<double>(assignment, 2);
			dataPoints.at<double>(idx, 0) = kNormalizedCorrespondences_.at<double>(pointIdx, 2);
			dataPoints.at<double>(idx, 1) = kNormalizedCorrespondences_.at<double>(pointIdx, 3);
			dataPoints.at<double>(idx, 2) = depth * kNormalizedCorrespondences_.at<double>(pointIdx, 0); // kPoint3d_.at<double>(assignment, 0);
			dataPoints.at<double>(idx, 3) = depth * kNormalizedCorrespondences_.at<double>(pointIdx, 1); // kPoint3d_.at<double>(assignment, 1);
			dataPoints.at<double>(idx, 4) = depth;
			
			// Saving the points for neighborhood calculation
			pointsForNeighborhood.at<double>(idx, 0) = kCorrespondences_.at<double>(pointIdx, 2);
			pointsForNeighborhood.at<double>(idx, 1) = kCorrespondences_.at<double>(pointIdx, 3);
			pointsForNeighborhood.at<double>(idx, 2) = 100 * depth * kNormalizedCorrespondences_.at<double>(pointIdx, 0); // kPoint3d_.at<double>(assignment, 0);
			pointsForNeighborhood.at<double>(idx, 3) = 100 * depth * kNormalizedCorrespondences_.at<double>(pointIdx, 1); // kPoint3d_.at<double>(assignment, 1);
			pointsForNeighborhood.at<double>(idx, 4) = 100 * depth;
		}

		// The sampler is used inside the local optimization
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::UniformSampler(&dataPoints));
	} else if (std::is_same<_MinimalSolver, gcransac::estimator::solver::ACP1PCayleySolver>())
	{
		// Initialize the data points
		dataPoints.create(kValidAssignmentNumber, 12, CV_64F);
		
		for (size_t idx = 0; idx < kValidAssignmentNumber; ++idx)
		{
			const size_t &pointIdx = validAssignments[idx];
			const int assignment = assignment2D3D_[pointIdx];
			
			// Reference image = Source image
			// Query image = Destination image
			Eigen::Vector3d projectedPoint, normal, point3d;
			Eigen::Matrix2d A;
			point3d << 
				transformedPoints3d.at<double>(assignment, 0), 
				transformedPoints3d.at<double>(assignment, 1), 
				transformedPoints3d.at<double>(assignment, 2);

			// Projective depth in the source (reference) image
			//projectedPoint = 
			//	kProjectionSource * point3d.homogeneous();
			double depth = point3d(2);

			// Normal in the coordinate system of the destination (reference) image
			normal << 
				transformedPoints3d.at<double>(assignment, 3), 
				transformedPoints3d.at<double>(assignment, 4), 
				transformedPoints3d.at<double>(assignment, 5);
			// normal = kSourcePose_.rotation() * normal;

			// Affine frame from the source to the destination image
			A << kNormalizedCorrespondences_.at<double>(pointIdx, 4), kNormalizedCorrespondences_.at<double>(pointIdx, 5),
				kNormalizedCorrespondences_.at<double>(pointIdx, 6), kNormalizedCorrespondences_.at<double>(pointIdx, 7);

			dataPoints.at<double>(idx, 0) = kNormalizedCorrespondences_.at<double>(pointIdx, 2);
			dataPoints.at<double>(idx, 1) = kNormalizedCorrespondences_.at<double>(pointIdx, 3);
			dataPoints.at<double>(idx, 2) = depth * kNormalizedCorrespondences_.at<double>(pointIdx, 0); // kPoint3d_.at<double>(assignment, 0);
			dataPoints.at<double>(idx, 3) = depth * kNormalizedCorrespondences_.at<double>(pointIdx, 1); // kPoint3d_.at<double>(assignment, 1);
			dataPoints.at<double>(idx, 4) = depth;
			dataPoints.at<double>(idx, 5) = normal(0);
			dataPoints.at<double>(idx, 6) = normal(1);
			dataPoints.at<double>(idx, 7) = normal(2);
			dataPoints.at<double>(idx, 8) =  A(0, 0);
			dataPoints.at<double>(idx, 9) =  A(0, 1);
			dataPoints.at<double>(idx, 10) = A(1, 0);
			dataPoints.at<double>(idx, 11) = A(1, 1);
			
			// Saving the points for neighborhood calculation
			pointsForNeighborhood.at<double>(idx, 0) = kCorrespondences_.at<double>(pointIdx, 2);
			pointsForNeighborhood.at<double>(idx, 1) = kCorrespondences_.at<double>(pointIdx, 3);
			pointsForNeighborhood.at<double>(idx, 2) = 100 * depth * kNormalizedCorrespondences_.at<double>(pointIdx, 0); // kPoint3d_.at<double>(assignment, 0);
			pointsForNeighborhood.at<double>(idx, 3) = 100 * depth * kNormalizedCorrespondences_.at<double>(pointIdx, 1); // kPoint3d_.at<double>(assignment, 1);
			pointsForNeighborhood.at<double>(idx, 4) = 100 * depth;
		}

		// Initialize the samplers
		main_sampler = std::unique_ptr<AbstractSampler>(new sampler::SinglePointSampler(&dataPoints, 1));

		//min_iteration_number = kValidAssignmentNumber;
		//max_iteration_number = kValidAssignmentNumber;
		lo_number = 100;
	}

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	neighborhood::FlannNeighborhoodGraph neighborhood(&pointsForNeighborhood, 20.0);

	// Apply Graph-cut RANSAC
	PnPEstimator estimator; // The estimator used for the pose fitting

	sampler::UniformSampler local_optimization_sampler(&dataPoints); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!main_sampler->isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<PnPEstimator> sprt_verification(
		dataPoints,
		estimator);

	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<PnPEstimator, 
		neighborhood::FlannNeighborhoodGraph> inlier_selector(&neighborhood);

	GCRANSAC<PnPEstimator,
		neighborhood::FlannNeighborhoodGraph,
		MSACScoringFunction<PnPEstimator>,
		preemption::SPRTPreemptiveVerfication<PnPEstimator>> gcransac;
	gcransac.settings.threshold = kInlierOutlierThreshold_ / kThresholdNormalizer; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = 0.999; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = lo_number;
	gcransac.settings.max_iteration_number = max_iteration_number; // The maximum number of iterations
	gcransac.settings.min_iteration_number = min_iteration_number; // The minimum number of iterations

	// Start GC-RANSAC
	gcransac.run(dataPoints, // The normalized points
		estimator,  // The estimator
		main_sampler.get(), // The sample used for selecting minimal samples in the main iteration
		&local_optimization_sampler, // The sampler used for selecting a minimal sample when doing the local optimization
		&neighborhood, // The neighborhood-graph
		model, // The obtained model parameters
		sprt_verification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	// Doing BA as a last step
	if (statistics.inliers.size() > 3)
	{
		std::vector<Model> models = { model };
		estimator::solver::PnPBundleAdjustment bundleAdjustment;
		if (bundleAdjustment.estimateModel(
			kNormalizedCorrespondences_, // The set of data points
			&statistics.inliers[0], // The sample used for the estimation
			statistics.inliers.size(), // The size of the sample
			models)) // The estimated model parameters
			model.descriptor = models[0].descriptor;
	}

	// Calculate relative pose
	const Eigen::Matrix3d kRelativeRotation = 
		kDestinationPose_.rotation() * kSourcePose_.rotation().transpose();
	const Eigen::Vector3d kRelativeTranslation = 
		kDestinationPose_.rotation() * (kSourcePose_.position() - kDestinationPose_.position());

	// The pose error
	double rotError = 180.0,
		posError = 0.0,
		transError = 180.0;

	// Calculate the errors
	rotError = rotationError(
		model.descriptor.block<3, 3>(0, 0),
		kRelativeRotation);

	transError = translationError(
		model.descriptor.rightCols<1>(),
		kRelativeTranslation);

	posError = (model.descriptor.rightCols<1>() - kRelativeTranslation).norm();

	// Print the statistics
	printf("%d\t%.3f\t%.3f\t%.3f\t%d\t%.3f\n", 
		_MinimalSolver::sampleSize(),
		rotError, transError, posError,
		static_cast<int>(statistics.inliers.size()),
		statistics.processing_time);

	savingMutex.lock();
	std::ofstream file("results.csv", std::fstream::app);
	file << 
		kScene_ << ";" <<
		"DoG + AffNet + HardNet (upright) SPRT" << ";" <<
		_MinimalSolver::sampleSize() << ";" <<
		kSNNThreshold_ << ";" <<
		kInlierOutlierThreshold_ << ";" <<
		kSpatialWeight_ << ";" <<
		kNormalAreaSize_ << ";" <<
		kNormalDirection_ << ";" <<
		rotError << ";" <<
		transError << ";" <<
		posError << ";" <<
		static_cast<int>(statistics.inliers.size()) << ";" <<
		statistics.processing_time << "\n";
	savingMutex.unlock();

	AbstractSampler *sampler_ptr = main_sampler.release();
	delete sampler_ptr;
}

template<typename _MinimalSolver>
void relativePoseEstimation(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const double &kSNNThreshold_,
	const double &kInlierOutlierThreshold_)
{
	// Calculate the threshold normalizer
	const double kThresholdNormalizer = 
		0.25 * (kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) + kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1));

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	cv::Mat emptyPoints(0, 4, CV_64F);
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&emptyPoints,
		{ 0, 0, 0, 0 },
		1);

	// The default estimator for essential matrix fitting
	typedef estimator::EssentialMatrixEstimator<_MinimalSolver, // The solver used for fitting a model to a minimal sample
		estimator::solver::EssentialMatrixFivePointSteweniusSolver> // The solver used for fitting a model to a non-minimal sample
		EssentialMatrixEstimator;

	// Apply Graph-cut RANSAC
	EssentialMatrixEstimator estimator(kIntrinsicsSource_,
		kIntrinsicsDestination_);
	std::vector<int> inliers;
	EssentialMatrix model;

	// Initializing SPRT test
	preemption::SPRTPreemptiveVerfication<EssentialMatrixEstimator> preemptiveVerification(
		kCorrespondences_,
		estimator,
		0.001);

	// Initializing the fast inlier selector object
	inlier_selector::EmptyInlierSelector<EssentialMatrixEstimator, 
		neighborhood::GridNeighborhoodGraph<4>> inlier_selector(&neighborhood);

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	sampler::UniformSampler mainSampler(&kNormalizedCorrespondences_); // The main sampler
	sampler::UniformSampler localOptimizationSampler(&kNormalizedCorrespondences_); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!mainSampler.isInitialized() ||
		!localOptimizationSampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	GCRANSAC<EssentialMatrixEstimator,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<EssentialMatrixEstimator>,
		preemption::SPRTPreemptiveVerfication<EssentialMatrixEstimator>> gcransac;
	gcransac.settings.threshold = kInlierOutlierThreshold_ / kThresholdNormalizer; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = 0.0; // The weight of the spatial coherence term
	gcransac.settings.confidence = 0.999; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = 1000; // The maximum number of iterations
	gcransac.settings.min_iteration_number = 1000; // The minimum number of iterations

	// Start GC-RANSAC
	gcransac.run(kNormalizedCorrespondences_,
		estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification,
		inlier_selector);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

    // The pose error
    double rotationError = 180.0,
		translationError = 180.0;

	// Calculate relative pose
	const Eigen::Matrix3d kRelativeRotation = 
		kDestinationPose_.rotation() * kSourcePose_.rotation().transpose();
	//const Eigen::Vector3d kRelativeTranslation = 
	//	kDestinationPose_.translation() - kSourcePose_.rotation().transpose() * kSourcePose_.translation();
	const Eigen::Vector3d kRelativeTranslation = 
		kDestinationPose_.rotation() * (kSourcePose_.position() - kDestinationPose_.position());

	// Calculate the errors
	calculateRelativePoseError(
		model.descriptor, // The currently tested essential matrix
		kIntrinsicsSource_, // The intrinsic parameters of the source camera
		kIntrinsicsDestination_, // The intrinsic parameters of the destination camera
		kRelativeRotation, // The ground truth relative rotation
		kRelativeTranslation, // The ground truth relative translation
		rotationError, // The rotation error
		translationError); // The translation error

	// Print the statistics
	printf("%d\t%.3f\t%.3f\t\t%d\t%.3f\n", 
		_MinimalSolver::sampleSize(),
		rotationError, translationError, 
		static_cast<int>(statistics.inliers.size()),
		statistics.processing_time);

	savingMutex.lock();
	std::ofstream file("results.csv", std::fstream::app);
	file << 
		_MinimalSolver::sampleSize() << ";" <<
		kSNNThreshold_ << ";" <<
		kInlierOutlierThreshold_ << ";" <<
		rotationError << ";" <<
		translationError <<  ";" <<
		";" <<
		static_cast<int>(statistics.inliers.size()) << ";" <<
		statistics.processing_time << "\n";
	savingMutex.unlock();
}

template <size_t _Method>
void selectOrientedPoints(
	const std::string &kSourceImageName_,
	const std::string &kDestinationImageName_,
	const std::string &kDatabaseName_,
	const double &kSNNThreshold_,
	const cv::Mat &kCorrespondences_,
	const CameraPose &kSourcePose_,
	const CameraPose &kDestinationPose_,
	const CameraIntrinsics &kSourceIntrinsics_,
	const CameraIntrinsics &kDestinationIntrinsics_,
	const cv::Mat &kPoints3d_,
	const double &kPointAssignmentThreshold_,
	std::vector<int> &assignment_)
{	
	std::string databaseLabel = kSourceImageName_ + "-" + kDestinationImageName_ + "-" + std::to_string(kSNNThreshold_);
	std::replace(databaseLabel.begin(), databaseLabel.end(), '/', '-');

	cv::Mat assignmentMat;
	if (datareading::readData(
		kDatabaseName_,
		databaseLabel,
		assignmentMat))
	{
		for (size_t pointIdx = 0; pointIdx < assignmentMat.rows; ++pointIdx)
			assignment_.emplace_back(assignmentMat.at<int>(pointIdx));
		return;
	}

	// Select the closest 3D point by sending a ray through the pixel in the first image
	if constexpr (_Method == 0)
	{
		const size_t kPointNumber = kCorrespondences_.rows;
		assignment_.resize(kPointNumber, -1);
		const Eigen::Matrix3d kInverseSourceRotation = 
			kSourcePose_.rotation().transpose();
		const Eigen::Vector3d &sourcePosition = kSourcePose_.position();
		Eigen::Vector3d ray, point3d, diff;
		double pointToRayDistance,
			closestPointToRayDistance,
			pointToCameraDistance,
			closestPointDistance;

//#pragma omp parallel for num_threads(kCoreNumber)
		for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
		{
			closestPointToRayDistance = std::numeric_limits<double>::max();
			closestPointDistance = std::numeric_limits<double>::max();

			// Calculate the ray direction
			ray(0) = kCorrespondences_.at<double>(pointIdx, 0);
			ray(1) = kCorrespondences_.at<double>(pointIdx, 1);
			ray(2) = 1;
			ray = kInverseSourceRotation * ray.normalized();

			// Iterate through the 3D points and find the ones that are close to the ray
			for (size_t point3dIdx = 0; point3dIdx < kPoints3d_.rows; ++point3dIdx)
			{
				// Point-to-ray distance
				point3d(0) = kPoints3d_.at<double>(point3dIdx, 0);
				point3d(1) = kPoints3d_.at<double>(point3dIdx, 1);
				point3d(2) = kPoints3d_.at<double>(point3dIdx, 2);

				diff = point3d - sourcePosition;
				pointToRayDistance = ray.cross(diff).norm();
				closestPointToRayDistance = MIN(closestPointToRayDistance, pointToRayDistance);
				if (pointToRayDistance < kPointAssignmentThreshold_)
				{
					pointToCameraDistance = diff.norm();
					if (pointToCameraDistance < closestPointDistance)
					{
						closestPointDistance = pointToCameraDistance;
						assignment_[pointIdx] = point3dIdx;
					}
				}
			}

			/*if (assignment_[pointIdx] > -1)
			{
				Eigen::Vector3d pp = sourcePosition + ray;
				Eigen::Vector3d qq;
				qq(0) = kPoints3d_.at<double>(assignment_[pointIdx], 0);
				qq(1) = kPoints3d_.at<double>(assignment_[pointIdx], 1);
				qq(2) = kPoints3d_.at<double>(assignment_[pointIdx], 2);

				std::ofstream ff("point_selection.txt", std::fstream::app);
				ff << sourcePosition(0) << " " << sourcePosition(1) << " " << sourcePosition(2) << " "
				 	<< 255 << " " << 0 << " " << 0 << "\n";
				ff << pp(0) << " " << pp(1) << " " << pp(2) << " "
				 	<< 0 << " " << 255 << " " << 0 << "\n";
				ff << qq(0) << " " << qq(1) << " " << qq(2) << " "
				 	<< 0 << " " << 0 << " " << 255 << "\n";
				ff.close();
			}*/
		}
	} else if (_Method == 1)
	{
		const size_t kPointNumber = kCorrespondences_.rows;
		assignment_.resize(kPointNumber, -1);
		const Eigen::Matrix3d kInverseSourceRotation = 
			kSourcePose_.rotation().transpose();
		const Eigen::Matrix3d kInverseDestinationRotation = 
			kDestinationPose_.rotation().transpose();
		const Eigen::Vector3d &sourcePosition = 
			kSourcePose_.position();
		const Eigen::Vector3d &destinationPosition = 
			kDestinationPose_.position();
		Eigen::Vector3d raySource, rayDestination,
			point3d, diff1, diff2;
		double pointToRayDistance1,
			pointToRayDistance2,
			closestPointToRayDistance,
			pointToCameraDistance,
			closestPointDistance;

//#pragma omp parallel for num_threads(kCoreNumber)
		for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
		{
			closestPointToRayDistance = std::numeric_limits<double>::max();
			closestPointDistance = std::numeric_limits<double>::max();

			// Calculate the ray direction in the source image
			raySource(0) = kCorrespondences_.at<double>(pointIdx, 0);
			raySource(1) = kCorrespondences_.at<double>(pointIdx, 1);
			raySource(2) = 1;
			raySource = kInverseSourceRotation * raySource.normalized();
			
			// Calculate the ray direction in the destination image
			rayDestination(0) = kCorrespondences_.at<double>(pointIdx, 2);
			rayDestination(1) = kCorrespondences_.at<double>(pointIdx, 3);
			rayDestination(2) = 1;
			rayDestination = kInverseDestinationRotation * rayDestination.normalized();

			// Iterate through the 3D points and find the ones that are close to the ray
			for (size_t point3dIdx = 0; point3dIdx < kPoints3d_.rows; ++point3dIdx)
			{
				// Point-to-ray distance
				point3d(0) = kPoints3d_.at<double>(point3dIdx, 0);
				point3d(1) = kPoints3d_.at<double>(point3dIdx, 1);
				point3d(2) = kPoints3d_.at<double>(point3dIdx, 2);

				diff1 = point3d - sourcePosition;
				diff2 = point3d - destinationPosition;
				
				pointToRayDistance1 = raySource.cross(diff1).norm();
				pointToRayDistance2 = rayDestination.cross(diff2).norm();

				if (pointToRayDistance1 + pointToRayDistance2 < 2 * kPointAssignmentThreshold_ &&
					pointToRayDistance1 + pointToRayDistance2 < closestPointDistance)
				{
					closestPointDistance = pointToRayDistance1 + pointToRayDistance2;
					assignment_[pointIdx] = point3dIdx;
				}
			}

			/*if (assignment_[pointIdx] > -1)
			{
				Eigen::Vector3d pp = sourcePosition + raySource;
				Eigen::Vector3d qq;
				qq(0) = kPoints3d_.at<double>(assignment_[pointIdx], 0);
				qq(1) = kPoints3d_.at<double>(assignment_[pointIdx], 1);
				qq(2) = kPoints3d_.at<double>(assignment_[pointIdx], 2);

				std::ofstream ff("point_selection.txt", std::fstream::app);
				ff << sourcePosition(0) << " " << sourcePosition(1) << " " << sourcePosition(2) << " "
				 	<< 255 << " " << 0 << " " << 0 << "\n";
				ff << pp(0) << " " << pp(1) << " " << pp(2) << " "
				 	<< 0 << " " << 255 << " " << 0 << "\n";
					
				pp = destinationPosition + rayDestination;
				ff << destinationPosition(0) << " " << destinationPosition(1) << " " << destinationPosition(2) << " "
				 	<< 255 << " " << 255 << " " << 0 << "\n";
				ff << pp(0) << " " << pp(1) << " " << pp(2) << " "
				 	<< 0 << " " << 255 << " " << 0 << "\n";
				ff << qq(0) << " " << qq(1) << " " << qq(2) << " "
				 	<< 0 << " " << 0 << " " << 255 << "\n";
				ff.close();
			}*/
		}
	} else if (_Method == 2)
	{
		const size_t kPointNumber = kCorrespondences_.rows;
		assignment_.resize(kPointNumber, -1);
		Eigen::Vector3d diff;

		const Eigen::Matrix<double, 3, 4> kProjectionSource = kSourcePose_.projectionMatrix(),
			kProjectionDestination = kDestinationPose_.projectionMatrix();

		// Calculate relative pose
		const Eigen::Matrix3d kRelativeRotation = 
			kDestinationPose_.rotation() * kSourcePose_.rotation().transpose();
		const Eigen::Vector3d kRelativeTranslation = 
			kDestinationPose_.rotation() * (kSourcePose_.position() - kDestinationPose_.position());

		Eigen::Matrix3d translationCrossProduct;
		translationCrossProduct << 0, -kRelativeTranslation(2), kRelativeTranslation(1), 
		kRelativeTranslation(2), 0, -kRelativeTranslation(0),
		-kRelativeTranslation(1), kRelativeTranslation(0), 0;

		Eigen::Matrix3d kEssentialMatrix = translationCrossProduct * kRelativeRotation;
		
		// The default estimator for essential matrix fitting
		gcransac::utils::DefaultEssentialMatrixEstimator estimator(
			Eigen::Matrix3d::Identity(),
			Eigen::Matrix3d::Identity());

		for (int pointIdx = 0; pointIdx < kPointNumber; ++pointIdx)
		{ 
			// Check if the correspondence is consistent with the epipolar geometry
			double residual = estimator.residual(kCorrespondences_.row(pointIdx),
				kEssentialMatrix);

			if (residual > kPointAssignmentThreshold_)
			{
				assignment_[pointIdx] = 
					round((kPoints3d_.rows - 1.0) * static_cast<double>(rand()) / RAND_MAX);
				continue;
			}

			Eigen::Vector4d triangulatedPoint;
			optimalTriangulation(
				kProjectionSource,
				kProjectionDestination,
				kCorrespondences_.row(pointIdx),
				triangulatedPoint);

			if (triangulatedPoint(3) < 0)
			{
				assignment_[pointIdx] = 
					round((kPoints3d_.rows - 1.0) * static_cast<double>(rand()) / RAND_MAX);
				continue;
			}

			double pointDistance = 0,
				bestDistance = std::numeric_limits<double>::max();
			Eigen::Vector3d triangulatedPoint3d = triangulatedPoint.hnormalized(),
				point3d;
				
			// Iterate through the 3D points and find the ones that are close to the ray
			for (size_t point3dIdx = 0; point3dIdx < kPoints3d_.rows; ++point3dIdx)
			{
				// Point-to-ray distance
				point3d(0) = kPoints3d_.at<double>(point3dIdx, 0);
				point3d(1) = kPoints3d_.at<double>(point3dIdx, 1);
				point3d(2) = kPoints3d_.at<double>(point3dIdx, 2);

				diff = point3d - triangulatedPoint3d;
				pointDistance = diff.norm();

				if (pointDistance < bestDistance)
				{
					bestDistance = pointDistance;
					assignment_[pointIdx] = point3dIdx;
				}
			}
			
			/*if (assignment_[pointIdx] > -1)
			{
				const Eigen::Vector3d &sourcePosition = 
					kSourcePose_.position();
				const Eigen::Vector3d &destinationPosition = 
					kDestinationPose_.position();
					
				const Eigen::Matrix3d kInverseSourceRotation = 
					kSourcePose_.rotation().transpose();
				const Eigen::Matrix3d kInverseDestinationRotation = 
					kDestinationPose_.rotation().transpose();

				// Calculate the ray direction in the source image
				Eigen::Vector3d raySource, rayDestination;
				raySource(0) = kCorrespondences_.at<double>(pointIdx, 0);
				raySource(1) = kCorrespondences_.at<double>(pointIdx, 1);
				raySource(2) = 1;
				raySource = kInverseSourceRotation * raySource.normalized();
				
				// Calculate the ray direction in the destination image
				rayDestination(0) = kCorrespondences_.at<double>(pointIdx, 2);
				rayDestination(1) = kCorrespondences_.at<double>(pointIdx, 3);
				rayDestination(2) = 1;
				rayDestination = kInverseDestinationRotation * rayDestination.normalized();

				Eigen::Vector3d pp = sourcePosition + raySource;
				Eigen::Vector3d qq;
				qq(0) = kPoints3d_.at<double>(assignment_[pointIdx], 0);
				qq(1) = kPoints3d_.at<double>(assignment_[pointIdx], 1);
				qq(2) = kPoints3d_.at<double>(assignment_[pointIdx], 2);

				std::ofstream ff("point_selection.txt", std::fstream::app);
				ff << sourcePosition(0) << " " << sourcePosition(1) << " " << sourcePosition(2) << " "
					<< 255 << " " << 0 << " " << 0 << "\n";
				ff << pp(0) << " " << pp(1) << " " << pp(2) << " "
					<< 0 << " " << 255 << " " << 0 << "\n";
					
				pp = destinationPosition + rayDestination;
				ff << destinationPosition(0) << " " << destinationPosition(1) << " " << destinationPosition(2) << " "
					<< 255 << " " << 255 << " " << 0 << "\n";
				ff << pp(0) << " " << pp(1) << " " << pp(2) << " "
					<< 0 << " " << 255 << " " << 0 << "\n";
				ff << qq(0) << " " << qq(1) << " " << qq(2) << " "
					<< 0 << " " << 0 << " " << 255 << "\n";
				ff.close();
			}	*/		
		}

		//while (1);
	}

	// Save the assignment to the database
	assignmentMat = cv::Mat(assignment_.size(), 1, CV_32S, &assignment_[0]);
	datareading::writeResults(
		kDatabaseName_,
		assignmentMat,
		databaseLabel);
}

void loadCalibrations(
	const std::string &kPath_,
	std::map<std::string, CameraIntrinsics> &calibrations_)
{
	std::ifstream file(kPath_);
	if (!file.is_open())
	{
		fprintf(stderr, "A problem occured when opening file '%s'.\n", kPath_.c_str());
		return;
	}

	std::string imageName;
	double focalLength,
		principalPointX,
		principalPointY;

	while (file >> imageName >> 
		focalLength >> 
		principalPointX >> 
		principalPointY)
	{
		auto iterator = calibrations_.find(imageName);
		if (iterator != std::end(calibrations_))
		{
			fprintf(stderr, "Image '%s' has multiple poses.\n", imageName.c_str());
			continue;	
		}
		
		calibrations_[imageName] =
			CameraIntrinsics(focalLength, principalPointX, principalPointY);
	}
}

template <size_t _HeaderLines>
void loadPoses(
	const std::string &kPath_,
	std::map<std::string, CameraPose> &poses_)
{
	std::ifstream file(kPath_);
	if (!file.is_open())
	{
		fprintf(stderr, "A problem occured when opening file '%s'.\n", kPath_.c_str());
		return;
	}

	std::string str;
	for (size_t lineIdx = 0; lineIdx < _HeaderLines; ++lineIdx)
		std::getline(file, str);

	Eigen::Quaterniond quaternion;
	Eigen::Vector4d qVec;
	Eigen::Vector3d position;
	while (file >> str >> 
		position(0) >> position(1) >> position(2) >>
		qVec(0) >> qVec(1) >> qVec(2) >> qVec(3))
	{
		auto iterator = poses_.find(str);
		if (iterator != std::end(poses_))
		{
			fprintf(stderr, "Image '%s' has multiple poses.\n", str.c_str());
			continue;	
		}

		quaternion = 
			Eigen::Quaterniond(qVec(0), qVec(1), qVec(2), qVec(3));
		poses_[str] = 
			CameraPose(quaternion, 
				position);
	}

	file.close();
}

void loadImagePairs(
	const std::string &kPath_,
	std::vector<std::pair<std::string, std::string>> &imagePairs_)
{
	std::ifstream file(kPath_);
	if (!file.is_open())
	{
		fprintf(stderr, "A problem occured when opening file '%s'.\n", kPath_.c_str());
		return;
	}

	imagePairs_.clear();	
	std::string imageNameSource, imageNameDestination;
	while (file >> imageNameSource >> imageNameDestination)
		imagePairs_.emplace_back(std::make_pair(imageNameSource, imageNameDestination));
	file.close();
}

void loadOrientedPoints(
	const std::string &kPath_,
	std::vector<double> &values_)
{
	std::ifstream file(kPath_);
	if (!file.is_open())
	{
		fprintf(stderr, "A problem occured when opening file '%s'.\n", kPath_.c_str());
		return;
	}

	values_.clear();
	double value;
	while (file >> value)
		values_.emplace_back(value);
	file.close();
}
