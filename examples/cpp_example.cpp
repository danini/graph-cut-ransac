#include <vector>	
#include <thread>
#include "utils.h"
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>

#include "GCRANSAC.h"

#include "neighborhood/flann_neighborhood_graph.h"
#include "neighborhood/grid_neighborhood_graph.h"

#include "samplers/uniform_sampler.h"
#include "samplers/prosac_sampler.h"
#include "samplers/progressive_napsac_sampler.h"

#include "estimators/fundamental_estimator.h"
#include "estimators/homography_estimator.h"
#include "estimators/essential_estimator.h"
#include "estimators/rigid_transformation_estimator.h"

#include "preemption/preemption_sprt.h"
#include "types.h"

#include "estimators/solver_fundamental_matrix_seven_point.h"
#include "estimators/solver_fundamental_matrix_eight_point.h"
#include "estimators/solver_rigid_transformation_svd.h"
#include "estimators/solver_homography_four_point.h"
#include "estimators/solver_essential_matrix_five_point_stewenius.h"
#include "estimators/solver_p3p.h"
#include "estimators/solver_dls_pnp.h"

#include "estimators/solver_essential_matrix_one_focal_four_point.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif 

struct stat info;

template<typename _EstimatorClass>
void runTest(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const Eigen::Matrix3d &kGravitySource_,
	const Eigen::Matrix3d &kGravityDestination_,
	Eigen::Matrix3d &kGroundTruthRotation_,
	Eigen::Vector3d &kGroundTruthTranslation_,
	const double kGroundTruthFocalLength_,
	const double kInlierOutlierThreshold_,
	const double kSpatialWeight_,
	const double kConfidence_,
	const size_t kMaximumIterations_,
	const size_t kMinimumIterations_);

void detectFeatures(
	const std::string &kWorkspacePath_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	cv::Mat& correspondences_);

void poseError(
	const cv::Mat &R1_,
	const cv::Mat &R2_,
	const cv::Mat &t1_,
	const cv::Mat &t2_,
	double &rotationError_,
	double &translationError_);

using namespace gcransac;

int main(int argc, const char* argv[])
{
	srand(static_cast<int>(time(NULL)));

	// The paths for the example data
	const std::string 
		kPath = "/home/danini/research/solver_e4f/graph-cut-ransac/examples/test_data/",
		kSourceFrameName = "00001",
		kDestinationFrameName = "00011";
	const std::string 
		kSourceImagePath = "Frame" + kSourceFrameName + ".png",
		kDestinationImagePath = "Frame" + kDestinationFrameName + ".png";
	const double kFocalLength = 1610;

	// Read the extrinsic calibration files
	Eigen::Matrix3d rotation,
		gravitySource,
		gravityDestination;
	Eigen::Vector3d translation;

	rotation << 0.982917620073483, -0.0173730906779992, -0.183224255679684, 
		0.00850123428401611, 0.998757905766926, -0.0490955668437875,
		0.183849615623010, 0.0466992653942067, 0.981844436480094;
	translation << -1.39756904123826, -0.101853038003336, 1.91622330042648;

	gravitySource << 0.996777262084430, -0.0802190114091898, 0, 
		0.0799956742118808, 0.994002141372462, 0.0745683247387754, 
		-0.00598179729298400, -0.0743280105713392, 0.997215906885592;

	gravityDestination << 0.998238541426086, -0.0593280238295505, 0, 
		0.0590255622873692, 0.993149398905985, 0.100847679444502, 
		-0.00598309352923827, -0.100670040434885, 0.994901877348042;

	// Read the images
	const cv::Mat kSourceImage = cv::imread(kPath + kSourceImagePath);

	if (kSourceImage.empty())
	{
		fprintf(stderr, "Image '%s' is not found.", kSourceImagePath.c_str());
		return 0;	
	}

	const cv::Mat kDestinationImage = cv::imread(kPath + kDestinationImagePath);

	if (kDestinationImage.empty())
	{
		fprintf(stderr, "Image '%s' is not found.", kDestinationImagePath.c_str());
		return 0;	
	}
		
	// Find or load correspondences
	cv::Mat correspondences;
	detectFeatures(
		kPath + kSourceFrameName + "_" + kDestinationFrameName + "_matches.txt",
		kSourceImage,
		kDestinationImage,
		correspondences);

	// Compose the intrinsic camera matrices
	Eigen::Matrix3d intrinsicsSource = Eigen::Matrix3d::Zero(),
		intrinsicsDestination = Eigen::Matrix3d::Zero();

	intrinsicsSource(0, 0) = kFocalLength;
	intrinsicsSource(1, 1) = kFocalLength;
	intrinsicsSource(0, 2) = kSourceImage.cols / 2.0;
	intrinsicsSource(1, 2) = kSourceImage.rows / 2.0;
	intrinsicsSource(2, 2) = 1.0;
	
	intrinsicsDestination(0, 0) = kFocalLength;
	intrinsicsDestination(1, 1) = kFocalLength;
	intrinsicsDestination(0, 2) = kDestinationImage.cols / 2.0;
	intrinsicsDestination(1, 2) = kDestinationImage.rows / 2.0;
	intrinsicsDestination(2, 2) = 1.0;

	// Normalizing the point correspondences
	cv::Mat normalizedCorrespondences(correspondences.size(), correspondences.type());
	utils::normalizeCorrespondences(
			correspondences,
			intrinsicsSource,
			intrinsicsDestination,
			normalizedCorrespondences);

	// The default estimator class
	typedef estimator::EssentialMatrixEstimator<estimator::solver::EssentialOnefocal4PC, // The solver used for fitting a model to a minimal sample
		estimator::solver::EssentialMatrixBundleAdjustmentSolver> // The solver used for fitting a model to a non-minimal sample
		EstimatorClass;
		
	// Run tests
	runTest<EstimatorClass>(
			correspondences, normalizedCorrespondences,
			kSourceImage, kDestinationImage,
			intrinsicsSource, intrinsicsDestination, 
			gravitySource, gravityDestination,
			rotation, translation, kFocalLength,
			0.75, 0.0, 0.999, 10000, 20);
			
	runTest<gcransac::utils::DefaultFundamentalMatrixEstimator>(
			correspondences, normalizedCorrespondences,
			kSourceImage, kDestinationImage,
			intrinsicsSource, intrinsicsDestination, 
			gravitySource, gravityDestination,
			rotation, translation, kFocalLength,
			0.75, 0.0, 0.999, 10000, 20);
	return 0;
}

template<typename _EstimatorClass>
void runTest(
	const cv::Mat &kCorrespondences_,
	const cv::Mat &kNormalizedCorrespondences_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	const Eigen::Matrix3d &kIntrinsicsSource_,
	const Eigen::Matrix3d &kIntrinsicsDestination_,
	const Eigen::Matrix3d &kGravitySource_,
	const Eigen::Matrix3d &kGravityDestination_,
	Eigen::Matrix3d &kGroundTruthRotation_,
	Eigen::Vector3d &kGroundTruthTranslation_,
	const double kGroundTruthFocalLength_,
	const double kInlierOutlierThreshold_,
	const double kSpatialWeight_,
	const double kConfidence_,
	const size_t kMaximumIterations_,
	const size_t kMinimumIterations_)
{	
	// Initializing the estimator
	std::unique_ptr<_EstimatorClass> estimator;
	if constexpr (std::is_same<gcransac::utils::DefaultFundamentalMatrixEstimator, _EstimatorClass>())
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass());
	else
		estimator = std::unique_ptr<_EstimatorClass>(new _EstimatorClass(kIntrinsicsSource_, kIntrinsicsDestination_));

	// Setting the gravity if the solver needs it
	if (estimator->getMinimalSolver()->needsGravity())
		estimator->getMinimalSolver()->setGravity(
			kGravitySource_,
			kGravityDestination_);
			
	if (estimator->getNonMinimalSolver()->needsGravity())
		estimator->getNonMinimalSolver()->setGravity(
			kGravitySource_,
			kGravityDestination_);

	// Normalize the threshold by the average of the focal lengths
	const double kNormalizedThreshold =
		kInlierOutlierThreshold_ / ((kIntrinsicsSource_(0, 0) + kIntrinsicsSource_(1, 1) +
			kIntrinsicsDestination_(0, 0) + kIntrinsicsDestination_(1, 1)) / 4.0);

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph<4> neighborhood(&kCorrespondences_,
		{ kSourceImage_.cols / static_cast<double>(4),
			kSourceImage_.rows / static_cast<double>(4),
			kDestinationImage_.cols / static_cast<double>(4),
			kDestinationImage_.rows / static_cast<double>(4) },
		4);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}

	// Apply Graph-cut RANSAC
	std::vector<int> inliers;
	EssentialMatrix model;

	// Initializing SPRT test
	preemption::EmptyPreemptiveVerfication<_EstimatorClass> preemptiveVerification;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	//sampler::ProsacSampler mainSampler(&kCorrespondences_, _EstimatorClass::sampleSize());  
	sampler::UniformSampler mainSampler(&kCorrespondences_); // The local optimization sampler is used inside the local optimization
	sampler::UniformSampler localOptimizationSampler(&kCorrespondences_); // The local optimization sampler is used inside the local optimization

	// Checking if the samplers are initialized successfully.
	if (!mainSampler.isInitialized() ||
		!localOptimizationSampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return;
	}

	GCRANSAC<_EstimatorClass,
		neighborhood::GridNeighborhoodGraph<4>,
		MSACScoringFunction<_EstimatorClass>,
		preemption::EmptyPreemptiveVerfication<_EstimatorClass>> gcransac;
	gcransac.settings.threshold = kNormalizedThreshold; // The inlier-outlier threshold
	gcransac.settings.spatial_coherence_weight = kSpatialWeight_; // The weight of the spatial coherence term
	gcransac.settings.confidence = kConfidence_; // The required confidence in the results
	gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
	gcransac.settings.max_iteration_number = kMaximumIterations_; // The maximum number of iterations
	gcransac.settings.min_iteration_number = kMinimumIterations_; // The minimum number of iterations
	gcransac.settings.neighborhood_sphere_radius = 4; // The radius of the neighborhood ball

	// Start GC-RANSAC
	gcransac.run(kNormalizedCorrespondences_,
		*estimator,
		&mainSampler,
		&localOptimizationSampler,
		&neighborhood,
		model,
		preemptiveVerification);

	// Get the statistics of the results
	const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

	double rotationError = std::numeric_limits<double>::max(),
		translationError = std::numeric_limits<double>::max();

	if (statistics.inliers.size() <= _EstimatorClass::sampleSize())
		model.descriptor = Eigen::Matrix3d::Identity();

	// Calculate the pose error
	Eigen::Matrix3d essentialMatrix = model.descriptor.block<3, 3>(0, 0);
	cv::Mat cv_EssentialMatrix(3, 3, CV_64F, essentialMatrix.data());
	cv_EssentialMatrix = cv_EssentialMatrix.t();
	cv::Mat cv_Rotation, cv_Translation;

	const int &kPointNumber = kNormalizedCorrespondences_.rows;
	std::vector<uchar> inlierMask(kPointNumber, 0);
	for (const auto &inlierIdx : statistics.inliers)
		inlierMask[inlierIdx] = 1;

	cv::recoverPose(cv_EssentialMatrix, 
		kNormalizedCorrespondences_(cv::Rect(0, 0, 2, kPointNumber)),
		kNormalizedCorrespondences_(cv::Rect(2, 0, 2, kPointNumber)),
		cv::Mat::eye(3, 3, CV_64F),
		cv_Rotation,
		cv_Translation,
		inlierMask);

	cv::Mat cv_GroundTruthRotation(3, 3, CV_64F, kGroundTruthRotation_.data());
	cv_GroundTruthRotation = cv_GroundTruthRotation.t();
	cv::Mat cv_GroundTruthTranslation(1, 3, CV_64F, kGroundTruthTranslation_.data());
	cv_GroundTruthTranslation = cv_GroundTruthTranslation.t();

	poseError(
		cv_Rotation,
		cv_GroundTruthRotation,
		cv_Translation,
		cv_GroundTruthTranslation,
		rotationError,
		translationError);

	// Print the statistics
	printf("Elapsed time = %f secs\n", statistics.processing_time);
	printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
	printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
	printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
	printf("Number of iterations = %d\n", static_cast<int>(statistics.iteration_number));
	printf("Rotation error = %f degrees\n", rotationError);
	printf("Translation error = %f degrees\n\n", translationError);
}

void poseError(
	const cv::Mat &R1_,
	const cv::Mat &R2_,
	const cv::Mat &t1_,
	const cv::Mat &t2_,
	double &rotationError_,
	double &translationError_)
{
	// Calculate angle between provided rotations
	cv::Mat R12 = R2_ * R1_.t();
	cv::Mat rotationVector;
	cv::Rodrigues(R12, rotationVector);

	rotationError_ = cv::norm(rotationVector) * 180 / M_PI;
	
	// calculate angle between provided translations
	double translationError1 = t2_.dot(t1_) / cv::norm(t2_) / cv::norm(t1_);
	translationError1 = MAX(-1.0, MIN(1.0, translationError1));
	translationError1 = acos(translationError1) * 180 / M_PI;
	
	double translationError2 = t2_.dot(-t1_) / cv::norm(t2_) / cv::norm(t1_);
	translationError2 = MAX(-1.0, MIN(1.0, translationError2));
	translationError2 = acos(translationError1) * 180 / M_PI;

	translationError_ = MIN(translationError1, translationError2);
}

void detectFeatures(
	const std::string &kWorkspacePath_,
	const cv::Mat &kSourceImage_,
	const cv::Mat &kDestinationImage_,
	cv::Mat& correspondences_)
{
	if (gcransac::utils::loadPointsFromFile(correspondences_,
		kWorkspacePath_.c_str()))
	{
		printf("Match number = %d\n", correspondences_.rows);
		return;
	}

	cv::Mat descriptors[2];
	std::vector<cv::KeyPoint> keypoints[2];

	cv::Ptr<cv::SiftFeatureDetector> sift = cv::SIFT::create(10000, 3, 0.0, 10000.);
	sift->detect(kSourceImage_, keypoints[0]);
	sift->compute(kSourceImage_, keypoints[0], descriptors[0]);

	sift->detect(kDestinationImage_, keypoints[1]);
	sift->compute(kDestinationImage_, keypoints[1], descriptors[1]);
	 
	// Normalize the descriptor vectors to get RootSIFT
	for (auto& descriptor : descriptors)
	{
		for (size_t row = 0; row < descriptor.rows; ++row)
		{
			descriptor.row(row) *= 1.0 / cv::norm(descriptor.row(row), cv::NORM_L1);
			for (size_t col = 0; col < descriptor.cols; ++col)
				descriptor.at<float>(row, col) = std::sqrt(descriptor.at<float>(row, col));
		}
	}

	// Do brute-force matching from the source to the destination image
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<std::vector<cv::DMatch>> matches;
	matcher->knnMatch(descriptors[0], descriptors[1], matches, 2);

	// Do brute-force matching from the destination to the source image
	cv::Ptr<cv::DescriptorMatcher> matcher_opposite = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
	std::vector<std::vector<cv::DMatch>> matches_opposite;
	matcher_opposite->knnMatch(descriptors[1], descriptors[0], matches_opposite, 1);

	std::vector<std::pair<double, const std::vector<cv::DMatch>*>> good_matches;

	// Do mutual nearest neighbor search
	for (size_t i = 0; i < matches.size(); ++i)
		if ((matches[i][0].distance < 0.9 * matches[i][1].distance) &&
			(matches[i][0].queryIdx == matches_opposite[matches[i][0].trainIdx][0].trainIdx)) // We increased threshold for mutual snn check
			good_matches.emplace_back(std::make_pair(matches[i][0].distance / matches[i][1].distance, &matches[i]));

	// Sort the correspondences according to their distance.
	// This is done for using PROSAC sampling
	std::sort(good_matches.begin(), good_matches.end());

	// Create the container for the correspondences
	correspondences_.create(good_matches.size(), 4, CV_64F);
	double* correspondences_ptr = reinterpret_cast<double*>(correspondences_.data);

	// Fill the container by the selected matched
	for (const auto& match_ptr : good_matches)
	{
		const std::vector<cv::DMatch>& match = *match_ptr.second;

		*(correspondences_ptr++) = keypoints[0][match[0].queryIdx].pt.x;
		*(correspondences_ptr++) = keypoints[0][match[0].queryIdx].pt.y;
		*(correspondences_ptr++) = keypoints[1][match[0].trainIdx].pt.x;
		*(correspondences_ptr++) = keypoints[1][match[0].trainIdx].pt.y;
	}

	gcransac::utils::savePointsToFile(correspondences_, 
		kWorkspacePath_.c_str());
	printf("Match number = %d\n", correspondences_.rows);
}