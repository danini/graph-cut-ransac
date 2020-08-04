#include <vector>
#include <string>

int find6DPose_(
	std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters);

int find6DPoseEPOS_(
	std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<double>& cameraParams,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double &score,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int min_iters,
	int max_iters,
	double sphere_radius,
	double scaling_from_millimeters,
	double minimum_coverage,
	double min_triangle_area,
	bool apply_numerical_optimization,
	int num_models);

int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&F,
                           int h1, int w1, int h2, int w2,
						   double spatial_coherence_weight,
                           double threshold,
                           double conf,
                           int max_iters);

int findEssentialMatrix_(std::vector<double>& srcPts_norm,
                           std::vector<double>& dstPts_norm,
                           std::vector<bool>& inliers,
                           std::vector<double>&E,
                               std::vector<double>& src_K,
                           std::vector<double>& dst_K,
                           int h1, int w1, int h2, int w2,
						   double spatial_coherence_weight,
                           double threshold,
                           double conf,
                           int max_iters);


int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    int h1, int w1, int h2, int w2,
				    double spatial_coherence_weight,
                    double threshold,
                    double conf,
                    int max_iters);
