#include <vector>
#include <string>

int findRigidTransform_(std::vector<double>& points1,
	std::vector<double>& points2,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt);

int find6DPose_(
	std::vector<double>& imagePoints,
	std::vector<double>& worldPoints,
	std::vector<bool>& inliers,
	std::vector<double> &pose,
	double spatial_coherence_weight,
	double threshold,
	double conf,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt);

int findFundamentalMatrix_(std::vector<double>& srcPts,
                           std::vector<double>& dstPts,
                           std::vector<bool>& inliers,
                           std::vector<double>&F,
                           int h1, int w1, int h2, int w2,
						   double spatial_coherence_weight,
                           double threshold,
                           double conf,
                           int max_iters,
						   bool use_sprt,
						   double min_inlier_ratio_for_sprt);

 int findLine2D_(std::vector<double>& srcPts,
                 std::vector<bool>& inliers,
                 std::vector<double>&abc,
                 int w1, int h1,
		             double threshold,
                 double conf,
                 int max_iters,
								 double spatial_coherence_weight,
	               bool use_sprt,
		   			 		 double min_inlier_ratio_for_sprt);


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
                           int max_iters,
						   bool use_sprt,
						   double min_inlier_ratio_for_sprt,
						   int sampler_id);


int findHomography_(std::vector<double>& srcPts,
                    std::vector<double>& dstPts,
                    std::vector<bool>& inliers,
                    std::vector<double>& H,
                    int h1, int w1, int h2, int w2,
				    double spatial_coherence_weight,
                    double threshold,
                    double conf,
                    int max_iters,
					bool use_sprt,
					double min_inlier_ratio_for_sprt);
