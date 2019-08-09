#pragma once

#include "neighborhood_graph.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include "estimator.h"

class FlannNeighborhoodGraph : NeighborhoodGraph<cv::Mat>
{
public:
	// The possible methods for building the neighborhood graph.
	// "SearchType::RadiusSearch" uses a hypersphere around each point with a manually set radius.
	// "SearchType::KNN" applies the k-nearest-neighbors algorithm with a manually set k value.
	enum SearchType { RadiusSearch, KNN };

protected:
	// The radius used as the radius of the neighborhood ball.
	double matching_radius;

	// The k value used in the k-nearest-neighbors algorithm.
	size_t knn;

	// The container consisting of the found neighbors for each point.
	std::vector<std::vector<size_t>> neighbours;

	// The method used for building the neighborhood graph.
	SearchType search_type;

public:
	FlannNeighborhoodGraph() : NeighborhoodGraph() {}

	FlannNeighborhoodGraph(const cv::Mat const *container_, // The pointer of the container consisting of the data points.
		double matching_radius_, // The radius used as the radius of the neighborhood ball.
		size_t knn_ = 0, // The k value used in the k-nearest-neighbors algorithm.
		SearchType search_type_ = SearchType::RadiusSearch) : // The method used for building the neighborhood graph.
		matching_radius(matching_radius_),
		search_type(search_type_),
		knn(knn_),
		NeighborhoodGraph(container_)
	{
		initialized = initialize(container);
	}

	bool initialize(const cv::Mat const *container_);
	const std::vector<size_t> &getNeighbors(size_t point_idx_) const;
};

bool FlannNeighborhoodGraph::initialize(const cv::Mat const *container_) 
{
	// Compute the neighborhood graph
	// TODO: replace by nanoflann
	std::vector<std::vector<cv::DMatch>> tmp_neighbours;
	cv::FlannBasedMatcher flann(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(6));

	if (container_->type() == CV_32F)
	{
		flann.radiusMatch(*container_, // The point set 
			*container_, // The point set 
			tmp_neighbours, // The estimated neighborhood graph
			static_cast<float>(matching_radius)); // The radius of the neighborhood ball
	}
	else
	{
		cv::Mat tmp_points;
		container_->convertTo(tmp_points, CV_32F); // OpenCV's FLANN dies if the points are doubles
		flann.radiusMatch(tmp_points, // The point set converted to floats
			tmp_points, // The point set converted to floats
			tmp_neighbours, // The estimated neighborhood graph
			static_cast<float>(matching_radius)); // The radius of the neighborhood ball
	}

	// Count the edges in the neighborhood graph
	neighbours.resize(tmp_neighbours.size());
	for (size_t i = 0; i < tmp_neighbours.size(); ++i)
	{
		const size_t n = tmp_neighbours[i].size() - 1;
		neighbor_number += static_cast<int>(n);

		neighbours[i].resize(n);
		for (size_t j = 0; j < n; ++j)
			neighbours[i][j] = tmp_neighbours[i][j + 1].trainIdx;
	}

	return neighbor_number > 0;
}

const std::vector<size_t> &FlannNeighborhoodGraph::getNeighbors(size_t point_idx_) const
{
	return neighbours[point_idx_];
}