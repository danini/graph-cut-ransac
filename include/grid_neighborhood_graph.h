#pragma once

#include "neighborhood_graph.h"
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include "estimator.h"

class GridCell {
public:
	const size_t idx_along_axis_x1, 
		idx_along_axis_y1, 
		idx_along_axis_x2, 
		idx_along_axis_y2;
	const size_t index;

	GridCell(
		size_t idx_along_axis_x1_, 
		size_t idx_along_axis_y1_, 
		size_t idx_along_axis_x2_, 
		size_t idx_along_axis_y2_,
		size_t cell_number_along_axis_x1_, 
		size_t cell_number_along_axis_y1_, 
		size_t cell_number_along_axis_x2_, 
		size_t cell_number_along_axis_y2_) : 
		idx_along_axis_x1(idx_along_axis_x1_),
		idx_along_axis_y1(idx_along_axis_y1_),
		idx_along_axis_x2(idx_along_axis_x2_),
		idx_along_axis_y2(idx_along_axis_y2_),
		index(idx_along_axis_x1_ +
			cell_number_along_axis_x1_ * idx_along_axis_y1_ +
			cell_number_along_axis_x1_ * cell_number_along_axis_y1_ * idx_along_axis_x2_ +
			cell_number_along_axis_x1_ * cell_number_along_axis_y1_ * cell_number_along_axis_x2_ * idx_along_axis_y2_)
	{
	}

	GridCell(
		size_t idx_along_axis_x1_, 
		size_t idx_along_axis_y1_,
		size_t idx_along_axis_x2_,
		size_t idx_along_axis_y2_,
		size_t cell_number_along_all_axes_) : 
		idx_along_axis_x1(idx_along_axis_x1_),
		idx_along_axis_y1(idx_along_axis_y1_),
		idx_along_axis_x2(idx_along_axis_x2_),
		idx_along_axis_y2(idx_along_axis_y2_),
		index(idx_along_axis_x1_ +
			cell_number_along_all_axes_ * idx_along_axis_y1_ +
			cell_number_along_all_axes_ * cell_number_along_all_axes_ * idx_along_axis_x2_ +
			cell_number_along_all_axes_ * cell_number_along_all_axes_ * cell_number_along_all_axes_ * idx_along_axis_y2_)
	{
	}

	bool operator==(const GridCell &o) const {
		return 
			idx_along_axis_x1 == o.idx_along_axis_x1 && 
			idx_along_axis_y1 == o.idx_along_axis_y1 && 
			idx_along_axis_x2 == o.idx_along_axis_x2 && 
			idx_along_axis_y2 == o.idx_along_axis_y2;
	}

	bool operator<(const GridCell &o) const {
		if (idx_along_axis_x1 < o.idx_along_axis_x1) 
			return true;

		if (idx_along_axis_x1 == o.idx_along_axis_x1 && 
			idx_along_axis_y1 < o.idx_along_axis_y1) 
			return true;

		if (idx_along_axis_x1 == o.idx_along_axis_x1 && 
			idx_along_axis_y1 == o.idx_along_axis_y1 && 
			idx_along_axis_x2 < o.idx_along_axis_x2) 
			return true;

		if (idx_along_axis_x1 == o.idx_along_axis_x1 && 
			idx_along_axis_y1 == o.idx_along_axis_y1 && 
			idx_along_axis_x2 == o.idx_along_axis_x2 && 
			idx_along_axis_y2 < o.idx_along_axis_y2) 
			return true;

		return false;
	}
};

namespace std {
	template<> struct hash<GridCell>
	{
		std::size_t operator()(const GridCell& coord) const noexcept
		{
			return coord.index;
		}
	};
}

class GridNeighborhoodGraph : NeighborhoodGraph<cv::Mat>
{
protected:
	double cell_width_source_image, // The width of a cell in the source image.
		cell_height_source_image, // The height of a cell in the source image.
		cell_width_destination_image, // The width of a cell in the destination image.
		cell_height_destination_image;  // The height of a cell in the destination image.

	// Number of cells along the image axes.
	size_t cell_number_along_all_axes; 
	
	// The grid is stored in the HashMap where the key is defined
	// via the cell coordinates.
	std::unordered_map<GridCell, std::vector<size_t>> grid;

	// The pointer to the cell (i.e. key in the grid) for each point.
	// It is faster to store them than to recreate the cell structure
	// whenever needed.
	std::vector<const GridCell *> cells_of_points;

public:
	GridNeighborhoodGraph() : NeighborhoodGraph() {}

	GridNeighborhoodGraph(const cv::Mat const *container_, // The pointer of the container consisting of the data points.
		const double cell_width_source_image_,
		const double cell_height_source_image_,
		const double cell_width_destination_image_,
		const double cell_height_destination_image_,
		const size_t cell_number_along_all_axes_) :
		cell_width_source_image(cell_width_source_image_),
		cell_height_source_image(cell_height_source_image_),
		cell_width_destination_image(cell_width_destination_image_),
		cell_height_destination_image(cell_height_destination_image_),
		cell_number_along_all_axes(cell_number_along_all_axes_),
		NeighborhoodGraph(container_)
	{
		initialized = initialize(container);
	}

	bool initialize(const cv::Mat const *container_);
	const std::vector<size_t> &getNeighbors(size_t point_idx_) const;
};

bool GridNeighborhoodGraph::initialize(const cv::Mat const *container_)
{
	// The number of points
	const size_t point_number = container_->rows;

	// Pointer to the coordinates
	const double *points_ptr = 
		reinterpret_cast<double *>(container_->data);

	// Iterate through the points and put each into the grid.
	for (size_t row = 0; row < point_number; ++row)
	{
		// The cell index along axis X in the source image.
		const size_t idx_along_axis_x1 = *(points_ptr++) / cell_width_source_image;
		// The cell index along axis Y in the source image.
		const size_t idx_along_axis_y1 = *(points_ptr++) / cell_height_source_image;
		// The cell index along axis X in the destination image.
		const size_t idx_along_axis_x2 = *(points_ptr++) / cell_width_destination_image;
		// The cell index along axis Y in the destination image.
		const size_t idx_along_axis_y2 = *(points_ptr++) / cell_height_destination_image;

		// Constructing the cell structure which is used in the HashMap.
		const GridCell cell(idx_along_axis_x1,
			idx_along_axis_y1,
			idx_along_axis_x2,
			idx_along_axis_y2,
			cell_number_along_all_axes);

		// Add the current point's index to the grid.
		grid[cell].push_back(row);
	}

	cells_of_points.resize(point_number);
	for (const auto &element : grid) 
	{
		const GridCell *cell = &element.first;
		for (const size_t &point_idx : element.second)
			cells_of_points[point_idx] = cell;
	}

	return neighbor_number > 0;
}

const std::vector<size_t> &GridNeighborhoodGraph::getNeighbors(size_t point_idx_) const
{
	const GridCell *cell = cells_of_points[point_idx_];
	return grid.at(*cell);
}