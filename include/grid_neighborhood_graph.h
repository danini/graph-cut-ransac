#pragma once

#include "neighborhood_graph.h"
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>
#include "estimator.h"

// The cell structure used in the HashMap
class GridCell {
public:
	const size_t idx_along_axis_x1, // The cell index along axis X in the source image
		idx_along_axis_y1, // The cell index along axis Y in the source image
		idx_along_axis_x2, // The cell index along axis X in the destination image
		idx_along_axis_y2; // The cell index along axis Y in the destination image
	const size_t index; // The index of the cell used in the hashing function

	GridCell(
		size_t idx_along_axis_x1_, // The cell index along axis X in the source image
		size_t idx_along_axis_y1_, // The cell index along axis Y in the source image 
		size_t idx_along_axis_x2_, // The cell index along axis X in the destination image 
		size_t idx_along_axis_y2_, // The cell index along axis Y in the destination image
		size_t cell_number_along_axis_x1_, // The number of cells along axis X in the source image
		size_t cell_number_along_axis_y1_, // The number of cells along axis Y in the source image
		size_t cell_number_along_axis_x2_, // The number of cells along axis X in the destination image 
		size_t cell_number_along_axis_y2_) : // The number of cells along axis Y in the destination image
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
		size_t idx_along_axis_x1_, // The cell index along axis X in the source image
		size_t idx_along_axis_y1_, // The cell index along axis Y in the source image 
		size_t idx_along_axis_x2_, // The cell index along axis X in the destination image 
		size_t idx_along_axis_y2_, // The cell index along axis Y in the destination image
		size_t cell_number_along_all_axes_) : // The number of cells along all axis in both images
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

	// Two cells are equal of their indices along all axes in both
	// images are equal.
	bool operator==(const GridCell &o) const {
		return 
			idx_along_axis_x1 == o.idx_along_axis_x1 && 
			idx_along_axis_y1 == o.idx_along_axis_y1 && 
			idx_along_axis_x2 == o.idx_along_axis_x2 && 
			idx_along_axis_y2 == o.idx_along_axis_y2;
	}

	// The cells are ordered in ascending order along axes X1 Y1 X2 Y2
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
			// The cell's index value is used in the hashing function
			return coord.index;
		}
	};
}

class GridNeighborhoodGraph : public NeighborhoodGraph<cv::Mat>
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

	// The pointer to the cell (i.e., key in the grid) for each point.
	// It is faster to store them than to recreate the cell structure
	// whenever is needed.
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
	inline const std::vector<size_t> &getNeighbors(size_t point_idx_) const;
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
		const GridCell cell(idx_along_axis_x1, // The cell index along axis X in the source image
			idx_along_axis_y1, // The cell index along axis Y in the source image 
			idx_along_axis_x2, // The cell index along axis X in the destination image
			idx_along_axis_y2, // The cell index along axis Y in the destination image
			cell_number_along_all_axes); // The number of cells along all axis in both images

		// Add the current point's index to the grid.
		grid[cell].push_back(row);
	}

	// Iterate through all cells and store the corresponding
	// cell pointer for each point.
	cells_of_points.resize(point_number);
	for (const auto &element : grid) 
	{
		// Get the pointer of the cell.
		const GridCell *cell = &element.first;

		// Increase the edge number in the neighborhood graph.
		// All possible pairs of points in each cell are neighbors,
		// therefore, the neighbor number is "n choose 2" for the
		// current cell.
		const size_t n = element.second.size();
		neighbor_number += n * (n - 1) / 2;

		// Iterate through all points in the cell.
		for (const size_t &point_idx : element.second)
			cells_of_points[point_idx] = cell; // Store the cell pointer for each contained point.
	}

	return neighbor_number > 0;
}

inline const std::vector<size_t> &GridNeighborhoodGraph::getNeighbors(size_t point_idx_) const
{
	// Get the pointer of the cell in which the point is.
	const GridCell *cell = cells_of_points[point_idx_];
	// Return the vector containing all the points in the cell.
	return grid.at(*cell);
}