#pragma once

#include <vector>
#include <Eigen/Eigen>
#include "estimator.h"

template <typename _DataContainer>
class NeighborhoodGraph
{
protected:
	// The pointer of the container consisting of the data points from which
	// the neighborhood graph is constructed.
	const _DataContainer const *container; 

	// The number of neighbors, i.e., edges in the neighborhood graph.
	size_t neighbor_number;

	// A flag indicating if the initialization was successfull.
	bool initialized;

public:
	NeighborhoodGraph() : initialized(false) {}

	NeighborhoodGraph(const _DataContainer const *container_) :
		neighbor_number(0),
		container(container_)
	{
	}

	// A function to initialize and create the neighbordhood graph.
	virtual bool initialize(const _DataContainer const *container_) = 0;

	// Returns the neighbors of the current point in the graph.
	inline virtual const std::vector<size_t> &getNeighbors(size_t point_idx_) const = 0;
	
	// Returns the number of edges in the neighborhood graph.
	size_t getNeighborNumber() const { return neighbor_number; }

	// Returns if the initialization was successfull.
	bool isInitialized() const { return initialized; }
};