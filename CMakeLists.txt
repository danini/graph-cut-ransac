cmake_minimum_required(VERSION 3.8)

project(GraphCutRANSAC LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if OPENMP should be enabled
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 3.0 REQUIRED)

# ==============================================================================
# Find Eigen
# ==============================================================================
find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
include_directories (
	${PROJECT_SOURCE_DIR}/include
)

# ==============================================================================
# Structure: Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_GCRANSAC
	"include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GCRANSAC 
	"include/*.cpp"
)

source_group(third_party\\GCOptimization FILES
	include/GCoptimization.cpp
	include/LinkedBlockList.cpp
	include/graph.cpp
	include/maxflow.cpp
	include/block.h
	include/energy.h
	include/GCoptimization.h
	include/graph.h
	include/LinkedBlockList.h
)

source_group(Models FILES
	include/solver_engine.h
	include/solver_essential_matrix_five_point_stewenius.h
	include/solver_fundamental_matrix_eight_point.h
	include/solver_fundamental_matrix_seven_point.h
	include/solver_homography_four_point.h
	include/essential_estimator.h
	include/estimator.h
	include/fundamental_estimator.h
	include/homography_estimator.h
	include/model.h
)

source_group(NeighborhoodGraphs FILES
	include/flann_neighborhood_graph.h
	include/grid_neighborhood_graph.h
	include/neighborhood_graph.h
)

source_group(Samplers FILES
	include/progressive_napsac_sampler.h
	include/prosac_sampler.h
	include/sampler.h
	include/uniform_sampler.h
)

source_group(Utilities FILES
	include/utils.h
	include/types.h
	include/uniform_random_generator.h
)

add_library(${PROJECT_NAME} STATIC
	${HDRS_GCRANSAC}
	${SRCS_GCRANSAC}
)

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
)

target_link_libraries (${PROJECT_NAME}
	Eigen3::Eigen)
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

	add_executable(SampleProject
		src/main.cpp)
		
	target_link_libraries (SampleProject 
		Eigen3::Eigen)
		
	target_link_libraries(SampleProject 
		${OpenCV_LIBS}
		${PROJECT_NAME}
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
