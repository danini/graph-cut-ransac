cmake_minimum_required(VERSION 3.8)

project(GraphCut_RANSAC LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 3.3 REQUIRED)

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
	${PROJECT_SOURCE_DIR}/include/GraphCut_RANSAC
)

# ==============================================================================
# Structure: Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_GCRANSAC
	"include/GraphCut_RANSAC/*.h"
	"include/GraphCut_RANSAC/*.cpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GCRANSAC 
	"src/GraphCut_RANSAC/*.cpp"
)

source_group(third_pary\\GCOptimization FILES
	src/GraphCut_RANSAC/GCoptimization.cpp
	include/GraphCut_RANSAC/graph.cpp
	src/GraphCut_RANSAC/LinkedBlockList.cpp
	include/GraphCut_RANSAC/maxflow.cpp
	include/GraphCut_RANSAC/block.h
	include/GraphCut_RANSAC/energy.h
	include/GraphCut_RANSAC/GCoptimization.h
	include/GraphCut_RANSAC/graph.h
	include/GraphCut_RANSAC/LinkedBlockList.h
)

source_group(third_pary\\Theia FILES
	include/GraphCut_RANSAC/prosac.h
	include/GraphCut_RANSAC/prosac_sampler.h
	include/GraphCut_RANSAC/quality_measurement.h
	include/GraphCut_RANSAC/sample_consensus_estimator.h
	include/GraphCut_RANSAC/sampler.h
	include/GraphCut_RANSAC/inline_support.h
	include/GraphCut_RANSAC/estimator.h
)

add_library(${PROJECT_NAME} STATIC
	${HDRS_GCRANSAC}
	${SRCS_GCRANSAC}
)

target_link_libraries(${PROJECT_NAME} 
	${OpenCV_LIBS}
)

# ==============================================================================
# Structure: Applications
# ==============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


# five_point
file(GLOB_RECURSE HDRSSRCS_FIVE_POINT 
	"src/Sample_FundamentalEstimator/five_point/*.*"
)
source_group(third_pary\\five_point FILES
	${HDRSSRCS_FIVE_POINT}
)

# estimators
source_group("Estimators" FILES
	src/Sample_FundamentalEstimator/essential_estimator.h
	src/Sample_FundamentalEstimator/fundamental_estimator.h
	src/Sample_FundamentalEstimator/homography_estimator.h
	src/Sample_FundamentalEstimator/line_estimator.h
)


add_executable(Sample_FundamentalEstimator
	${HDRSSRCS_FIVE_POINT}
	src/Sample_FundamentalEstimator/essential_estimator.h
	src/Sample_FundamentalEstimator/fundamental_estimator.h
	src/Sample_FundamentalEstimator/homography_estimator.h
	src/Sample_FundamentalEstimator/line_estimator.h
	src/Sample_FundamentalEstimator/main.cpp)
	
target_link_libraries(Sample_FundamentalEstimator 
	${OpenCV_LIBS}
	${PROJECT_NAME}
)
add_dependencies(Sample_FundamentalEstimator ${PROJECT_NAME})
