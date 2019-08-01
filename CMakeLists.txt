cmake_minimum_required(VERSION 3.8)

project(GraphCutRANSAC LANGUAGES CXX)

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
find_package(OpenCV 3.0 REQUIRED)

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
	"include/*.cpp"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_GCRANSAC 
	"src/*.cpp"
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

source_group(third_party\\Theia FILES
	include/prosac_sampler.h
	include/sampler.h
	include/estimator.h
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
	"src/five_point/*.h"
	"src/five_point/*.cpp"
)
source_group(third_party\\five_point FILES
	${HDRSSRCS_FIVE_POINT}
)

# estimators
source_group("Estimators" FILES
	src/essential_estimator.h
	src/fundamental_estimator.h
	src/homography_estimator.h
)

add_executable(SampleProject
	${HDRSSRCS_FIVE_POINT}
	src/essential_estimator.h
	src/fundamental_estimator.h
	src/homography_estimator.h
	src/main.cpp)
	
target_link_libraries(SampleProject 
	${OpenCV_LIBS}
	${PROJECT_NAME}
)
add_dependencies(SampleProject ${PROJECT_NAME})
